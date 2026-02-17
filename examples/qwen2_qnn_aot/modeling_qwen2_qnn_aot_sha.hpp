// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// The optimization splits large Q/K/V projections into per-head projections,
// allowing QNN to optimize each head separately, reducing AOT compilation time
// and improving HTP performance.

#pragma once

#include "mllm/core/TensorStorage.hpp"
#include "mllm/mllm.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/utils/Enumerate.hpp"
#include "mllm/compile/ir/Trace.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/models/qwen3/configuration_qwen3.hpp"

namespace mllm::models::qwen2::sha {

namespace ptq {

Tensor QDQ_CONSTANT(nn::Module* m, Tensor in, const std::string& qdq_name_in_pytorch) {
  std::string scale_name = qdq_name_in_pytorch + ".scale";
  std::string zp_name = qdq_name_in_pytorch + ".zero_point";
  switch (in.dtype()) {
    case kFloat32:
    case kUInt16PerTensorAsy: {
      auto scale = m->getTopParameterFile()->pull(scale_name);
      auto zp = m->getTopParameterFile()->pull(zp_name);
      in.attach("scale", scale.impl(), true);
      in.attach("zero_point", zp.impl(), true);
      break;
    }
    default: {
      MLLM_ERROR_EXIT(ExitCode::kCoreError, "Can't Process dtype={}", nameOfType(in.dtype()));
    }
  }
  return in;
}

Tensor QDQ(nn::Module* m, Tensor in, const std::string& qdq_name_in_pytorch) {
  std::string scale_name = m->getModuleName() + "." + qdq_name_in_pytorch + ".fake_quant.scale";
  std::string zp_name = m->getModuleName() + "." + qdq_name_in_pytorch + ".fake_quant.zero_point";

  if (m->getModuleName().empty()) {
    scale_name = qdq_name_in_pytorch + ".fake_quant.scale";
    zp_name = qdq_name_in_pytorch + ".fake_quant.zero_point";
  } else {
    scale_name = m->getModuleName() + "." + qdq_name_in_pytorch + ".fake_quant.scale";
    zp_name = m->getModuleName() + "." + qdq_name_in_pytorch + ".fake_quant.zero_point";
  }

  switch (in.dtype()) {
    case kUInt16PerTensorAsy: {
      auto scale = m->getTopParameterFile()->pull(scale_name);
      auto zp = m->getTopParameterFile()->pull(zp_name);
      in.attach("scale", scale.impl(), true);
      in.attach("zero_point", zp.impl(), true);
      break;
    }
    // For Constant!
    case kFloat32: {
      MLLM_RT_ASSERT_EQ(in.rank(), 1);
      MLLM_RT_ASSERT_EQ(in.size(-1), 1);
      auto scale = m->getTopParameterFile()->pull(scale_name);
      auto zp = m->getTopParameterFile()->pull(zp_name);
      in.attach("scale", scale.impl(), true);
      in.attach("zero_point", zp.impl(), true);
      break;
    }
    default: {
      MLLM_ERROR_EXIT(ExitCode::kCoreError, "Can't Process dtype={}", nameOfType(in.dtype()));
    }
  }

  return in;
}

Tensor QDQ_KV(nn::Module* m, Tensor in, const std::string& qdq_name_in_pytorch) {
  auto scale_name = m->getModuleName() + "." + qdq_name_in_pytorch + ".fake_quant.scale";
  auto zp_name = m->getModuleName() + "." + qdq_name_in_pytorch + ".fake_quant.zero_point";

  // The inputs is int8 sym. which means zero_point should be changed.
  switch (in.dtype()) {
    case kUInt8PerTensorSym: {
      auto scale = m->getTopParameterFile()->pull(scale_name);
      auto zp = m->getTopParameterFile()->pull(zp_name);
      MLLM_RT_ASSERT_EQ(zp.item<mllm_int32_t>(), 128);

      // Is 128! not 127!
      auto new_zp = Tensor::constant(128, kInt32).setName(zp_name).setMemType(kParamsNormal);
      in.attach("scale", scale.impl(), true);
      in.attach("zero_point", new_zp.impl(), true);
      break;
    }
    default: {
      MLLM_ERROR_EXIT(ExitCode::kCoreError, "Can't Process dtype={}", nameOfType(in.dtype()));
    }
  }

  return in;
}

Tensor QDQ_ROPE(nn::Module* m, Tensor in, const std::string& qdq_name_in_pytorch) {
  auto scale_name = m->getModuleName() + "." + qdq_name_in_pytorch + ".fake_quant.scale";
  auto zp_name = m->getModuleName() + "." + qdq_name_in_pytorch + ".fake_quant.zero_point";

  (void)in.__unsafeSetDType(kUInt16PerTensorAsy);

  switch (in.dtype()) {
    case kUInt16PerTensorAsy: {
      auto scale = m->getTopParameterFile()->pull(scale_name);
      auto zp = m->getTopParameterFile()->pull(zp_name);
      in.attach("scale", scale.impl(), true);
      in.attach("zero_point", zp.impl(), true);
      break;
    }
    default: {
      MLLM_ERROR_EXIT(ExitCode::kCoreError, "Can't Process dtype={}", nameOfType(in.dtype()));
    }
  }

  return in;
}

}  // namespace ptq

Tensor rotateHalf(Tensor x, nn::Module* m, const std::string& qdq_name_in_pytorch) {  // NOLINT
  // X is [x, x, x, D]
  auto D = x.size(-1);
  auto x1 = x.slice({kAll, kAll, kAll, {kAll, D / 2}}, /*ssa=*/true);
  auto x2 = x.slice({kAll, kAll, kAll, {D / 2, kAll}}, /*ssa=*/true);
  return nn::functional::concat({ptq::QDQ(m, -x2, qdq_name_in_pytorch), x1}, -1);
}

using vi32 = std::vector<int32_t>;
#define CONV2D_PROPERTY vi32{1, 1}, vi32{1, 1}, vi32{0, 0}, vi32{1, 1}, false, aops::Conv2DOpImplType::kQNN_LPBQ_w4a16o16_G32

// Using Conv2D to replace Linear.
// Conv2D Filter Weight is [1, 1, In, Out]
// Conv2D Activation is [N, H=1, W=Seq, In]

class Qwen2MLP final : public nn::Module {
  nn::Conv2D gate_proj_;
  nn::Conv2D up_proj_;
  nn::Conv2D down_proj_;
  nn::SiLU silu_;
  int hidden_size_;
  int intermediate_size_;

 public:
  Qwen2MLP() = default;
  Qwen2MLP(const std::string& name, const qwen3::Qwen3Config& cfg) : nn::Module(name) {
    gate_proj_ = reg<nn::Conv2D>("gate_proj", cfg.hidden_size, cfg.intermediate_size, CONV2D_PROPERTY);
    silu_ = reg<nn::SiLU>("act");
    up_proj_ = reg<nn::Conv2D>("up_proj", cfg.hidden_size, cfg.intermediate_size, CONV2D_PROPERTY);
    down_proj_ = reg<nn::Conv2D>("down_proj", cfg.intermediate_size, cfg.hidden_size, CONV2D_PROPERTY);
    hidden_size_ = cfg.hidden_size;
    intermediate_size_ = cfg.intermediate_size;
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = inputs[0];
    x = ptq::QDQ(this, x, "up_proj_input_qdq");
    x = x.view({1, 1, -1, hidden_size_}, true);

    auto up_result = ptq::QDQ(this, up_proj_(x), "up_proj_output_qdq").view({1, -1, intermediate_size_}, true);
    auto gate_result = ptq::QDQ(this, gate_proj_(x), "gate_proj_output_qdq").view({1, -1, intermediate_size_}, true);

    // SiLU
    gate_result = ptq::QDQ(this, (gate_result * ptq::QDQ(this, nn::functional::sigmoid(gate_result), "sigmoid_output_qdq")),
                           "act_output_qdq");

    auto o = ptq::QDQ(this, gate_result * up_result, "down_proj_input_qdq");
    o = o.view({1, 1, -1, intermediate_size_}, true);
    o = down_proj_(o).view({1, -1, hidden_size_}, true);

    return {o};
  }
};

// ============================================================================
// Single Head Attention (SHA) Implementation
// ============================================================================
//
// This class implements SHA where each attention head has its own separate
// Conv2D projection, instead of one large MHA projection that processes all
// heads at once.
//
// Benefits:
// 1. Reduces QNN AOT compilation time
// 2. Improves HTP runtime performance
// 3. Enables better memory locality per head
//
// Note: Qwen2 does NOT have RMSNorm after Q/K projection (unlike Qwen3)

class Qwen2AttentionSHA final : public nn::Module {
  // Per-head Q projections: num_attention_heads Conv2D(hidden_size, head_dim)
  std::vector<nn::Conv2D> q_projs_;
  // Per-head K projections: num_key_value_heads Conv2D(hidden_size, head_dim)
  std::vector<nn::Conv2D> k_projs_;
  // Per-head V projections: num_key_value_heads Conv2D(hidden_size, head_dim)
  std::vector<nn::Conv2D> v_projs_;
  // Single O projection remains unchanged (concatenated heads -> hidden_size)
  nn::Conv2D o_proj_;

  nn::CausalMask mask_;
  nn::Softmax softmax_;

  int hidden_size_;
  int head_dim_;
  int num_attention_heads_;
  int num_key_value_heads_;
  int num_key_value_groups_;
  float scale_;

 public:
  Qwen2AttentionSHA() = default;

  Qwen2AttentionSHA(const std::string& name, const qwen3::Qwen3Config& cfg) : nn::Module(name) {
    hidden_size_ = cfg.hidden_size;
    num_attention_heads_ = cfg.num_attention_heads;
    num_key_value_heads_ = cfg.num_key_value_heads;
    head_dim_ = cfg.head_dim;
    num_key_value_groups_ = num_attention_heads_ / num_key_value_heads_;
    scale_ = (1.f / sqrtf((float)head_dim_));

    // Register per-head Q projections
    for (int h = 0; h < num_attention_heads_; ++h) {
      q_projs_.emplace_back(reg<nn::Conv2D>("q_proj." + std::to_string(h), hidden_size_, head_dim_, CONV2D_PROPERTY));
    }

    // Register per-head K projections
    for (int h = 0; h < num_key_value_heads_; ++h) {
      k_projs_.emplace_back(reg<nn::Conv2D>("k_proj." + std::to_string(h), hidden_size_, head_dim_, CONV2D_PROPERTY));
    }

    // Register per-head V projections
    for (int h = 0; h < num_key_value_heads_; ++h) {
      v_projs_.emplace_back(reg<nn::Conv2D>("v_proj." + std::to_string(h), hidden_size_, head_dim_, CONV2D_PROPERTY));
    }

    // O projection remains the same (combines all heads)
    o_proj_ = reg<nn::Conv2D>("o_proj", head_dim_ * num_attention_heads_, hidden_size_, CONV2D_PROPERTY);

    mask_ = reg<nn::CausalMask>("mask");
    softmax_ = reg<nn::Softmax>("softmax", -1);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden_states = inputs[0];
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto causal_mask = inputs[3];
    const auto& past_key = inputs[4];    // [B, num_kv_heads, D, S]
    const auto& past_value = inputs[5];  // [B, num_kv_heads, S, D]

    // [B, S, D] - shared QDQ for input to all Q/K/V projections
    hidden_states = ptq::QDQ(this, hidden_states, "q_proj_input_qdq");
    hidden_states = hidden_states.view({1, 1, -1, hidden_size_}, true);

    // ========================================================================
    // Per-head Q/K/V Projections
    // ========================================================================
    // This is the key SHA optimization: instead of one large projection for all
    // heads, we have separate smaller projections per head.

    // Compute per-head Q projections: each outputs [1, 1, S, head_dim]
    std::vector<Tensor> query_states_per_head;
    for (int h = 0; h < num_attention_heads_; ++h) {
      auto q_h = q_projs_[h](hidden_states);
      q_h = q_h.view({1, 1, -1, head_dim_}, /*ssa=*/true);
      query_states_per_head.push_back(q_h);
    }

    // Compute per-head K projections
    std::vector<Tensor> key_states_per_head;
    for (int h = 0; h < num_key_value_heads_; ++h) {
      auto k_h = k_projs_[h](hidden_states);
      k_h = k_h.view({1, 1, -1, head_dim_}, /*ssa=*/true);
      key_states_per_head.push_back(k_h);
    }

    // Compute per-head V projections
    std::vector<Tensor> value_states_per_head;
    for (int h = 0; h < num_key_value_heads_; ++h) {
      auto v_h = v_projs_[h](hidden_states);
      v_h = v_h.view({1, 1, -1, head_dim_}, /*ssa=*/true);
      value_states_per_head.push_back(v_h);
    }

    // ========================================================================
    // Reshape and Transpose for RoPE
    // ========================================================================
    // Qwen2 does NOT have RMSNorm here (unlike Qwen3)
    // Directly apply RoPE after reshaping to [B, H, S, D] format
    // Each head tensor is [1, 1, S, head_dim], need to reshape to [1, 1, S, head_dim] for RoPE
    // (The shape is already correct, but we need to ensure QDQ is applied)

    auto cos = llm_embedding_cos.unsqueeze(1, true);
    auto sin = llm_embedding_sin.unsqueeze(1, true);

    // Apply QDQ and RoPE per Q head
    // Each query_states_per_head[h] is [1, 1, S, head_dim]
    for (int h = 0; h < num_attention_heads_; ++h) {
      std::string h_str = std::to_string(h);
      query_states_per_head[h] = ptq::QDQ(this, query_states_per_head[h], "q_proj_output_qdq_h" + h_str);
      // Reshape to [1, 1, S, head_dim] for RoPE (already correct shape)
      query_states_per_head[h] =
          ptq::QDQ(this,
                   ptq::QDQ(this, query_states_per_head[h] * cos, "q_rope_mul_0_output_qdq_h" + h_str)
                       + ptq::QDQ(this, rotateHalf(query_states_per_head[h], this, "q_rope_neg_half_qdq_h" + h_str) * sin,
                                  "q_rope_mul_1_output_qdq_h" + h_str),
                   "q_rope_add_0_output_qdq_h" + h_str);
    }

    // Apply QDQ and RoPE per K head
    // Each key_states_per_head[h] is [1, 1, S, head_dim]
    for (int h = 0; h < num_key_value_heads_; ++h) {
      std::string h_str = std::to_string(h);
      key_states_per_head[h] = ptq::QDQ(this, key_states_per_head[h], "k_proj_output_qdq_h" + h_str);
      // Reshape to [1, 1, S, head_dim] for RoPE (already correct shape)
      key_states_per_head[h] =
          ptq::QDQ(this,
                   ptq::QDQ(this, key_states_per_head[h] * cos, "k_rope_mul_0_output_qdq_h" + h_str)
                       + ptq::QDQ(this, rotateHalf(key_states_per_head[h], this, "k_rope_neg_half_qdq_h" + h_str) * sin,
                                  "k_rope_mul_1_output_qdq_h" + h_str),
                   "k_rope_add_0_output_qdq_h" + h_str);
    }

    // ========================================================================
    // KV Cache Processing per head
    // ========================================================================

    std::vector<Tensor> new_key_per_head;
    std::vector<Tensor> new_value_per_head;
    std::vector<Tensor> key_cache_per_head;
    std::vector<Tensor> value_cache_per_head;

    for (int h = 0; h < num_key_value_heads_; ++h) {
      std::string h_str = std::to_string(h);

      // K: De-quantize and re-quantize to int8
      auto k_h = key_states_per_head[h].to(kFloat32);
      k_h = k_h.to(kUInt8PerTensorSym);
      k_h = ptq::QDQ_KV(this, k_h, "k_cast_to_int8_qdq_h" + h_str);
      k_h = k_h.transpose(2, 3);  // [B, 1, D, S]

      // V: Quantize to int16 then int8
      auto v_h = ptq::QDQ(this, value_states_per_head[h], "v_cast_to_int16_qdq_h" + h_str);
      v_h = v_h.to(kFloat32);
      v_h = v_h.to(kUInt8PerTensorSym);
      v_h = ptq::QDQ_KV(this, v_h, "v_cast_to_int8_qdq_h" + h_str);

      new_key_per_head.push_back(k_h);
      new_value_per_head.push_back(v_h);

      // Slice past cache for this head
      auto past_k_h = past_key.slice({kAll, {h, h + 1}, kAll, kAll}, true);
      auto past_v_h = past_value.slice({kAll, {h, h + 1}, kAll, kAll}, true);

      // Concat current with past
      key_cache_per_head.push_back(nn::functional::concat({past_k_h, k_h}, -1));
      value_cache_per_head.push_back(nn::functional::concat({past_v_h, v_h}, 2));
    }

    // ========================================================================
    // Per-head Attention Computation
    // ========================================================================
    // Each Q head computes attention with its corresponding KV head (GQA support)
    // For GQA, multiple Q heads share the same KV head

    std::vector<Tensor> attn_outputs;

    for (int h = 0; h < num_attention_heads_; ++h) {
      std::string h_str = std::to_string(h);
      int kv_head_idx = h / num_key_value_groups_;

      const auto& q_h = query_states_per_head[h];
      const auto& kh = key_cache_per_head[kv_head_idx];
      const auto& vh = value_cache_per_head[kv_head_idx];

      // QK^T
      auto attn = ptq::QDQ(this, nn::functional::matmul(q_h, kh), "qk_matmul_output_qdq_h" + h_str);

      // Scale
      auto scale = Tensor::constant(scale_, kFloat32);
      scale = ptq::QDQ(this, scale, "scaling_qdq_h" + h_str);
      attn = ptq::QDQ(this, attn.mulConstant(scale), "mul_0_output_qdq_h" + h_str);

      // Masked Softmax
      auto attn_min = ptq::QDQ(this, attn.min(-1, true), "reduce_min_output_qdq_h" + h_str);
      auto minus_value = Tensor::constant(-20, kFloat32);
      minus_value = ptq::QDQ(this, minus_value, "neg_20_qdq_h" + h_str);
      auto attn_vv = ptq::QDQ(this, attn_min.addConstant(minus_value), "minus_0_output_qdq_h" + h_str);
      auto zero_constant = Tensor::constant(0.f, kFloat32);
      zero_constant = ptq::QDQ_CONSTANT(this, zero_constant, "constant_zero");
      attn = nn::functional::where(causal_mask.equalConstant(zero_constant), attn, attn_vv);
      attn = ptq::QDQ(this, attn, "where_attn_qdq_h" + h_str);
      attn = ptq::QDQ(this, nn::functional::softmax(attn, -1), "softmax_output_qdq_h" + h_str);

      // Output: attn @ V
      auto y_h = ptq::QDQ(this, nn::functional::matmul(attn, vh), "attn_value_matmul_output_qdq_h" + h_str);
      attn_outputs.push_back(y_h);
    }

    // ========================================================================
    // Concatenate and Output Projection
    // ========================================================================

    // Concat all head outputs: [B, num_heads, S, D]
    auto y = nn::functional::concat(attn_outputs, 1);

    // Reshape and apply O projection
    y = y.transpose(1, 2).view({1, 1, -1, num_attention_heads_ * head_dim_}, /*ssa=*/true);
    y = o_proj_(y).view({1, -1, hidden_size_}, true);

    // Concat new keys and values back to original format
    auto new_key = nn::functional::concat(new_key_per_head, 1);
    auto new_value = nn::functional::concat(new_value_per_head, 1);

    return {y, new_key, new_value};
  }

  int layer_idx_;
};

class Qwen2DecoderSHA final : public nn::Module {
 public:
  int layer_idx_;
  Qwen2AttentionSHA self_attn_;
  Qwen2MLP mlp_;
  nn::RMSNorm input_layer_norm_;
  nn::RMSNorm post_attention_layer_norm_;

  Qwen2DecoderSHA() = default;

  Qwen2DecoderSHA(const std::string& name, const qwen3::Qwen3Config& cfg, int layer_idx) : nn::Module(name) {
    layer_idx_ = layer_idx;
    self_attn_ = reg<Qwen2AttentionSHA>("self_attn", cfg);
    mlp_ = reg<Qwen2MLP>("mlp", cfg);
    input_layer_norm_ = reg<nn::RMSNorm>("input_layernorm", cfg.rms_norm_eps);
    post_attention_layer_norm_ = reg<nn::RMSNorm>("post_attention_layernorm", cfg.rms_norm_eps);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto causal_mask = inputs[3];
    auto past_key = inputs[4];
    auto past_value = inputs[5];

    auto hidden_states = inputs[0];
    if (layer_idx_ != 0) { hidden_states = ptq::QDQ(this, hidden_states, "input_layernorm_input_qdq"); }
    auto residual = hidden_states;
    hidden_states = input_layer_norm_(hidden_states);
    auto _ = self_attn_(hidden_states, llm_embedding_sin, llm_embedding_cos, causal_mask, past_key, past_value);
    hidden_states = _[0];
    hidden_states = ptq::QDQ(this, residual + ptq::QDQ(this, hidden_states, "add_0_lhs_input_qdq"), "add_0_output_qdq");
    residual = hidden_states;
    hidden_states = post_attention_layer_norm_(hidden_states);
    hidden_states = mlp_(hidden_states)[0];
    hidden_states = residual + ptq::QDQ(this, hidden_states, "add_1_lhs_input_qdq");
    return {hidden_states, _[1], _[2]};
  }
};

class Qwen2TextSHA final : public nn::Module {
  nn::ModuleListWithIdx<Qwen2DecoderSHA> decode_blocks_;
  nn::RMSNorm norm_;
  nn::Embedding embedding_;
  nn::Param rope_sin_;
  nn::Param rope_cos_;
  int32_t num_hidden_layers_;
  int32_t hidden_size_;

 public:
  Qwen2TextSHA() = default;

  Qwen2TextSHA(const std::string& name, const qwen3::Qwen3Config& cfg) : nn::Module(name) {
    num_hidden_layers_ = cfg.num_hidden_layers;
    hidden_size_ = cfg.hidden_size;
    decode_blocks_ = reg<nn::ModuleListWithIdx<Qwen2DecoderSHA>>("layers", cfg.num_hidden_layers, cfg);
    for (auto [idx, b] : enumerate(decode_blocks_.list())) { b.self_attn_.layer_idx_ = idx; }
    norm_ = reg<nn::RMSNorm>("norm", cfg.rms_norm_eps);
    embedding_ = reg<nn::Embedding>("embed_tokens", cfg.vocab_size, cfg.hidden_size);
    rope_sin_ = reg<nn::Param>("mllm_max_sin_embedding", "model.mllm_max_sin_embedding");
    rope_cos_ = reg<nn::Param>("mllm_max_cos_embedding", "model.mllm_max_cos_embedding");
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto& blocks = decode_blocks_.list();

    // X is already embedded
    auto x = embedding_(inputs[0]);

    const auto& position_ids = inputs[1];
    auto causal_mask = inputs[2];

    // clang-format off
    auto llm_embedding_sin = nn::functional::gather(ptq::QDQ_ROPE(this, rope_sin_(), "sin_embedding_input_qdq"), 1, position_ids);
    auto llm_embedding_cos = nn::functional::gather(ptq::QDQ_ROPE(this, rope_cos_(), "cos_embedding_input_qdq"), 1, position_ids);
    // clang-format on

    std::vector<Tensor> keys;
    std::vector<Tensor> values;
    for (auto [index, block] : enumerate(blocks)) {
      auto pk = inputs[3 + index];
      auto pv = inputs[3 + index + num_hidden_layers_];
      auto _ = block(x, llm_embedding_sin, llm_embedding_cos, causal_mask, pk, pv);
      x = _[0];
      keys.push_back(_[1]);
      values.push_back(_[2]);
    }

    x = norm_(ptq::QDQ(this, x, "norm_input_qdq"));
    x = x.view({1, 1, -1, hidden_size_}, true);

    auto ret = std::vector<Tensor>{x};
    for (const auto& item : keys) { ret.push_back(item); }
    for (const auto& item : values) { ret.push_back(item); }

    return ret;
  }
};

class Qwen2ForCausalLM_SHA : public ARGeneration, public nn::Module {
 public:
  explicit Qwen2ForCausalLM_SHA(const qwen3::Qwen3Config& cfg) : cfg(cfg) {
    eos_token_id_ = cfg.end_of_text_token_id;
    max_length_ = cfg.max_cache_length;
    tie_word_embeddings_ = cfg.tie_word_embeddings;

    llm = reg<Qwen2TextSHA>("model", cfg);

    if (cfg.tie_word_embeddings) {
      // NOTE:
      // model.lm_head.weight is quantization weights of model.embed_tokens.weight
      lm_head_ = reg<nn::Conv2D>("lm_head", cfg.hidden_size, cfg.vocab_size, CONV2D_PROPERTY);
    }
  }

  IROutput trace(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override {
    // Things we need to return
    ir::IRContext::ptr_t llm_ir = nullptr;

    auto sequence = input.at("sequence");
    auto causal_mask = input.at("causal_mask");

    std::vector<Tensor> kv_caches;

    // Append Key
    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
      auto past_key_name = "past_key_" + std::to_string(i);
      if (input.count(past_key_name)) {
        kv_caches.push_back(input.at(past_key_name));
      } else {
        throw std::runtime_error("Missing KV cache for layer " + std::to_string(i));
      }
    }

    // Append Value
    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
      auto past_value_name = "past_value_" + std::to_string(i);
      if (input.count(past_value_name)) {
        kv_caches.push_back(input.at(past_value_name));
      } else {
        throw std::runtime_error("Missing KV cache for layer " + std::to_string(i));
      }
    }

    // Generate position_ids for the current sequence
    auto batch_size = sequence.shape()[0];
    auto seq_len = sequence.shape()[1];

    Tensor position_ids = Tensor::nil();
    if (input.count("position_ids")) {
      // Use existing position_ids for decode phase
      position_ids = input.at("position_ids");

      // For decode phase, increment the last position
      if (seq_len == 1) {
        auto last_pos = *position_ids.offsettedPtr<int32_t>({position_ids.shape()[1] - 1});
        position_ids = Tensor::empty({1}, kInt32, kCPU).alloc();
        *position_ids.offsettedPtr<int32_t>({0}) = last_pos + 1;
      }
    } else {
      // Generate position_ids for prefill phase
      position_ids = Tensor::empty({seq_len}, kInt32, kCPU).alloc();
      auto position_ids_ptr = position_ids.ptr<int32_t>();
      for (int s = 0; s < seq_len; ++s) { position_ids_ptr[s] = s; }
    }

    ir::lowlevel::traceStart();

    // Build inputs for llm: sequence, llm_embedding_sin, llm_embedding_cos, causal_mask, then all KV caches
    std::vector<Tensor> llm_inputs = {sequence, position_ids, causal_mask};
    llm_inputs.insert(llm_inputs.end(), kv_caches.begin(), kv_caches.end());

    sequence = llm(llm_inputs)[0];
    sequence = lm_head_(ptq::QDQ(this, sequence, "lm_head_input_qdq"));
    sequence = ptq::QDQ(this, sequence, "lm_head_output_qdq");
    ir::lowlevel::traceComment("    ╔═════╗   ");
    ir::lowlevel::traceComment("   ║  o o  ║  ");
    ir::lowlevel::traceComment("   ║   ▽   ║  ");
    ir::lowlevel::traceComment("   ╚═════╝   ");
    ir::lowlevel::traceComment("    ║   ║     ");
    ir::lowlevel::traceComment("   ╱╩╦╦╩╲    ");
    llm_ir = ir::lowlevel::traceStop();

    return {{"model", llm_ir}};
  }

  ARGenerationOutputPast forward(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override { return {}; }

 private:
  const qwen3::Qwen3Config& cfg;
  Qwen2TextSHA llm;
  nn::Conv2D lm_head_;
  bool tie_word_embeddings_;
};

// ============================================================================
// Weight Slicing Utilities for SHA
// ============================================================================
//
// These functions are used during the compile phase to slice the original
// MHA weights into per-head SHA weights.
//
// Note: Qwen2 does NOT have q_norm and k_norm (RMSNorm), so we don't need
// to slice those parameters.

/**
 * @brief Prepares the parameter file by slicing MHA weights into SHA weights.
 *
 * This function takes the original parameter file with MHA weights and creates
 * new per-head weights for the SHA model.
 *
 * Original weight layout for Conv2D: [out_channels, in_channels, 1, 1]
 * - q_proj.weight: [num_heads * head_dim, hidden_size, 1, 1]
 * - k_proj.weight: [num_kv_heads * head_dim, hidden_size, 1, 1]
 * - v_proj.weight: [num_kv_heads * head_dim, hidden_size, 1, 1]
 *
 * For LPBQ quantization, also need to slice:
 * - scale1: flattened scale for block quantization
 * - scale2: flattened scale for block quantization
 *
 * SHA weight layout:
 * - q_proj.{h}.weight: [head_dim, hidden_size, 1, 1] for h in [0, num_heads)
 * - q_proj.{h}.scale1: sliced scale for head h
 * - q_proj.{h}.scale2: sliced scale for head h
 * - Similar for k_proj and v_proj
 */
inline void prepareParametersForSHA(const ParameterFile::ptr_t& params, const qwen3::Qwen3Config& cfg) {
  int num_heads = cfg.num_attention_heads;
  int num_kv_heads = cfg.num_key_value_heads;
  int head_dim = cfg.head_dim;
  int num_layers = cfg.num_hidden_layers;

  // Helper lambda to slice and push Conv2D params (weight, scale1, scale2)
  // For LPBQ, scale1 and scale2 are flattened along the output channel dimension
  // Scale size per head = total_scale_size / num_heads_for_this_proj
  auto sliceAndPushConv2DParams = [&](const std::string& orig_name_prefix, const std::string& new_name_prefix,
                                      int total_out_channels, int out_channels_per_head, int num_splits) {
    // Process weight: HWIO format [H=1, W=1, In_channels, Out_channels]
    // For q_proj: [1, 1, hidden_size, num_heads * head_dim]
    // Slice on the last dimension (Out_channels)
    std::string orig_weight_name = orig_name_prefix + ".weight";
    if (params->has(orig_weight_name)) {
      auto orig_weight = params->pull(orig_weight_name);

      for (int h = 0; h < num_splits; ++h) {
        std::string new_weight_name = new_name_prefix + "." + std::to_string(h) + ".weight";
        int start_idx = h * out_channels_per_head;
        int end_idx = (h + 1) * out_channels_per_head;
        // HWIO format: slice on dim 3 (Out_channels)
        auto sliced = orig_weight.slice({kAll, kAll, kAll, {start_idx, end_idx}}, false);
        params->push(new_weight_name, sliced.contiguous().setMemType(kParamsNormal).setName(new_weight_name));
      }
    }

    // Process scale1: flattened, size = total_out_channels / block_size (or similar)
    // Slice index: (total_scale_size / num_splits) * h
    std::string orig_scale1_name = orig_name_prefix + ".scale1";
    if (params->has(orig_scale1_name)) {
      auto orig_scale1 = params->pull(orig_scale1_name);
      int total_scale_size = orig_scale1.numel();
      int scale_per_head = total_scale_size / num_splits;

      for (int h = 0; h < num_splits; ++h) {
        std::string new_scale1_name = new_name_prefix + "." + std::to_string(h) + ".scale1";
        int start_idx = h * scale_per_head;
        int end_idx = (h + 1) * scale_per_head;
        auto sliced = orig_scale1.slice({{start_idx, end_idx}}, false);
        params->push(new_scale1_name, sliced.contiguous().setMemType(kParamsNormal).setName(new_scale1_name));
      }
    }

    // Process scale2: flattened, same logic as scale1
    std::string orig_scale2_name = orig_name_prefix + ".scale2";
    if (params->has(orig_scale2_name)) {
      auto orig_scale2 = params->pull(orig_scale2_name);
      int total_scale_size = orig_scale2.numel();
      int scale_per_head = total_scale_size / num_splits;

      for (int h = 0; h < num_splits; ++h) {
        std::string new_scale2_name = new_name_prefix + "." + std::to_string(h) + ".scale2";
        int start_idx = h * scale_per_head;
        int end_idx = (h + 1) * scale_per_head;
        auto sliced = orig_scale2.slice({{start_idx, end_idx}}, false);
        params->push(new_scale2_name, sliced.contiguous().setMemType(kParamsNormal).setName(new_scale2_name));
      }
    }
  };

  for (int layer = 0; layer < num_layers; ++layer) {
    std::string layer_prefix = "model.layers." + std::to_string(layer) + ".self_attn.";

    // Process Q projection: split into num_heads parts
    sliceAndPushConv2DParams(layer_prefix + "q_proj", layer_prefix + "q_proj", num_heads * head_dim, head_dim, num_heads);

    // Process K projection: split into num_kv_heads parts
    sliceAndPushConv2DParams(layer_prefix + "k_proj", layer_prefix + "k_proj", num_kv_heads * head_dim, head_dim, num_kv_heads);

    // Process V projection: split into num_kv_heads parts
    sliceAndPushConv2DParams(layer_prefix + "v_proj", layer_prefix + "v_proj", num_kv_heads * head_dim, head_dim, num_kv_heads);

    // ========================================================================
    // Duplicate QDQ parameters for each head
    // ========================================================================
    // The original MHA uses shared QDQ params for all heads. For SHA, we
    // duplicate these to per-head versions using "_h{N}" suffix naming.
    // This allows each head to have its own quantization parameters.

    auto copyQDQParams = [&](const std::string& base_name, const std::string& new_base_name, int count) {
      std::string scale_name = layer_prefix + base_name + ".fake_quant.scale";
      std::string zp_name = layer_prefix + base_name + ".fake_quant.zero_point";

      if (params->has(scale_name)) {
        auto scale = params->pull(scale_name);
        auto zp = params->pull(zp_name);

        for (int h = 0; h < count; ++h) {
          std::string new_scale_name = layer_prefix + new_base_name + std::to_string(h) + ".fake_quant.scale";
          std::string new_zp_name = layer_prefix + new_base_name + std::to_string(h) + ".fake_quant.zero_point";
          // QDQ scale/zp are typically scalar or small tensors, clone to ensure contiguous
          params->push(new_scale_name, scale.contiguous().setMemType(kParamsNormal).setName(new_scale_name));
          params->push(new_zp_name, zp.contiguous().setMemType(kParamsNormal).setName(new_zp_name));
        }
      }
    };

    // Copy QDQ params for Q-related nodes (per Q head)
    copyQDQParams("q_proj_output_qdq", "q_proj_output_qdq_h", num_heads);
    copyQDQParams("q_rope_mul_0_output_qdq", "q_rope_mul_0_output_qdq_h", num_heads);
    copyQDQParams("q_rope_mul_1_output_qdq", "q_rope_mul_1_output_qdq_h", num_heads);
    copyQDQParams("q_rope_neg_half_qdq", "q_rope_neg_half_qdq_h", num_heads);
    copyQDQParams("q_rope_add_0_output_qdq", "q_rope_add_0_output_qdq_h", num_heads);

    // Copy QDQ params for K-related nodes (per KV head)
    copyQDQParams("k_proj_output_qdq", "k_proj_output_qdq_h", num_kv_heads);
    copyQDQParams("k_rope_mul_0_output_qdq", "k_rope_mul_0_output_qdq_h", num_kv_heads);
    copyQDQParams("k_rope_mul_1_output_qdq", "k_rope_mul_1_output_qdq_h", num_kv_heads);
    copyQDQParams("k_rope_neg_half_qdq", "k_rope_neg_half_qdq_h", num_kv_heads);
    copyQDQParams("k_rope_add_0_output_qdq", "k_rope_add_0_output_qdq_h", num_kv_heads);
    copyQDQParams("k_cast_to_int8_qdq", "k_cast_to_int8_qdq_h", num_kv_heads);

    // Copy QDQ params for V-related nodes (per KV head)
    copyQDQParams("v_cast_to_int16_qdq", "v_cast_to_int16_qdq_h", num_kv_heads);
    copyQDQParams("v_cast_to_int8_qdq", "v_cast_to_int8_qdq_h", num_kv_heads);

    // Copy QDQ params for attention computation (per Q head)
    copyQDQParams("qk_matmul_output_qdq", "qk_matmul_output_qdq_h", num_heads);
    copyQDQParams("scaling_qdq", "scaling_qdq_h", num_heads);
    copyQDQParams("mul_0_output_qdq", "mul_0_output_qdq_h", num_heads);
    copyQDQParams("reduce_min_output_qdq", "reduce_min_output_qdq_h", num_heads);
    copyQDQParams("neg_20_qdq", "neg_20_qdq_h", num_heads);
    copyQDQParams("minus_0_output_qdq", "minus_0_output_qdq_h", num_heads);
    copyQDQParams("where_attn_qdq", "where_attn_qdq_h", num_heads);
    copyQDQParams("softmax_output_qdq", "softmax_output_qdq_h", num_heads);
    copyQDQParams("attn_value_matmul_output_qdq", "attn_value_matmul_output_qdq_h", num_heads);
  }
}

}  // namespace mllm::models::qwen2::sha
