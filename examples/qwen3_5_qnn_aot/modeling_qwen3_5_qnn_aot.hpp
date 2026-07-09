// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/mllm.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/compile/ir/Trace.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/models/qwen3_5/configuration_qwen3_5.hpp"

namespace mllm::models::qwen3_5 {

// ============================================================================
// QNN AOT model for Qwen3.5 full attention layers ONLY.
//
// This traces a single full-attention decoder layer into a QNN graph.
// GDN layers, embedding, and lm_head stay on CPU at runtime.
//
// Key differences from Qwen3 QNN AOT:
//   - Partial RoPE: only first rotary_dim=64 of head_dim=256 dims are rotated
//   - Output gating: Q proj is 2x wide, second half is sigmoid gate
//   - Pre-baked RMSNorm: weights already have +1.0 (no add_unit_offset)
// ============================================================================

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
  std::string scale_name, zp_name;
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

  switch (in.dtype()) {
    case kUInt8PerTensorSym: {
      auto scale = m->getTopParameterFile()->pull(scale_name);
      auto zp = m->getTopParameterFile()->pull(zp_name);
      MLLM_RT_ASSERT_EQ(zp.item<mllm_int32_t>(), 128);
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

// Rotate half — same as Qwen3 but only operates on rotary_dim subset
Tensor rotateHalf(Tensor x, nn::Module* m, const std::string& qdq_name_in_pytorch) {  // NOLINT
  auto D = x.size(-1);
  auto x1 = x.slice({kAll, kAll, kAll, {kAll, D / 2}}, /*ssa=*/true);
  auto x2 = x.slice({kAll, kAll, kAll, {D / 2, kAll}}, /*ssa=*/true);
  return nn::functional::concat({ptq::QDQ(m, -x2, qdq_name_in_pytorch), x1}, -1);
}

using vi32 = std::vector<int32_t>;
#define CONV2D_PROPERTY vi32{1, 1}, vi32{1, 1}, vi32{0, 0}, vi32{1, 1}, false, aops::Conv2DOpImplType::kQNN_LPBQ_w4a16o16_G16

// ---------------------------------------------------------------------------
// MLP — identical to Qwen3 pattern
// ---------------------------------------------------------------------------

class Qwen3_5MLP final : public nn::Module {
  nn::Conv2D gate_proj_;
  nn::Conv2D up_proj_;
  nn::Conv2D down_proj_;
  int hidden_size_;
  int intermediate_size_;

 public:
  Qwen3_5MLP() = default;
  Qwen3_5MLP(const std::string& name, const Qwen3_5Config& cfg) : nn::Module(name) {
    gate_proj_ = reg<nn::Conv2D>("gate_proj", cfg.hidden_size, cfg.intermediate_size, CONV2D_PROPERTY);
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

    // SiLU = x * sigmoid(x)
    gate_result = ptq::QDQ(this, (gate_result * ptq::QDQ(this, nn::functional::sigmoid(gate_result), "sigmoid_output_qdq")),
                           "act_output_qdq");

    auto o = ptq::QDQ(this, gate_result * up_result, "down_proj_input_qdq");
    o = o.view({1, 1, -1, intermediate_size_}, true);
    o = down_proj_(o).view({1, -1, hidden_size_}, true);

    return {o};
  }
};

// ---------------------------------------------------------------------------
// Full Attention with partial RoPE and output gating
// ---------------------------------------------------------------------------

class Qwen3_5Attention final : public nn::Module {
  nn::Conv2D q_proj_;
  nn::Conv2D k_proj_;
  nn::Conv2D v_proj_;
  nn::Conv2D o_proj_;
  nn::RMSNorm rms_norm_q_;
  nn::RMSNorm rms_norm_k_;
  nn::CausalMask mask_;
  nn::Softmax softmax_;

  int hidden_size_;
  int head_dim_;
  int rotary_dim_;
  int num_attention_heads_;
  int num_key_value_heads_;
  int num_key_value_groups_;
  bool attn_output_gate_;
  float scale_;

 public:
  int layer_idx_ = 0;

  Qwen3_5Attention() = default;

  Qwen3_5Attention(const std::string& name, const Qwen3_5Config& cfg) : nn::Module(name) {
    hidden_size_ = cfg.hidden_size;
    num_attention_heads_ = cfg.num_attention_heads;
    num_key_value_heads_ = cfg.num_key_value_heads;
    head_dim_ = cfg.head_dim;
    rotary_dim_ = cfg.rotary_dim();
    num_key_value_groups_ = num_attention_heads_ / num_key_value_heads_;
    attn_output_gate_ = cfg.attn_output_gate;
    scale_ = (1.f / sqrtf((float)head_dim_));

    // Q projection is 2x wide when output gating is enabled
    int q_proj_out = head_dim_ * num_attention_heads_;
    if (attn_output_gate_) { q_proj_out *= 2; }

    q_proj_ = reg<nn::Conv2D>("q_proj", hidden_size_, q_proj_out, CONV2D_PROPERTY);
    k_proj_ = reg<nn::Conv2D>("k_proj", hidden_size_, head_dim_ * num_key_value_heads_, CONV2D_PROPERTY);
    v_proj_ = reg<nn::Conv2D>("v_proj", hidden_size_, head_dim_ * num_key_value_heads_, CONV2D_PROPERTY);
    o_proj_ = reg<nn::Conv2D>("o_proj", head_dim_ * num_attention_heads_, hidden_size_, CONV2D_PROPERTY);

    // Pre-baked RMSNorm (no add_unit_offset since weights already have +1.0)
    rms_norm_q_ = reg<nn::RMSNorm>("q_norm", cfg.rms_norm_eps);
    rms_norm_k_ = reg<nn::RMSNorm>("k_norm", cfg.rms_norm_eps);

    mask_ = reg<nn::CausalMask>("mask");
    softmax_ = reg<nn::Softmax>("softmax", -1);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden_states = inputs[0];
    auto llm_embedding_sin = inputs[1];  // [1, S, rotary_dim]
    auto llm_embedding_cos = inputs[2];  // [1, S, rotary_dim]
    auto causal_mask = inputs[3];
    auto past_key = inputs[4];
    auto past_value = inputs[5];

    // [B, S, D]
    hidden_states = ptq::QDQ(this, hidden_states, "q_proj_input_qdq");
    hidden_states = hidden_states.view({1, 1, -1, hidden_size_}, true);

    // Projections via Conv2D
    auto q_raw = q_proj_(hidden_states);
    auto key_states = k_proj_(hidden_states);
    auto value_states = v_proj_(hidden_states);

    // Output gating: split Q into query + gate
    Tensor gate;
    Tensor query_states;
    if (attn_output_gate_) {
      // q_raw: [1, 1, S, num_heads * head_dim * 2] from Conv2D
      q_raw = q_raw.view({1, -1, num_attention_heads_, head_dim_ * 2}, /*ssa=*/true);
      query_states = q_raw.slice({kAll, kAll, kAll, {kAll, head_dim_}}, /*ssa=*/true);
      gate = q_raw.slice({kAll, kAll, kAll, {head_dim_, kAll}}, /*ssa=*/true);
      // gate stays as [B, S, H, D] — we'll use it after attention
      gate = ptq::QDQ(this, gate.transpose(1, 2), "gate_transpose_qdq");  // [B, H, S, D]
    } else {
      query_states = q_raw.view({1, -1, num_attention_heads_, head_dim_}, /*ssa=*/true);
    }

    // [B, H, S, D]
    query_states = query_states.view({1, -1, num_attention_heads_, head_dim_}, /*ssa=*/true).transpose(1, 2);
    key_states = key_states.view({1, -1, num_key_value_heads_, head_dim_}, /*ssa=*/true).transpose(1, 2);
    value_states = value_states.view({1, -1, num_key_value_heads_, head_dim_}, /*ssa=*/true).transpose(1, 2);

    // QK RMSNorm
    query_states = rms_norm_q_(ptq::QDQ(this, query_states, "q_norm_input_qdq"));
    key_states = rms_norm_k_(ptq::QDQ(this, key_states, "k_norm_input_qdq"));

    query_states = ptq::QDQ(this, query_states, "q_norm_output_qdq");
    key_states = ptq::QDQ(this, key_states, "k_norm_output_qdq");

    // ================================================================
    // Partial RoPE: only rotate first rotary_dim=64 of head_dim=256
    // ================================================================
    // Slice out rotary and pass-through parts
    auto q_rot = query_states.slice({kAll, kAll, kAll, {kAll, rotary_dim_}}, /*ssa=*/true);     // [B,H,S,64]
    auto q_pass = query_states.slice({kAll, kAll, kAll, {rotary_dim_, kAll}}, /*ssa=*/true);     // [B,H,S,192]
    auto k_rot = key_states.slice({kAll, kAll, kAll, {kAll, rotary_dim_}}, /*ssa=*/true);
    auto k_pass = key_states.slice({kAll, kAll, kAll, {rotary_dim_, kAll}}, /*ssa=*/true);

    // Apply RoPE to rotary part
    auto cos = llm_embedding_cos.unsqueeze(1, true);  // [1, 1, S, rotary_dim]
    auto sin = llm_embedding_sin.unsqueeze(1, true);

    auto q_rot_applied =
        ptq::QDQ(this,
                 ptq::QDQ(this, q_rot * cos, "q_rope_mul_0_output_qdq")
                     + ptq::QDQ(this, rotateHalf(q_rot, this, "q_rope_neg_half_qdq") * sin, "q_rope_mul_1_output_qdq"),
                 "q_rope_add_0_output_qdq");

    auto k_rot_applied =
        ptq::QDQ(this,
                 ptq::QDQ(this, k_rot * cos, "k_rope_mul_0_output_qdq")
                     + ptq::QDQ(this, rotateHalf(k_rot, this, "k_rope_neg_half_qdq") * sin, "k_rope_mul_1_output_qdq"),
                 "k_rope_add_0_output_qdq");

    // Concat rotated + pass-through back to full head_dim
    query_states = nn::functional::concat({q_rot_applied, ptq::QDQ(this, q_pass, "q_pass_qdq")}, -1);
    key_states = nn::functional::concat({k_rot_applied, ptq::QDQ(this, k_pass, "k_pass_qdq")}, -1);

    query_states = ptq::QDQ(this, query_states, "q_rope_cat_output_qdq");
    key_states = ptq::QDQ(this, key_states, "k_rope_cat_output_qdq");

    // KV quantization for cache
    key_states = key_states.to(kFloat32);
    key_states = key_states.to(kUInt8PerTensorSym);
    key_states = ptq::QDQ_KV(this, key_states, "k_cast_to_int8_qdq");

    // [B, H, D, S]
    key_states = key_states.transpose(2, 3);

    value_states = ptq::QDQ(this, value_states, "v_cast_to_int16_qdq");
    value_states = value_states.to(kFloat32);
    value_states = value_states.to(kUInt8PerTensorSym);
    value_states = ptq::QDQ_KV(this, value_states, "v_cast_to_int8_qdq");

    // KV cache concat
    auto kh = nn::functional::concat({past_key, key_states}, -1);     // [B, H, D, S]
    auto vh = nn::functional::concat({past_value, value_states}, 2);  // [B, H, S, D]

    // GQA repeat
    kh = kh.repeat(num_key_value_groups_, 1);
    vh = vh.repeat(num_key_value_groups_, 1);

    // Attention
    auto attn = ptq::QDQ(this, nn::functional::matmul(query_states, kh), "qk_matmul_output_qdq");
    auto scale = Tensor::constant(scale_, kFloat32);
    scale = ptq::QDQ(this, scale, "scaling_qdq");
    attn = ptq::QDQ(this, attn.mulConstant(scale), "mul_0_output_qdq");

    // Masked Softmax
    auto attn_min = ptq::QDQ(this, attn.min(-1, true), "reduce_min_output_qdq");
    auto minus_value = Tensor::constant(-20, kFloat32);
    minus_value = ptq::QDQ(this, minus_value, "neg_20_qdq");
    auto attn_vv = ptq::QDQ(this, attn_min.addConstant(minus_value), "minus_0_output_qdq");
    auto zero_constant = Tensor::constant(0.f, kFloat32);
    zero_constant = ptq::QDQ_CONSTANT(this, zero_constant, "constant_zero");
    attn = nn::functional::where(causal_mask.equalConstant(zero_constant), attn, attn_vv);
    attn = ptq::QDQ(this, attn, "where_attn_qdq");
    attn = ptq::QDQ(this, nn::functional::softmax(attn, -1), "softmax_output_qdq");

    auto y = ptq::QDQ(this, nn::functional::matmul(attn, vh), "attn_value_matmul_output_qdq");

    // Apply output gating: y = y * sigmoid(gate)
    if (attn_output_gate_) {
      auto gate_sig = ptq::QDQ(this, nn::functional::sigmoid(gate), "attn_gate_sigmoid_qdq");
      y = ptq::QDQ(this, y * gate_sig, "attn_gate_mul_qdq");
    }

    // [B, S, H*D]
    y = y.transpose(1, 2).view({1, 1, -1, num_attention_heads_ * head_dim_}, /*ssa=*/true);
    y = o_proj_(y).view({1, -1, hidden_size_}, true);

    return {y, key_states, value_states};
  }
};

// ---------------------------------------------------------------------------
// Single full-attention decoder layer (for per-layer QNN tracing)
// ---------------------------------------------------------------------------

class Qwen3_5FullAttnDecoder final : public nn::Module {
 public:
  int layer_idx_;
  Qwen3_5Attention self_attn_;
  Qwen3_5MLP mlp_;
  nn::RMSNorm input_layer_norm_;
  nn::RMSNorm post_attention_layer_norm_;

  Qwen3_5FullAttnDecoder() = default;

  Qwen3_5FullAttnDecoder(const std::string& name, const Qwen3_5Config& cfg, int layer_idx)
      : nn::Module(name), layer_idx_(layer_idx) {
    self_attn_ = reg<Qwen3_5Attention>("self_attn", cfg);
    self_attn_.layer_idx_ = layer_idx;
    mlp_ = reg<Qwen3_5MLP>("mlp", cfg);
    // Pre-baked RMSNorm (no add_unit_offset)
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
    hidden_states = ptq::QDQ(this, hidden_states, "input_layernorm_input_qdq");
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

// ---------------------------------------------------------------------------
// Module hierarchy wrappers to match weight name prefix:
//   model.language_model.layers.{i}.self_attn.q_proj.weight
//
// Structure: Qwen3_5SingleLayerForQNN (unnamed)
//            └─ "model" (Qwen3_5ModelShell)
//                └─ "language_model" (Qwen3_5LMShell)
//                    └─ "layers.{i}" (Qwen3_5FullAttnDecoder)
// ---------------------------------------------------------------------------

class Qwen3_5LMShell final : public nn::Module {
 public:
  Qwen3_5FullAttnDecoder decoder_;
  nn::Param rope_sin_;
  nn::Param rope_cos_;

  Qwen3_5LMShell() = default;
  Qwen3_5LMShell(const std::string& name, const Qwen3_5Config& cfg, int actual_layer_idx)
      : nn::Module(name) {
    std::string layer_name = "layers." + std::to_string(actual_layer_idx);
    decoder_ = reg<Qwen3_5FullAttnDecoder>(layer_name, cfg, actual_layer_idx);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    return decoder_(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5]);
  }
};

class Qwen3_5ModelShell final : public nn::Module {
 public:
  Qwen3_5LMShell lm_;
  nn::Param rope_sin_;
  nn::Param rope_cos_;

  Qwen3_5ModelShell() = default;
  Qwen3_5ModelShell(const std::string& name, const Qwen3_5Config& cfg, int actual_layer_idx)
      : nn::Module(name) {
    lm_ = reg<Qwen3_5LMShell>("language_model", cfg, actual_layer_idx);
    rope_sin_ = reg<nn::Param>("mllm_max_sin_embedding", "model.mllm_max_sin_embedding");
    rope_cos_ = reg<nn::Param>("mllm_max_cos_embedding", "model.mllm_max_cos_embedding");
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    return lm_(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5]);
  }
};

// ---------------------------------------------------------------------------
// Per-layer trace model: wraps a single decoder layer for QNN compilation
// ---------------------------------------------------------------------------

class Qwen3_5SingleLayerForQNN : public ARGeneration, public nn::Module {
 public:
  Qwen3_5SingleLayerForQNN(const Qwen3_5Config& cfg, int actual_layer_idx)
      : cfg_(cfg), actual_layer_idx_(actual_layer_idx) {
    shell_ = reg<Qwen3_5ModelShell>("model", cfg, actual_layer_idx);
  }

  IROutput trace(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override {
    ir::IRContext::ptr_t layer_ir = nullptr;

    auto hidden_states = input.at("hidden_states");
    auto causal_mask = input.at("causal_mask");
    auto position_ids = input.at("position_ids");
    auto past_key = input.at("past_key");
    auto past_value = input.at("past_value");

    ir::lowlevel::traceStart();

    // Gather RoPE for current positions — only rotary_dim columns
    // Use shell_.lm_ as context since RoPE QDQ lives at model.language_model.{sin,cos}_embedding_input_qdq
    auto llm_embedding_sin = nn::functional::gather(
        ptq::QDQ_ROPE(&shell_.lm_, shell_.rope_sin_(), "sin_embedding_input_qdq"), 1, position_ids);
    auto llm_embedding_cos = nn::functional::gather(
        ptq::QDQ_ROPE(&shell_.lm_, shell_.rope_cos_(), "cos_embedding_input_qdq"), 1, position_ids);

    auto outputs = shell_.lm_.decoder_(
        hidden_states, llm_embedding_sin, llm_embedding_cos, causal_mask, past_key, past_value);

    layer_ir = ir::lowlevel::traceStop();
    return {{"model", layer_ir}};
  }

  ARGenerationOutputPast forward(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override {
    return {};
  }

 private:
  const Qwen3_5Config& cfg_;
  int actual_layer_idx_;
  Qwen3_5ModelShell shell_;
};

}  // namespace mllm::models::qwen3_5
