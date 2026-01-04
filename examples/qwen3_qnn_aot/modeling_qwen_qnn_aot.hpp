// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/mllm.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/utils/Enumerate.hpp"
#include "mllm/compile/ir/Trace.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/models/qwen3/configuration_qwen3.hpp"

namespace mllm::models::qwen3 {

Tensor rotateHalf(Tensor x) {  // NOLINT
  // X is [x, x, x, D]
  auto D = x.size(-1);
  auto x1 = x[{kAll, kAll, kAll, {kAll, D / 2}}];
  auto x2 = x[{kAll, kAll, kAll, {D / 2, kAll}}];
  return nn::functional::concat({-x2, x1}, -1);
}

namespace ptq {

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
      in.attach("scale", scale.impl());
      in.attach("zero_point", zp.impl());
      break;
    }
    // For Constant!
    case kFloat32: {
      MLLM_RT_ASSERT_EQ(in.rank(), 1);
      MLLM_RT_ASSERT_EQ(in.size(-1), 1);
      auto scale = m->getTopParameterFile()->pull(scale_name);
      auto zp = m->getTopParameterFile()->pull(zp_name);
      in.attach("scale", scale.impl());
      in.attach("zero_point", zp.impl());
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
      MLLM_RT_ASSERT_EQ(zp.item<mllm_int32_t>(), 0);

      // Is 128! not 127!
      auto new_zp = Tensor::constant(128, kInt32).setName(zp_name).setMemType(kParamsNormal);
      in.attach("scale", scale.impl());
      in.attach("zero_point", new_zp.impl());
      break;
    }
    default: {
      MLLM_ERROR_EXIT(ExitCode::kCoreError, "Can't Process dtype={}", nameOfType(in.dtype()));
    }
  }

  return in;
}

}  // namespace ptq

inline auto makeRoPEInvFreq(int output_dim, float rope_theta) -> Tensor {
  auto inv_freq = Tensor::empty({output_dim / 2}, kFloat32, kCPU).alloc();
  auto inv_freq_ptr = inv_freq.ptr<float>();
  for (int i = 0; i < output_dim / 2; i++) { inv_freq_ptr[i] = 1.0 / std::pow(rope_theta, 2.0 * i / output_dim); }
  return inv_freq;
}

inline auto makeRotaryPosEmbedding(Tensor& position_ids, const Tensor& inv_freq,
                                   float attention_scaling = 1.0f) -> std::pair<Tensor, Tensor> {
  auto batch_size = position_ids.shape()[0];
  auto seq_len = position_ids.shape()[1];
  auto inv_freq_len = inv_freq.shape()[0];
  auto dim = inv_freq_len * 2;

  // Create freqs tensor: position_ids @ inv_freq
  auto freqs = Tensor::empty({batch_size, seq_len, inv_freq_len}, kFloat32, kCPU).alloc();
  auto freqs_ptr = freqs.ptr<float>();
  auto position_ids_ptr = position_ids.ptr<int64_t>();
  auto inv_freq_ptr = inv_freq.ptr<float>();

  // Compute freqs = position_ids[:, :, None] @ inv_freq[None, :]
  for (int b = 0; b < batch_size; ++b) {
    for (int s = 0; s < seq_len; ++s) {
      auto pos = position_ids_ptr[b * seq_len + s];
      for (int d = 0; d < inv_freq_len; ++d) {
        freqs_ptr[b * seq_len * inv_freq_len + s * inv_freq_len + d] = static_cast<float>(pos) * inv_freq_ptr[d];
      }
    }
  }

  // Create sin and cos tensors with shape [batch_size, seq_len, dim]
  auto sin_emb = Tensor::empty({batch_size, seq_len, dim}, kFloat32, kCPU).alloc();
  auto cos_emb = Tensor::empty({batch_size, seq_len, dim}, kFloat32, kCPU).alloc();
  auto sin_ptr = sin_emb.ptr<float>();
  auto cos_ptr = cos_emb.ptr<float>();

  // Compute sin and cos embeddings: emb = [freqs, freqs]
  for (int b = 0; b < batch_size; ++b) {
    for (int s = 0; s < seq_len; ++s) {
      for (int d = 0; d < inv_freq_len; ++d) {
        auto freq = freqs_ptr[b * seq_len * inv_freq_len + s * inv_freq_len + d];
        auto sin_val = std::sin(freq) * attention_scaling;
        auto cos_val = std::cos(freq) * attention_scaling;

        // Store the same values in both halves: [freqs, freqs]
        sin_ptr[b * seq_len * dim + s * dim + d] = sin_val;
        sin_ptr[b * seq_len * dim + s * dim + d + inv_freq_len] = sin_val;
        cos_ptr[b * seq_len * dim + s * dim + d] = cos_val;
        cos_ptr[b * seq_len * dim + s * dim + d + inv_freq_len] = cos_val;
      }
    }
  }

  return {sin_emb, cos_emb};
}

class Qwen3MLP final : public nn::Module {
  nn::Linear gate_proj_;
  nn::Linear up_proj_;
  nn::Linear down_proj_;
  nn::SiLU silu_;

 public:
  Qwen3MLP() = default;
  Qwen3MLP(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
    gate_proj_ = reg<nn::Linear>("gate_proj", cfg.hidden_size, cfg.intermediate_size, false, cfg.linear_impl_type);
    silu_ = reg<nn::SiLU>("act");
    up_proj_ = reg<nn::Linear>("up_proj", cfg.hidden_size, cfg.intermediate_size, false, cfg.linear_impl_type);
    down_proj_ = reg<nn::Linear>("down_proj", cfg.intermediate_size, cfg.hidden_size, false, cfg.linear_impl_type);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = inputs[0];
    x = ptq::QDQ(this, x, "up_proj_input_qdq");
    auto up_result = ptq::QDQ(this, up_proj_(x), "up_proj_output_qdq");
    auto gate_result = ptq::QDQ(this, gate_proj_(x), "gate_proj_output_qdq");

    // SiLU
    gate_result = ptq::QDQ(this, (gate_result * ptq::QDQ(this, nn::functional::sigmoid(gate_result), "sigmoid_output_qdq")),
                           "act_output_qdq");

    auto o = ptq::QDQ(this, gate_result * up_result, "down_proj_input_qdq");
    o = down_proj_(o);

    return {o};
  }
};

class Qwen3Attention final : public nn::Module {
  nn::Linear q_proj_;
  nn::Linear k_proj_;
  nn::Linear v_proj_;
  nn::Linear o_proj_;
  nn::RMSNorm rms_norm_q_;
  nn::RMSNorm rms_norm_k_;
  nn::CausalMask mask_;
  nn::Softmax softmax_;

  int hidden_size_;
  int head_dim_;
  int num_attention_heads_;
  int num_key_value_heads_;
  int num_key_value_groups_;
  float scale_;

 public:
  Qwen3Attention() = default;

  Qwen3Attention(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
    hidden_size_ = cfg.hidden_size;
    num_attention_heads_ = cfg.num_attention_heads;
    num_key_value_heads_ = cfg.num_key_value_heads;
    head_dim_ = cfg.head_dim;
    num_key_value_groups_ = num_attention_heads_ / num_key_value_heads_;
    scale_ = (1.f / sqrtf((float)head_dim_));

    // clang-format off
    q_proj_ = reg<nn::Linear>("q_proj", hidden_size_, head_dim_ * num_attention_heads_, cfg.attention_bias);
    k_proj_ = reg<nn::Linear>("k_proj", hidden_size_, head_dim_ * num_key_value_heads_, cfg.attention_bias);
    v_proj_ = reg<nn::Linear>("v_proj", hidden_size_, head_dim_ * num_key_value_heads_, cfg.attention_bias);
    o_proj_ = reg<nn::Linear>("o_proj", head_dim_ * num_attention_heads_, hidden_size_, cfg.attention_bias);
    // clang-format on

    rms_norm_q_ = reg<nn::RMSNorm>("q_norm", cfg.rms_norm_eps);
    rms_norm_k_ = reg<nn::RMSNorm>("k_norm", cfg.rms_norm_eps);

    mask_ = reg<nn::CausalMask>("mask");
    softmax_ = reg<nn::Softmax>("softmax", -1);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden_states = inputs[0];
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto causal_mask = inputs[3];
    auto past_key = inputs[4];
    auto past_value = inputs[5];

    hidden_states = ptq::QDQ(this, hidden_states, "q_proj_input_qdq");

    // [B, S, H * D]
    auto query_states = q_proj_(hidden_states);
    auto key_states = k_proj_(hidden_states);
    auto value_states = v_proj_(hidden_states);

    // [B, H, S, D]
    query_states = query_states.view({1, -1, num_attention_heads_, head_dim_}).transpose(1, 2);
    key_states = key_states.view({1, -1, num_key_value_heads_, head_dim_}).transpose(1, 2);
    value_states = value_states.view({1, -1, num_key_value_heads_, head_dim_}).transpose(1, 2);

    // [B, H, S, D]
    query_states = rms_norm_q_(ptq::QDQ(this, query_states, "q_norm_input_qdq"));
    key_states = rms_norm_k_(ptq::QDQ(this, key_states, "k_norm_input_qdq"));

    query_states = ptq::QDQ(this, query_states, "q_norm_output_qdq");
    key_states = ptq::QDQ(this, key_states, "k_norm_output_qdq");

    // [B, H, S, D]
    auto cos = llm_embedding_cos.unsqueeze(1);
    auto sin = llm_embedding_sin.unsqueeze(1);
    query_states = ptq::QDQ(this,
                            ptq::QDQ(this, query_states * cos, "q_rope_mul_0_output_qdq")
                                + ptq::QDQ(this, rotateHalf(query_states) * sin, "q_rope_mul_1_output_qdq"),
                            "q_rope_add_0_output_qdq");
    key_states = ptq::QDQ(this,
                          ptq::QDQ(this, key_states * cos, "k_rope_mul_0_output_qdq")
                              + ptq::QDQ(this, rotateHalf(key_states) * sin, "k_rope_mul_1_output_qdq"),
                          "k_rope_add_0_output_qdq");

    // De-quantization and quantization again
    key_states = key_states.to(kFloat16);
    key_states = key_states.to(kUInt8PerTensorSym);
    key_states = ptq::QDQ_KV(this, key_states, "k_cast_to_int8_qdq");

    // [B, H, D, S]
    key_states = key_states.transpose(2, 3);

    // Handle KV Cache
    value_states = ptq::QDQ(this, value_states, "v_cast_to_int16_qdq");
    value_states = value_states.to(kFloat16);
    value_states = value_states.to(kUInt8PerTensorSym);
    value_states = ptq::QDQ_KV(this, value_states, "v_cast_to_int8_qdq");

    auto kh = nn::functional::concat({past_key, key_states}, -1);     // [B, H, D, S]
    auto vh = nn::functional::concat({past_value, value_states}, 2);  // [B, H, S, D]

    // Repeat
    kh = kh.repeat(num_key_value_groups_, 1);
    vh = vh.repeat(num_key_value_groups_, 1);

    // Attn
    auto attn = ptq::QDQ(this, nn::functional::matmul(query_states, kh), "qk_matmul_output_qdq");
    auto scale = Tensor::constant(scale_, kFloat32);
    scale = ptq::QDQ(this, scale, "scaling_qdq");
    attn = ptq::QDQ(this, attn.mulConstant(scale), "mul_0_output_qdq");

    // Masked Softmax
    auto attn_min = ptq::QDQ(this, attn.min(-1, true), "reduce_min_output_qdq");
    auto minus_value = Tensor::constant(-20, kFloat32);
    minus_value = ptq::QDQ(this, minus_value, "neg_20_qdq");
    attn = nn::functional::where(causal_mask.equal(0.f), attn, attn_min.addConstant(minus_value));
    attn = ptq::QDQ(this, nn::functional::softmax(attn, -1), "softmax_output_qdq");
    auto y = ptq::QDQ(this, nn::functional::matmul(attn, vh), "attn_value_matmul_output_qdq");
    y = y.transpose(1, 2).view({1, -1, num_attention_heads_ * head_dim_});
    y = o_proj_(y);

    return {y, key_states, value_states};
  }

  int layer_idx_;
};

class Qwen3Decoder final : public nn::Module {
 public:
  Qwen3Attention self_attn_;
  Qwen3MLP mlp_;
  nn::RMSNorm input_layer_norm_;
  nn::RMSNorm post_attention_layer_norm_;

  Qwen3Decoder() = default;

  Qwen3Decoder(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
    self_attn_ = reg<Qwen3Attention>("self_attn", cfg);
    mlp_ = reg<Qwen3MLP>("mlp", cfg);
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

class Qwen3Text final : public nn::Module {
  nn::ModuleList<Qwen3Decoder> decode_blocks_;
  nn::RMSNorm norm_;
  nn::Embedding embedding_;
  nn::Param rope_sin_;
  nn::Param rope_cos_;
  int32_t num_hidden_layers_;

 public:
  Qwen3Text() = default;

  Qwen3Text(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
    num_hidden_layers_ = cfg.num_hidden_layers;
    decode_blocks_ = reg<nn::ModuleList<Qwen3Decoder>>("layers", cfg.num_hidden_layers, cfg);
    for (auto [idx, b] : enumerate(decode_blocks_.list())) { b.self_attn_.layer_idx_ = idx; }
    norm_ = reg<nn::RMSNorm>("norm", cfg.rms_norm_eps);
    embedding_ = reg<nn::Embedding>("embed_tokens", cfg.vocab_size, cfg.hidden_size);
    rope_sin_ = reg<nn::Param>("rope_sin", "rope_sin");
    rope_cos_ = reg<nn::Param>("rope_cos", "rope_cos");
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto& blocks = decode_blocks_.list();

    // X is already embedded
    auto x = embedding_(inputs[0]);

    // Quantization
    x = x.to(kUInt16PerTensorAsy);

    auto position_ids = inputs[1];
    auto causal_mask = inputs[2];
    position_ids = position_ids.squeeze(0);
    auto llm_embedding_sin = rope_sin_()[{{0}, position_ids, {kAll}}];
    auto llm_embedding_cos = rope_cos_()[{{0}, position_ids, {kAll}}];

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

    auto ret = std::vector<Tensor>{x};
    for (const auto& item : keys) { ret.push_back(item); }
    for (const auto& item : values) { ret.push_back(item); }

    return ret;
  }
};

class Qwen3ForCausalLM : public ARGeneration, public nn::Module {
 public:
  explicit Qwen3ForCausalLM(const Qwen3Config& cfg) : cfg(cfg) {
    eos_token_id_ = cfg.end_of_text_token_id;
    max_length_ = cfg.max_cache_length;
    tie_word_embeddings_ = cfg.tie_word_embeddings;

    llm = reg<Qwen3Text>("model", cfg);

    if (cfg.tie_word_embeddings) {
      // NOTE:
      // model.lm_head.weight is quantization weights of model.embed_tokens.weight
      lm_head_ = reg<nn::Linear>("lm_head", cfg.hidden_size, cfg.vocab_size, false, cfg.linear_impl_type);
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
        // If KV cache doesn't exist, we need to handle this case
        // For now, we'll create empty tensors or handle it appropriately
        // This might need adjustment based on your initialization logic
        throw std::runtime_error("Missing KV cache for layer " + std::to_string(i));
      }
    }

    // Append Value
    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
      auto past_value_name = "past_value_" + std::to_string(i);
      if (input.count(past_value_name)) {
        kv_caches.push_back(input.at(past_value_name));
      } else {
        // If KV cache doesn't exist, we need to handle this case
        // For now, we'll create empty tensors or handle it appropriately
        // This might need adjustment based on your initialization logic
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
        auto last_pos = *position_ids.offsettedPtr<int32_t>({0, position_ids.shape()[1] - 1});
        position_ids = Tensor::empty({batch_size, 1}, kInt32, kCPU).alloc();
        *position_ids.offsettedPtr<int32_t>({0, 0}) = last_pos + 1;
      }
    } else {
      // Generate position_ids for prefill phase
      position_ids = Tensor::empty({batch_size, seq_len}, kInt32, kCPU).alloc();
      auto position_ids_ptr = position_ids.ptr<int32_t>();
      for (int s = 0; s < seq_len; ++s) { position_ids_ptr[s] = s; }
    }

    ir::lowlevel::traceStart();

    // Build inputs for llm: sequence, llm_embedding_sin, llm_embedding_cos, causal_mask, then all KV caches
    std::vector<Tensor> llm_inputs = {sequence, position_ids, causal_mask};
    llm_inputs.insert(llm_inputs.end(), kv_caches.begin(), kv_caches.end());

    sequence = llm(llm_inputs)[0];
    sequence = lm_head_(ptq::QDQ(this, sequence, "lm_head_input_qdq"));
    ptq::QDQ(this, sequence, "lm_head_output_qdq");
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
  const Qwen3Config& cfg;
  Qwen3Text llm;
  nn::Linear lm_head_;
  bool tie_word_embeddings_;
};

}  // namespace mllm::models::qwen3
