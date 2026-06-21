#pragma once

#include <cmath>

#include "mllm/models/qwen_npu/configuration_qwen_npu.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/layers/KVCache.hpp"
#include "mllm/utils/Enumerate.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/Tensor.hpp"

namespace mllm::models::qwen_npu::utils {
using mllm::kCPU;
using mllm::kFloat32;
using mllm::Tensor;

inline auto makeRoPEInvFreq(int output_dim, float rope_theta) -> Tensor {
  auto inv_freq = Tensor::empty({output_dim / 2}, kFloat32, kCPU).alloc();
  auto inv_freq_ptr = inv_freq.ptr<float>();
  for (int i = 0; i < output_dim / 2; i++) {
    inv_freq_ptr[i] = 1.0f / std::pow(rope_theta, 2.0f * static_cast<float>(i) / static_cast<float>(output_dim));
  }
  return inv_freq;
}

inline auto makeRotaryPosEmbedding(const Tensor& position_ids, const Tensor& inv_freq,
                                   float attention_scaling = 1.0f) -> std::pair<Tensor, Tensor> {
  auto batch_size = position_ids.shape()[0];
  auto seq_len = position_ids.shape()[1];
  auto inv_freq_len = inv_freq.shape()[0];
  auto dim = inv_freq_len * 2;

  // Create freqs tensor: position_ids[:, :, None] @ inv_freq[None, :]
  // Shape: [batch_size, seq_len, inv_freq_len]
  auto freqs = Tensor::empty({batch_size, seq_len, inv_freq_len}, kFloat32, kCPU).alloc();
  auto freqs_ptr = freqs.ptr<float>();
  auto position_ids_ptr = position_ids.ptr<int64_t>();
  auto inv_freq_ptr = inv_freq.ptr<float>();

  // Compute freqs = position_ids * inv_freq (broadcasted)
  for (int b = 0; b < batch_size; ++b) {
    for (int s = 0; s < seq_len; ++s) {
      auto pos = static_cast<float>(position_ids_ptr[b * seq_len + s]);
      for (int d = 0; d < inv_freq_len; ++d) {
        freqs_ptr[b * seq_len * inv_freq_len + s * inv_freq_len + d] = pos * inv_freq_ptr[d];
      }
    }
  }

  // Create sin and cos tensors with shape [batch_size, seq_len, dim]
  auto sin_emb = Tensor::empty({batch_size, seq_len, dim}, kFloat32, kCPU).alloc();
  auto cos_emb = Tensor::empty({batch_size, seq_len, dim}, kFloat32, kCPU).alloc();
  auto sin_ptr = sin_emb.ptr<float>();
  auto cos_ptr = cos_emb.ptr<float>();

  // Compute sin and cos embeddings: emb = [freqs, freqs] (duplicate frequencies for both halves)
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
}  // namespace mllm::models::qwen_npu::utils

namespace mllm::models::qwen_npu {

// Simple MLP block: gate_proj -> SiLU -> up_proj -> elementwise mul -> down_proj
class QwenMLPCPU final : public nn::Module {
 public:
  QwenMLPCPU() = default;

  QwenMLPCPU(const std::string& name, const QwenNPUConfig& cfg) : nn::Module(name) {
    gate_proj_ = reg<nn::Linear>("gate_proj", cfg.hidden_size, cfg.intermediate_size, false, cfg.linear_impl_type);
    up_proj_ = reg<nn::Linear>("up_proj", cfg.hidden_size, cfg.intermediate_size, false, cfg.linear_impl_type);
    down_proj_ = reg<nn::Linear>("down_proj", cfg.intermediate_size, cfg.hidden_size, false, cfg.linear_impl_type);
    silu_ = reg<nn::SiLU>("act");
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& /*args*/) override {
    auto x = gate_proj_(inputs[0]);
    x = silu_(x);
    auto y = up_proj_(inputs[0]);
    x = x * y;
    x = down_proj_(x);
    return {x};
  }

 private:
  nn::Linear gate_proj_;
  nn::Linear up_proj_;
  nn::Linear down_proj_;
  nn::SiLU silu_;
};

// Qwen2 doesn't use q_norm/k_norm
class QwenAttentionCPU final : public nn::Module {
 public:
  QwenAttentionCPU() = default;

  QwenAttentionCPU(const std::string& name, const QwenNPUConfig& cfg) : nn::Module(name) {
    hidden_size_ = cfg.hidden_size;
    num_attention_heads_ = cfg.num_attention_heads;
    num_key_value_heads_ = cfg.num_key_value_heads;
    head_dim_ = cfg.head_dim;

    q_proj_ = reg<nn::Linear>("q_proj", hidden_size_, head_dim_ * num_attention_heads_, true, cfg.linear_impl_type);
    k_proj_ = reg<nn::Linear>("k_proj", hidden_size_, head_dim_ * num_key_value_heads_, true, cfg.linear_impl_type);
    v_proj_ = reg<nn::Linear>("v_proj", hidden_size_, head_dim_ * num_key_value_heads_, true, cfg.linear_impl_type);
    o_proj_ = reg<nn::Linear>("o_proj", head_dim_ * num_attention_heads_, hidden_size_, false, cfg.linear_impl_type);

    q_rope_ = reg<nn::RoPE>("q_rope", cfg.rope_theta, cfg.max_position_embeddings);
    k_rope_ = reg<nn::RoPE>("k_rope", cfg.rope_theta, cfg.max_position_embeddings);

    mask_ = reg<nn::CausalMask>("mask");
    softmax_ = reg<nn::Softmax>("softmax", -1);

    kv_cache_ =
        reg<nn::KVCache>("kv_cache", 0 /*only hold 1 layer*/, num_attention_heads_, num_key_value_heads_, head_dim_, false);
  }

  void set_layer_idx(int layer_idx) {
    layer_idx_ = layer_idx;
    kv_cache_.setLayerIndex(layer_idx);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& /*args*/) override {
    // inputs: [hidden_states, sin, cos]
    auto hidden_states = inputs[0];
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];

    const int B = hidden_states.shape()[0];
    const int S = hidden_states.shape()[1];

    // Project to Q/K/V: [B, S, H] -> [B, S, Q/K/V_dim]
    auto query_states = q_proj_(hidden_states);
    auto key_states = k_proj_(hidden_states);
    auto value_states = v_proj_(hidden_states);

    // Reshape to [B, S, H, D] then transpose to [B, H, S, D] as in Qwen2Attention
    query_states = query_states.view({B, S, num_attention_heads_, head_dim_}).transpose(1, 2);
    key_states = key_states.view({B, S, num_key_value_heads_, head_dim_}).transpose(1, 2);
    value_states = value_states.view({B, S, num_key_value_heads_, head_dim_}).transpose(1, 2);

    // Apply RoPE
    query_states = q_rope_(query_states, llm_embedding_sin, llm_embedding_cos);
    key_states = k_rope_(key_states, llm_embedding_sin, llm_embedding_cos);

    auto kv_outputs = kv_cache_(key_states, value_states);
    key_states = kv_outputs[0];
    value_states = kv_outputs[1];

    // Attention weights: [B,H,S,S] where S for query is current input length, S for key is cached length
    auto attn_weights = nn::functional::matmul(query_states, key_states, /*trans_a=*/false, /*trans_b=*/true)
                        * (1.f / std::sqrt(static_cast<float>(head_dim_)));

    attn_weights = mask_(attn_weights);
    // HF Qwen2 computes softmax in float32, then converts back to original dtype
    auto attn_weights_dtype = attn_weights.dtype();
    attn_weights = softmax_(attn_weights.to(kFloat32)).to(attn_weights_dtype);

    // Attention output: [B,H,S,S] @ [B,H,S,D] -> [B,H,S,D]
    auto attn_output = nn::functional::matmul(attn_weights, value_states);
    // -> [B,S,H,D] -> [B,S,H*D]
    attn_output = attn_output.transpose(1, 2).view({B, S, num_attention_heads_ * head_dim_});

    // Output projection: [B,S,H] -> [B,S,H]
    attn_output = o_proj_(attn_output);

    return {attn_output};
  }

  nn::KVCache& getKVCache() { return kv_cache_; }

 private:
  int hidden_size_ = 0;
  int num_attention_heads_ = 0;
  int num_key_value_heads_ = 0;
  int head_dim_ = 0;

  nn::Linear q_proj_;
  nn::Linear k_proj_;
  nn::Linear v_proj_;
  nn::Linear o_proj_;

  nn::RoPE q_rope_;
  nn::RoPE k_rope_;
  nn::CausalMask mask_;
  nn::Softmax softmax_;
  nn::KVCache kv_cache_;

  int layer_idx_ = 0;
};

class QwenDecoderCPU final : public nn::Module {
 public:
  QwenAttentionCPU self_attn_;
  QwenMLPCPU mlp_;

  nn::KVCache& getKVCache() { return self_attn_.getKVCache(); }

  QwenDecoderCPU() = default;

  QwenDecoderCPU(const std::string& name, const QwenNPUConfig& cfg) : nn::Module(name) {
    self_attn_ = reg<QwenAttentionCPU>("self_attn", cfg);
    mlp_ = reg<QwenMLPCPU>("mlp", cfg);
    input_layernorm_ = reg<nn::RMSNorm>("input_layernorm", cfg.rms_norm_eps);
    post_attention_layernorm_ = reg<nn::RMSNorm>("post_attention_layernorm", cfg.rms_norm_eps);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& /*args*/) override {
    // inputs: [x, sin, cos]
    auto hidden_states = inputs[0];
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];

    auto residual = hidden_states;
    auto x_norm = input_layernorm_(hidden_states);  // [B, S, H]

    auto attn_output = self_attn_(x_norm, llm_embedding_sin, llm_embedding_cos)[0];
    hidden_states = residual + attn_output;

    residual = hidden_states;
    auto y_norm = post_attention_layernorm_(hidden_states);
    auto mlp_output = mlp_(y_norm)[0];
    hidden_states = residual + mlp_output;

    return {hidden_states};
  }

 private:
  nn::RMSNorm input_layernorm_;
  nn::RMSNorm post_attention_layernorm_;
};

class QwenTextCPU final : public nn::Module {
 public:
  QwenTextCPU() = default;

  QwenTextCPU(const std::string& name, const QwenNPUConfig& cfg) : nn::Module(name) {
    embedding_ = reg<nn::Embedding>("embed_tokens", cfg.vocab_size, cfg.hidden_size);
    decode_blocks_ = reg<nn::ModuleList<QwenDecoderCPU>>("layers", cfg.num_hidden_layers, cfg);
    for (auto [idx, b] : enumerate(decode_blocks_.list())) { b.self_attn_.set_layer_idx(static_cast<int>(idx)); }
    norm_ = reg<nn::RMSNorm>("norm", cfg.rms_norm_eps);

    auto inv_freq = utils::makeRoPEInvFreq(cfg.hidden_size / cfg.num_attention_heads, cfg.rope_theta);
    registerBuffer("inv_freq", inv_freq);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& /*args*/) override {
    auto& blocks = decode_blocks_.list();

    // inputs: [input_embeddings, sin, cos]
    auto x = inputs[0];
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    for (auto& block : blocks) { x = block(x, llm_embedding_sin, llm_embedding_cos)[0]; }

    x = norm_(x);
    return {x};
  }

  nn::Embedding& embedding() { return embedding_; }
  nn::ModuleList<QwenDecoderCPU>& decode_blocks() { return decode_blocks_; }

 private:
  nn::ModuleList<QwenDecoderCPU> decode_blocks_;
  nn::RMSNorm norm_;
  nn::Embedding embedding_;
};

class QwenForCausalLMCPU : public nn::Module, public ARGeneration {
 public:
  QwenForCausalLMCPU() = default;

  explicit QwenForCausalLMCPU(const std::string& name, const QwenNPUConfig& cfg) : cfg_(cfg), nn::Module(name) {
    text_model_ = reg<QwenTextCPU>("model", cfg_);
    lm_head_ = reg<nn::Linear>("lm_head", cfg_.hidden_size, cfg_.vocab_size, false, cfg_.linear_impl_type);

    eos_token_id_ = static_cast<int>(cfg_.eos_token_id);
    max_length_ = cfg_.max_cache_length;
  }

  ARGenerationOutputPast forward(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override {
    auto sequence = input.at("sequence");
    const int B = sequence.shape()[0];
    const int S = sequence.shape()[1];

    auto real_seq = args.count("seq_len") ? args.at("seq_len").get<int>() : S;

    Tensor position_ids;
    if (input.count("position_ids")) {
      position_ids = input.at("position_ids");
      if (S == 1) {
        auto last_pos = *position_ids.offsettedPtr<int64_t>({0, position_ids.shape()[1] - 1});
        position_ids = Tensor::empty({B, 1}, kInt64, kCPU).alloc();
        *position_ids.offsettedPtr<int64_t>({0, 0}) = last_pos + 1;
      }
    } else {
      position_ids = Tensor::empty({B, S}, kInt64, kCPU).alloc();
      auto ptr = position_ids.ptr<int64_t>();
      for (int b = 0; b < B; ++b) {
        for (int s = 0; s < S; ++s) { ptr[b * S + s] = s; }
      }
    }

    const auto& inv_freq = text_model_.getBuffer("inv_freq");
    auto [llm_embedding_sin, llm_embedding_cos] = utils::makeRotaryPosEmbedding(position_ids, inv_freq, 1.0f);

    auto input_embeddings = text_model_.embedding()(sequence);

    auto hidden_states = text_model_(input_embeddings, llm_embedding_sin, llm_embedding_cos)[0];

    // Clip to last valid position
    { hidden_states = hidden_states[{kAll, {real_seq - 1}, kAll}]; }

    auto logits = lm_head_(hidden_states);

    return {
        {"sequence", logits},
        {"position_ids", position_ids},
    };
  }

  QwenTextCPU& text_model() { return text_model_; }

 private:
  QwenNPUConfig cfg_;
  QwenTextCPU text_model_;
  nn::Linear lm_head_;
};

}  // namespace mllm::models::qwen_npu
