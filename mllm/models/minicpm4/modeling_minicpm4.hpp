// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/mllm.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/nn/lmcache/StaticCache.hpp"
#include "mllm/models/minicpm4/configuration_minicpm4.hpp"
#include "mllm/utils/Enumerate.hpp"
#include "mllm/models/ARGeneration.hpp"
#include <cmath>

namespace mllm::models::minicpm4 {

inline auto makeRoPEInvFreqWithLongRoPE(int output_dim, float rope_theta, const MiniCPM4Config& cfg, int seq_len) -> Tensor {
  auto inv_freq = Tensor::empty({output_dim / 2}, kFloat32, kCPU).alloc();
  auto inv_freq_ptr = inv_freq.ptr<float>();

  for (int i = 0; i < output_dim / 2; i++) { inv_freq_ptr[i] = 1.0 / std::pow(rope_theta, 2.0 * i / output_dim); }

  if (!cfg.short_factor.empty()) {
    const auto& ext_factors = (seq_len > cfg.original_max_position_embeddings) ? cfg.long_factor : cfg.short_factor;

    for (int i = 0; i < output_dim / 2; i++) { inv_freq_ptr[i] = inv_freq_ptr[i] / ext_factors[i]; }
  }

  return inv_freq;
}

inline float computeLongRoPEScalingFactor(const MiniCPM4Config& cfg) {
  if (cfg.rope_scaling_type != "longrope") { return 1.0f; }
  float scale = (float)cfg.max_position_embeddings / cfg.original_max_position_embeddings;
  return std::sqrt(1.0f + std::log(scale) / std::log((float)cfg.original_max_position_embeddings));
}

inline auto makeRotaryPosEmbedding(Tensor& position_ids, const Tensor& inv_freq, float scaling_factor = 1.0f)
    -> std::pair<Tensor, Tensor> {
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
        auto sin_val = std::sin(freq) * scaling_factor;
        auto cos_val = std::cos(freq) * scaling_factor;

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

class MiniCPM4MLP final : public nn::Module {
  nn::Linear gate_proj_;
  nn::Linear up_proj_;
  nn::Linear down_proj_;
  nn::SiLU silu_;

 public:
  MiniCPM4MLP() = default;
  MiniCPM4MLP(const std::string& name, const MiniCPM4Config& cfg) : nn::Module(name) {
    gate_proj_ = reg<nn::Linear>("gate_proj", cfg.hidden_size, cfg.intermediate_size, false, cfg.linear_impl_type);
    silu_ = reg<nn::SiLU>("act");
    up_proj_ = reg<nn::Linear>("up_proj", cfg.hidden_size, cfg.intermediate_size, false, cfg.linear_impl_type);
    down_proj_ = reg<nn::Linear>("down_proj", cfg.intermediate_size, cfg.hidden_size, false, cfg.linear_impl_type);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    // SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
    auto x = gate_proj_(inputs[0]);
    x = silu_(x);
    auto y = up_proj_(inputs[0]);
    x = x * y;
    x = down_proj_(x);
    return {x};
  }
};

class MiniCPM4Attention final : public nn::Module {
  nn::Linear q_proj_;
  nn::Linear k_proj_;
  nn::Linear v_proj_;
  nn::Linear o_proj_;
  nn::RoPE q_rope_;
  nn::RoPE k_rope_;
  nn::CausalMask mask_;
  nn::Softmax softmax_;

  int hidden_size_;
  int head_dim_;
  int num_attention_heads_;
  int num_key_value_heads_;
  int num_key_value_groups_;

 public:
  MiniCPM4Attention() = default;

  MiniCPM4Attention(const std::string& name, const MiniCPM4Config& cfg) : nn::Module(name) {
    hidden_size_ = cfg.hidden_size;
    num_attention_heads_ = cfg.num_attention_heads;
    num_key_value_heads_ = cfg.num_key_value_heads;
    head_dim_ = hidden_size_ / num_attention_heads_;
    num_key_value_groups_ = num_attention_heads_ / num_key_value_heads_;

    q_proj_ = reg<nn::Linear>("q_proj", hidden_size_, hidden_size_, cfg.attention_bias, cfg.linear_impl_type);
    k_proj_ =
        reg<nn::Linear>("k_proj", hidden_size_, head_dim_ * num_key_value_heads_, cfg.attention_bias, cfg.linear_impl_type);
    v_proj_ =
        reg<nn::Linear>("v_proj", hidden_size_, head_dim_ * num_key_value_heads_, cfg.attention_bias, cfg.linear_impl_type);
    o_proj_ = reg<nn::Linear>("o_proj", hidden_size_, hidden_size_, cfg.attention_bias, cfg.linear_impl_type);

    q_rope_ = reg<nn::RoPE>("q_rope", cfg.rope_theta, cfg.max_position_embeddings);
    k_rope_ = reg<nn::RoPE>("k_rope", cfg.rope_theta, cfg.max_position_embeddings);

    mask_ = reg<nn::CausalMask>("mask");
    softmax_ = reg<nn::Softmax>("softmax", -1);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = inputs[0];
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto past_kv_cache = args[0].get<nn::StaticCache*>();

    // [B, S, H * D]
    auto query_states = q_proj_(x);
    auto key_states = k_proj_(x);
    auto value_states = v_proj_(x);

    int B = inputs[0].shape()[0];
    int S = inputs[0].shape()[1];

    // [B, S, H, D]
    query_states = query_states.view({B, S, num_attention_heads_, head_dim_});
    key_states = key_states.view({B, S, num_key_value_heads_, head_dim_});
    value_states = value_states.view({B, S, num_key_value_heads_, head_dim_});

    // [B, H, S, D]
    query_states = query_states.transpose(1, 2);
    key_states = key_states.transpose(1, 2);
    value_states = value_states.transpose(1, 2);

    // Apply RoPE
    query_states = q_rope_(query_states, llm_embedding_sin, llm_embedding_cos);
    key_states = k_rope_(key_states, llm_embedding_sin, llm_embedding_cos);

    // Update KV cache
    auto [key_states_new, value_states_new] = past_kv_cache->updateKVCache(layer_idx_, key_states, value_states);
    key_states = key_states_new;
    value_states = value_states_new;

    // Compute attention
    Tensor attn;
    if (key_states.dtype() == kFloat32) {
      // [B, H, S, S]
      attn = nn::functional::matmul(query_states, key_states, false, true) * (1.f / sqrtf(head_dim_));
      attn = mask_(attn);
      attn = softmax_(attn);
    } else if (key_states.dtype() == kFloat16) {
      attn = nn::functional::matmul(query_states.to(kFloat32), key_states.to(kFloat32), false, true) * (1.f / sqrtf(head_dim_));
      attn = mask_(attn);
      attn = softmax_(attn);
      attn = attn.to(kFloat16);
    }

    // Apply attention to values
    // [B, H, S, S] @ [B, H, S, D] -> [B, H, S, D]
    auto output = nn::functional::matmul(attn, value_states);
    // [B, H, S, D] -> [B, S, H, D] -> [B, S, H * D]
    output = output.transpose(1, 2).view({B, S, num_attention_heads_ * head_dim_});
    output = o_proj_(output);

    return {output};
  }

  int layer_idx_;
};

class MiniCPM4Decoder final : public nn::Module {
 public:
  MiniCPM4Attention self_attn_;
  MiniCPM4MLP mlp_;
  nn::RMSNorm input_layer_norm_;
  nn::RMSNorm post_attention_layer_norm_;

  float scale_depth_;
  int num_hidden_layers_;

  MiniCPM4Decoder() = default;

  MiniCPM4Decoder(const std::string& name, const MiniCPM4Config& cfg) : nn::Module(name) {
    self_attn_ = reg<MiniCPM4Attention>("self_attn", cfg);
    mlp_ = reg<MiniCPM4MLP>("mlp", cfg);
    input_layer_norm_ = reg<nn::RMSNorm>("input_layernorm", cfg.rms_norm_eps);
    post_attention_layer_norm_ = reg<nn::RMSNorm>("post_attention_layernorm", cfg.rms_norm_eps);

    scale_depth_ = cfg.scale_depth;
    num_hidden_layers_ = cfg.num_hidden_layers;
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto& kv_cache = args[0];

    // Compute residual scaling factor
    float residual_scale = scale_depth_ / std::sqrt(static_cast<float>(num_hidden_layers_));

    // Self attention with residual
    auto residual = inputs[0];
    auto x = input_layer_norm_(residual);
    x = self_attn_(x, llm_embedding_sin, llm_embedding_cos, kv_cache)[0];
    // Apply scaled residual: residual + x * scale
    auto tmp = residual + x * residual_scale;

    // MLP with residual
    x = post_attention_layer_norm_(tmp);
    x = mlp_(x)[0];
    // Apply scaled residual: residual + x * scale
    x = tmp + x * residual_scale;

    return {x};
  }
};

class MiniCPM4Text final : public nn::Module {
 public:
  nn::ModuleList<MiniCPM4Decoder> decode_blocks_;
  nn::RMSNorm norm_;
  nn::Embedding embedding_;
  float scale_emb_;

  MiniCPM4Text() = default;

  MiniCPM4Text(const std::string& name, const MiniCPM4Config& cfg) : nn::Module(name) {
    decode_blocks_ = reg<nn::ModuleList<MiniCPM4Decoder>>("layers", cfg.num_hidden_layers, cfg);
    for (auto [idx, b] : enumerate(decode_blocks_.list())) { b.self_attn_.layer_idx_ = idx; }
    norm_ = reg<nn::RMSNorm>("norm", cfg.rms_norm_eps);
    embedding_ = reg<nn::Embedding>("embed_tokens", cfg.vocab_size, cfg.hidden_size);
    scale_emb_ = cfg.scale_emb;
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto& blocks = decode_blocks_.list();

    // X is embedded and scaled
    auto x = embedding_(inputs[0]);
    // Apply embedding scaling (MiniCPM specific)
    if (scale_emb_ != 1.0f) { x = x * scale_emb_; }

    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto& kv_cache = args[0];

    for (auto& block : blocks) { x = block(x, llm_embedding_sin, llm_embedding_cos, kv_cache)[0]; }

    x = norm_(x);

    return {x};
  }
};

class MiniCPM4ForCausalLM : public ARGeneration, public nn::Module {
 public:
  explicit MiniCPM4ForCausalLM(const MiniCPM4Config& cfg) : cfg(cfg) {
    kv_cache_ = nn::StaticCache(cfg.max_cache_length, cfg.num_hidden_layers,
                                cfg.num_attention_heads,                    // q_heads
                                cfg.num_key_value_heads,                    // kv_heads
                                cfg.hidden_size / cfg.num_attention_heads,  // kv_dim
                                kFloat32,                                   // k_dtype
                                kFloat32,                                   // v_dtype
                                kCPU,                                       // device_type
                                false                                       // use_fa2
    );
    eos_token_id_ = cfg.eos_token_id;
    max_length_ = cfg.max_cache_length;
    tie_word_embeddings_ = cfg.tie_word_embeddings;

    llm = reg<MiniCPM4Text>("model", cfg);

    // Only create lm_head if NOT using tied embeddings
    if (!cfg.tie_word_embeddings) {
      lm_head_ = reg<nn::Linear>("lm_head", cfg.hidden_size, cfg.vocab_size, false, cfg.linear_impl_type);
    }

    longrope_scaling_factor_ = computeLongRoPEScalingFactor(cfg);
  }

  ARGenerationOutputPast forward(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override {
    auto sequence = input.at("sequence");

    // Generate position_ids for the current sequence
    auto batch_size = sequence.shape()[0];
    auto seq_len = sequence.shape()[1];

    auto inv_freq = makeRoPEInvFreqWithLongRoPE(cfg.hidden_size / cfg.num_attention_heads, cfg.rope_theta, cfg, seq_len);

    Tensor position_ids = Tensor::nil();
    if (input.count("position_ids")) {
      // Use existing position_ids for decode phase
      position_ids = input.at("position_ids");

      // For decode phase, increment the last position
      if (seq_len == 1) {
        auto last_pos = *position_ids.offsettedPtr<int64_t>({0, position_ids.shape()[1] - 1});
        position_ids = Tensor::empty({batch_size, 1}, kInt64, kCPU).alloc();
        *position_ids.offsettedPtr<int64_t>({0, 0}) = last_pos + 1;
      }
    } else {
      // Generate position_ids for prefill phase
      position_ids = Tensor::empty({batch_size, seq_len}, kInt64, kCPU).alloc();
      auto position_ids_ptr = position_ids.ptr<int64_t>();
      for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) { position_ids_ptr[b * seq_len + s] = s; }
      }
    }

    // Generate RoPE embeddings using the inv_freq buffer
    auto [llm_embedding_sin, llm_embedding_cos] = makeRotaryPosEmbedding(position_ids, inv_freq, longrope_scaling_factor_);

    sequence = llm(sequence, llm_embedding_sin, llm_embedding_cos, AnyValue(&kv_cache_))[0];

    // clip x to one seq length
    {
      auto S = sequence.shape()[1];
      sequence = sequence[{kAll, {S - 1}, kAll}];
    }

    // Apply MiniCPM specific scaling before output
    // logits = lm_head(hidden_states / (hidden_size / dim_model_base))
    if (cfg.dim_model_base != 1.0f) {
      float scale_factor = cfg.hidden_size / cfg.dim_model_base;
      sequence = sequence / scale_factor;
    }

    if (tie_word_embeddings_) {
      auto embedding_weight = llm.embedding_.weight();
      sequence = nn::functional::matmul(sequence, embedding_weight, false, true);
    } else {
      sequence = lm_head_(sequence);
    }

    return {
        {"sequence", sequence},
        {"position_ids", position_ids},
    };
  }

  inline nn::StaticCache& kvCache() { return kv_cache_; }

 private:
  const MiniCPM4Config& cfg;
  MiniCPM4Text llm;
  nn::Linear lm_head_;
  bool tie_word_embeddings_;
  nn::StaticCache kv_cache_;
  float longrope_scaling_factor_;
};

}  // namespace mllm::models::minicpm4
