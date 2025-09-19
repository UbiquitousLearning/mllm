// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <memory>
#include "mllm/mllm.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/nn/lmcache/StaticCache.hpp"
#include "mllm/utils/Enumerate.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/models/llama/configuration_llama.hpp"

namespace mllm::models::llama {

inline auto makeRoPEInvFreq(int output_dim, float rope_theta) -> Tensor {
  auto inv_freq = Tensor::empty({output_dim / 2}, kFloat32, kCPU).alloc();
  auto inv_freq_ptr = inv_freq.ptr<float>();
  for (int i = 0; i < output_dim / 2; i++) { inv_freq_ptr[i] = 1.0f / std::pow(rope_theta, 2.0f * i / output_dim); }
  return inv_freq;
}

inline auto makeRotaryPosEmbedding(Tensor& position_ids, const Tensor& inv_freq, float attention_scaling = 1.0f)
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

class LlamaMLP final : public nn::Module {
  nn::Linear gate_proj_;
  nn::Linear up_proj_;
  nn::Linear down_proj_;
  nn::SiLU silu_;

 public:
  LlamaMLP() = default;
  LlamaMLP(const std::string& name, const LLaMAConfig& cfg) : nn::Module(name) {
    gate_proj_ = reg<nn::Linear>("gate_proj", cfg.hidden_size, cfg.intermediate_size, false, cfg.linear_impl_type);
    silu_ = reg<nn::SiLU>("act");
    up_proj_ = reg<nn::Linear>("up_proj", cfg.hidden_size, cfg.intermediate_size, false, cfg.linear_impl_type);
    down_proj_ = reg<nn::Linear>("down_proj", cfg.intermediate_size, cfg.hidden_size, false, cfg.linear_impl_type);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = gate_proj_(inputs[0]);
    x = silu_(x);
    auto y = up_proj_(inputs[0]);
    x = x * y;
    x = down_proj_(x);
    return {x};
  }
};

class LlamaAttention final : public nn::Module {
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
  float rope_theta_;
  int max_position_embeddings_;

 public:
  LlamaAttention() = default;

  LlamaAttention(const std::string& name, const LLaMAConfig& cfg) : nn::Module(name) {
    hidden_size_ = cfg.hidden_size;
    num_attention_heads_ = cfg.num_attention_heads;
    num_key_value_heads_ = cfg.num_key_value_heads;
    head_dim_ = hidden_size_ / num_attention_heads_;
    num_key_value_groups_ = num_attention_heads_ / num_key_value_heads_;
    rope_theta_ = cfg.rope_theta;
    max_position_embeddings_ = cfg.max_position_embeddings;

    q_proj_ =
        reg<nn::Linear>("q_proj", hidden_size_, head_dim_ * num_attention_heads_, cfg.attention_bias, cfg.linear_impl_type);
    k_proj_ =
        reg<nn::Linear>("k_proj", hidden_size_, head_dim_ * num_key_value_heads_, cfg.attention_bias, cfg.linear_impl_type);
    v_proj_ =
        reg<nn::Linear>("v_proj", hidden_size_, head_dim_ * num_key_value_heads_, cfg.attention_bias, cfg.linear_impl_type);
    o_proj_ =
        reg<nn::Linear>("o_proj", head_dim_ * num_attention_heads_, hidden_size_, cfg.attention_bias, cfg.linear_impl_type);

    q_rope_ = reg<nn::RoPE>("q_rope", cfg.rope_theta, cfg.max_position_embeddings);
    k_rope_ = reg<nn::RoPE>("k_rope", cfg.rope_theta, cfg.max_position_embeddings);

    mask_ = reg<nn::CausalMask>("mask");
    softmax_ = reg<nn::Softmax>("softmax", -1);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = inputs[0];
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    const auto& causal_mask = inputs[3];
    auto past_kv_cache = args[0].get<nn::AbstractStaticCache*>();

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

    query_states = q_rope_(query_states, llm_embedding_sin, llm_embedding_cos);
    key_states = k_rope_(key_states, llm_embedding_sin, llm_embedding_cos);

    // [B, H, S, D]
    auto [k, v] = past_kv_cache->updateKVCache(layer_idx_, key_states, value_states);
    key_states = k;
    value_states = v;

    // attention weight
    // [B, H, S, S]
    Tensor attn;
    if (key_states.dtype() == kFloat32) {
      // attention weight
      // [B, H, S, S]
      attn = nn::functional::matmul(query_states, key_states, false, true) * (1.f / sqrtf(head_dim_));
      if (causal_mask) {
        attn = attn + causal_mask;
      } else {
        attn = mask_(attn);
      }
      attn = softmax_(attn);
    } else if (key_states.dtype() == kFloat16) {
      attn = nn::functional::matmul(query_states.to(kFloat32), key_states.to(kFloat32), false, true) * (1.f / sqrtf(head_dim_));
      attn = mask_(attn);
      attn = softmax_(attn);
      if (causal_mask) {
        attn = attn + causal_mask;
      } else {
        attn = mask_(attn);
      }
      attn = attn.to(kFloat16);
    }

    // attn output
    // [B, H, S, S] @ [B, H, S, D] -> [B, H, S, D]
    auto output = nn::functional::matmul(attn, value_states);
    // [B, H, S, D] -> [B, S, H, D] -> [B, S, H * D]
    output = output.transpose(1, 2).view({B, S, num_attention_heads_ * head_dim_});
    output = o_proj_(output);
    return {output};
  }

  int layer_idx_;
};

class LlamaDecoder final : public nn::Module {
 public:
  LlamaAttention self_attn_;
  LlamaMLP mlp_;
  nn::RMSNorm input_layer_norm_;
  nn::RMSNorm post_attention_layer_norm_;

  LlamaDecoder() = default;

  LlamaDecoder(const std::string& name, const LLaMAConfig& cfg) : nn::Module(name) {
    self_attn_ = reg<LlamaAttention>("self_attn", cfg);
    mlp_ = reg<LlamaMLP>("mlp", cfg);
    input_layer_norm_ = reg<nn::RMSNorm>("input_layernorm", cfg.rms_norm_eps);
    post_attention_layer_norm_ = reg<nn::RMSNorm>("post_attention_layernorm", cfg.rms_norm_eps);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto causal_mask = inputs[3];
    auto& kv_cache = args[0];

    auto x = input_layer_norm_(inputs[0]);

    x = self_attn_(x, kv_cache, llm_embedding_sin, llm_embedding_cos, causal_mask)[0];
    auto tmp = x + inputs[0];
    x = post_attention_layer_norm_(tmp);
    x = mlp_(x)[0];
    x = x + tmp;

    return {x};
  }
};

class LlamaText final : public nn::Module {
  nn::ModuleList<LlamaDecoder> decode_blocks_;
  nn::RMSNorm norm_;

 public:
  LlamaText() = default;

  LlamaText(const std::string& name, const LLaMAConfig& cfg) : nn::Module(name) {
    // decode_blocks_ = reg<nn::ModuleList<LlamaDecoder>>("layers", cfg.num_hidden_layers, cfg);
    decode_blocks_ = reg<nn::ModuleList<LlamaDecoder>>("layers", cfg.num_hidden_layers, cfg);
    for (auto [idx, b] : enumerate(decode_blocks_.list())) { b.self_attn_.layer_idx_ = idx; }

    norm_ = reg<nn::RMSNorm>("norm", cfg.rms_norm_eps);
    embedding_ = reg<nn::Embedding>("embed_tokens", cfg.vocab_size, cfg.hidden_size);

    // Initialize inv_freq for RoPE
    auto inv_freq = makeRoPEInvFreq(cfg.hidden_size / cfg.num_attention_heads, cfg.rope_theta);
    registerBuffer("inv_freq", inv_freq);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto& blocks = decode_blocks_.list();

    // X is already embedded
    auto x = inputs[0];
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto causal_mask = inputs[3];
    auto& kv_cache = args[0];

    for (auto& block : blocks) { x = block(x, llm_embedding_sin, llm_embedding_cos, causal_mask, kv_cache)[0]; }
    x = norm_(x);

    return {x};
  }

  nn::Embedding embedding_;
};

class LlamaForCausalLM : public nn::Module, public ARGeneration {
 public:
  LlamaForCausalLM() = default;
  explicit LlamaForCausalLM(const std::string& name, const LLaMAConfig& cfg, bool mask_by_tensor = false)
      : cfg(cfg), nn::Module(name), mask_by_tensor_(mask_by_tensor) {
    kv_cache_ = std::make_unique<nn::StaticCache>(cfg.max_position_embeddings, cfg.num_hidden_layers,
                                                  cfg.num_attention_heads,                    // q_heads
                                                  cfg.num_key_value_heads,                    // kv_heads
                                                  cfg.hidden_size / cfg.num_attention_heads,  // kv_dims
                                                  kFloat32,                                   // k_dtype
                                                  kFloat32,                                   // v_dtype
                                                  kCPU,                                       // device_type
                                                  false                                       // use_fa2
    );

    model = reg<LlamaText>("model", cfg);
    if (!cfg.tie_word_embeddings) {
      lm_head_ = reg<nn::Linear>("lm_head", cfg.hidden_size, cfg.vocab_size, false, cfg.linear_impl_type);
    }
    tie_word_embeddings_ = cfg.tie_word_embeddings;
  }

  ARGenerationOutputPast forward(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override {
    auto sequence = input.at("sequence");

    // Generate position_ids for the current sequence
    auto batch_size = sequence.shape()[0];
    auto seq_len = sequence.shape()[1];

    Tensor position_ids;
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
    auto [llm_embedding_sin, llm_embedding_cos] = makeRotaryPosEmbedding(position_ids, model.getBuffer("inv_freq"), 1.0f);

    // Create Causal Mask.
    Tensor causal_mask = Tensor::nil();
    // Initialize Causal Mask.
    if (mask_by_tensor_) {
      auto h_len = sequence.shape()[1];
      causal_mask = Tensor::zeros({1, 1, h_len, h_len}, kFloat32, kCPU);
      auto mask_data = causal_mask.ptr<mllm_fp32_t>();
      float min_value = -1e12;
      auto indices =
          std::views::iota(0, h_len) | std::views::transform([h_len](int64_t i) {
            return std::views::iota(i + 1, h_len) | std::views::transform([i, h_len](int64_t j) { return i * h_len + j; });
          })
          | std::views::join;
      std::ranges::for_each(indices, [mask_data, min_value](int64_t index) { mask_data[index] = min_value; });
    }

    // Calculate the text embeddings
    auto input_embeddings = model.embedding_(sequence);

    auto hidden_states =
        model(input_embeddings, llm_embedding_sin, llm_embedding_cos, causal_mask, AnyValue(kv_cache_.get()))[0];

    // clip hidden_states to one seq length
    {
      auto S = hidden_states.shape()[1];
      hidden_states = hidden_states[{kAll, {S - 1}, kAll}];
    }

    Tensor logits;
    if (!tie_word_embeddings_) { logits = lm_head_(hidden_states); }

    return {
        {"sequence", logits},
        {"position_ids", position_ids},
    };
  }

  LLaMAConfig cfg;
  LlamaText model;
  nn::Linear lm_head_;
  bool tie_word_embeddings_;
  bool mask_by_tensor_;

 private:
  std::unique_ptr<nn::AbstractStaticCache> kv_cache_;
};

}  // namespace mllm::models::llama