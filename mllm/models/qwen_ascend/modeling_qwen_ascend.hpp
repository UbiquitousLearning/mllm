// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

// QwenAscend model requires Ascend backend
#ifndef MLLM_BUILD_ASCEND_BACKEND
#error "QwenAscend model requires MLLM_BUILD_ASCEND_BACKEND to be enabled"
#endif

#include <cstdlib>
#include <limits>

#include "mllm/mllm.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/models/qwen_ascend/configuration_qwen_ascend.hpp"
#include "mllm/backends/ascend/ops/AscendKVCacheOp.hpp"
#include "mllm/backends/ascend/AscendCommon.hpp"

namespace mllm::models::qwen_ascend {

class QwenAscendMLP final : public nn::Module {
  nn::Linear gate_proj_;
  nn::Linear up_proj_;
  nn::Linear down_proj_;
  nn::SiLU silu_;

 public:
  QwenAscendMLP() = default;
  QwenAscendMLP(const std::string& name, const QwenAscendConfig& cfg) : nn::Module(name) {
    gate_proj_ = reg<nn::Linear>("gate_proj", cfg.hidden_size, cfg.intermediate_size, false);
    silu_ = reg<nn::SiLU>("act");
    up_proj_ = reg<nn::Linear>("up_proj", cfg.hidden_size, cfg.intermediate_size, false);
    down_proj_ = reg<nn::Linear>("down_proj", cfg.intermediate_size, cfg.hidden_size, false);
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

class QwenAscendAttention final : public nn::Module {
  nn::Linear q_proj_;
  nn::Linear k_proj_;
  nn::Linear v_proj_;
  nn::Linear o_proj_;
  nn::RMSNorm rms_norm_q_;
  nn::RMSNorm rms_norm_k_;
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
  QwenAscendAttention() = default;

  QwenAscendAttention(const std::string& name, const QwenAscendConfig& cfg) : nn::Module(name) {
    hidden_size_ = cfg.hidden_size;
    num_attention_heads_ = cfg.num_attention_heads;
    num_key_value_heads_ = cfg.num_key_value_heads;
    head_dim_ = cfg.head_dim;
    num_key_value_groups_ = num_attention_heads_ / num_key_value_heads_;

    q_proj_ = reg<nn::Linear>("q_proj", hidden_size_, head_dim_ * num_attention_heads_, cfg.attention_bias);
    k_proj_ = reg<nn::Linear>("k_proj", hidden_size_, head_dim_ * num_key_value_heads_, cfg.attention_bias);
    v_proj_ = reg<nn::Linear>("v_proj", hidden_size_, head_dim_ * num_key_value_heads_, cfg.attention_bias);
    o_proj_ = reg<nn::Linear>("o_proj", head_dim_ * num_attention_heads_, hidden_size_, cfg.attention_bias);

    rms_norm_q_ = reg<nn::RMSNorm>("q_norm", cfg.rms_norm_eps);
    rms_norm_k_ = reg<nn::RMSNorm>("k_norm", cfg.rms_norm_eps);

    q_rope_ = reg<nn::RoPE>("q_rope", cfg.rope_theta, cfg.max_position_embeddings);
    k_rope_ = reg<nn::RoPE>("k_rope", cfg.rope_theta, cfg.max_position_embeddings);

    mask_ = reg<nn::CausalMask>("mask");
    softmax_ = reg<nn::Softmax>("softmax", -1);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = inputs[0];
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto past_kv_cache = args[0].get<mllm::ascend::AscendKVCache*>();

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

    // [B, S, H, D]
    query_states = rms_norm_q_(query_states);
    key_states = rms_norm_k_(key_states);

    // Ascend RoPE currently reads input as [B, S, H, D], so apply RoPE before
    // transposing to attention layout.
    // [B, S, H, D]
    query_states = q_rope_(query_states, llm_embedding_sin, llm_embedding_cos);
    key_states = k_rope_(key_states, llm_embedding_sin, llm_embedding_cos);

    // [B, S, H, D] -> [B, H, S, D]
    query_states = query_states.transpose(1, 2);
    key_states = key_states.transpose(1, 2);
    value_states = value_states.transpose(1, 2);

    // Update KV cache and get cached key/value states
    // AscendKVCache now handles GQA repeat internally - returns [B, q_heads, S, D]
    auto [key_cached, value_cached] = past_kv_cache->updateKVCache(layer_idx_, key_states, value_states);

    // attention weight [B, H, S_q, S_kv]
    Tensor attn = nn::functional::matmul(query_states, key_cached, false, true) * (1.f / sqrtf(head_dim_));

    // Apply causal mask
    attn = mask_(attn);
    attn = softmax_(attn);

    // attn output [B, H, S_q, S_kv] @ [B, H, S_kv, D] -> [B, H, S_q, D]
    auto output = nn::functional::matmul(attn, value_cached);
    // [B, H, S, D] -> [B, S, H, D] -> [B, S, H * D]
    output = output.transpose(1, 2).view({B, S, num_attention_heads_ * head_dim_});
    output = o_proj_(output);

    return {output};
  }

  int layer_idx_;
};

class QwenAscendDecoder final : public nn::Module {
 public:
  QwenAscendAttention self_attn_;
  QwenAscendMLP mlp_;
  nn::RMSNorm input_layer_norm_;
  nn::RMSNorm post_attention_layer_norm_;

  QwenAscendDecoder() = default;

  QwenAscendDecoder(const std::string& name, const QwenAscendConfig& cfg) : nn::Module(name) {
    self_attn_ = reg<QwenAscendAttention>("self_attn", cfg);
    mlp_ = reg<QwenAscendMLP>("mlp", cfg);
    input_layer_norm_ = reg<nn::RMSNorm>("input_layernorm", cfg.rms_norm_eps);
    post_attention_layer_norm_ = reg<nn::RMSNorm>("post_attention_layernorm", cfg.rms_norm_eps);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto& kv_cache = args[0];

    auto x = input_layer_norm_(inputs[0]);
    x = self_attn_(x, llm_embedding_sin, llm_embedding_cos, kv_cache)[0];
    auto tmp = x + inputs[0];
    x = post_attention_layer_norm_(tmp);
    x = mlp_(x)[0];
    x = x + tmp;
    return {x};
  }
};

class QwenAscendText final : public nn::Module {
  nn::ModuleList<QwenAscendDecoder> decode_blocks_;
  nn::RMSNorm norm_;
  nn::Embedding embedding_;

 public:
  QwenAscendText() = default;

  QwenAscendText(const std::string& name, const QwenAscendConfig& cfg) : nn::Module(name) {
    decode_blocks_ = reg<nn::ModuleList<QwenAscendDecoder>>("layers", cfg.num_hidden_layers, cfg);
    for (int idx = 0; idx < decode_blocks_.list().size(); ++idx) {
      decode_blocks_.list()[idx].self_attn_.layer_idx_ = idx;
    }
    norm_ = reg<nn::RMSNorm>("norm", cfg.rms_norm_eps);
    embedding_ = reg<nn::Embedding>("embed_tokens", cfg.vocab_size, cfg.hidden_size);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto& blocks = decode_blocks_.list();

    // X is already embedded
    auto x = embedding_(inputs[0]);

    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto& kv_cache = args[0];

    for (auto& block : blocks) { x = block(x, llm_embedding_sin, llm_embedding_cos, kv_cache)[0]; }

    x = norm_(x);

    return {x};
  }

  // Get embedding weight for tied word embeddings (lm_head shares embed_tokens weight)
  [[nodiscard]] Tensor embeddingWeight() const { return embedding_.weight(); }
};

// Helper functions for RoPE
inline auto makeRoPEInvFreq(int output_dim, float rope_theta) -> Tensor {
  auto inv_freq = Tensor::empty({output_dim / 2}, kFloat32, kCPU).alloc();
  auto inv_freq_ptr = inv_freq.ptr<float>();
  for (int i = 0; i < output_dim / 2; i++) {
    inv_freq_ptr[i] = 1.0f / std::pow(rope_theta, 2.0f * i / output_dim);
  }
  return inv_freq;
}

inline auto makeRotaryPosEmbedding(Tensor& position_ids, const Tensor& inv_freq,
                                   float attention_scaling = 1.0f) -> std::pair<Tensor, Tensor> {
  auto batch_size = position_ids.shape()[0];
  auto seq_len = position_ids.shape()[1];
  auto inv_freq_len = inv_freq.shape()[0];
  auto dim = inv_freq_len * 2;

  auto freqs = Tensor::empty({batch_size, seq_len, inv_freq_len}, kFloat32, kCPU).alloc();
  auto freqs_ptr = freqs.ptr<float>();
  auto position_ids_ptr = position_ids.ptr<int64_t>();
  auto inv_freq_ptr = inv_freq.ptr<float>();

  for (int b = 0; b < batch_size; ++b) {
    for (int s = 0; s < seq_len; ++s) {
      auto pos = position_ids_ptr[b * seq_len + s];
      for (int d = 0; d < inv_freq_len; ++d) {
        freqs_ptr[b * seq_len * inv_freq_len + s * inv_freq_len + d] = static_cast<float>(pos) * inv_freq_ptr[d];
      }
    }
  }

  auto sin_emb = Tensor::empty({batch_size, seq_len, dim}, kFloat32, kCPU).alloc();
  auto cos_emb = Tensor::empty({batch_size, seq_len, dim}, kFloat32, kCPU).alloc();
  auto sin_ptr = sin_emb.ptr<float>();
  auto cos_ptr = cos_emb.ptr<float>();

  for (int b = 0; b < batch_size; ++b) {
    for (int s = 0; s < seq_len; ++s) {
      for (int d = 0; d < inv_freq_len; ++d) {
        auto freq = freqs_ptr[b * seq_len * inv_freq_len + s * inv_freq_len + d];
        auto sin_val = std::sin(freq) * attention_scaling;
        auto cos_val = std::cos(freq) * attention_scaling;

        sin_ptr[b * seq_len * dim + s * dim + d] = sin_val;
        sin_ptr[b * seq_len * dim + s * dim + d + inv_freq_len] = sin_val;
        cos_ptr[b * seq_len * dim + s * dim + d] = cos_val;
        cos_ptr[b * seq_len * dim + s * dim + d + inv_freq_len] = cos_val;
      }
    }
  }

  return {sin_emb, cos_emb};
}

class QwenAscendForCausalLM : public models::ARGeneration, public nn::Module {
 public:
  explicit QwenAscendForCausalLM(const QwenAscendConfig& cfg) : cfg_(cfg), cached_max_seq_len_(0) {
    // Initialize Ascend-specific KV cache with GQA support
    // Cache will handle GQA repeat internally and return repeated tensors
    int num_key_value_groups = cfg.num_attention_heads / cfg.num_key_value_heads;
    kv_cache_ = mllm::ascend::AscendKVCache(
        cfg.max_cache_length,
        cfg.num_hidden_layers,
        cfg.num_key_value_heads,   // kv_heads (NOT q_heads)
        cfg.head_dim,
        kFloat16,                  // dtype - match Ascend model dtype
        num_key_value_groups       // GQA repeat factor
    );

    eos_token_id_ = cfg.eos_token_id;
    max_length_ = cfg.max_cache_length;
    tie_word_embeddings_ = cfg.tie_word_embeddings;

    llm_ = reg<QwenAscendText>("model", cfg);

    // When tie_word_embeddings is true, lm_head shares weights with embed_tokens
    // so we don't create a separate lm_head layer - we'll use embedding weight in forward()
    // Only create lm_head when tie_word_embeddings is false
    if (!cfg.tie_word_embeddings) {
      lm_head_ = reg<nn::Linear>("lm_head", cfg.hidden_size, cfg.vocab_size, false);
    }

    // Init inv freq for RoPE
    inv_freq_ = makeRoPEInvFreq(cfg.head_dim, cfg.rope_theta);
  }

  // ARGeneration interface
  models::ARGenerationOutputPast forward(const models::ARGenerationOutputPast& input,
                                         const models::ARGenerationArgs& args) override {
    auto sequence = input.at("sequence");

    auto batch_size = sequence.shape()[0];
    auto seq_len = sequence.shape()[1];

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
        for (int s = 0; s < seq_len; ++s) {
          position_ids_ptr[b * seq_len + s] = s;
        }
      }
    }

    // Generate or reuse cached RoPE embeddings
    Tensor sin_emb, cos_emb;
    std::tie(sin_emb, cos_emb) = getRoPEEmbeddings(position_ids);

    // Move input_ids to Ascend (will be converted by Embedding layer)
    Tensor input_ids_device = sequence.to(kAscend);

    // Forward through model with KV cache
    auto hidden_states = llm_(input_ids_device, sin_emb, cos_emb, AnyValue(&kv_cache_))[0];

    // Get last token logits
    auto S = hidden_states.shape()[1];
    hidden_states = hidden_states[{kAll, {S - 1}, kAll}];

    // Compute logits using lm_head
    if (tie_word_embeddings_) {
      // Use embedding weight transposed for output projection
      // hidden_states: [B, 1, hidden_size], embed_weight: [vocab_size, hidden_size]
      // output: [B, 1, vocab_size] = hidden_states @ embed_weight^T
      auto embed_weight = llm_.embeddingWeight();
      hidden_states = nn::functional::matmul(hidden_states, embed_weight, false, true);
    } else {
      // Use separate lm_head layer
      hidden_states = lm_head_(hidden_states);
    }

    // Convert to FP32 first (while still on Ascend), then move to CPU
    // This ensures proper dtype conversion before device transfer
    hidden_states = hidden_states.to(kCPU).to(kFloat32);

    return {
        {"sequence", hidden_states},
        {"position_ids", position_ids},
    };
  }

  // Simple forward for backward compatibility (clears cache each time)
  Tensor forward(const Tensor& input_ids) {
    kv_cache_.clearCache();
    auto result = forward({{"sequence", input_ids}}, {});
    return result.at("sequence");
  }

  // Access to KV cache for manual management
  inline mllm::ascend::AscendKVCache& kvCache() { return kv_cache_; }

  // Clear KV cache (call before new conversation)
  void clearCache() {
    kv_cache_.clearCache();
    // Reset RoPE cache
    cached_max_seq_len_ = 0;
    cached_sin_emb_ = Tensor::nil();
    cached_cos_emb_ = Tensor::nil();
  }

  // DEBUG: Access to embedding weight for verification
  [[nodiscard]] Tensor debugEmbeddingWeight() const { return llm_.embeddingWeight(); }

  [[nodiscard]] bool enableChatTimingFallbackFix() const override { return true; }

 private:
  // Get or compute RoPE embeddings with caching
  std::pair<Tensor, Tensor> getRoPEEmbeddings(const Tensor& position_ids) {
    auto batch_size = position_ids.shape()[0];
    auto seq_len = position_ids.shape()[1];

    // Find the maximum position in position_ids
    auto pos_ptr = position_ids.ptr<int64_t>();
    int64_t max_pos = pos_ptr[0];
    for (int i = 1; i < batch_size * seq_len; ++i) {
      if (pos_ptr[i] > max_pos) {
        max_pos = pos_ptr[i];
      }
    }

    // Check if we need to recompute the cache
    if (cached_max_seq_len_ <= max_pos || cached_sin_emb_.isNil() || cached_cos_emb_.isNil()) {
      // Allocate cache for a larger size to reduce recomputation
      // Use max_cache_length_ as the upper bound
      int cache_size = std::min(static_cast<int>(max_pos + 1) * 2, static_cast<int>(cfg_.max_cache_length));

      // Generate position_ids for the cache
      auto cache_position_ids = Tensor::empty({1, cache_size}, kInt64, kCPU).alloc();
      auto cache_pos_ptr = cache_position_ids.ptr<int64_t>();
      for (int i = 0; i < cache_size; ++i) {
        cache_pos_ptr[i] = i;
      }

      // Compute RoPE embeddings on CPU in FP32
      auto [sin_cpu, cos_cpu] = makeRotaryPosEmbedding(cache_position_ids, inv_freq_, 1.0f);

      // Convert to FP16 and move to Ascend device
      cached_sin_emb_ = sin_cpu.to(kFloat16).to(kAscend);
      cached_cos_emb_ = cos_cpu.to(kFloat16).to(kAscend);
      cached_max_seq_len_ = cache_size;
    }

    // Extract embeddings for the requested positions
    // For prefill (seq_len > 1): position_ids = [0, 1, 2, ..., seq_len-1]
    // For decode (seq_len = 1): position_ids = [current_pos]
    // We need to gather the embeddings at these specific positions

    // Simple case: contiguous positions starting from 0 (prefill phase)
    if (seq_len > 1 && pos_ptr[0] == 0) {
      // Verify it's contiguous
      bool is_contiguous = true;
      for (int i = 1; i < seq_len; ++i) {
        if (pos_ptr[i] != i) {
          is_contiguous = false;
          break;
        }
      }

      if (is_contiguous) {
        // Fast path: just slice [0:seq_len]
        auto sin_slice = cached_sin_emb_[{kAll, {kAll, seq_len}, kAll}];
        auto cos_slice = cached_cos_emb_[{kAll, {kAll, seq_len}, kAll}];

        if (batch_size > 1) {
          sin_slice = sin_slice.repeat(batch_size, 0);
          cos_slice = cos_slice.repeat(batch_size, 0);
        }

        return {sin_slice, cos_slice};
      }
    }

    // General case: gather embeddings at specific positions
    // For decode phase: position_ids = [current_pos], we need cached_emb[:, current_pos:current_pos+1, :]
    auto dim = cached_sin_emb_.shape()[2];
    auto sin_result = Tensor::empty({batch_size, seq_len, dim}, kFloat16, kAscend).alloc();
    auto cos_result = Tensor::empty({batch_size, seq_len, dim}, kFloat16, kAscend).alloc();

    // Copy embeddings for each position
    for (int b = 0; b < batch_size; ++b) {
      for (int s = 0; s < seq_len; ++s) {
        int64_t pos = pos_ptr[b * seq_len + s];
        // Slice cached_emb[:, pos:pos+1, :] and copy to result[:, s:s+1, :]
        auto sin_pos = cached_sin_emb_[{kAll, {static_cast<int32_t>(pos), static_cast<int32_t>(pos + 1)}, kAll}];
        auto cos_pos = cached_cos_emb_[{kAll, {static_cast<int32_t>(pos), static_cast<int32_t>(pos + 1)}, kAll}];

        // Copy to result (this is not optimal but correct)
        // TODO: optimize with a single gather operation
        const size_t copy_size = dim * sizeof(mllm_fp16_t);
        auto ret = aclrtMemcpy(
            static_cast<char*>(sin_result.ptr<void>()) + (b * seq_len + s) * copy_size,
            copy_size,
            sin_pos.ptr<void>(),
            copy_size,
            ACL_MEMCPY_DEVICE_TO_DEVICE
        );
        MLLM_ACL_CHECK(ret);

        ret = aclrtMemcpy(
            static_cast<char*>(cos_result.ptr<void>()) + (b * seq_len + s) * copy_size,
            copy_size,
            cos_pos.ptr<void>(),
            copy_size,
            ACL_MEMCPY_DEVICE_TO_DEVICE
        );
        MLLM_ACL_CHECK(ret);
      }
    }

    mllm::ascend::syncGlobalAtbStream();

    return {sin_result, cos_result};
  }

  const QwenAscendConfig& cfg_;
  QwenAscendText llm_;
  nn::Linear lm_head_;
  Tensor inv_freq_;
  bool tie_word_embeddings_;
  mllm::ascend::AscendKVCache kv_cache_;

  // RoPE cache to avoid recomputing sin/cos embeddings every forward pass
  Tensor cached_sin_emb_;
  Tensor cached_cos_emb_;
  int cached_max_seq_len_;
};

}  // namespace mllm::models::qwen_ascend
