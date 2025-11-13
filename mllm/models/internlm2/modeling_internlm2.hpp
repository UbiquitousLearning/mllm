// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cmath>

#include "fmt/base.h"
#include "mllm/mllm.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/models/internlm2/configuration_internlm2.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/lmcache/StaticCache.hpp"
#include "mllm/utils/Enumerate.hpp"

namespace mllm::models::internlm2 {

inline Tensor makeRoPEInvFreq(int output_dim, float rope_theta, float linear_scale = 1.0f) {
  auto inv_freq = Tensor::empty({output_dim / 2}, kFloat32, kCPU).alloc();
  auto inv_freq_ptr = inv_freq.ptr<float>();
  for (int i = 0; i < output_dim / 2; ++i) {
    auto base = 1.0f / std::pow(rope_theta, 2.0f * static_cast<float>(i) / static_cast<float>(output_dim));
    inv_freq_ptr[i] = base / linear_scale;
  }
  return inv_freq;
}

inline std::pair<Tensor, Tensor> makeRotaryPosEmbedding(Tensor& position_ids, const Tensor& inv_freq) {
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
        auto sin_val = std::sin(freq);
        auto cos_val = std::cos(freq);

        sin_ptr[b * seq_len * dim + s * dim + d] = sin_val;
        sin_ptr[b * seq_len * dim + s * dim + d + inv_freq_len] = sin_val;
        cos_ptr[b * seq_len * dim + s * dim + d] = cos_val;
        cos_ptr[b * seq_len * dim + s * dim + d + inv_freq_len] = cos_val;
      }
    }
  }

  return {sin_emb, cos_emb};
}

class InternLM2MLP final : public nn::Module {
  nn::Linear w1_;
  nn::Linear w3_;
  nn::Linear w2_;
  nn::SiLU silu_;

 public:
  InternLM2MLP() = default;

  InternLM2MLP(const std::string& name, const InternLM2Config& cfg) : nn::Module(name) {
    w1_ = reg<nn::Linear>("w1", cfg.hidden_size, cfg.intermediate_size, cfg.bias, cfg.linear_impl_type);
    silu_ = reg<nn::SiLU>("act");
    w3_ = reg<nn::Linear>("w3", cfg.hidden_size, cfg.intermediate_size, cfg.bias, cfg.linear_impl_type);
    w2_ = reg<nn::Linear>("w2", cfg.intermediate_size, cfg.hidden_size, cfg.bias, cfg.linear_impl_type);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& /*args*/) override {
    auto x = w1_(inputs[0]);
    x = silu_(x);
    auto y = w3_(inputs[0]);
    x = x * y;
    x = w2_(x);
    return {x};
  }
};

class InternLM2Attention final : public nn::Module {
  nn::Linear wqkv_;
  nn::Linear o_proj_;
  nn::RoPE q_rope_;
  nn::RoPE k_rope_;
  nn::CausalMask mask_;
  nn::Softmax softmax_;

  int hidden_size_ = 0;
  int head_dim_ = 0;
  int num_attention_heads_ = 0;
  int num_key_value_heads_ = 0;
  int num_key_value_groups_ = 0;

 public:
  InternLM2Attention() = default;

  InternLM2Attention(const std::string& name, const InternLM2Config& cfg) : nn::Module(name) {
    hidden_size_ = cfg.hidden_size;
    num_attention_heads_ = cfg.num_attention_heads;
    num_key_value_heads_ = cfg.num_key_value_heads;
    head_dim_ = hidden_size_ / num_attention_heads_;
    num_key_value_groups_ = num_attention_heads_ / num_key_value_heads_;

    auto out_features = (num_attention_heads_ + 2 * num_key_value_heads_) * head_dim_;
    wqkv_ = reg<nn::Linear>("wqkv", hidden_size_, out_features, cfg.bias, cfg.linear_impl_type);
    o_proj_ = reg<nn::Linear>("wo", num_attention_heads_ * head_dim_, hidden_size_, cfg.bias, cfg.linear_impl_type);

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

    auto qkv = wqkv_(x);

    int B = x.shape()[0];
    int S = x.shape()[1];

    qkv = qkv.view({B, S, num_key_value_heads_, num_key_value_groups_ + 2, head_dim_});

    // now we have to contiguous before reshape, this is why the model is not efficient
    auto query_blocks = qkv[{kAll, kAll, kAll, {0, num_key_value_groups_}, kAll}].contiguous();
    query_blocks = query_blocks.view({B, S, num_key_value_heads_ * num_key_value_groups_, head_dim_});
    auto query_states = query_blocks.permute({0, 2, 1, 3});  // [B, num_heads, S, D]

    auto key_blocks = qkv[{kAll, kAll, kAll, {num_key_value_groups_, num_key_value_groups_ + 1}, kAll}].contiguous().squeeze(3);
    auto key_states = key_blocks.permute({0, 2, 1, 3});  // [B, num_kv_heads, S, D]

    auto value_blocks =
        qkv[{kAll, kAll, kAll, {num_key_value_groups_ + 1, num_key_value_groups_ + 2}, kAll}].contiguous().squeeze(3);
    auto value_states = value_blocks.permute({0, 2, 1, 3});

    query_states = q_rope_(query_states, llm_embedding_sin, llm_embedding_cos);
    key_states = k_rope_(key_states, llm_embedding_sin, llm_embedding_cos);

    auto [cached_k, cached_v] = past_kv_cache->updateKVCache(layer_idx_, key_states, value_states);
    key_states = cached_k;
    value_states = cached_v;

    auto scale = 1.f / std::sqrt(static_cast<float>(head_dim_));
    auto attn = nn::functional::matmul(query_states, key_states, false, true) * scale;
    attn = mask_(attn);
    attn = softmax_(attn);

    auto output = nn::functional::matmul(attn, value_states);
    output = output.transpose(1, 2).view({B, S, num_attention_heads_ * head_dim_});
    output = o_proj_(output);

    return {output};
  }

  int layer_idx_ = 0;
};

class InternLM2Decoder final : public nn::Module {
 public:
  InternLM2Attention self_attn_;
  InternLM2MLP mlp_;
  nn::RMSNorm input_layer_norm_;
  nn::RMSNorm post_attention_layer_norm_;

 public:
  InternLM2Decoder() = default;

  InternLM2Decoder(const std::string& name, const InternLM2Config& cfg) : nn::Module(name) {
    self_attn_ = reg<InternLM2Attention>("attention", cfg);
    mlp_ = reg<InternLM2MLP>("feed_forward", cfg);
    input_layer_norm_ = reg<nn::RMSNorm>("attention_norm", cfg.rms_norm_eps);
    post_attention_layer_norm_ = reg<nn::RMSNorm>("ffn_norm", cfg.rms_norm_eps);
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

class InternLM2Model final : public nn::Module {
  nn::Embedding embedding_;
  nn::ModuleList<InternLM2Decoder> layers_;
  nn::RMSNorm norm_;

 public:
  InternLM2Model() = default;

  InternLM2Model(const std::string& name, const InternLM2Config& cfg) : nn::Module(name) {
    embedding_ = reg<nn::Embedding>("tok_embeddings", cfg.vocab_size, cfg.hidden_size);
    layers_ = reg<nn::ModuleList<InternLM2Decoder>>("layers", cfg.num_hidden_layers, cfg);
    for (auto [idx, block] : enumerate(layers_.list())) { block.self_attn_.layer_idx_ = idx; }
    norm_ = reg<nn::RMSNorm>("norm", cfg.rms_norm_eps);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = embedding_(inputs[0]);
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto& kv_cache = args[0];

    for (auto& block : layers_.list()) { x = block(x, llm_embedding_sin, llm_embedding_cos, kv_cache)[0]; }

    x = norm_(x);
    return {x};
  }
};

class InternLM2ForCausalLM : public ARGeneration, public nn::Module {
 public:
  explicit InternLM2ForCausalLM(const InternLM2Config& cfg)
      : cfg_(cfg), rope_linear_scale_(cfg.rope_scaling_type == "linear" ? cfg.rope_scaling_factor : 1.0f) {
    kv_cache_ = nn::StaticCache(cfg.max_cache_length, cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads,
                                cfg.head_dim, kFloat32, kFloat32, kCPU, false);

    eos_token_id_ = cfg.eos_token_id;
    max_length_ = cfg.max_cache_length;

    decoder_ = reg<InternLM2Model>("model", cfg);
    lm_head_ = reg<nn::Linear>("output", cfg.hidden_size, cfg.vocab_size, cfg.bias, cfg.linear_impl_type);

    tie_word_embeddings_ = cfg.tie_word_embeddings;
    rope_scaling_type_ = cfg.rope_scaling_type;
    rope_scaling_factor_ = cfg.rope_scaling_factor;

    auto inv_freq = makeRoPEInvFreq(cfg.head_dim, cfg.rope_theta, rope_linear_scale_);
    registerBuffer("inv_freq_base", inv_freq);
  }

  ARGenerationOutputPast forward(const ARGenerationOutputPast& input, const ARGenerationArgs& /*args*/) override {
    auto sequence = input.at("sequence");

    auto batch_size = sequence.shape()[0];
    auto seq_len = sequence.shape()[1];

    Tensor position_ids;
    if (input.count("position_ids")) {
      position_ids = input.at("position_ids");
      if (seq_len == 1) {
        auto last_pos = *position_ids.offsettedPtr<int64_t>({0, position_ids.shape()[1] - 1});
        position_ids = Tensor::empty({batch_size, 1}, kInt64, kCPU).alloc();
        *position_ids.offsettedPtr<int64_t>({0, 0}) = last_pos + 1;
      }
    } else {
      position_ids = Tensor::empty({batch_size, seq_len}, kInt64, kCPU).alloc();
      auto position_ids_ptr = position_ids.ptr<int64_t>();
      for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) { position_ids_ptr[b * seq_len + s] = s; }
      }
    }

    auto max_position = int64_t{0};
    auto pos_ptr = position_ids.ptr<int64_t>();
    for (size_t i = 0; i < position_ids.numel(); ++i) { max_position = std::max(max_position, pos_ptr[i]); }
    max_position += 1;

    Tensor inv_freq = getBuffer("inv_freq_base");
    if (rope_scaling_type_ == "dynamic" && rope_scaling_factor_ > 1.0f && max_position > cfg_.max_position_embeddings) {
      auto factor = rope_scaling_factor_;
      auto base = cfg_.rope_theta
                  * std::pow((factor * static_cast<float>(max_position) / static_cast<float>(cfg_.max_position_embeddings))
                                 - (factor - 1.0f),
                             static_cast<float>(cfg_.head_dim) / static_cast<float>(cfg_.head_dim - 2));
      inv_freq = makeRoPEInvFreq(cfg_.head_dim, base, rope_linear_scale_);
    }

    auto [llm_embedding_sin, llm_embedding_cos] = makeRotaryPosEmbedding(position_ids, inv_freq);

    auto hidden_states = decoder_(sequence, llm_embedding_sin, llm_embedding_cos, AnyValue(&kv_cache_))[0];

    auto S = hidden_states.shape()[1];
    hidden_states = hidden_states[{kAll, {S - 1}, kAll}];

    auto logits = lm_head_(hidden_states);

    return {
        {"sequence", logits},
        {"position_ids", position_ids},
    };
  }

  nn::StaticCache& kvCache() { return kv_cache_; }

 private:
  InternLM2Config cfg_;
  InternLM2Model decoder_;
  nn::Linear lm_head_;
  nn::StaticCache kv_cache_;
  bool tie_word_embeddings_ = false;
  std::string rope_scaling_type_;
  float rope_scaling_factor_ = 1.0f;
  float rope_linear_scale_ = 1.0f;
};

}  // namespace mllm::models::internlm2