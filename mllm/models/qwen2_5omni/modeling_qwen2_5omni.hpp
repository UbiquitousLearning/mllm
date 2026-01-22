// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <cmath>

#include "mllm/mllm.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/nn/lmcache/StaticCache.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/utils/Enumerate.hpp"

#include "mllm/models/qwen2_5omni/configuration_qwen2_5omni.hpp"

namespace mllm::models::qwen2_5omni {

inline auto makeMultimodalRoPEInvFreq(int output_dim, float rope_theta) -> Tensor {
  auto inv_freq = Tensor::empty({output_dim / 2}, kFloat32, kCPU).alloc();
  auto inv_freq_ptr = inv_freq.ptr<float>();
  for (int i = 0; i < output_dim / 2; i++) { inv_freq_ptr[i] = 1.0f / std::pow(rope_theta, 2.0f * i / output_dim); }
  return inv_freq;
}

inline auto makeMultimodalPositionEmbedding(Tensor& position_ids, const Tensor& inv_freq, int seq_len, int output_dim,
                                            const std::vector<int32_t>& mrope_section) -> std::pair<Tensor, Tensor> {
  MLLM_RT_ASSERT_EQ(position_ids.shape().size(), 3);
  MLLM_RT_ASSERT_EQ(position_ids.shape()[1], 1);

  Tensor tmp_sin = Tensor::empty({3, position_ids.shape()[2], inv_freq.shape()[0] * 2}).alloc();
  Tensor tmp_cos = Tensor::empty({3, position_ids.shape()[2], inv_freq.shape()[0] * 2}).alloc();

  for (int b = 0; b < 3; ++b) {
    for (int d = 0; d < inv_freq.shape()[0]; ++d) {
      for (int s = 0; s < position_ids.shape()[2]; ++s) {
        auto value = inv_freq.ptr<float>()[d] * (*position_ids.offsettedPtr<int64_t>({b, 0, s}));
        *tmp_cos.offsettedPtr<float>({b, s, d}) = cosf(value);
        *tmp_cos.offsettedPtr<float>({b, s, d + inv_freq.shape()[0]}) = cosf(value);
        *tmp_sin.offsettedPtr<float>({b, s, d}) = sinf(value);
        *tmp_sin.offsettedPtr<float>({b, s, d + inv_freq.shape()[0]}) = sinf(value);
      }
    }
  }

  Tensor sin = Tensor::nil();
  Tensor cos = Tensor::nil();

  if (!mrope_section.empty()) {
    auto double_rope_section = mrope_section;
    for (int i : mrope_section) { double_rope_section.push_back(i); }

    int num_rows = tmp_sin.shape()[1];
    int num_cols = tmp_sin.shape()[2];

    sin = Tensor::empty({num_rows, num_cols}, kFloat32, kCPU).alloc();
    cos = Tensor::empty({num_rows, num_cols}, kFloat32, kCPU).alloc();

    std::vector<int> start_cols;
    int current_start = 0;
    start_cols.push_back(current_start);
    for (int s : double_rope_section) {
      current_start += s;
      start_cols.push_back(current_start);
    }

    for (int j = 0; j < static_cast<int>(double_rope_section.size()); ++j) {
      int layer = j % 3;
      int s_j = double_rope_section[j];
      int start_col_in = start_cols[j];
      int start_col_out = start_cols[j];
      for (int row = 0; row < num_rows; ++row) {
        auto in_cos_row_ptr = tmp_cos.offsettedPtr<float>({layer, row, 0});
        auto out_cos_row_ptr = cos.offsettedPtr<float>({row, 0});
        for (int c = 0; c < s_j; ++c) { out_cos_row_ptr[start_col_out + c] = in_cos_row_ptr[start_col_in + c]; }

        auto in_sin_row_ptr = tmp_sin.offsettedPtr<float>({layer, row, 0});
        auto out_sin_row_ptr = sin.offsettedPtr<float>({row, 0});
        for (int c = 0; c < s_j; ++c) { out_sin_row_ptr[start_col_out + c] = in_sin_row_ptr[start_col_in + c]; }
      }
    }
  } else {
    sin = tmp_sin;
    cos = tmp_cos;
  }

  return {sin, cos};
}

class Qwen2_5OmniMLP final : public nn::Module {
  nn::Linear gate_proj_;
  nn::Linear up_proj_;
  nn::Linear down_proj_;
  nn::SiLU silu_;

 public:
  Qwen2_5OmniMLP() = default;
  Qwen2_5OmniMLP(const std::string& name, const Qwen2_5OmniConfig& cfg) : nn::Module(name) {
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

class Qwen2_5OmniAttention final : public nn::Module {
  nn::Linear q_proj_;
  nn::Linear k_proj_;
  nn::Linear v_proj_;
  nn::Linear o_proj_;
  nn::MultimodalRoPE q_rope_;
  nn::MultimodalRoPE k_rope_;
  nn::CausalMask mask_;
  nn::Softmax softmax_;

  int hidden_size_;
  int head_dim_;
  int num_attention_heads_;
  int num_key_value_heads_;
  int num_key_value_groups_;

 public:
  Qwen2_5OmniAttention() = default;

  Qwen2_5OmniAttention(const std::string& name, const Qwen2_5OmniConfig& cfg) : nn::Module(name) {
    hidden_size_ = cfg.hidden_size;
    num_attention_heads_ = cfg.num_attention_heads;
    num_key_value_heads_ = cfg.num_key_value_heads;
    head_dim_ = hidden_size_ / num_attention_heads_;
    num_key_value_groups_ = num_attention_heads_ / num_key_value_heads_;

    q_proj_ = reg<nn::Linear>("q_proj", hidden_size_, head_dim_ * num_attention_heads_, true, cfg.linear_impl_type);
    k_proj_ = reg<nn::Linear>("k_proj", hidden_size_, head_dim_ * num_key_value_heads_, true, cfg.linear_impl_type);
    v_proj_ = reg<nn::Linear>("v_proj", hidden_size_, head_dim_ * num_key_value_heads_, true, cfg.linear_impl_type);
    o_proj_ = reg<nn::Linear>("o_proj", head_dim_ * num_attention_heads_, hidden_size_, false, cfg.linear_impl_type);

    q_rope_ = reg<nn::MultimodalRoPE>(
        "q_rope", aops::Qwen2VLMultimodalRoPEOpOptions{.rope_theta = cfg.rope_theta,
                                                       .max_position_embeddings = cfg.max_position_embeddings,
                                                       .mrope_section = cfg.mrope_section});
    k_rope_ = reg<nn::MultimodalRoPE>(
        "k_rope", aops::Qwen2VLMultimodalRoPEOpOptions{.rope_theta = cfg.rope_theta,
                                                       .max_position_embeddings = cfg.max_position_embeddings,
                                                       .mrope_section = cfg.mrope_section});

    mask_ = reg<nn::CausalMask>("mask");
    softmax_ = reg<nn::Softmax>("softmax", -1);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = inputs[0];
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto past_kv_cache = args[0].get<nn::StaticCache*>();

    auto query_states = q_proj_(x);
    auto key_states = k_proj_(x);
    auto value_states = v_proj_(x);

    int B = inputs[0].shape()[0];
    int S = inputs[0].shape()[1];

    query_states = query_states.view({B, S, num_attention_heads_, head_dim_});
    key_states = key_states.view({B, S, num_key_value_heads_, head_dim_});
    value_states = value_states.view({B, S, num_key_value_heads_, head_dim_});

    query_states = query_states.transpose(1, 2);
    key_states = key_states.transpose(1, 2);
    value_states = value_states.transpose(1, 2);

    query_states = q_rope_(query_states, llm_embedding_sin, llm_embedding_cos);
    key_states = k_rope_(key_states, llm_embedding_sin, llm_embedding_cos);

    auto [k, v] = past_kv_cache->updateKVCache(layer_idx_, key_states, value_states);
    key_states = k;
    value_states = v;

    Tensor attn;
    if (key_states.dtype() == kFloat32) {
      attn = nn::functional::matmul(query_states, key_states, false, true) * (1.f / sqrtf(head_dim_));
      attn = mask_(attn);
      attn = softmax_(attn);
    } else if (key_states.dtype() == kFloat16) {
      attn = nn::functional::matmul(query_states.to(kFloat32), key_states.to(kFloat32), false, true) * (1.f / sqrtf(head_dim_));
      attn = mask_(attn);
      attn = softmax_(attn);
      attn = attn.to(kFloat16);
    }

    auto output = nn::functional::matmul(attn, value_states);
    output = output.transpose(1, 2).view({B, S, num_attention_heads_ * head_dim_});
    output = o_proj_(output);
    return {output};
  }

  int layer_idx_;
};

class Qwen2_5OmniDecoder final : public nn::Module {
 public:
  Qwen2_5OmniAttention self_attn_;
  Qwen2_5OmniMLP mlp_;
  nn::RMSNorm input_layer_norm_;
  nn::RMSNorm post_attention_layer_norm_;

  Qwen2_5OmniDecoder() = default;

  Qwen2_5OmniDecoder(const std::string& name, const Qwen2_5OmniConfig& cfg) : nn::Module(name) {
    self_attn_ = reg<Qwen2_5OmniAttention>("self_attn", cfg);
    mlp_ = reg<Qwen2_5OmniMLP>("mlp", cfg);
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

class Qwen2_5OmniText final : public nn::Module {
  nn::ModuleList<Qwen2_5OmniDecoder> decode_blocks_;
  nn::RMSNorm norm_;

 public:
  Qwen2_5OmniText() = default;

  Qwen2_5OmniText(const std::string& name, const Qwen2_5OmniConfig& cfg) : nn::Module(name) {
    decode_blocks_ = reg<nn::ModuleList<Qwen2_5OmniDecoder>>("layers", cfg.num_hidden_layers, cfg);
    for (auto [idx, b] : enumerate(decode_blocks_.list())) { b.self_attn_.layer_idx_ = idx; }

    norm_ = reg<nn::RMSNorm>("norm", cfg.rms_norm_eps);
    embedding_ = reg<nn::Embedding>("embed_tokens", cfg.vocab_size, cfg.hidden_size);

    auto inv = makeMultimodalRoPEInvFreq(cfg.hidden_size / cfg.num_attention_heads, cfg.rope_theta);
    registerBuffer("inv_freq", inv);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto& blocks = decode_blocks_.list();
    auto x = inputs[0];
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto& kv_cache = args[0];

    for (auto& block : blocks) { x = block(x, llm_embedding_sin, llm_embedding_cos, kv_cache)[0]; }
    x = norm_(x);

    return {x};
  }

  nn::Embedding embedding_;
};

class Qwen2_5OmniThinker final : public nn::Module {
 public:
  Qwen2_5OmniThinker() = default;
  Qwen2_5OmniThinker(const std::string& name, const Qwen2_5OmniConfig& cfg) : nn::Module(name) {
    model_ = reg<Qwen2_5OmniText>("model", cfg);
    lm_head_ = reg<nn::Linear>("lm_head", cfg.hidden_size, cfg.vocab_size, false, cfg.linear_impl_type);
  }

  Qwen2_5OmniText model_;
  nn::Linear lm_head_;
};

class Qwen2_5OmniForCausalLM : public ARGeneration {
 public:
  explicit Qwen2_5OmniForCausalLM(const Qwen2_5OmniConfig& cfg) : cfg_(cfg), thinker_("thinker", cfg) {
    kv_cache_ = nn::StaticCache(cfg.max_cache_length, cfg.num_hidden_layers,
                                cfg.num_attention_heads,
                                cfg.num_key_value_heads,
                                cfg.hidden_size / cfg.num_attention_heads,
                                kFloat32,
                                kFloat32,
                                kCPU,
                                false);
    eos_token_id_ = cfg.eos_token_id;
    max_length_ = cfg.max_cache_length;
  }

  ARGenerationOutputPast forward(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override {
    auto sequence = input.at("sequence");

    auto input_embeddings = thinker_.model_.embedding_(sequence);

    Tensor position_ids = Tensor::nil();
    if (input.count("position_ids")) {
      position_ids = input.at("position_ids");
    }
    position_ids = getPositionIds(sequence, position_ids);

    auto [llm_embedding_sin, llm_embedding_cos] =
        makeMultimodalPositionEmbedding(position_ids, thinker_.model_.getBuffer("inv_freq"), cfg_.max_position_embeddings,
                                        cfg_.hidden_size / cfg_.num_attention_heads, cfg_.mrope_section);

    auto hidden_states = thinker_.model_(input_embeddings, llm_embedding_sin, llm_embedding_cos, AnyValue(&kv_cache_))[0];
    auto seq_len = hidden_states.shape()[1];
    auto last_hidden = hidden_states[{kAll, {seq_len - 1}, kAll}];
    auto logits = thinker_.lm_head_(last_hidden);

    return {
        {"sequence", logits},
        {"position_ids", position_ids},
    };
  }

  Qwen2_5OmniThinker thinker_;

 private:
  Tensor getPositionIds(Tensor& input_ids, Tensor& position_ids) const {
    MLLM_RT_ASSERT_EQ(input_ids.shape().size(), 2);

    if (!position_ids.isNil()) {
      auto last_pos = *position_ids.offsettedPtr<int64_t>({0, 0, position_ids.shape()[2] - 1});
      auto ret_position_ids = Tensor::empty({3, 1, 1}, kInt64, kCPU).alloc();
      *ret_position_ids.offsettedPtr<int64_t>({0, 0, 0}) = last_pos + 1;
      *ret_position_ids.offsettedPtr<int64_t>({1, 0, 0}) = last_pos + 1;
      *ret_position_ids.offsettedPtr<int64_t>({2, 0, 0}) = last_pos + 1;
      return ret_position_ids;
    }

    auto B = input_ids.shape()[0];
    auto S = input_ids.shape()[1];
    MLLM_RT_ASSERT_EQ(B, 1);

    Tensor out = Tensor::empty({3, B, S}, kInt64, kCPU).alloc();
    for (int d = 0; d < 3; ++d) {
      auto out_ptr = out.offsettedPtr<int64_t>({d, 0, 0});
      for (int64_t s = 0; s < S; ++s) { out_ptr[s] = s; }
    }
    return out;
  }

  const Qwen2_5OmniConfig& cfg_;
  nn::StaticCache kv_cache_;
};

}  // namespace mllm::models::qwen2_5omni
