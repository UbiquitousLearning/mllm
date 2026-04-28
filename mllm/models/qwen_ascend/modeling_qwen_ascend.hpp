// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#ifndef MLLM_BUILD_ASCEND_BACKEND
#error "QwenAscend model requires MLLM_BUILD_ASCEND_BACKEND to be enabled"
#endif

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "mllm/mllm.hpp"
#include "mllm/backends/ascend/ops/AscendKVCacheOp.hpp"
#include "mllm/backends/ascend/ops/AscendLinearOp.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/models/qwen_ascend/configuration_qwen_ascend.hpp"
#include "mllm/models/qwen_ascend/qwen_ascend_graph_ops.hpp"
#include "mllm/models/qwen_ascend/qwen_ascend_rope.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"

namespace mllm::models::qwen_ascend {

class QwenAscendDecoder;

inline mllm::ascend::AscendLinearOp* getAscendLinearOpPtr(const nn::Linear& linear) {
  return dynamic_cast<mllm::ascend::AscendLinearOp*>(linear.impl()->getInstancedOp().get());
}

inline mllm::ascend::AscendLinearOp& checkedAscendW8A8LinearOp(const nn::Linear& linear) {
  auto* op = getAscendLinearOpPtr(linear);
  MLLM_RT_ASSERT(op != nullptr);
  MLLM_RT_ASSERT(op->isW8A8());
  return *op;
}

class QwenAscendMLP final : public nn::Module {
  friend class QwenAscendDecoder;

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
  friend class QwenAscendDecoder;

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
  bool attention_bias_ = false;
  float rms_norm_epsilon_ = 1e-5f;

 public:
  QwenAscendAttention() = default;

  QwenAscendAttention(const std::string& name, const QwenAscendConfig& cfg) : nn::Module(name) {
    hidden_size_ = cfg.hidden_size;
    num_attention_heads_ = cfg.num_attention_heads;
    num_key_value_heads_ = cfg.num_key_value_heads;
    head_dim_ = cfg.head_dim;
    num_key_value_groups_ = num_attention_heads_ / num_key_value_heads_;
    attention_bias_ = cfg.attention_bias;
    rms_norm_epsilon_ = cfg.rms_norm_eps;

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
    (void)inputs[3];
    auto past_kv_cache = args[0].get<mllm::ascend::AscendKVCache*>();

    auto query_states = q_proj_(x);
    auto key_states = k_proj_(x);
    auto value_states = v_proj_(x);

    int batch_size = inputs[0].shape()[0];
    int seq_len = inputs[0].shape()[1];

    query_states = query_states.view({batch_size, seq_len, num_attention_heads_, head_dim_});
    key_states = key_states.view({batch_size, seq_len, num_key_value_heads_, head_dim_});
    value_states = value_states.view({batch_size, seq_len, num_key_value_heads_, head_dim_});

    query_states = rms_norm_q_(query_states);
    key_states = rms_norm_k_(key_states);

    // Ascend RoPE currently reads [B, S, H, D], so apply RoPE before transpose.
    query_states = q_rope_(query_states, llm_embedding_sin, llm_embedding_cos);
    key_states = k_rope_(key_states, llm_embedding_sin, llm_embedding_cos);

    query_states = query_states.transpose(1, 2);
    key_states = key_states.transpose(1, 2);
    value_states = value_states.transpose(1, 2);

    auto [key_cached, value_cached] = past_kv_cache->updateKVCache(layer_idx_, key_states, value_states);
    if (num_key_value_groups_ > 1) {
      key_cached = mllm::ascend::repeatInterleaveForGQA(key_cached, num_key_value_groups_);
      value_cached = mllm::ascend::repeatInterleaveForGQA(value_cached, num_key_value_groups_);
    }

    Tensor attn = nn::functional::matmul(query_states, key_cached, false, true) * (1.f / sqrtf(head_dim_));
    attn = mask_(attn);
    attn = softmax_(attn);

    auto output = nn::functional::matmul(attn, value_cached);
    output = output.transpose(1, 2).view({batch_size, seq_len, num_attention_heads_ * head_dim_});
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
    attention_bias_ = cfg.attention_bias;
    rms_norm_epsilon_ = cfg.rms_norm_eps;
    max_cache_length_ = cfg.max_cache_length;
    decoder_graph_runner_.configure(cfg.max_cache_length);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override;

 private:
  bool attention_bias_ = false;
  float rms_norm_epsilon_ = 1e-5f;
  int32_t max_cache_length_ = 0;
  QwenAscendDecoderGraphRunner decoder_graph_runner_;

  bool isW8A8Mode() const;
  bool canUseGraph(const Tensor& hidden_states) const;
  void ensureGraphExecutor();
  void buildDecoderGraph();
  void buildDecoderGraphFP16();
  void buildDecoderGraphW8A8();
};

}  // namespace mllm::models::qwen_ascend

#include "mllm/models/qwen_ascend/qwen_ascend_decoder_graph.hpp"

namespace mllm::models::qwen_ascend {

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
    auto x = embedding_(inputs[0]);

    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto local_rope_pos_ids = inputs[3];
    auto& kv_cache = args[0];

    for (auto& block : blocks) {
      x = block(x, llm_embedding_sin, llm_embedding_cos, local_rope_pos_ids, kv_cache)[0];
    }

    x = norm_(x);
    return {x};
  }

  [[nodiscard]] Tensor embeddingWeight() const { return embedding_.weight(); }
};

class QwenAscendForCausalLM : public models::ARGeneration, public nn::Module {
 public:
  explicit QwenAscendForCausalLM(const QwenAscendConfig& cfg) : cfg_(cfg) {
    int num_key_value_groups = cfg.num_attention_heads / cfg.num_key_value_heads;
    kv_cache_ = mllm::ascend::AscendKVCache(
        cfg.max_cache_length,
        cfg.num_hidden_layers,
        cfg.num_key_value_heads,
        cfg.head_dim,
        kFloat16,
        num_key_value_groups);

    eos_token_id_ = cfg.eos_token_id;
    max_length_ = cfg.max_cache_length;
    tie_word_embeddings_ = cfg.tie_word_embeddings;

    llm_ = reg<QwenAscendText>("model", cfg);
    if (!cfg.tie_word_embeddings) {
      lm_head_ = reg<nn::Linear>("lm_head", cfg.hidden_size, cfg.vocab_size, false);
    }

    inv_freq_ = makeRoPEInvFreq(cfg.head_dim, cfg.rope_theta);
  }

  models::ARGenerationOutputPast forward(const models::ARGenerationOutputPast& input,
                                         const models::ARGenerationArgs& args) override {
    auto sequence = input.at("sequence");

    auto batch_size = sequence.shape()[0];
    auto seq_len = sequence.shape()[1];

    Tensor position_ids = Tensor::nil();
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
        for (int s = 0; s < seq_len; ++s) {
          position_ids_ptr[b * seq_len + s] = s;
        }
      }
    }

    auto [sin_emb, cos_emb] = rope_cache_.getEmbeddings(position_ids, inv_freq_, cfg_.max_cache_length);
    auto local_rope_pos_ids = makeLocalRoPEPositionIds(batch_size, seq_len);
    Tensor input_ids_device = sequence.to(kAscend);
    auto hidden_states = llm_(input_ids_device, sin_emb, cos_emb, local_rope_pos_ids, AnyValue(&kv_cache_))[0];

    auto hidden_seq_len = hidden_states.shape()[1];
    hidden_states = hidden_states[{kAll, {hidden_seq_len - 1}, kAll}];

    if (tie_word_embeddings_) {
      auto embed_weight = llm_.embeddingWeight();
      hidden_states = nn::functional::matmul(hidden_states, embed_weight, false, true);
    } else {
      hidden_states = lm_head_(hidden_states);
    }

    hidden_states = hidden_states.to(kCPU).to(kFloat32);

    return {
        {"sequence", hidden_states},
        {"position_ids", position_ids},
    };
  }

  Tensor forward(const Tensor& input_ids) {
    kv_cache_.clearCache();
    auto result = forward({{"sequence", input_ids}}, {});
    return result.at("sequence");
  }

  inline mllm::ascend::AscendKVCache& kvCache() { return kv_cache_; }

  void clearCache() {
    kv_cache_.clearCache();
    rope_cache_.clear();
  }

  void perfSummary() override {
    if (llm_decode_end_time_.time_since_epoch().count() == 0 && ar_steps_ > 1) {
      decodeEventEndTimePoint();
    }
    models::ARGeneration::perfSummary();
  }

 private:
  const QwenAscendConfig& cfg_;
  QwenAscendText llm_;
  nn::Linear lm_head_;
  Tensor inv_freq_;
  bool tie_word_embeddings_ = false;
  mllm::ascend::AscendKVCache kv_cache_;
  QwenAscendRoPECache rope_cache_;
};

}  // namespace mllm::models::qwen_ascend
