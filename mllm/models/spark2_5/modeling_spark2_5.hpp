// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once
#include "mllm/mllm.hpp"
#include "mllm/utils/Enumerate.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/models/common/rope_tables.hpp"
#include "mllm/models/spark2_5/configuration_spark2_5.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/nn/layers/GroupedQueryAttention.hpp"
#include "mllm/nn/lmcache/KVHeadStaticCache.hpp"

namespace mllm::models::spark2_5 {

class SparkMLP final : public nn::Module {
 public:
  SparkMLP() = default;
  SparkMLP(const std::string& name, const SparkConfig& c) : Module(name) {
    gate_ = reg<nn::Linear>("gate_proj", c.hidden_size, c.intermediate_size, false, c.linear_impl_type);
    up_ = reg<nn::Linear>("up_proj", c.hidden_size, c.intermediate_size, false, c.linear_impl_type);
    down_ = reg<nn::Linear>("down_proj", c.intermediate_size, c.hidden_size, false, c.linear_impl_type);
    gelu_ = reg<nn::GELU>("act", aops::GELUOpOptions{.approximate = false});
  }
  std::vector<Tensor> forward(const std::vector<Tensor>& x, const std::vector<AnyValue>&) override {
    return {down_(gelu_(gate_(x[0])) * up_(x[0]))};
  }

 private:
  nn::Linear gate_, up_, down_;
  nn::GELU gelu_;
};

class SparkAttention final : public nn::Module {
 public:
  SparkAttention() = default;
  SparkAttention(const std::string& name, const SparkConfig& c) : Module(name), c_(c) {
    qkv_ = reg<nn::Linear>("q_k_v_proj", c.hidden_size, (c.num_attention_heads + 2 * c.num_key_value_heads) * c.head_dim, false,
                           c.linear_impl_type);
    gate_ = reg<nn::Linear>("g_proj", c.hidden_size, c.num_attention_heads, false);
    out_ = reg<nn::Linear>("out_proj", c.num_attention_heads * c.head_dim, c.hidden_size, false, c.linear_impl_type);
    sigmoid_ = reg<nn::Sigmoid>("gate_sigmoid", aops::SigmoidOpOptions{.approximate = false});
    full_q_ = reg<nn::RoPE>("full_q_rope", c.full_theta, c.max_position_embeddings, c.rotary_dim);
    full_k_ = reg<nn::RoPE>("full_k_rope", c.full_theta, c.max_position_embeddings, c.rotary_dim);
    sliding_q_ = reg<nn::RoPE>("sliding_q_rope", c.sliding_theta, c.max_position_embeddings, c.head_dim);
    sliding_k_ = reg<nn::RoPE>("sliding_k_rope", c.sliding_theta, c.max_position_embeddings, c.head_dim);
    full_attn_ = reg<nn::GroupedQueryAttention>("full_attention", aops::GroupedQueryAttentionImplementation::kDirectStrided);
    sliding_attn_ = reg<nn::GroupedQueryAttention>("sliding_attention",
                                                   aops::GroupedQueryAttentionImplementation::kDirectStrided, c.sliding_window);
  }
  void reset() {
    history_k_ = Tensor();
    history_v_ = Tensor();
  }
  int32_t retainedTokens() const { return history_k_.isNil() ? 0 : history_k_.shape()[2]; }
  std::vector<Tensor> forward(const std::vector<Tensor>& x, const std::vector<AnyValue>& args) override {
    const int32_t s = x[0].shape()[1], qdim = c_.num_attention_heads * c_.head_dim,
                  kvdim = c_.num_key_value_heads * c_.head_dim;
    auto qkv = qkv_(x[0]);
    auto q = qkv[{kAll, kAll, {0, qdim}}].contiguous().view({1, s, c_.num_attention_heads, c_.head_dim}).transpose(1, 2);
    auto k =
        qkv[{kAll, kAll, {qdim, qdim + kvdim}}].contiguous().view({1, s, c_.num_key_value_heads, c_.head_dim}).transpose(1, 2);
    auto v = qkv[{kAll, kAll, {qdim + kvdim, qdim + 2 * kvdim}}]
                 .contiguous()
                 .view({1, s, c_.num_key_value_heads, c_.head_dim})
                 .transpose(1, 2);
    Tensor output;
    if (sliding) {
      q = sliding_q_(q, x[3], x[4]);
      k = sliding_k_(k, x[3], x[4]);
      if (!history_k_.isNil()) {
        k = nn::functional::concat({history_k_, k}, 2);
        v = nn::functional::concat({history_v_, v}, 2);
      }
      output = sliding_attn_(q, k, v);
      // Retain W-1 past tokens for the next call, after all current queries have attended.
      const int32_t end = k.shape()[2], begin = std::max(0, end - c_.sliding_window + 1);
      history_k_ = k[{kAll, kAll, {begin, end}, kAll}].contiguous().clone();
      history_v_ = v[{kAll, kAll, {begin, end}, kAll}].contiguous().clone();
    } else {
      q = full_q_(q, x[1], x[2]);
      k = full_k_(k, x[1], x[2]);
      auto* cache = args[0].get<nn::KVHeadStaticCache*>();
      auto kv = cache->updateKVCache(full_slot, k, v);
      output = full_attn_(q, kv[0], kv[1]);
    }
    auto gate = sigmoid_(gate_(x[0])).view({1, s, c_.num_attention_heads, 1});
    output = (output.transpose(1, 2) * gate).view({1, s, qdim});
    return {out_(output)};
  }
  bool sliding = false;
  int32_t full_slot = 0;

 private:
  SparkConfig c_;
  nn::Linear qkv_, gate_, out_;
  nn::Sigmoid sigmoid_;
  nn::RoPE full_q_, full_k_, sliding_q_, sliding_k_;
  nn::GroupedQueryAttention full_attn_, sliding_attn_;
  Tensor history_k_, history_v_;
};

class SparkDecoder final : public nn::Module {
 public:
  SparkDecoder() = default;
  SparkDecoder(const std::string& name, const SparkConfig& c) : Module(name) {
    attn = reg<SparkAttention>("self_attn", c);
    mlp_ = reg<SparkMLP>("mlp", c);
    in_norm_ = reg<nn::RMSNorm>("input_layernorm", c.rms_norm_eps);
    post_norm_ = reg<nn::RMSNorm>("post_attention_layernorm", c.rms_norm_eps);
  }
  std::vector<Tensor> forward(const std::vector<Tensor>& x, const std::vector<AnyValue>& args) override {
    auto h = attn(in_norm_(x[0]), x[1], x[2], x[3], x[4], args[0])[0] + x[0];
    return {h + mlp_(post_norm_(h))[0]};
  }
  SparkAttention attn;

 private:
  SparkMLP mlp_;
  nn::RMSNorm in_norm_, post_norm_;
};

class SparkText final : public nn::Module {
 public:
  SparkText() = default;
  SparkText(const std::string& name, const SparkConfig& c) : Module(name) {
    embedding_ = reg<nn::Embedding>("embedding", c.vocab_size, c.hidden_size);
    layers_ = reg<nn::ModuleList<SparkDecoder>>("layers", c.num_hidden_layers, c);
    int slot = 0;
    for (auto [i, layer] : enumerate(layers_.list())) {
      layer.attn.sliding = c.layer_types[i] == "sliding_attention";
      if (!layer.attn.sliding) layer.attn.full_slot = slot++;
    }
    norm_ = reg<nn::RMSNorm>("norm", c.rms_norm_eps);
  }
  std::vector<Tensor> forward(const std::vector<Tensor>& x, const std::vector<AnyValue>& args) override {
    auto h = embedding_(x[0]);
    for (auto& layer : layers_.list()) h = layer(h, x[1], x[2], x[3], x[4], args[0])[0];
    return {norm_(h)};
  }
  void reset() {
    for (auto& layer : layers_.list()) layer.attn.reset();
  }

 private:
  nn::Embedding embedding_;
  nn::ModuleList<SparkDecoder> layers_;
  nn::RMSNorm norm_;
};

class SparkForCausalLM final : public ARGeneration, public nn::Module {
 public:
  explicit SparkForCausalLM(const SparkConfig& c)
      : c_(c), cache_(c.max_cache_length, c.full_layers, c.num_key_value_heads, c.head_dim) {
    eos_token_id_ = 1;
    max_length_ = c.max_cache_length;
    model_ = reg<SparkText>("model", c);
    // Converter derives this Linear's weights from the same checkpoint embedding.
    head_ = reg<nn::Linear>("lm_head", c.hidden_size, c.vocab_size, false, c.linear_impl_type);
    registerBuffer("full_inv_freq", common::makeRoPEInvFreq(c.rotary_dim, c.full_theta));
    registerBuffer("sliding_inv_freq", common::makeRoPEInvFreq(c.head_dim, c.sliding_theta));
  }
  ARGenerationOutputPast forward(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override {
    auto tokens = input.at("sequence");
    const auto shape = tokens.shape();
    if (shape.size() != 2 || shape[0] != 1 || shape[1] <= 0 || tokens.dtype() != kInt64 || tokens.device() != kCPU
        || !tokens.isContiguous())
      throw std::invalid_argument("Spark expects contiguous int64 CPU tokens [1,S]");
    if (shape[1] > c_.max_cache_length - position_) throw std::invalid_argument("Spark context capacity exceeded");
    for (size_t i = 0; i < tokens.numel(); ++i)
      if (tokens.ptr<int64_t>()[i] < 0 || tokens.ptr<int64_t>()[i] >= c_.vocab_size)
        throw std::invalid_argument("Spark token outside vocabulary");
    auto positions = Tensor::empty({1, shape[1]}, kInt64, kCPU).alloc();
    for (int i = 0; i < shape[1]; ++i) positions.ptr<int64_t>()[i] = position_ + i;
    auto [fs, fc] = common::makeRotaryPosEmbedding(positions, getBuffer("full_inv_freq"));
    auto [ss, sc] = common::makeRotaryPosEmbedding(positions, getBuffer("sliding_inv_freq"));
    try {
      auto h = model_(tokens, fs, fc, ss, sc, AnyValue(&cache_))[0];
      position_ += shape[1];
      if (!all_logits) h = h[{kAll, {shape[1] - 1}, kAll}];
      return {{"sequence", head_(h)}, {"position_ids", positions}};
    } catch (...) {
      resetState();
      throw;
    }
  }
  void resetState() {
    cache_.clearCache();
    model_.reset();
    position_ = 0;
  }
  bool all_logits = false;

 private:
  SparkConfig c_;
  SparkText model_;
  nn::Linear head_;
  nn::KVHeadStaticCache cache_;
  int32_t position_ = 0;
};
}  // namespace mllm::models::spark2_5
