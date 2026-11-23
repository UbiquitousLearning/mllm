// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mllm/core/Tensor.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/models/common/rope_tables.hpp"
#include "mllm/models/lfm2/configuration_lfm2.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/lmcache/KVHeadStaticCache.hpp"

namespace mllm::models::lfm2 {

// Rotation itself remains the registered nn::RoPE operation; only the
// immutable analytical tables are materialized here.
using common::makeRoPEInvFreq;
using common::makeRotaryPosEmbedding;

inline constexpr int32_t kLfm2KaiDecodeThreadCap = 4;
inline constexpr int32_t kLfm2KaiPrefillThreadCap = 6;

inline auto makeLfm2LinearOptions(int32_t in_channels, int32_t out_channels, bool bias, aops::LinearImplTypes impl_type)
    -> aops::LinearOpOptions {
  return {.in_channels = in_channels,
          .out_channels = out_channels,
          .bias = bias,
          .impl_type = impl_type,
          // Source-bound OnePlus 13T screening selected six workers for I8MM
          // prefill and four workers to avoid decode GEMV oversubscription.
          .kai_w4a32_decode_thread_cap = kLfm2KaiDecodeThreadCap,
          .kai_w4a32_prefill_thread_cap = kLfm2KaiPrefillThreadCap};
}

class Lfm2MLP final : public nn::Module {
 public:
  Lfm2MLP() = default;
  Lfm2MLP(const std::string& name, const Lfm2Config& cfg) : nn::Module(name) {
    gate_up_proj_ = reg<nn::ParallelLinear>(
        "gate_up_proj", cfg.hidden_size, std::vector<int32_t>{cfg.intermediate_size, cfg.intermediate_size},
        std::vector<std::string>{"w1", "w3"}, false, cfg.linear_impl_type, kLfm2KaiDecodeThreadCap, kLfm2KaiPrefillThreadCap);
    w2_ = reg<nn::Linear>("w2", makeLfm2LinearOptions(cfg.intermediate_size, cfg.hidden_size, false, cfg.linear_impl_type));
    silu_ = reg<nn::SiLU>("silu");
  }
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>&) override {
    auto gate_up = gate_up_proj_(inputs[0]);
    return {w2_(silu_(gate_up[0]) * gate_up[1])};
  }

 private:
  nn::ParallelLinear gate_up_proj_;
  nn::Linear w2_;
  nn::SiLU silu_;
};

class Lfm2Attention final : public nn::Module {
 public:
  Lfm2Attention() = default;
  Lfm2Attention(const std::string& name, const Lfm2Config& cfg) : nn::Module(name) {
    use_tiled_fa2_prefill_ = cfg.use_tiled_fa2_prefill;
    hidden_size_ = cfg.hidden_size;
    head_dim_ = cfg.head_dim;
    query_heads_ = cfg.num_attention_heads;
    kv_heads_ = cfg.num_key_value_heads;
    qkv_proj_ = reg<nn::ParallelLinear>(
        "qkv_proj", hidden_size_, std::vector<int32_t>{query_heads_ * head_dim_, kv_heads_ * head_dim_, kv_heads_ * head_dim_},
        std::vector<std::string>{"q_proj", "k_proj", "v_proj"}, false, cfg.linear_impl_type, kLfm2KaiDecodeThreadCap,
        kLfm2KaiPrefillThreadCap);
    out_proj_ =
        reg<nn::Linear>("out_proj", makeLfm2LinearOptions(query_heads_ * head_dim_, hidden_size_, false, cfg.linear_impl_type));
    q_layernorm_ = reg<nn::RMSNorm>("q_layernorm", cfg.norm_eps, false);
    k_layernorm_ = reg<nn::RMSNorm>("k_layernorm", cfg.norm_eps, false);
    q_rope_ = reg<nn::RoPE>("q_rope", cfg.rope_theta, cfg.max_position_embeddings, head_dim_);
    k_rope_ = reg<nn::RoPE>("k_rope", cfg.rope_theta, cfg.max_position_embeddings, head_dim_);
    gqa_ = reg<nn::GroupedQueryAttention>("gqa", aops::GroupedQueryAttentionImplementation::kDirectStrided);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    const auto& x = inputs[0];
    const int32_t batch = x.shape()[0];
    const int32_t sequence = x.shape()[1];
    auto qkv = qkv_proj_(x);
    auto query = qkv[0].view({batch, sequence, query_heads_, head_dim_});
    auto key = qkv[1].view({batch, sequence, kv_heads_, head_dim_});
    auto value = qkv[2].view({batch, sequence, kv_heads_, head_dim_});
    query = q_layernorm_(query).transpose(1, 2);
    key = k_layernorm_(key).transpose(1, 2);
    value = value.transpose(1, 2);
    query = q_rope_(query, inputs[1], inputs[2]);
    key = k_rope_(key, inputs[1], inputs[2]);

    auto* cache = args.at(0).get<nn::KVHeadStaticCache*>();
    auto updated = cache->updateKVCache(logical_cache_slot_, key, value);
    // Keep the established direct order by default and for single-token decode.
    constexpr auto layout = aops::FlashAttention2Layout::kBHSD;
    auto output =
        use_tiled_fa2_prefill_ && nn::functional::supportsCpuFlashAttention2Prefill(query, updated[0], updated[1], layout)
            ? nn::functional::flashAttention2(query, updated[0], updated[1], layout)
            : gqa_(query, updated[0], updated[1]);
    output = output.transpose(1, 2).contiguous().view({batch, sequence, query_heads_ * head_dim_});
    return {out_proj_(output)};
  }

  int32_t logical_cache_slot_ = 0;

 private:
  bool use_tiled_fa2_prefill_ = false;
  int32_t hidden_size_ = 0;
  int32_t head_dim_ = 0;
  int32_t query_heads_ = 0;
  int32_t kv_heads_ = 0;
  nn::ParallelLinear qkv_proj_;
  nn::Linear out_proj_;
  nn::RMSNorm q_layernorm_;
  nn::RMSNorm k_layernorm_;
  nn::RoPE q_rope_;
  nn::RoPE k_rope_;
  nn::GroupedQueryAttention gqa_;
};

class Lfm2ShortConv final : public nn::Module {
 public:
  Lfm2ShortConv() = default;
  Lfm2ShortConv(const std::string& name, const Lfm2Config& cfg) : nn::Module(name) {
    hidden_size_ = cfg.hidden_size;
    kernel_size_ = cfg.conv_L_cache;
    in_proj_ =
        reg<nn::Linear>("in_proj", makeLfm2LinearOptions(hidden_size_, 3 * hidden_size_, cfg.conv_bias, cfg.linear_impl_type));
    conv_ = reg<nn::CausalDepthwiseConv1D>("conv", hidden_size_, kernel_size_, cfg.conv_bias, true,
                                           aops::CausalDepthwiseConv1DAccumulationOrder::kHistoryFirst);
    out_proj_ =
        reg<nn::Linear>("out_proj", makeLfm2LinearOptions(hidden_size_, hidden_size_, cfg.conv_bias, cfg.linear_impl_type));
  }

  // The causal kernel consumes only K - 1 historical samples. The former
  // generic concat/Conv1D path stored K samples but never read the oldest one.
  void resetState(int32_t batch_size) { state_ = Tensor::zeros({batch_size, hidden_size_, kernel_size_ - 1}, kFloat32, kCPU); }

  [[nodiscard]] const Tensor& state() const { return state_; }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>&) override {
    const auto& input = inputs[0];
    const int32_t batch = input.shape()[0];
    const int32_t sequence = input.shape()[1];
    if (input.dtype() != kFloat32 || input.device() != kCPU) {
      throw std::invalid_argument("LFM2 short convolution currently requires float32 CPU activations");
    }
    if (state_.isNil() || state_.shape()[0] != batch) resetState(batch);

    auto projected = in_proj_(input);
    auto b = projected[{kAll, kAll, {0, hidden_size_}}].contiguous();
    auto c = projected[{kAll, kAll, {hidden_size_, 2 * hidden_size_}}].contiguous();
    auto x = projected[{kAll, kAll, {2 * hidden_size_, 3 * hidden_size_}}].contiguous();
    auto bx = b * x;
    auto [convolved, updated_state] = conv_(bx, state_);
    state_ = std::move(updated_state);
    return {out_proj_(c * convolved)};
  }

 private:
  int32_t hidden_size_ = 0;
  int32_t kernel_size_ = 0;
  nn::Linear in_proj_;
  nn::CausalDepthwiseConv1D conv_;
  nn::Linear out_proj_;
  Tensor state_;
};

class Lfm2AttentionDecoder final : public nn::Module {
 public:
  Lfm2AttentionDecoder() = default;
  Lfm2AttentionDecoder(const std::string& name, const Lfm2Config& cfg) : nn::Module(name) {
    self_attn_ = reg<Lfm2Attention>("self_attn", cfg);
    feed_forward_ = reg<Lfm2MLP>("feed_forward", cfg);
    operator_norm_ = reg<nn::RMSNorm>("operator_norm", cfg.norm_eps, false);
    ffn_norm_ = reg<nn::RMSNorm>("ffn_norm", cfg.norm_eps, false);
  }
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto residual = inputs[0];
    auto hidden = residual + self_attn_(operator_norm_(residual), inputs[1], inputs[2], args.at(0))[0];
    return {hidden + feed_forward_(ffn_norm_(hidden))[0]};
  }
  Lfm2Attention self_attn_;

 private:
  Lfm2MLP feed_forward_;
  nn::RMSNorm operator_norm_;
  nn::RMSNorm ffn_norm_;
};

class Lfm2ConvDecoder final : public nn::Module {
 public:
  Lfm2ConvDecoder() = default;
  Lfm2ConvDecoder(const std::string& name, const Lfm2Config& cfg) : nn::Module(name) {
    conv_ = reg<Lfm2ShortConv>("conv", cfg);
    feed_forward_ = reg<Lfm2MLP>("feed_forward", cfg);
    operator_norm_ = reg<nn::RMSNorm>("operator_norm", cfg.norm_eps, false);
    ffn_norm_ = reg<nn::RMSNorm>("ffn_norm", cfg.norm_eps, false);
  }
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>&) override {
    auto residual = inputs[0];
    auto hidden = residual + conv_(operator_norm_(residual))[0];
    return {hidden + feed_forward_(ffn_norm_(hidden))[0]};
  }
  Lfm2ShortConv conv_;

 private:
  Lfm2MLP feed_forward_;
  nn::RMSNorm operator_norm_;
  nn::RMSNorm ffn_norm_;
};

class Lfm2Model final : public nn::Module {
 public:
  Lfm2Model() = default;
  Lfm2Model(const std::string& name, const Lfm2Config& cfg) : nn::Module(name) {
    embed_tokens_ = reg<nn::Embedding>("embed_tokens", cfg.vocab_size, cfg.hidden_size);
    int32_t attention_index = 0;
    int32_t conv_index = 0;
    for (int32_t physical_layer = 0; physical_layer < cfg.num_hidden_layers; ++physical_layer) {
      const auto layer_name = "layers." + std::to_string(physical_layer);
      if (cfg.isAttentionLayer(physical_layer)) {
        auto layer = reg<Lfm2AttentionDecoder>(layer_name, cfg);
        layer.self_attn_.logical_cache_slot_ = attention_index;
        attention_layers_.push_back(std::move(layer));
        layer_kind_.push_back(0);
        layer_dispatch_.push_back(attention_index++);
      } else {
        conv_layers_.push_back(reg<Lfm2ConvDecoder>(layer_name, cfg));
        layer_kind_.push_back(1);
        layer_dispatch_.push_back(conv_index++);
      }
    }
    embedding_norm_ = reg<nn::RMSNorm>("embedding_norm", cfg.norm_eps, false);
  }

  void resetConvStates(int32_t batch_size) {
    for (auto& layer : conv_layers_) layer.conv_.resetState(batch_size);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden = embed_tokens_(inputs[0]);
    for (size_t layer = 0; layer < layer_kind_.size(); ++layer) {
      if (layer_kind_[layer] == 0) {
        hidden = attention_layers_[layer_dispatch_[layer]](hidden, inputs[1], inputs[2], args.at(0))[0];
      } else {
        hidden = conv_layers_[layer_dispatch_[layer]](hidden)[0];
      }
    }
    return {embedding_norm_(hidden)};
  }

 private:
  nn::Embedding embed_tokens_;
  nn::RMSNorm embedding_norm_;
  std::vector<Lfm2AttentionDecoder> attention_layers_;
  std::vector<Lfm2ConvDecoder> conv_layers_;
  std::vector<int32_t> layer_kind_;
  std::vector<int32_t> layer_dispatch_;
};

class Lfm2ForCausalLM final : public ARGeneration, public nn::Module {
 public:
  explicit Lfm2ForCausalLM(const Lfm2Config& cfg)
      : kv_cache_(cfg.max_cache_length, cfg.numAttentionLayers(), cfg.num_key_value_heads, cfg.head_dim) {
    eos_token_id_ = cfg.eos_token_id;
    max_length_ = cfg.max_cache_length;
    model_ = reg<Lfm2Model>("model", cfg);
    // The converter creates this packed alias from the tied embedding. Keeping
    // it explicit lets the mobile W4A32 output projection use the standard Linear path.
    lm_head_ =
        reg<nn::Linear>("lm_head_out", makeLfm2LinearOptions(cfg.hidden_size, cfg.vocab_size, false, cfg.linear_impl_type));
    registerBuffer("inv_freq", makeRoPEInvFreq(cfg.head_dim, cfg.rope_theta));
  }

  ARGenerationOutputPast forward(const ARGenerationOutputPast& input, const ARGenerationArgs&) override {
    auto sequence = input.at("sequence");
    const auto shape = sequence.shape();
    if (shape.size() != 2 || shape[0] != 1 || shape[1] <= 0 || sequence.dtype() != kInt64 || sequence.device() != kCPU) {
      throw std::invalid_argument("LFM2 CPU expects a non-empty rank-2 int64 CPU sequence with batch size 1");
    }
    const int32_t sequence_length = shape[1];
    const int32_t cached_tokens = kv_cache_.getCurrentSeqCnt(0);
    if (sequence_length > max_length_ || cached_tokens > max_length_ - sequence_length) {
      throw std::invalid_argument("LFM2 sequence exceeds the configured cache capacity");
    }

    // Generation orchestration: position IDs and analytical sin/cos buffers
    // are inputs to the registered nn::RoPE operations, not backend kernels.
    Tensor position_ids;
    if (input.count("position_ids")) {
      position_ids = input.at("position_ids");
      if (sequence_length == 1) {
        const auto previous = *position_ids.offsettedPtr<int64_t>({0, position_ids.shape()[1] - 1});
        position_ids = Tensor::empty({1, 1}, kInt64, kCPU).alloc();
        *position_ids.ptr<int64_t>() = previous + 1;
      }
    } else {
      if (cached_tokens != 0) {
        throw std::invalid_argument(
            "LFM2 continuation with a non-empty KV cache requires the position_ids returned by the previous step");
      }
      position_ids = Tensor::empty({1, sequence_length}, kInt64, kCPU).alloc();
      for (int32_t index = 0; index < sequence_length; ++index) position_ids.ptr<int64_t>()[index] = index;
    }
    auto [sin, cos] = makeRotaryPosEmbedding(position_ids, getBuffer("inv_freq"));
    sequence = model_(sequence, sin, cos, AnyValue(&kv_cache_))[0];
    sequence = sequence[{kAll, {sequence.shape()[1] - 1}, kAll}];
    sequence = lm_head_(sequence);
    return {{"sequence", sequence}, {"position_ids", position_ids}};
  }

  void resetState(int32_t batch_size = 1) {
    if (batch_size != 1) { throw std::invalid_argument("LFM2 CPU currently supports batch size 1 only"); }
    kv_cache_.clearCache();
    model_.resetConvStates(batch_size);
  }
  [[nodiscard]] nn::KVHeadStaticCache& kvCache() { return kv_cache_; }

 private:
  Lfm2Model model_;
  nn::Linear lm_head_;
  nn::KVHeadStaticCache kv_cache_;
};

}  // namespace mllm::models::lfm2
