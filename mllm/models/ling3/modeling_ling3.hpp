// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/mllm.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/models/ling3/configuration_ling3.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/lmcache/StaticCache.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mllm::models::ling3 {

// Model-local tensor preparation: Ling-3 uses an adjacent-pair checkpoint RoPE
// layout that must be converted to mllm's half-split attention layout.  These
// helpers prepare RoPE/cache tensors only; reusable compute is dispatched
// through registered nn layers and functional operations below.
inline Tensor makeLing3RoPEInvFreq(int rotary_dim, float rope_theta) {
  auto inv_freq = Tensor::empty({rotary_dim / 2}, kFloat32, kCPU).alloc();
  auto* values = inv_freq.ptr<float>();
  for (int index = 0; index < rotary_dim / 2; ++index) {
    values[index] = 1.0F / std::pow(rope_theta, 2.0F * static_cast<float>(index) / static_cast<float>(rotary_dim));
  }
  return inv_freq;
}

inline std::pair<Tensor, Tensor> makeLing3RotaryEmbedding(const Tensor& position_ids, const Tensor& inv_freq) {
  if (position_ids.shape().size() != 2 || position_ids.dtype() != kInt64 || inv_freq.shape().size() != 1
      || inv_freq.dtype() != kFloat32) {
    throw std::invalid_argument("Ling-3 RoPE received invalid position ids or frequencies");
  }
  const int batch = position_ids.shape()[0];
  const int sequence = position_ids.shape()[1];
  const int half_dim = inv_freq.shape()[0];
  const int rotary_dim = half_dim * 2;
  auto cos = Tensor::empty({batch, sequence, rotary_dim}, kFloat32, kCPU).alloc();
  auto sin = Tensor::empty({batch, sequence, rotary_dim}, kFloat32, kCPU).alloc();
  const auto* positions = position_ids.ptr<int64_t>();
  const auto* frequencies = inv_freq.ptr<float>();
  auto* cos_values = cos.ptr<float>();
  auto* sin_values = sin.ptr<float>();
  for (int b = 0; b < batch; ++b) {
    for (int s = 0; s < sequence; ++s) {
      for (int dim = 0; dim < half_dim; ++dim) {
        const float angle = static_cast<float>(positions[b * sequence + s]) * frequencies[dim];
        const float cos_value = std::cos(angle);
        const float sin_value = std::sin(angle);
        const std::size_t base = (static_cast<std::size_t>(b) * sequence + s) * rotary_dim;
        cos_values[base + dim] = cos_value;
        cos_values[base + half_dim + dim] = cos_value;
        sin_values[base + dim] = sin_value;
        sin_values[base + half_dim + dim] = sin_value;
      }
    }
  }
  return {cos, sin};
}

// Converts adjacent-pair RoPE storage [x0, x1, x2, x3, ...] to the
// half-split layout used by the official Bailing implementation while applying
// the rotation. Input/output are [B, H, S, D]; cos/sin are [B, S, D].
inline Tensor applyLing3InterleavedRoPE(Tensor input, const Tensor& cos, const Tensor& sin) {
  if (input.dtype() != kFloat32 || cos.dtype() != kFloat32 || sin.dtype() != kFloat32 || input.shape().size() != 4
      || cos.shape().size() != 3 || sin.shape() != cos.shape()) {
    throw std::invalid_argument("Ling-3 interleaved RoPE currently requires float32 tensors");
  }
  const int batch = input.shape()[0];
  const int heads = input.shape()[1];
  const int sequence = input.shape()[2];
  const int dim = input.shape()[3];
  if (dim <= 0 || dim % 2 != 0 || cos.shape()[0] != batch || cos.shape()[1] != sequence || cos.shape()[2] != dim) {
    throw std::invalid_argument("Ling-3 interleaved RoPE received incompatible shapes");
  }
  auto source = input.contiguous();
  auto output = Tensor::empty(input.shape(), kFloat32, kCPU).alloc();
  const auto* source_values = source.ptr<float>();
  const auto* cos_values = cos.ptr<float>();
  const auto* sin_values = sin.ptr<float>();
  auto* output_values = output.ptr<float>();
  const int half_dim = dim / 2;
  for (int b = 0; b < batch; ++b) {
    for (int h = 0; h < heads; ++h) {
      for (int s = 0; s < sequence; ++s) {
        const std::size_t vector_base = ((static_cast<std::size_t>(b) * heads + h) * sequence + s) * dim;
        const std::size_t rope_base = (static_cast<std::size_t>(b) * sequence + s) * dim;
        for (int pair = 0; pair < half_dim; ++pair) {
          const float even = source_values[vector_base + pair * 2];
          const float odd = source_values[vector_base + pair * 2 + 1];
          const float cos_value = cos_values[rope_base + pair];
          const float sin_value = sin_values[rope_base + pair];
          output_values[vector_base + pair] = even * cos_value - odd * sin_value;
          output_values[vector_base + half_dim + pair] = odd * cos_value + even * sin_value;
        }
      }
    }
  }
  return output;
}

inline Tensor padLing3ValuesForCache(Tensor input, int output_dim) {
  if (input.dtype() != kFloat32 || input.shape().size() != 4 || input.shape()[3] <= 0 || input.shape()[3] > output_dim) {
    throw std::invalid_argument("Ling-3 MLA value-cache padding received an invalid tensor");
  }
  auto source = input.contiguous();
  auto output = Tensor::zeros({input.shape()[0], input.shape()[1], input.shape()[2], output_dim}, kFloat32, kCPU);
  const int input_dim = input.shape()[3];
  const std::size_t vectors = static_cast<std::size_t>(input.shape()[0]) * input.shape()[1] * input.shape()[2];
  const auto* source_values = source.ptr<float>();
  auto* output_values = output.ptr<float>();
  for (std::size_t vector = 0; vector < vectors; ++vector) {
    std::memcpy(output_values + vector * output_dim, source_values + vector * input_dim,
                static_cast<std::size_t>(input_dim) * sizeof(float));
  }
  return output;
}

class Ling3MLP final : public nn::Module {
 public:
  Ling3MLP() = default;
  Ling3MLP(const std::string& name, const Ling3Config& config, const std::optional<int32_t>& intermediate_size = std::nullopt)
      : nn::Module(name) {
    const int32_t intermediate = intermediate_size.value_or(config.intermediate_size);
    gate_proj_ = reg<nn::Linear>("gate_proj", config.hidden_size, intermediate, false, config.linear_impl_type);
    up_proj_ = reg<nn::Linear>("up_proj", config.hidden_size, intermediate, false, config.linear_impl_type);
    down_proj_ = reg<nn::Linear>("down_proj", intermediate, config.hidden_size, false, config.linear_impl_type);
    activation_ = reg<nn::SiLU>("act");
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    return {down_proj_(activation_(gate_proj_(inputs[0])) * up_proj_(inputs[0]))};
  }

 private:
  nn::Linear gate_proj_;
  nn::Linear up_proj_;
  nn::Linear down_proj_;
  nn::SiLU activation_;
};

class Ling3MoEGate final : public nn::Module {
 public:
  Ling3MoEGate() = default;
  Ling3MoEGate(const std::string& name, const Ling3Config& config) : nn::Module(name) {
    hidden_size_ = config.hidden_size;
    num_experts_ = config.num_experts;
    top_k_ = config.num_experts_per_tok;
    num_groups_ = config.n_group;
    top_groups_ = config.topk_group;
    routed_scaling_factor_ = config.routed_scaling_factor;
    weight_ = reg<nn::Param>("weight", getModuleName() + ".weight");
    expert_bias_ = reg<nn::Param>("expert_bias", getModuleName() + ".expert_bias");
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden = inputs[0];
    if (hidden.dtype() != kFloat32 || hidden.shape().size() != 3 || hidden.shape()[2] != hidden_size_
        || weight_.weight().dtype() != kFloat32 || expert_bias_.weight().dtype() != kFloat32) {
      throw std::invalid_argument("Ling-3 MoE router requires float32 tensors with official shapes");
    }
    // Model-level orchestration: the official Ling grouped top-k rule includes
    // expert bias and a top-2-per-group score, which the existing generic mllm
    // MoE layers do not represent.  Matmul and every expert MLP remain normal
    // nn operations; this loop only produces routing ids and weights.
    hidden = hidden.view({-1, hidden_size_});
    auto logits = nn::functional::matmul(hidden, weight_.weight(), false, true).contiguous();
    const int tokens = hidden.shape()[0];
    auto topk_ids = Tensor::empty({tokens, top_k_}, kInt32, kCPU).alloc();
    auto topk_weights = Tensor::empty({tokens, top_k_}, kFloat32, kCPU).alloc();
    const auto* logits_values = logits.ptr<float>();
    const auto* bias_values = expert_bias_.weight().ptr<float>();
    auto* id_values = topk_ids.ptr<int32_t>();
    auto* weight_values = topk_weights.ptr<float>();
    const int experts_per_group = num_experts_ / num_groups_;

    std::vector<float> scores(static_cast<std::size_t>(num_experts_));
    std::vector<std::pair<float, int32_t>> ranked_groups(static_cast<std::size_t>(num_groups_));
    std::vector<std::pair<float, int32_t>> ranked_experts;
    ranked_experts.reserve(static_cast<std::size_t>(top_groups_ * experts_per_group));
    for (int token = 0; token < tokens; ++token) {
      for (int expert = 0; expert < num_experts_; ++expert) {
        const float value = logits_values[static_cast<std::size_t>(token) * num_experts_ + expert];
        scores[expert] = value >= 0.0F ? 1.0F / (1.0F + std::exp(-value)) : std::exp(value) / (1.0F + std::exp(value));
      }
      for (int group = 0; group < num_groups_; ++group) {
        float largest = -std::numeric_limits<float>::infinity();
        float second_largest = -std::numeric_limits<float>::infinity();
        for (int offset = 0; offset < experts_per_group; ++offset) {
          const int expert = group * experts_per_group + offset;
          const float routed_score = scores[expert] + bias_values[expert];
          if (routed_score > largest) {
            second_largest = largest;
            largest = routed_score;
          } else if (routed_score > second_largest) {
            second_largest = routed_score;
          }
        }
        ranked_groups[group] = {largest + second_largest, group};
      }
      std::partial_sort(ranked_groups.begin(), ranked_groups.begin() + top_groups_, ranked_groups.end(),
                        [](const auto& lhs, const auto& rhs) {
                          return lhs.first != rhs.first ? lhs.first > rhs.first : lhs.second < rhs.second;
                        });
      ranked_experts.clear();
      for (int group_index = 0; group_index < top_groups_; ++group_index) {
        const int group = ranked_groups[group_index].second;
        for (int offset = 0; offset < experts_per_group; ++offset) {
          const int expert = group * experts_per_group + offset;
          ranked_experts.emplace_back(scores[expert] + bias_values[expert], expert);
        }
      }
      std::partial_sort(ranked_experts.begin(), ranked_experts.begin() + top_k_, ranked_experts.end(),
                        [](const auto& lhs, const auto& rhs) {
                          return lhs.first != rhs.first ? lhs.first > rhs.first : lhs.second < rhs.second;
                        });
      float score_sum = 1.0e-20F;
      for (int route = 0; route < top_k_; ++route) { score_sum += scores[ranked_experts[route].second]; }
      for (int route = 0; route < top_k_; ++route) {
        const int expert = ranked_experts[route].second;
        id_values[token * top_k_ + route] = expert;
        weight_values[token * top_k_ + route] = scores[expert] / score_sum * routed_scaling_factor_;
      }
    }
    return {topk_ids, topk_weights};
  }

 private:
  int32_t hidden_size_ = 0;
  int32_t num_experts_ = 0;
  int32_t top_k_ = 0;
  int32_t num_groups_ = 0;
  int32_t top_groups_ = 0;
  float routed_scaling_factor_ = 1.0F;
  nn::Param weight_;
  nn::Param expert_bias_;
};

class Ling3SparseMoE final : public nn::Module {
 public:
  Ling3SparseMoE() = default;
  Ling3SparseMoE(const std::string& name, const Ling3Config& config) : nn::Module(name) {
    top_k_ = config.num_experts_per_tok;
    experts_ = reg<nn::ModuleList<Ling3MLP>>("experts", config.num_experts, config,
                                             std::optional<int32_t>(config.moe_intermediate_size));
    gate_ = reg<Ling3MoEGate>("gate", config);
    shared_experts_ =
        reg<Ling3MLP>("shared_experts", config,
                      std::optional<int32_t>(config.moe_shared_expert_intermediate_size * config.num_shared_experts));
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto identity = inputs[0];
    const auto original_shape = identity.shape();
    auto routing = gate_(identity);
    auto hidden = identity.view({-1, identity.shape().back()});
    auto routed = moeInfer(hidden, routing[0], routing[1]).view(original_shape);
    return {routed + shared_experts_(identity)[0]};
  }

 private:
  Tensor moeInfer(const Tensor& input, Tensor& topk_ids, Tensor& topk_weights) {
    const int tokens = topk_ids.shape()[0];
    const int hidden_size = input.shape()[1];
    auto routed_output = Tensor::empty({tokens, top_k_, hidden_size}, kFloat32, kCPU).alloc();
    const auto* id_values = topk_ids.ptr<int32_t>();
    auto* routed_values = routed_output.ptr<float>();
    // Model-level sparse dispatch: keep each routed expert invocation at M=1.
    // KAI's dynamic activation
    // quantization is row-local, and this avoids platform-dependent grouped
    // prefill results while retaining the same token/expert routing contract.
    for (int token = 0; token < tokens; ++token) {
      auto expert_input = input[{{token, token + 1}, kAll}];
      for (int route = 0; route < top_k_; ++route) {
        const int expert = id_values[token * top_k_ + route];
        auto expert_output = experts_.list()[expert](expert_input)[0].contiguous();
        std::memcpy(routed_values + (static_cast<std::size_t>(token) * top_k_ + route) * hidden_size,
                    expert_output.ptr<float>(), static_cast<std::size_t>(hidden_size) * sizeof(float));
      }
    }
    return routed_output.mul_(topk_weights.unsqueeze(-1)).sum(1).to(routed_output.dtype());
  }

  int32_t top_k_ = 0;
  nn::ModuleList<Ling3MLP> experts_;
  Ling3MoEGate gate_;
  Ling3MLP shared_experts_;
};

class Ling3KimiDeltaAttention final : public nn::Module {
 public:
  // The official checkpoint was validated against the current-first
  // accumulation order (current tap first, then the K - 1 history taps).
  static constexpr auto kConvAccumulationOrder = aops::CausalDepthwiseConv1DAccumulationOrder::kCurrentFirst;

  Ling3KimiDeltaAttention() = default;
  Ling3KimiDeltaAttention(const std::string& name, const Ling3Config& config) : nn::Module(name) {
    hidden_size_ = config.hidden_size;
    num_heads_ = config.num_attention_heads;
    head_dim_ = config.head_dim;
    projection_size_ = num_heads_ * head_dim_;
    conv_size_ = config.short_conv_kernel_size;
    safe_gate_ = config.kda_safe_gate;
    lower_bound_ = config.kda_lower_bound;

    q_proj_ = reg<nn::Linear>("q_proj", hidden_size_, projection_size_, false, config.linear_impl_type);
    k_proj_ = reg<nn::Linear>("k_proj", hidden_size_, projection_size_, false, config.linear_impl_type);
    v_proj_ = reg<nn::Linear>("v_proj", hidden_size_, projection_size_, false, config.linear_impl_type);
    // Stateful depthwise causal short convolutions.  The registered operation
    // owns the official `{q,k,v}_conv1d.weight` parameters and the explicit
    // [B, C, K - 1] history; the current-first accumulation order reuses the
    // existing GDN causal-convolution kernel.
    q_conv1d_ = reg<nn::CausalDepthwiseConv1D>("q_conv1d", projection_size_, conv_size_, /*bias=*/false,
                                               /*state_inplace=*/true, kConvAccumulationOrder);
    k_conv1d_ = reg<nn::CausalDepthwiseConv1D>("k_conv1d", projection_size_, conv_size_, /*bias=*/false,
                                               /*state_inplace=*/true, kConvAccumulationOrder);
    v_conv1d_ = reg<nn::CausalDepthwiseConv1D>("v_conv1d", projection_size_, conv_size_, /*bias=*/false,
                                               /*state_inplace=*/true, kConvAccumulationOrder);
    f_proj_ = reg<nn::Linear>("f_proj", hidden_size_, projection_size_, false, config.linear_impl_type);
    b_proj_ = reg<nn::Linear>("b_proj", hidden_size_, num_heads_, false, config.linear_impl_type);
    g_proj_ = reg<nn::Linear>("g_proj", hidden_size_, projection_size_, false, config.linear_impl_type);
    A_log_ = reg<nn::Param>("A_log", getModuleName() + ".A_log");
    dt_bias_ = reg<nn::Param>("dt_bias", getModuleName() + ".dt_bias");
    o_norm_ = reg<nn::RMSNorm>("o_norm", config.rms_norm_eps, false);
    o_proj_ = reg<nn::Linear>("o_proj", projection_size_, hidden_size_, false, config.linear_impl_type);
    silu_ = reg<nn::SiLU>("conv_activation");
    sigmoid_ = reg<nn::Sigmoid>("gate_activation");
    kda_ = reg<nn::KimiDeltaAttention>("kda", safe_gate_, lower_bound_, /*state_inplace=*/true);
  }

  void resetState(int batch_size) {
    if (batch_size <= 0) { throw std::invalid_argument("Ling-3 KDA reset requires a positive batch size"); }
    // Module-owned request state.  State transitions themselves are explicit
    // outputs of the registered causal-convolution and KDA operations.
    recurrent_state_ = Tensor::zeros({batch_size, num_heads_, head_dim_, head_dim_}, kFloat32, kCPU);
    q_conv_state_ = Tensor::zeros({batch_size, projection_size_, conv_size_ - 1}, kFloat32, kCPU);
    k_conv_state_ = Tensor::zeros({batch_size, projection_size_, conv_size_ - 1}, kFloat32, kCPU);
    v_conv_state_ = Tensor::zeros({batch_size, projection_size_, conv_size_ - 1}, kFloat32, kCPU);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden = inputs[0];
    const int batch = hidden.shape()[0];
    const int sequence = hidden.shape()[1];
    if (recurrent_state_.isNil() || recurrent_state_.shape()[0] != batch) { resetState(batch); }

    auto q = q_proj_(hidden).contiguous();
    auto k = k_proj_(hidden).contiguous();
    auto v = v_proj_(hidden).contiguous();
    auto gate_logits = f_proj_(hidden).contiguous();
    auto beta = sigmoid_(b_proj_(hidden)).contiguous();
    const auto a_log = A_log_.weight();
    const auto dt_bias = dt_bias_.weight();
    if (q.dtype() != kFloat32 || k.dtype() != kFloat32 || v.dtype() != kFloat32 || gate_logits.dtype() != kFloat32
        || beta.dtype() != kFloat32 || a_log.dtype() != kFloat32 || dt_bias.dtype() != kFloat32) {
      throw std::invalid_argument("Ling-3 KDA requires float32 recurrent activations and parameters");
    }

    auto [q_conv, updated_q_conv_state] = q_conv1d_(q, q_conv_state_);
    auto [k_conv, updated_k_conv_state] = k_conv1d_(k, k_conv_state_);
    auto [v_conv, updated_v_conv_state] = v_conv1d_(v, v_conv_state_);
    q_conv_state_ = std::move(updated_q_conv_state);
    k_conv_state_ = std::move(updated_k_conv_state);
    v_conv_state_ = std::move(updated_v_conv_state);
    q = silu_(q_conv).view({batch, sequence, num_heads_, head_dim_}).contiguous();
    k = silu_(k_conv).view({batch, sequence, num_heads_, head_dim_}).contiguous();
    v = silu_(v_conv).view({batch, sequence, num_heads_, head_dim_}).contiguous();
    gate_logits = gate_logits.view({batch, sequence, num_heads_, head_dim_}).contiguous();
    auto [output, updated_state] = kda_(q, k, v, gate_logits, beta, a_log, dt_bias, recurrent_state_);
    recurrent_state_ = std::move(updated_state);

    output = output.view({batch * sequence * num_heads_, head_dim_});
    auto output_gate = sigmoid_(g_proj_(hidden).view({batch * sequence * num_heads_, head_dim_}));
    output = o_norm_(output) * output_gate;
    output = output.view({batch, sequence, projection_size_});
    return {o_proj_(output)};
  }

 private:
  int32_t hidden_size_ = 0;
  int32_t num_heads_ = 0;
  int32_t head_dim_ = 0;
  int32_t projection_size_ = 0;
  int32_t conv_size_ = 0;
  bool safe_gate_ = true;
  float lower_bound_ = -5.0F;
  nn::Linear q_proj_, k_proj_, v_proj_;
  nn::CausalDepthwiseConv1D q_conv1d_, k_conv1d_, v_conv1d_;
  nn::Linear f_proj_, b_proj_, g_proj_;
  nn::Param A_log_, dt_bias_;
  nn::RMSNorm o_norm_;
  nn::Linear o_proj_;
  nn::SiLU silu_;
  nn::Sigmoid sigmoid_;
  nn::KimiDeltaAttention kda_;
  Tensor recurrent_state_, q_conv_state_, k_conv_state_, v_conv_state_;
};

class Ling3MultiLatentAttention final : public nn::Module {
 public:
  int32_t cache_layer_index_ = 0;

  Ling3MultiLatentAttention() = default;
  Ling3MultiLatentAttention(const std::string& name, const Ling3Config& config) : nn::Module(name) {
    hidden_size_ = config.hidden_size;
    num_heads_ = config.num_attention_heads;
    q_lora_rank_ = config.q_lora_rank;
    kv_lora_rank_ = config.kv_lora_rank;
    qk_rope_dim_ = config.qk_rope_head_dim;
    qk_nope_dim_ = config.qk_nope_head_dim;
    qk_dim_ = config.qk_head_dim;
    value_dim_ = config.v_head_dim;
    q_a_proj_ = reg<nn::Linear>("q_a_proj", hidden_size_, q_lora_rank_, config.use_qkv_bias, config.linear_impl_type);
    q_a_layernorm_ = reg<nn::RMSNorm>("q_a_layernorm", config.rms_norm_eps, false);
    q_b_proj_ = reg<nn::Linear>("q_b_proj", q_lora_rank_, num_heads_ * qk_dim_, false, config.linear_impl_type);
    kv_a_proj_ = reg<nn::Linear>("kv_a_proj_with_mqa", hidden_size_, kv_lora_rank_ + qk_rope_dim_, config.use_qkv_bias,
                                 config.linear_impl_type);
    kv_a_layernorm_ = reg<nn::RMSNorm>("kv_a_layernorm", config.rms_norm_eps, false);
    kv_b_proj_ =
        reg<nn::Linear>("kv_b_proj", kv_lora_rank_, num_heads_ * (qk_nope_dim_ + value_dim_), false, config.linear_impl_type);
    g_proj_ = reg<nn::Linear>("g_proj", hidden_size_, num_heads_, false, config.linear_impl_type);
    dense_ = reg<nn::Linear>("dense", num_heads_ * value_dim_, hidden_size_, config.use_qkv_bias, config.linear_impl_type);
    mask_ = reg<nn::CausalMask>("mask");
    softmax_ = reg<nn::Softmax>("softmax", -1);
    sigmoid_ = reg<nn::Sigmoid>("gate_sigmoid");
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden = inputs[0];
    const auto cos = inputs[1];
    const auto sin = inputs[2];
    auto* cache = args[0].get<nn::StaticCache*>();
    const int batch = hidden.shape()[0];
    const int sequence = hidden.shape()[1];

    auto query = q_b_proj_(q_a_layernorm_(q_a_proj_(hidden))).view({batch, sequence, num_heads_, qk_dim_}).transpose(1, 2);
    auto q_nope = query[{kAll, kAll, kAll, {0, qk_nope_dim_}}].contiguous();
    auto q_rope = query[{kAll, kAll, kAll, {qk_nope_dim_, qk_dim_}}].contiguous();

    auto compressed_kv = kv_a_proj_(hidden);
    auto kv_latent = compressed_kv[{kAll, kAll, {0, kv_lora_rank_}}].contiguous();
    auto k_rope = compressed_kv[{kAll, kAll, {kv_lora_rank_, kv_lora_rank_ + qk_rope_dim_}}]
                      .contiguous()
                      .view({batch, sequence, 1, qk_rope_dim_})
                      .transpose(1, 2);
    auto kv_expanded =
        kv_b_proj_(kv_a_layernorm_(kv_latent)).view({batch, sequence, num_heads_, qk_nope_dim_ + value_dim_}).transpose(1, 2);
    auto k_nope = kv_expanded[{kAll, kAll, kAll, {0, qk_nope_dim_}}].contiguous();
    auto value = kv_expanded[{kAll, kAll, kAll, {qk_nope_dim_, qk_nope_dim_ + value_dim_}}].contiguous();

    q_rope = applyLing3InterleavedRoPE(q_rope, cos, sin);
    k_rope = applyLing3InterleavedRoPE(k_rope, cos, sin).repeat(num_heads_, 1);
    query = nn::functional::concat({q_nope, q_rope}, -1);
    auto key = nn::functional::concat({k_nope, k_rope}, -1);
    auto padded_value = padLing3ValuesForCache(value, qk_dim_);
    auto cached = cache->updateKVCache(cache_layer_index_, key, padded_value);
    key = cached[0];
    padded_value = cached[1];

    auto attention = nn::functional::matmul(query, key, false, true) * (1.0F / std::sqrt(static_cast<float>(qk_dim_)));
    attention = softmax_(mask_(attention));
    auto output = nn::functional::matmul(attention, padded_value);
    output = output[{kAll, kAll, kAll, {0, value_dim_}}].contiguous().transpose(1, 2);
    auto gate = sigmoid_(g_proj_(hidden)).unsqueeze(-1);
    output = output * gate;
    output = output.view({batch, sequence, num_heads_ * value_dim_});
    return {dense_(output)};
  }

 private:
  int32_t hidden_size_ = 0, num_heads_ = 0, q_lora_rank_ = 0, kv_lora_rank_ = 0;
  int32_t qk_rope_dim_ = 0, qk_nope_dim_ = 0, qk_dim_ = 0, value_dim_ = 0;
  nn::Linear q_a_proj_, q_b_proj_, kv_a_proj_, kv_b_proj_, g_proj_, dense_;
  nn::RMSNorm q_a_layernorm_, kv_a_layernorm_;
  nn::CausalMask mask_;
  nn::Softmax softmax_;
  nn::Sigmoid sigmoid_;
};

class Ling3KDAdecoder final : public nn::Module {
 public:
  Ling3KimiDeltaAttention attention_;

  Ling3KDAdecoder() = default;
  Ling3KDAdecoder(const std::string& name, const Ling3Config& config, int32_t layer_index) : nn::Module(name) {
    attention_ = reg<Ling3KimiDeltaAttention>("attention", config);
    input_layernorm_ = reg<nn::RMSNorm>("input_layernorm", config.rms_norm_eps, false);
    post_attention_layernorm_ = reg<nn::RMSNorm>("post_attention_layernorm", config.rms_norm_eps, false);
    use_moe_ = layer_index >= config.first_k_dense_replace;
    if (use_moe_) {
      moe_ = reg<Ling3SparseMoE>("mlp", config);
    } else {
      dense_mlp_ = reg<Ling3MLP>("mlp", config);
    }
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden = inputs[0];
    auto attention_output = attention_(input_layernorm_(hidden))[0];
    hidden = hidden + attention_output;
    auto mlp_input = post_attention_layernorm_(hidden);
    return {hidden + (use_moe_ ? moe_(mlp_input)[0] : dense_mlp_(mlp_input)[0])};
  }

 private:
  bool use_moe_ = false;
  nn::RMSNorm input_layernorm_, post_attention_layernorm_;
  Ling3MLP dense_mlp_;
  Ling3SparseMoE moe_;
};

class Ling3MLADecoder final : public nn::Module {
 public:
  Ling3MultiLatentAttention attention_;

  Ling3MLADecoder() = default;
  Ling3MLADecoder(const std::string& name, const Ling3Config& config, int32_t layer_index) : nn::Module(name) {
    attention_ = reg<Ling3MultiLatentAttention>("attention", config);
    input_layernorm_ = reg<nn::RMSNorm>("input_layernorm", config.rms_norm_eps, false);
    post_attention_layernorm_ = reg<nn::RMSNorm>("post_attention_layernorm", config.rms_norm_eps, false);
    use_moe_ = layer_index >= config.first_k_dense_replace;
    if (use_moe_) {
      moe_ = reg<Ling3SparseMoE>("mlp", config);
    } else {
      dense_mlp_ = reg<Ling3MLP>("mlp", config);
    }
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden = inputs[0];
    auto attention_output = attention_(input_layernorm_(hidden), inputs[1], inputs[2], args[0])[0];
    hidden = hidden + attention_output;
    auto mlp_input = post_attention_layernorm_(hidden);
    return {hidden + (use_moe_ ? moe_(mlp_input)[0] : dense_mlp_(mlp_input)[0])};
  }

 private:
  bool use_moe_ = false;
  nn::RMSNorm input_layernorm_, post_attention_layernorm_;
  Ling3MLP dense_mlp_;
  Ling3SparseMoE moe_;
};

class Ling3Model final : public nn::Module {
 public:
  Ling3Model() = default;
  Ling3Model(const std::string& name, const Ling3Config& config) : nn::Module(name) {
    word_embeddings_ = reg<nn::Embedding>("word_embeddings", config.vocab_size, config.hidden_size);
    int32_t kda_index = 0;
    int32_t mla_index = 0;
    for (int32_t layer = 0; layer < config.num_hidden_layers; ++layer) {
      const std::string layer_name = "layers." + std::to_string(layer);
      if (config.isFullAttentionLayer(layer)) {
        auto decoder = reg<Ling3MLADecoder>(layer_name, config, layer);
        decoder.attention_.cache_layer_index_ = mla_index;
        mla_layers_.push_back(std::move(decoder));
        layer_types_.push_back(0);
        layer_dispatch_.push_back(mla_index++);
      } else {
        kda_layers_.push_back(reg<Ling3KDAdecoder>(layer_name, config, layer));
        layer_types_.push_back(1);
        layer_dispatch_.push_back(kda_index++);
      }
    }
    norm_ = reg<nn::RMSNorm>("norm", config.rms_norm_eps, false);
  }

  void resetKDAStates(int32_t batch_size) {
    for (auto& layer : kda_layers_) { layer.attention_.resetState(batch_size); }
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden = word_embeddings_(inputs[0]);
    for (size_t layer = 0; layer < layer_types_.size(); ++layer) {
      if (layer_types_[layer] == 0) {
        hidden = mla_layers_[layer_dispatch_[layer]](hidden, inputs[1], inputs[2], args[0])[0];
      } else {
        hidden = kda_layers_[layer_dispatch_[layer]](hidden)[0];
      }
    }
    return {norm_(hidden)};
  }

 private:
  nn::Embedding word_embeddings_;
  nn::RMSNorm norm_;
  std::vector<Ling3MLADecoder> mla_layers_;
  std::vector<Ling3KDAdecoder> kda_layers_;
  std::vector<int32_t> layer_types_;
  std::vector<int32_t> layer_dispatch_;
};

class Ling3ForCausalLM final : public ARGeneration, public nn::Module {
 public:
  explicit Ling3ForCausalLM(const Ling3Config& config) {
    if (!hasOfficialLing3TinyArchitecture(config)) {
      throw std::invalid_argument("Ling-3 CPU currently supports inclusionAI/Ling-3.0-tiny only");
    }
    kv_cache_ = nn::StaticCache(config.max_cache_length, config.numFullAttentionLayers(), config.num_attention_heads,
                                config.num_key_value_heads, config.qk_head_dim, kFloat32, kFloat32, kCPU, false);
    model_ = reg<Ling3Model>("model", config);
    lm_head_ = reg<nn::Linear>("lm_head", config.hidden_size, config.vocab_size, false, config.linear_impl_type);
    registerBuffer("inv_freq", makeLing3RoPEInvFreq(config.qk_rope_head_dim, config.rope_theta));
    eos_token_id_ = config.eos_token_id;
    additional_eos_token_ids_.insert(config.end_of_text_token_id);
    max_length_ = config.max_cache_length;
  }

  ARGenerationOutputPast forward(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override {
    auto sequence = input.at("sequence");
    if (sequence.shape().size() != 2 || sequence.shape()[0] != 1 || sequence.shape()[1] <= 0 || sequence.dtype() != kInt64
        || sequence.device() != kCPU) {
      throw std::invalid_argument("Ling-3 CPU expects a non-empty rank-2 int64 CPU sequence with batch size 1");
    }
    const int sequence_length = sequence.shape()[1];
    const int cached_tokens = kv_cache_.getCurrentSeqCnt(0);
    if (sequence_length > max_length_ || cached_tokens > max_length_ - sequence_length) {
      throw std::invalid_argument("Ling-3 sequence exceeds the configured KV-cache capacity");
    }

    // Autoregressive input orchestration; position ids feed the model-local
    // checkpoint-layout RoPE preparation above.
    auto position_ids = Tensor::empty({1, sequence_length}, kInt64, kCPU).alloc();
    auto* positions = position_ids.ptr<int64_t>();
    if (input.count("position_ids") != 0 && sequence_length == 1) {
      auto previous_positions = input.at("position_ids");
      previous_positions = previous_positions.contiguous();
      positions[0] = previous_positions.ptr<int64_t>()[previous_positions.numel() - 1] + 1;
    } else {
      for (int index = 0; index < sequence_length; ++index) { positions[index] = cached_tokens + index; }
    }
    auto [cos, sin] = makeLing3RotaryEmbedding(position_ids, getBuffer("inv_freq"));
    auto hidden = model_(sequence, cos, sin, AnyValue(&kv_cache_))[0];
    hidden = hidden[{kAll, {sequence_length - 1}, kAll}];
    return {{"sequence", lm_head_(hidden)}, {"position_ids", position_ids}};
  }

  void resetState(int32_t batch_size = 1) {
    if (batch_size != 1) { throw std::invalid_argument("Ling-3 CPU currently supports batch size 1 only"); }
    kv_cache_.clearCache();
    model_.resetKDAStates(batch_size);
  }

  nn::StaticCache& kvCache() { return kv_cache_; }

 private:
  Ling3Model model_;
  nn::Linear lm_head_;
  nn::StaticCache kv_cache_;
};

}  // namespace mllm::models::ling3
