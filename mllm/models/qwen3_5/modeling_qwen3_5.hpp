// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Qwen3.5 hybrid text model. The configured layer schedule interleaves GDN
// (Gated Delta Network) layers with full-attention layers. GDN layers use
// linear attention with recurrent state; full-attention layers use GQA with
// partial RoPE and output gating.

#pragma once

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "mllm/mllm.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/nn/lmcache/StaticCache.hpp"
#include "mllm/models/qwen3_5/configuration_qwen3_5.hpp"
#include "mllm/models/qwen3_5/modeling_qwen3_5_vision.hpp"
#include "mllm/models/qwen3_5/multimodal_qwen3_5.hpp"
#include "mllm/utils/Enumerate.hpp"
#include "mllm/models/ARGeneration.hpp"

namespace mllm::models::qwen3_5 {

// ---------------------------------------------------------------------------
// RoPE helpers (partial rotation: only first rotary_dim dims are rotated)
// ---------------------------------------------------------------------------

inline auto makeRoPEInvFreq(int output_dim, float rope_theta) -> Tensor {
  auto inv_freq = Tensor::empty({output_dim / 2}, kFloat32, kCPU).alloc();
  auto inv_freq_ptr = inv_freq.ptr<float>();
  for (int i = 0; i < output_dim / 2; i++) { inv_freq_ptr[i] = 1.0 / std::pow(rope_theta, 2.0 * i / output_dim); }
  return inv_freq;
}

inline auto makeRotaryPosEmbedding(Tensor& position_ids, const Tensor& inv_freq, float attention_scaling = 1.0f)
    -> std::pair<Tensor, Tensor> {
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

// ---------------------------------------------------------------------------
// MLP (shared by both full attention and GDN decoder layers)
// ---------------------------------------------------------------------------

class Qwen3_5MLP final : public nn::Module {
  nn::Linear gate_proj_;
  nn::Linear up_proj_;
  nn::Linear down_proj_;
  nn::SiLU silu_;

 public:
  Qwen3_5MLP() = default;
  Qwen3_5MLP(const std::string& name, const Qwen3_5Config& cfg) : nn::Module(name) {
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

// ---------------------------------------------------------------------------
// Full Attention (GQA with partial RoPE, QK-norm, output gate)
// ---------------------------------------------------------------------------

class Qwen3_5FullAttention final : public nn::Module {
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
  nn::Sigmoid sigmoid_;

  int hidden_size_;
  int head_dim_;
  int num_attention_heads_;
  int num_key_value_heads_;
  bool attn_output_gate_;
  bool use_tiled_fa2_prefill_ = false;

 public:
  Qwen3_5FullAttention() = default;

  Qwen3_5FullAttention(const std::string& name, const Qwen3_5Config& cfg) : nn::Module(name) {
    use_tiled_fa2_prefill_ = cfg.use_tiled_fa2_prefill;
    hidden_size_ = cfg.hidden_size;
    num_attention_heads_ = cfg.num_attention_heads;
    num_key_value_heads_ = cfg.num_key_value_heads;
    head_dim_ = cfg.head_dim;
    attn_output_gate_ = cfg.attn_output_gate;

    // Q projection is 2x wide when output gating is enabled (second half = gate)
    int q_proj_out = head_dim_ * num_attention_heads_;
    if (attn_output_gate_) { q_proj_out *= 2; }

    q_proj_ = reg<nn::Linear>("q_proj", hidden_size_, q_proj_out, cfg.attention_bias, cfg.linear_impl_type);
    k_proj_ =
        reg<nn::Linear>("k_proj", hidden_size_, head_dim_ * num_key_value_heads_, cfg.attention_bias, cfg.linear_impl_type);
    v_proj_ =
        reg<nn::Linear>("v_proj", hidden_size_, head_dim_ * num_key_value_heads_, cfg.attention_bias, cfg.linear_impl_type);
    o_proj_ =
        reg<nn::Linear>("o_proj", head_dim_ * num_attention_heads_, hidden_size_, cfg.attention_bias, cfg.linear_impl_type);

    // GemmaRMSNorm: add_unit_offset=true → weight = weight + 1
    rms_norm_q_ = reg<nn::RMSNorm>("q_norm", cfg.rms_norm_eps, /*add_unit_offset=*/true);
    rms_norm_k_ = reg<nn::RMSNorm>("k_norm", cfg.rms_norm_eps, /*add_unit_offset=*/true);

    // Partial RoPE: only rotate first rotary_dim dimensions
    int rotary_dim = cfg.rotary_dim();
    q_rope_ = reg<nn::RoPE>("q_rope", cfg.rope_theta, cfg.max_position_embeddings, rotary_dim);
    k_rope_ = reg<nn::RoPE>("k_rope", cfg.rope_theta, cfg.max_position_embeddings, rotary_dim);

    mask_ = reg<nn::CausalMask>("mask");
    softmax_ = reg<nn::Softmax>("softmax", -1);
    sigmoid_ = reg<nn::Sigmoid>("gate_sigmoid");
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = inputs[0];
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto past_kv_cache = args[0].get<nn::StaticCache*>();

    int B = x.shape()[0];
    int S = x.shape()[1];

    // Projections
    auto q_raw = q_proj_(x);
    auto key_states = k_proj_(x);
    auto value_states = v_proj_(x);

    // Split Q into actual Q and gate if output gating enabled
    Tensor gate;
    Tensor query_states;
    if (attn_output_gate_) {
      // q_raw: [B, S, num_heads * head_dim * 2]
      // Reshape to [B, S, num_heads, head_dim * 2], split on last dim
      auto q_reshaped = q_raw.view({B, S, num_attention_heads_, head_dim_ * 2});
      // First half: query, second half: gate — contiguous() needed for subsequent view
      query_states = q_reshaped[{kAll, kAll, kAll, {0, head_dim_}}].contiguous().view({B, S, num_attention_heads_ * head_dim_});
      gate = q_reshaped[{kAll, kAll, kAll, {head_dim_, head_dim_ * 2}}].contiguous().view(
          {B, S, num_attention_heads_ * head_dim_});
    } else {
      query_states = q_raw;
    }

    // [B, S, H, D]
    query_states = query_states.view({B, S, num_attention_heads_, head_dim_});
    key_states = key_states.view({B, S, num_key_value_heads_, head_dim_});
    value_states = value_states.view({B, S, num_key_value_heads_, head_dim_});

    // QK normalization (GemmaRMSNorm)
    query_states = rms_norm_q_(query_states);
    key_states = rms_norm_k_(key_states);

    // [B, H, S, D]
    query_states = query_states.transpose(1, 2);
    key_states = key_states.transpose(1, 2);
    value_states = value_states.transpose(1, 2);

    // Partial RoPE (only first rotary_dim dims are rotated)
    query_states = q_rope_(query_states, llm_embedding_sin, llm_embedding_cos);
    key_states = k_rope_(key_states, llm_embedding_sin, llm_embedding_cos);

    // KV cache update
    auto [key_states_new, value_states_new] = past_kv_cache->updateKVCache(layer_idx_, key_states, value_states);
    key_states = key_states_new;
    value_states = value_states_new;

    // Attention
    constexpr auto layout = aops::FlashAttention2Layout::kBHSD;
    Tensor output;
    if (use_tiled_fa2_prefill_
        && nn::functional::supportsCpuFlashAttention2Prefill(query_states, key_states, value_states, layout)) {
      output = nn::functional::flashAttention2(query_states, key_states, value_states, layout);
    } else {
      Tensor attn;
      if (key_states.dtype() == kFloat32) {
        attn = nn::functional::matmul(query_states, key_states, false, true) * (1.f / sqrtf(head_dim_));
        attn = mask_(attn);
        attn = softmax_(attn);
      } else if (key_states.dtype() == kFloat16) {
        attn =
            nn::functional::matmul(query_states.to(kFloat32), key_states.to(kFloat32), false, true) * (1.f / sqrtf(head_dim_));
        attn = mask_(attn);
        attn = softmax_(attn);
        attn = attn.to(kFloat16);
      } else {
        throw std::invalid_argument("Qwen3.5 full attention only supports float32 and float16 activations");
      }
      output = nn::functional::matmul(attn, value_states);
    }

    // [B, H, S, D] -> [B, S, H*D]
    output = output.transpose(1, 2).view({B, S, num_attention_heads_ * head_dim_});

    // Output gate: output = output * sigmoid(gate)
    if (attn_output_gate_) {
      gate = sigmoid_(gate);
      output = output * gate;
    }

    output = o_proj_(output);
    return {output};
  }

  int layer_idx_ = 0;
};

// ---------------------------------------------------------------------------
// GDN Layer (Gated Delta Network — linear attention with recurrent state)
// ---------------------------------------------------------------------------

class Qwen3_5GDNLayer final : public nn::Module {
  // Input projections
  nn::Linear in_proj_qkv_;  // hidden → key_dim*2 + value_dim
  nn::Linear in_proj_z_;    // hidden → value_dim (gate for output)
  nn::Linear in_proj_a_;    // hidden → num_v_heads (decay gate)
  nn::Linear in_proj_b_;    // hidden → num_v_heads (beta gate)

  // Stateful sequence primitives and their checkpoint-owned parameters.
  nn::CausalDepthwiseConv1D causal_conv_;
  nn::GatedDeltaRule gated_delta_rule_;

  // Learnable parameters for gating
  nn::Param A_log_;    // [num_v_heads]
  nn::Param dt_bias_;  // [num_v_heads]

  // Output
  nn::RMSNorm norm_;  // standard RMSNorm (no add_unit_offset)
  nn::Linear out_proj_;

  nn::SiLU silu_;

  int hidden_size_;
  int num_k_heads_;
  int num_v_heads_;
  int head_k_dim_;
  int head_v_dim_;
  int key_dim_;
  int value_dim_;
  int conv_kernel_size_;

  // Recurrent state: [num_v_heads, head_v_dim, head_k_dim] per batch element
  // Conv state: [conv_dim, kernel_size-1] per batch element
  // Allocated on first forward call.
  Tensor recurrent_state_;  // [B, H, V, K]
  Tensor conv_state_;       // [B, conv_dim, kernel_size-1]

 public:
  Qwen3_5GDNLayer() = default;

  Qwen3_5GDNLayer(const std::string& name, const Qwen3_5Config& cfg) : nn::Module(name) {
    hidden_size_ = cfg.hidden_size;
    num_k_heads_ = cfg.linear_num_key_heads;
    num_v_heads_ = cfg.linear_num_value_heads;
    head_k_dim_ = cfg.linear_key_head_dim;
    head_v_dim_ = cfg.linear_value_head_dim;
    key_dim_ = head_k_dim_ * num_k_heads_;
    value_dim_ = head_v_dim_ * num_v_heads_;
    conv_kernel_size_ = cfg.linear_conv_kernel_dim;

    // Projections
    in_proj_qkv_ = reg<nn::Linear>("in_proj_qkv", hidden_size_, key_dim_ * 2 + value_dim_, false, cfg.linear_impl_type);
    in_proj_z_ = reg<nn::Linear>("in_proj_z", hidden_size_, value_dim_, false, cfg.linear_impl_type);
    in_proj_a_ = reg<nn::Linear>("in_proj_a", hidden_size_, num_v_heads_, false, cfg.linear_impl_type);
    in_proj_b_ = reg<nn::Linear>("in_proj_b", hidden_size_, num_v_heads_, false, cfg.linear_impl_type);

    int conv_channels = key_dim_ * 2 + value_dim_;
    causal_conv_ = reg<nn::CausalDepthwiseConv1D>(
        "conv1d", conv_channels, conv_kernel_size_, /*bias=*/false, /*state_inplace=*/true,
        aops::CausalDepthwiseConv1DAccumulationOrder::kCurrentFirst);
    gated_delta_rule_ = reg<nn::GatedDeltaRule>("gated_delta_rule", /*state_inplace=*/true);

    // Learnable gating parameters (loaded from weight file)
    A_log_ = reg<nn::Param>("A_log", getModuleName() + ".A_log", Tensor::shape_t{num_v_heads_});
    dt_bias_ = reg<nn::Param>("dt_bias", getModuleName() + ".dt_bias", Tensor::shape_t{num_v_heads_});

    // Gated RMSNorm — standard (NOT GemmaRMSNorm, no add_unit_offset)
    norm_ = reg<nn::RMSNorm>("norm", cfg.rms_norm_eps, /*add_unit_offset=*/false);

    out_proj_ = reg<nn::Linear>("out_proj", value_dim_, hidden_size_, false, cfg.linear_impl_type);

    silu_ = reg<nn::SiLU>("silu");
  }

  void resetState(int batch_size) {
    if (batch_size <= 0) { throw std::invalid_argument("Qwen3.5 GDN reset batch size must be positive"); }
    int conv_dim = key_dim_ * 2 + value_dim_;
    recurrent_state_ = Tensor::empty({batch_size, num_v_heads_, head_v_dim_, head_k_dim_}, kFloat32, kCPU).alloc();
    conv_state_ = Tensor::empty({batch_size, conv_dim, conv_kernel_size_ - 1}, kFloat32, kCPU).alloc();
    std::memset(recurrent_state_.ptr<float>(), 0, batch_size * num_v_heads_ * head_v_dim_ * head_k_dim_ * sizeof(float));
    std::memset(conv_state_.ptr<float>(), 0, batch_size * conv_dim * (conv_kernel_size_ - 1) * sizeof(float));
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = inputs[0];  // [B, S, H]
    int B = x.shape()[0];
    int S = x.shape()[1];
    int conv_dim = key_dim_ * 2 + value_dim_;

    // Lazy init recurrent + conv state. A batch-size change starts a new
    // independent sequence set and therefore cannot reuse the old states.
    if (recurrent_state_.isNil() || recurrent_state_.shape()[0] != B) { resetState(B); }

    // Input projections
    auto mixed_qkv = in_proj_qkv_(x);  // [B, S, key_dim*2 + value_dim]
    auto z = in_proj_z_(x);            // [B, S, value_dim]
    auto a_proj = in_proj_a_(x);       // [B, S, num_v_heads]
    auto b_proj = in_proj_b_(x);       // [B, S, num_v_heads]

    // The projection output, convolution weights, gates and persistent states
    // intentionally stay in float32. Mobile quantized Linear implementations
    // dequantize to float32 before this numerically sensitive recurrence.
    auto mixed_qkv_pre = mixed_qkv.contiguous();
    auto a_log = A_log_();
    auto dt_bias = dt_bias_();
    if (mixed_qkv_pre.dtype() != kFloat32 || a_proj.dtype() != kFloat32 || b_proj.dtype() != kFloat32
        || a_log.dtype() != kFloat32 || dt_bias.dtype() != kFloat32) {
      throw std::invalid_argument("Qwen3.5 GDN currently requires float32 activations and recurrent parameters");
    }

    auto [conv_out, updated_conv_state] = causal_conv_(mixed_qkv_pre, conv_state_);
    conv_state_ = updated_conv_state;

    // Qwen3.5 applies SiLU to the depthwise-convolution result before
    // splitting it into q, k and v.
    mixed_qkv = silu_(conv_out);
    auto q = mixed_qkv[{kAll, kAll, {0, key_dim_}}].contiguous();
    auto k = mixed_qkv[{kAll, kAll, {key_dim_, key_dim_ * 2}}].contiguous();
    auto v = mixed_qkv[{kAll, kAll, {key_dim_ * 2, key_dim_ * 2 + value_dim_}}].contiguous();

    q = q.view({B, S, num_k_heads_, head_k_dim_});
    k = k.view({B, S, num_k_heads_, head_k_dim_});
    v = v.view({B, S, num_v_heads_, head_v_dim_});
    a_proj = a_proj.contiguous();
    b_proj = b_proj.contiguous();

    auto [output, updated_recurrent_state] = gated_delta_rule_(q, k, v, a_proj, b_proj, a_log, dt_bias, recurrent_state_);
    recurrent_state_ = updated_recurrent_state;

    // [B, S, num_v_heads, head_v_dim] -> [B, S, value_dim]
    output = output.view({B, S, value_dim_});

    // Gated RMSNorm: norm(output) * silu(z)
    output = output.view({B * S * num_v_heads_, head_v_dim_});
    z = z.view({B * S * num_v_heads_, head_v_dim_});
    output = norm_(output);
    z = silu_(z);
    output = output * z;
    output = output.view({B, S, value_dim_});

    output = out_proj_(output);
    return {output};
  }
};

// ---------------------------------------------------------------------------
// Decoder layers
// ---------------------------------------------------------------------------

class Qwen3_5FullAttentionDecoder final : public nn::Module {
 public:
  Qwen3_5FullAttention self_attn_;
  Qwen3_5MLP mlp_;
  nn::RMSNorm input_layer_norm_;           // GemmaRMSNorm
  nn::RMSNorm post_attention_layer_norm_;  // GemmaRMSNorm

  Qwen3_5FullAttentionDecoder() = default;

  Qwen3_5FullAttentionDecoder(const std::string& name, const Qwen3_5Config& cfg) : nn::Module(name) {
    self_attn_ = reg<Qwen3_5FullAttention>("self_attn", cfg);
    mlp_ = reg<Qwen3_5MLP>("mlp", cfg);
    input_layer_norm_ = reg<nn::RMSNorm>("input_layernorm", cfg.rms_norm_eps, /*add_unit_offset=*/true);
    post_attention_layer_norm_ = reg<nn::RMSNorm>("post_attention_layernorm", cfg.rms_norm_eps, /*add_unit_offset=*/true);
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

class Qwen3_5GDNDecoder final : public nn::Module {
 public:
  Qwen3_5GDNLayer linear_attn_;
  Qwen3_5MLP mlp_;
  nn::RMSNorm input_layer_norm_;
  nn::RMSNorm post_attention_layer_norm_;

  Qwen3_5GDNDecoder() = default;

  Qwen3_5GDNDecoder(const std::string& name, const Qwen3_5Config& cfg) : nn::Module(name) {
    linear_attn_ = reg<Qwen3_5GDNLayer>("linear_attn", cfg);
    mlp_ = reg<Qwen3_5MLP>("mlp", cfg);
    input_layer_norm_ = reg<nn::RMSNorm>("input_layernorm", cfg.rms_norm_eps, /*add_unit_offset=*/true);
    post_attention_layer_norm_ = reg<nn::RMSNorm>("post_attention_layernorm", cfg.rms_norm_eps, /*add_unit_offset=*/true);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = input_layer_norm_(inputs[0]);
    x = linear_attn_(x)[0];
    auto tmp = x + inputs[0];
    x = post_attention_layer_norm_(tmp);
    x = mlp_(x)[0];
    x = x + tmp;
    return {x};
  }
};

// ---------------------------------------------------------------------------
// Qwen3.5 Transformer backbone
// ---------------------------------------------------------------------------

class Qwen3_5Text final : public nn::Module {
  // We store both layer types in separate lists, dispatched by layer_types_
  std::vector<Qwen3_5FullAttentionDecoder> full_attn_layers_;
  std::vector<Qwen3_5GDNDecoder> gdn_layers_;
  std::vector<int> layer_dispatch_;  // 0 = full_attn, 1 = gdn, index into respective vector
  std::vector<int> layer_type_;      // 0 = full_attn, 1 = gdn

  nn::RMSNorm norm_;
  nn::Embedding embedding_;

 public:
  Qwen3_5Text() = default;

  Qwen3_5Text(const std::string& name, const Qwen3_5Config& cfg) : nn::Module(name) {
    embedding_ = reg<nn::Embedding>("embed_tokens", cfg.vocab_size, cfg.hidden_size);

    int gdn_count = 0;
    int attn_count = 0;
    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
      std::string layer_name = "layers." + std::to_string(i);
      if (cfg.isFullAttentionLayer(i)) {
        auto layer = reg<Qwen3_5FullAttentionDecoder>(layer_name, cfg);
        layer.self_attn_.layer_idx_ = attn_count;
        full_attn_layers_.push_back(std::move(layer));
        layer_type_.push_back(0);
        layer_dispatch_.push_back(attn_count++);
      } else {
        auto layer = reg<Qwen3_5GDNDecoder>(layer_name, cfg);
        gdn_layers_.push_back(std::move(layer));
        layer_type_.push_back(1);
        layer_dispatch_.push_back(gdn_count++);
      }
    }

    norm_ = reg<nn::RMSNorm>("norm", cfg.rms_norm_eps, /*add_unit_offset=*/true);
  }

  void resetGDNStates(int batch_size) {
    for (auto& gdn : gdn_layers_) { gdn.linear_attn_.resetState(batch_size); }
  }

  Tensor embed(Tensor input_ids) { return embedding_(input_ids); }

  std::vector<Tensor> forwardEmbeddings(Tensor x, Tensor llm_embedding_sin, Tensor llm_embedding_cos,
                                        const AnyValue& kv_cache) {
    std::vector<AnyValue> args = {kv_cache};
    return forwardEmbeddingsImpl(x, llm_embedding_sin, llm_embedding_cos, args);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = embedding_(inputs[0]);
    return forwardEmbeddingsImpl(x, inputs[1], inputs[2], args);
  }

 private:
  std::vector<Tensor> forwardEmbeddingsImpl(Tensor x, Tensor llm_embedding_sin, Tensor llm_embedding_cos,
                                            const std::vector<AnyValue>& args) {
    auto& kv_cache = args[0];

    for (size_t i = 0; i < layer_type_.size(); ++i) {
      if (layer_type_[i] == 0) {
        x = full_attn_layers_[layer_dispatch_[i]](x, llm_embedding_sin, llm_embedding_cos, kv_cache)[0];
      } else {
        x = gdn_layers_[layer_dispatch_[i]](x)[0];
      }
    }

    x = norm_(x);
    return {x};
  }
};

// ---------------------------------------------------------------------------
// Qwen3.5 Model wrapper (matches HF's model.language_model.* weight prefix)
// ---------------------------------------------------------------------------

class Qwen3_5Model final : public nn::Module {
  Qwen3_5Text language_model_;
  Qwen3_5VisionModel visual_;
  bool vision_enabled_ = false;

 public:
  Qwen3_5Model() = default;

  Qwen3_5Model(const std::string& name, const Qwen3_5Config& cfg) : nn::Module(name) {
    language_model_ = reg<Qwen3_5Text>("language_model", cfg);
    vision_enabled_ = cfg.vision_enabled;
    if (vision_enabled_) { visual_ = reg<Qwen3_5VisionModel>("visual", cfg); }
  }

  void resetGDNStates(int batch_size) { language_model_.resetGDNStates(batch_size); }
  Tensor embed(Tensor input_ids) { return language_model_.embed(input_ids); }
  Tensor encodeImage(Tensor pixel_values, Tensor image_grid_thw) {
    if (!vision_enabled_) { throw std::invalid_argument("Qwen3.5 model was built without a vision tower"); }
    return visual_(pixel_values, image_grid_thw)[0];
  }
  Tensor encodeVideo(Tensor pixel_values_videos, Tensor video_grid_thw) {
    if (!vision_enabled_) { throw std::invalid_argument("Qwen3.5 model was built without a vision tower"); }
    return visual_(pixel_values_videos, video_grid_thw)[0];
  }
  Tensor forwardEmbeddings(Tensor embeddings, Tensor sin, Tensor cos, const AnyValue& kv_cache) {
    return language_model_.forwardEmbeddings(embeddings, sin, cos, kv_cache)[0];
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    return language_model_(inputs[0], inputs[1], inputs[2], args[0]);
  }
};

// ---------------------------------------------------------------------------
// Qwen3.5ForCausalLM
// ---------------------------------------------------------------------------

class Qwen3_5ForCausalLM : public ARGeneration, public nn::Module {
 public:
  explicit Qwen3_5ForCausalLM(const Qwen3_5Config& cfg) {
    // Only full attention layers use the KV cache
    kv_cache_ = nn::StaticCache(cfg.max_cache_length, cfg.numFullAttentionLayers(), cfg.num_attention_heads,
                                cfg.num_key_value_heads, cfg.head_dim, kFloat32, kFloat32, kCPU, false);
    eos_token_id_ = cfg.im_end_token_id;
    additional_eos_token_ids_.insert(cfg.eos_token_id);
    max_length_ = cfg.max_cache_length;
    llm = reg<Qwen3_5Model>("model", cfg);
    lm_head_ = reg<nn::Linear>("lm_head_out", cfg.hidden_size, cfg.vocab_size, false, cfg.linear_impl_type);

    // RoPE inv_freq uses rotary_dim (partial_rotary_factor * head_dim)
    auto inv = makeRoPEInvFreq(cfg.rotary_dim(), cfg.rope_theta);
    registerBuffer("inv_freq", inv);
    vision_enabled_ = cfg.vision_enabled;
    image_token_id_ = cfg.image_token_id;
    video_token_id_ = cfg.video_token_id;
    vision_spatial_merge_size_ = cfg.vision_spatial_merge_size;
    mrope_section_ = cfg.mrope_section;
  }

  ARGenerationOutputPast forward(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override {
    auto sequence = input.at("sequence");
    const auto sequence_shape = sequence.shape();
    if (sequence_shape.size() != 2 || sequence_shape[0] != 1 || sequence_shape[1] <= 0 || sequence.dtype() != kInt64
        || sequence.device() != kCPU) {
      throw std::invalid_argument("Qwen3.5 CPU expects a non-empty rank-2 int64 CPU sequence with batch size 1");
    }
    auto batch_size = sequence_shape[0];
    auto seq_len = sequence_shape[1];
    if (kv_cache_.getLayerNums() > 0) {
      const auto cached_tokens = kv_cache_.getCurrentSeqCnt(0);
      if (seq_len > max_length_ || cached_tokens > max_length_ - seq_len) {
        throw std::invalid_argument("Qwen3.5 sequence exceeds the configured KV-cache capacity");
      }
    }

    const bool has_pixel_values = input.count("pixel_values") != 0;
    const bool has_image_grid = input.count("image_grid_thw") != 0;
    const bool has_pixel_values_videos = input.count("pixel_values_videos") != 0;
    const bool has_video_grid = input.count("video_grid_thw") != 0;
    const bool has_token_types = input.count("mm_token_type_ids") != 0;
    if (has_pixel_values != has_image_grid || has_pixel_values_videos != has_video_grid) {
      throw std::invalid_argument("Qwen3.5 multimodal prefill requires each pixel tensor with its matching grid");
    }
    const bool has_image = has_pixel_values;
    const bool has_video = has_pixel_values_videos;
    const bool has_multimodal = has_image || has_video;
    if (has_token_types != has_multimodal) {
      throw std::invalid_argument("Qwen3.5 multimodal prefill requires mm_token_type_ids");
    }
    if (has_multimodal && !vision_enabled_) {
      throw std::invalid_argument("Qwen3.5 multimodal input requires a config with vision_config");
    }

    Tensor position_ids = Tensor::nil();
    if (input.count("position_ids")) {
      position_ids = input.at("position_ids");
      if (seq_len == 1) { position_ids = advanceQwen3_5PositionIds(position_ids); }
    } else if (has_multimodal) {
      position_ids = makeQwen3_5MultimodalPositionIds(
          input.at("mm_token_type_ids"), has_image ? input.at("image_grid_thw") : Tensor::nil(),
          has_video ? input.at("video_grid_thw") : Tensor::nil(), vision_spatial_merge_size_);
    } else {
      position_ids = Tensor::empty({batch_size, seq_len}, kInt64, kCPU).alloc();
      auto position_ids_ptr = position_ids.ptr<int64_t>();
      for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) { position_ids_ptr[b * seq_len + s] = s; }
      }
    }

    std::pair<Tensor, Tensor> rotary_embeddings;
    if (position_ids.shape().size() == 3) {
      rotary_embeddings = makeQwen3_5InterleavedRotaryEmbedding(position_ids, getBuffer("inv_freq"), mrope_section_, 1.0F);
    } else {
      rotary_embeddings = makeRotaryPosEmbedding(position_ids, getBuffer("inv_freq"), 1.0F);
    }
    auto& [llm_embedding_sin, llm_embedding_cos] = rotary_embeddings;

    if (has_multimodal) {
      auto token_types = input.at("mm_token_type_ids");
      if (token_types.shape() != sequence.shape() || token_types.dtype() != kInt32 || token_types.device() != kCPU) {
        throw std::invalid_argument("Qwen3.5 mm_token_type_ids must match the input sequence");
      }
      const auto* input_ids = sequence.ptr<int64_t>();
      const auto* types = token_types.ptr<int32_t>();
      int32_t image_count = 0;
      int32_t video_count = 0;
      for (int32_t s = 0; s < seq_len; ++s) {
        if (types[s] < 0 || types[s] > 2) {
          throw std::invalid_argument("Qwen3.5 mm_token_type_ids contains an unsupported modality");
        }
        if ((input_ids[s] == image_token_id_) != (types[s] == 1)) {
          throw std::invalid_argument("Qwen3.5 image token IDs and modality token types disagree");
        }
        if ((input_ids[s] == video_token_id_) != (types[s] == 2)) {
          throw std::invalid_argument("Qwen3.5 video token IDs and modality token types disagree");
        }
        if (types[s] == 1) { ++image_count; }
        if (types[s] == 2) { ++video_count; }
      }

      auto input_embeddings = llm.embed(sequence);
      const auto validate_features = [&input_embeddings](const Tensor& features, int32_t placeholder_count) {
        return placeholder_count > 0 && features.shape().size() == 2 && features.shape()[0] == placeholder_count
               && input_embeddings.shape().size() == 3 && features.shape()[1] == input_embeddings.shape()[2]
               && features.dtype() == input_embeddings.dtype();
      };
      if (has_image) {
        auto image_embeddings = llm.encodeImage(input.at("pixel_values"), input.at("image_grid_thw"));
        if (!validate_features(image_embeddings, image_count)) {
          throw std::invalid_argument("Qwen3.5 image features and image placeholders do not match");
        }
        injectQwen3_5ImageEmbeddings(input_embeddings, image_embeddings, token_types);
      } else if (image_count != 0) {
        throw std::invalid_argument("Qwen3.5 image placeholders require image pixels");
      }
      if (has_video) {
        auto video_embeddings = llm.encodeVideo(input.at("pixel_values_videos"), input.at("video_grid_thw"));
        if (!validate_features(video_embeddings, video_count)) {
          throw std::invalid_argument("Qwen3.5 video features and video placeholders do not match");
        }
        injectQwen3_5VideoEmbeddings(input_embeddings, video_embeddings, token_types);
      } else if (video_count != 0) {
        throw std::invalid_argument("Qwen3.5 video placeholders require video pixels");
      }
      sequence = llm.forwardEmbeddings(input_embeddings, llm_embedding_sin, llm_embedding_cos, AnyValue(&kv_cache_));
    } else {
      sequence = llm(sequence, llm_embedding_sin, llm_embedding_cos, AnyValue(&kv_cache_))[0];
    }

    {
      auto S = sequence.shape()[1];
      sequence = sequence[{kAll, {S - 1}, kAll}];
    }
    sequence = lm_head_(sequence);

    return {
        {"sequence", sequence},
        {"position_ids", position_ids},
    };
  }

  inline nn::StaticCache& kvCache() { return kv_cache_; }

  void resetState(int batch_size = 1) {
    if (batch_size != 1) { throw std::invalid_argument("Qwen3.5 CPU currently supports batch size 1 only"); }
    kv_cache_.clearCache();
    llm.resetGDNStates(batch_size);
  }

 private:
  Qwen3_5Model llm;
  nn::Linear lm_head_;
  nn::StaticCache kv_cache_;
  bool vision_enabled_ = false;
  int64_t image_token_id_ = 248056;
  int64_t video_token_id_ = 248057;
  int32_t vision_spatial_merge_size_ = 2;
  std::vector<int32_t> mrope_section_ = {11, 11, 10};
};

}  // namespace mllm::models::qwen3_5
