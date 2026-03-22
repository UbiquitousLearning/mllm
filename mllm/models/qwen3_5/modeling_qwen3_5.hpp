// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Qwen3.5 hybrid model: 18 GDN (Gated Delta Network) layers + 6 full attention layers.
// GDN layers use linear attention with recurrent state; full attention layers use GQA
// with partial RoPE and output gating.

#pragma once

#include "mllm/mllm.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/nn/lmcache/StaticCache.hpp"
#include "mllm/models/qwen3_5/configuration_qwen3_5.hpp"
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
  int num_key_value_groups_;
  bool attn_output_gate_;

 public:
  Qwen3_5FullAttention() = default;

  Qwen3_5FullAttention(const std::string& name, const Qwen3_5Config& cfg) : nn::Module(name) {
    hidden_size_ = cfg.hidden_size;
    num_attention_heads_ = cfg.num_attention_heads;
    num_key_value_heads_ = cfg.num_key_value_heads;
    head_dim_ = cfg.head_dim;
    num_key_value_groups_ = num_attention_heads_ / num_key_value_heads_;
    attn_output_gate_ = cfg.attn_output_gate;

    // Q projection is 2x wide when output gating is enabled (second half = gate)
    int q_proj_out = head_dim_ * num_attention_heads_;
    if (attn_output_gate_) { q_proj_out *= 2; }

    q_proj_ = reg<nn::Linear>("q_proj", hidden_size_, q_proj_out, cfg.attention_bias, cfg.linear_impl_type);
    k_proj_ = reg<nn::Linear>("k_proj", hidden_size_, head_dim_ * num_key_value_heads_, cfg.attention_bias, cfg.linear_impl_type);
    v_proj_ = reg<nn::Linear>("v_proj", hidden_size_, head_dim_ * num_key_value_heads_, cfg.attention_bias, cfg.linear_impl_type);
    o_proj_ = reg<nn::Linear>("o_proj", head_dim_ * num_attention_heads_, hidden_size_, cfg.attention_bias, cfg.linear_impl_type);

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
      gate = q_reshaped[{kAll, kAll, kAll, {head_dim_, head_dim_ * 2}}].contiguous().view({B, S, num_attention_heads_ * head_dim_});
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

    // [B, H, S, D] -> [B, S, H*D]
    auto output = nn::functional::matmul(attn, value_states);
    output = output.transpose(1, 2).view({B, S, num_attention_heads_ * head_dim_});

    // Output gate: output = output * sigmoid(gate)
    if (attn_output_gate_) {
      gate = sigmoid_(gate);
      output = output * gate;
    }

    output = o_proj_(output);
    return {output};
  }

  int layer_idx_;
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

  // Causal Conv1D for sequence mixing
  nn::Conv1D conv1d_;

  // Output
  nn::RMSNorm norm_;  // gated RMSNorm (with add_unit_offset for Gemma style)
  nn::Linear out_proj_;

  nn::SiLU silu_;
  nn::Sigmoid sigmoid_;

  int hidden_size_;
  int num_k_heads_;
  int num_v_heads_;
  int head_k_dim_;
  int head_v_dim_;
  int key_dim_;
  int value_dim_;
  int conv_kernel_size_;

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

    // Causal Conv1D: groups = channels (depthwise), no bias, padding = kernel-1 for causal
    int conv_channels = key_dim_ * 2 + value_dim_;
    conv1d_ = reg<nn::Conv1D>("conv1d", conv_channels, conv_channels, conv_kernel_size_,
                              /*stride=*/1, /*padding=*/conv_kernel_size_ - 1, /*dilation=*/1,
                              /*groups=*/conv_channels, /*bias=*/false);

    // Gated RMSNorm (GemmaRMSNorm: add_unit_offset=true)
    norm_ = reg<nn::RMSNorm>("norm", cfg.rms_norm_eps, /*add_unit_offset=*/true);

    out_proj_ = reg<nn::Linear>("out_proj", value_dim_, hidden_size_, false, cfg.linear_impl_type);

    silu_ = reg<nn::SiLU>("silu");
    sigmoid_ = reg<nn::Sigmoid>("sigmoid");
  }

  // State parameters are registered as buffers by ForCausalLM
  // A_log: [num_v_heads], dt_bias: [num_v_heads]

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = inputs[0];  // [B, S, H]
    int B = x.shape()[0];
    int S = x.shape()[1];

    // Input projections
    auto mixed_qkv = in_proj_qkv_(x);  // [B, S, key_dim*2 + value_dim]
    auto z = in_proj_z_(x);             // [B, S, value_dim]
    auto a = in_proj_a_(x);             // [B, S, num_v_heads]
    auto b = in_proj_b_(x);             // [B, S, num_v_heads]

    // Causal Conv1D on mixed_qkv
    // Conv1D expects [B, C, L] layout
    mixed_qkv = mixed_qkv.transpose(1, 2);            // [B, C, S]
    mixed_qkv = conv1d_(mixed_qkv);                   // [B, C, S + padding]
    mixed_qkv = mixed_qkv[{kAll, kAll, {0, S}}];      // Causal trim to [B, C, S]
    mixed_qkv = mixed_qkv.transpose(1, 2);            // [B, S, C]

    // Split into q, k, v and apply SiLU activation
    // contiguous() needed because slice produces non-contiguous views
    auto q = silu_(mixed_qkv[{kAll, kAll, {0, key_dim_}}].contiguous());
    auto k = silu_(mixed_qkv[{kAll, kAll, {key_dim_, key_dim_ * 2}}].contiguous());
    auto v = mixed_qkv[{kAll, kAll, {key_dim_ * 2, key_dim_ * 2 + value_dim_}}].contiguous();

    // Reshape to heads
    q = q.view({B, S, num_k_heads_, head_k_dim_});    // [B, S, Hk, Dk]
    k = k.view({B, S, num_k_heads_, head_k_dim_});    // [B, S, Hk, Dk]
    v = v.view({B, S, num_v_heads_, head_v_dim_});    // [B, S, Hv, Dv]

    // Full GDN recurrence (not yet implemented):
    //   g_t = -exp(A_log) * softplus(a_t + dt_bias)
    //   beta_t = sigmoid(b_t)
    //   state_t = exp(g_t) * state_{t-1} + beta_t * (k_t outer v_t)
    //   output_t = q_t @ state_t
    // Simplified to standard linear attention for initial bringup.
    (void)a;
    (void)b;

    // Transpose to [B, H, S, D] for batched matmul
    q = q.transpose(1, 2);  // [B, Hk, S, Dk]
    k = k.transpose(1, 2);  // [B, Hk, S, Dk]
    v = v.transpose(1, 2);  // [B, Hv, S, Dv]

    // Simplified linear attention: O = softmax(Q K^T / sqrt(d)) V with causal mask
    // NOTE: A full GDN implementation uses the recurrent state form:
    //   g_t = -exp(A_log) * softplus(a_t + dt_bias)
    //   state_t = exp(g_t) * state_{t-1} + beta_t * (k_t outer v_t)
    //   output_t = q_t @ state_t
    // This simplified form is for initial CPU bringup; the recurrent kernel
    // will be implemented as a custom HTP op for QNN and as a fused CUDA kernel.
    auto attn_weights = nn::functional::matmul(q, k, false, true);  // [B, H, S, S]
    auto scale = 1.f / sqrtf(static_cast<float>(head_k_dim_));
    attn_weights = attn_weights * scale;
    auto output = nn::functional::matmul(attn_weights, v);  // [B, H, S, Dv]

    // [B, H, S, Dv] -> [B, S, H, Dv] -> [B, S, H*Dv]
    output = output.transpose(1, 2).view({B, S, value_dim_});

    // Gated RMSNorm: norm(output) * silu(z)
    // The norm operates per-head: reshape to [..., head_v_dim]
    output = output.view({B * S * num_v_heads_, head_v_dim_});
    z = z.view({B * S * num_v_heads_, head_v_dim_});
    output = norm_(output);
    // Gate with z (SiLU gating)
    z = silu_(z);
    output = output * z;
    output = output.view({B, S, value_dim_});

    output = out_proj_(output);
    return {output};
  }

  int layer_idx_;
  int gdn_layer_idx_;
};

// ---------------------------------------------------------------------------
// Decoder layers
// ---------------------------------------------------------------------------

class Qwen3_5FullAttentionDecoder final : public nn::Module {
 public:
  Qwen3_5FullAttention self_attn_;
  Qwen3_5MLP mlp_;
  nn::RMSNorm input_layer_norm_;   // GemmaRMSNorm
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
    auto& kv_cache = args[0];  // unused for GDN but kept for uniform interface

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
        layer.self_attn_.layer_idx_ = i;
        full_attn_layers_.push_back(std::move(layer));
        layer_type_.push_back(0);
        layer_dispatch_.push_back(attn_count++);
      } else {
        auto layer = reg<Qwen3_5GDNDecoder>(layer_name, cfg);
        layer.linear_attn_.layer_idx_ = i;
        layer.linear_attn_.gdn_layer_idx_ = gdn_count;
        gdn_layers_.push_back(std::move(layer));
        layer_type_.push_back(1);
        layer_dispatch_.push_back(gdn_count++);
      }
    }

    norm_ = reg<nn::RMSNorm>("norm", cfg.rms_norm_eps, /*add_unit_offset=*/true);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = embedding_(inputs[0]);
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto& kv_cache = args[0];

    for (size_t i = 0; i < layer_type_.size(); ++i) {
      if (layer_type_[i] == 0) {
        x = full_attn_layers_[layer_dispatch_[i]](x, llm_embedding_sin, llm_embedding_cos, kv_cache)[0];
      } else {
        x = gdn_layers_[layer_dispatch_[i]](x, llm_embedding_sin, llm_embedding_cos, kv_cache)[0];
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

 public:
  Qwen3_5Model() = default;

  Qwen3_5Model(const std::string& name, const Qwen3_5Config& cfg) : nn::Module(name) {
    language_model_ = reg<Qwen3_5Text>("language_model", cfg);
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
  explicit Qwen3_5ForCausalLM(const Qwen3_5Config& cfg) : cfg(cfg) {
    // Only full attention layers use the KV cache
    kv_cache_ = nn::StaticCache(cfg.max_cache_length, cfg.num_hidden_layers,
                                cfg.num_attention_heads,
                                cfg.num_key_value_heads,
                                cfg.head_dim,
                                kFloat32,
                                kFloat32,
                                kCPU,
                                false);
    eos_token_id_ = cfg.end_of_text_token_id;
    max_length_ = cfg.max_cache_length;
    tie_word_embeddings_ = cfg.tie_word_embeddings;

    llm = reg<Qwen3_5Model>("model", cfg);

    if (cfg.tie_word_embeddings) {
      lm_head_ = reg<nn::Linear>("lm_head_out", cfg.hidden_size, cfg.vocab_size, false, cfg.linear_impl_type);
    }

    // RoPE inv_freq uses rotary_dim (partial_rotary_factor * head_dim)
    auto inv = makeRoPEInvFreq(cfg.rotary_dim(), cfg.rope_theta);
    registerBuffer("inv_freq", inv);
  }

  ARGenerationOutputPast forward(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override {
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
        for (int s = 0; s < seq_len; ++s) { position_ids_ptr[b * seq_len + s] = s; }
      }
    }

    auto [llm_embedding_sin, llm_embedding_cos] = makeRotaryPosEmbedding(position_ids, getBuffer("inv_freq"), 1.0f);

    sequence = llm(sequence, llm_embedding_sin, llm_embedding_cos, AnyValue(&kv_cache_))[0];

    {
      auto S = sequence.shape()[1];
      sequence = sequence[{kAll, {S - 1}, kAll}];
    }
    if (tie_word_embeddings_) { sequence = lm_head_(sequence); }

    return {
        {"sequence", sequence},
        {"position_ids", position_ids},
    };
  }

  inline nn::StaticCache& kvCache() { return kv_cache_; }

 private:
  const Qwen3_5Config& cfg;
  Qwen3_5Model llm;
  nn::Linear lm_head_;
  bool tie_word_embeddings_;
  nn::StaticCache kv_cache_;
};

}  // namespace mllm::models::qwen3_5
