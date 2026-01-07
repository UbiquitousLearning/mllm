// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/mllm.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/nn/lmcache/StaticCache.hpp"
#include "mllm/models/qwen3_moe/configuration_qwen3_moe.hpp"
#include "mllm/utils/Enumerate.hpp"
#include "mllm/models/ARGeneration.hpp"

namespace mllm::models::qwen3_moe {

inline auto makeRoPEInvFreq(int output_dim, float rope_theta) -> Tensor {
  auto inv_freq = Tensor::empty({output_dim / 2}, kFloat32, kCPU).alloc();
  auto inv_freq_ptr = inv_freq.ptr<float>();
  for (int i = 0; i < output_dim / 2; i++) { inv_freq_ptr[i] = 1.0 / std::pow(rope_theta, 2.0 * i / output_dim); }
  return inv_freq;
}

inline auto makeRotaryPosEmbedding(Tensor& position_ids, const Tensor& inv_freq,
                                   float attention_scaling = 1.0f) -> std::pair<Tensor, Tensor> {
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

class Qwen3MoeMLP final : public nn::Module {
  nn::Linear gate_proj_;
  nn::Linear up_proj_;
  nn::Linear down_proj_;
  nn::SiLU act_;

  int hidden_size_;
  int intermediate_size_;

 public:
  Qwen3MoeMLP() = default;

  explicit Qwen3MoeMLP(const std::string& name, const Qwen3MoeConfig& config,
                         const std::optional<int>& hidden_size = std::nullopt,
                         const std::optional<int>& intermediate_size = std::nullopt)
      : nn::Module(name) {
    hidden_size_ = hidden_size.value_or(config.hidden_size);
    intermediate_size_ = intermediate_size.value_or(config.intermediate_size);

    // clang-format off
    gate_proj_ = reg<nn::Linear>("gate_proj", hidden_size_, intermediate_size_, false, config.linear_impl_type);
    up_proj_ = reg<nn::Linear>("up_proj", hidden_size_, intermediate_size_, false, config.linear_impl_type);
    down_proj_ = reg<nn::Linear>("down_proj", intermediate_size_, hidden_size_, false, config.linear_impl_type);
    act_ = reg<nn::SiLU>("act");
    // clang-format on
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    return {down_proj_(act_(gate_proj_(inputs[0])) * up_proj_(inputs[0]))};
  }
};

class MoEGate final : public nn::Module {
  int top_k_;
  int num_experts_;
  bool norm_topk_prob_;

  nn::Param weight_;

 public:
  MoEGate() = default;

  MoEGate(const std::string& name, const Qwen3MoeConfig& config) : nn::Module(name) {
    top_k_ = config.num_experts_per_tok;
    num_experts_ = config.num_experts;
    norm_topk_prob_ = config.norm_topk_prob;

    weight_ = reg<nn::Param>("weight", getModuleName() + ".weight");
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden_states = inputs[0];
    auto bsz = hidden_states.size(0);
    auto seq_len = hidden_states.size(1);
    auto h = hidden_states.size(2);

    // Compute gating score
    hidden_states = hidden_states.view({-1, h});
    // hidden_states and weight must in fp32 to keep precision !!!
    auto logits = nn::functional::matmul(hidden_states, weight_.weight(), false, true);
    auto scores = nn::functional::softmax(logits, -1);
    auto [topk_weight, topk_idx] = nn::functional::topk(scores, top_k_, -1, true, false);
    
    if(norm_topk_prob_){
      topk_weight = topk_weight / topk_weight.sum(-1, true);
    }

    return {topk_idx, topk_weight};
  }
};

class Qwen3MoE final : public nn::Module {
  int num_experts_per_tok_;
  nn::ModuleList<Qwen3MoeMLP> experts_;
  MoEGate gate_;

 public:
  Qwen3MoE() = default;

  Qwen3MoE(const std::string& name, const Qwen3MoeConfig& config) : nn::Module(name) {
    num_experts_per_tok_ = config.num_experts_per_tok;
    // Init experts
    experts_ = reg<nn::ModuleList<Qwen3MoeMLP>>("experts", config.num_experts, config, std::nullopt,
                                                  config.moe_intermediate_size);
    gate_ = reg<MoEGate>("gate", config);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden_states = inputs[0];
    auto identity = hidden_states;
    auto orig_shape = hidden_states.shape();
    auto topk_idx = Tensor::nil();
    auto topk_weight = Tensor::nil();
    auto gated_ret = gate_(hidden_states);
    topk_idx = gated_ret[0];
    topk_weight = gated_ret[1];
    hidden_states = hidden_states.view({-1, hidden_states.size(-1)});

    auto y = moeInfer(hidden_states, topk_idx, topk_weight).view(orig_shape);

    return {y};
  }

 private:
  Tensor moeInfer(const Tensor& x, Tensor& topk_ids, Tensor& topk_weights) {
    // x shape is [batch_size * seq, hidden_dim]
    
    auto cnts = Tensor::zeros({topk_ids.size(0), (int32_t)experts_.list().size()});
    // Do scatter_ operation
    {
      const int32_t* idx_ptr = topk_ids.ptr<mllm_int32_t>();
      float* cnt_ptr = cnts.ptr<mllm_fp32_t>();
      const int batch = topk_ids.size(0);
      const int k = topk_ids.size(1);
      const int n_exp = cnts.size(1);
      for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < k; ++j) {
          int32_t e = idx_ptr[b * k + j];
          MLLM_RT_ASSERT(e >= 0 && e < n_exp);
          cnt_ptr[b * n_exp + e] += 1.f;  // +1
        }
      }
    }
    auto tokens_per_expert = cnts.sum(0);
    auto idxs = topk_ids.view({-1}).argsort();

    // TODO this line maybe error
    auto sorted_tokens = x[{idxs / topk_ids.size(1), {kAll}}];
    
    std::vector<Tensor> outputs;
    int start_idx = 0;

    // tokens_per_expert shape is [num_experts]
    // Loop through each expert
    for (int i = 0; i < experts_.list().size(); ++i) {
      auto num_tokens = tokens_per_expert.ptr<mllm_fp32_t>()[i];
      auto end_idx = start_idx + (int32_t)num_tokens;
      if (num_tokens == 0) { continue; }
      auto& expert = experts_.list()[i];
      auto tokens_for_this_expert = sorted_tokens[{{start_idx, end_idx}, kAll}];
      auto expert_out = expert(tokens_for_this_expert)[0];
      outputs.push_back(expert_out);
      start_idx = end_idx;
    }

    auto outs = nn::functional::concat(outputs, 0);
    auto new_x = Tensor::emptyLike(outs).alloc();

    // indexed_write
    // python logic: new_x[idxs] = outs
    {
      const int32_t* idx_ptr = idxs.ptr<mllm_int32_t>();
      float* outs_ptr = outs.ptr<mllm_fp32_t>();
      float* new_x_ptr = new_x.ptr<mllm_fp32_t>();
      MLLM_RT_ASSERT_EQ(new_x.rank(), 2);
      MLLM_RT_ASSERT_EQ(new_x.size(0), idxs.size(0));
      auto dim = new_x.size(1);
      for (int i = 0; i < idxs.size(0); ++i) {
        int32_t idx = idx_ptr[i];
        std::memcpy(new_x_ptr + idx * dim, outs_ptr + i * dim, dim * sizeof(float));
      }
    }

    auto final_out_shape = topk_ids.shape();
    final_out_shape.emplace_back(-1);
    auto final_out =
        new_x.view(final_out_shape).to(topk_weights.dtype()).mul_(topk_weights.unsqueeze(-1)).sum(1).to(new_x.dtype());
    return final_out;
  }
};

class Qwen3MoeAttention final : public nn::Module {
  nn::Linear q_proj_;
  nn::Linear k_proj_;
  nn::Linear v_proj_;
  nn::Linear o_proj_;
  nn::RMSNorm rms_norm_q_;
  nn::RMSNorm rms_norm_k_;
  nn::RoPE q_rope_;
  nn::RoPE k_rope_;

  int hidden_size_;
  int head_dim_;
  int num_attention_heads_;
  int num_key_value_heads_;
  int num_key_value_groups_;

 public:
  Qwen3MoeAttention() = default;

  Qwen3MoeAttention(const std::string& name, const Qwen3MoeConfig& cfg) : nn::Module(name) {
    hidden_size_ = cfg.hidden_size;
    num_attention_heads_ = cfg.num_attention_heads;
    num_key_value_heads_ = cfg.num_key_value_heads;
    head_dim_ = cfg.head_dim;
    num_key_value_groups_ = num_attention_heads_ / num_key_value_heads_;

    // clang-format off
    q_proj_ = reg<nn::Linear>("q_proj", hidden_size_, head_dim_ * num_attention_heads_, cfg.attention_bias, cfg.linear_impl_type);
    k_proj_ = reg<nn::Linear>("k_proj", hidden_size_, head_dim_ * num_key_value_heads_, cfg.attention_bias, cfg.linear_impl_type).redirect();
    v_proj_ = reg<nn::Linear>("v_proj", hidden_size_, head_dim_ * num_key_value_heads_, cfg.attention_bias, cfg.linear_impl_type).redirect();
    o_proj_ = reg<nn::Linear>("o_proj", head_dim_ * num_attention_heads_, hidden_size_, cfg.attention_bias, cfg.linear_impl_type);
    // clang-format on

    rms_norm_q_ = reg<nn::RMSNorm>("q_norm", cfg.rms_norm_eps).inplace();
    rms_norm_k_ = reg<nn::RMSNorm>("k_norm", cfg.rms_norm_eps).inplace();

    // clang-format off
    q_rope_ = reg<nn::RoPE>("q_rope", cfg.rope_theta, cfg.max_position_embeddings, aops::RoPEOpOptionsInputType::kBSHD).inplace();
    k_rope_ = reg<nn::RoPE>("k_rope", cfg.rope_theta, cfg.max_position_embeddings, aops::RoPEOpOptionsInputType::kBSHD).inplace();
    // clang-format on
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = inputs[0];
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto past_kv_cache = args[0].get<nn::StaticCache*>();

    int B = inputs[0].shape()[0];
    int S = inputs[0].shape()[1];

    // Get KV cache for Key and Value first.
    // [B, S, H * D]
    auto [key_states_redirect, value_states_redirect] = past_kv_cache->preGetKVWriteLocation(layer_idx_, S);

    // [B, S, H * D]
    auto query_states = q_proj_(x);
    auto key_states = k_proj_(x, key_states_redirect);
    auto value_states = v_proj_(x, value_states_redirect);

    // [B, S, H, D]
    query_states = query_states.view({B, S, num_attention_heads_, head_dim_});
    key_states = key_states.view({B, S, num_key_value_heads_, head_dim_});

    // [B, S, H, D]
    query_states = rms_norm_q_(query_states);
    key_states = rms_norm_k_(key_states);

    // [B, S, H, D]
    query_states = q_rope_(query_states, llm_embedding_sin, llm_embedding_cos);
    key_states = k_rope_(key_states, llm_embedding_sin, llm_embedding_cos);

    // Get KV
    auto [K, V] = past_kv_cache->getKVCache(layer_idx_);

    // [B, S, H, D] FA2
    auto output = o_proj_(nn::functional::flashAttention2(query_states, K, V).view({B, S, num_attention_heads_ * head_dim_}));

    return {output};
  }

  int layer_idx_;
};

class Qwen3MoeDecoder final : public nn::Module {
  Qwen3MoeAttention self_attn_;
  nn::RMSNorm input_layer_norm_;
  nn::RMSNorm post_attention_layer_norm_;

  std::optional<Qwen3MoE> mlp_opt0_ = std::nullopt;
  std::optional<Qwen3MoeMLP> mlp_opt1_ = std::nullopt;

 public:
  int layer_idx_;

  Qwen3MoeDecoder() = default;

  Qwen3MoeDecoder(const std::string& name, const Qwen3MoeConfig& cfg, int layer_idx) : nn::Module(name) {
    layer_idx_ = layer_idx;

    self_attn_ = reg<Qwen3MoeAttention>("self_attn", cfg);
    self_attn_.layer_idx_ = layer_idx;

    bool is_mlp_only = std::find(cfg.mlp_only_layers.begin(), cfg.mlp_only_layers.end(), layer_idx) != cfg.mlp_only_layers.end();
    if ((!is_mlp_only) && (cfg.num_experts > 0 && (layer_idx_+1) % cfg.decoder_sparse_step == 0)) {
      mlp_opt0_ = reg<Qwen3MoE>("mlp", cfg);
    } else {
      mlp_opt1_ = reg<Qwen3MoeMLP>("mlp", cfg);
    }

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
    if(mlp_opt0_){
      x = mlp_opt0_.value()(x)[0];
    } else {
      x = mlp_opt1_.value()(x)[0];
    }
    x = x + tmp;
    return {x};
  }
};

class Qwen3MoeText final : public nn::Module {
  nn::Embedding embedding_;
  nn::ModuleListWithIdx<Qwen3MoeDecoder> decode_blocks_;
  nn::RMSNorm norm_;

 public:
  Qwen3MoeText() = default;

  explicit Qwen3MoeText(const std::string& name, const Qwen3MoeConfig& cfg) : nn::Module(name) {
    embedding_ = reg<nn::Embedding>("embed_tokens", cfg.vocab_size, cfg.hidden_size);
    decode_blocks_ = reg<nn::ModuleListWithIdx<Qwen3MoeDecoder>>("layers", cfg.num_hidden_layers, cfg);
    norm_ = reg<nn::RMSNorm>("norm", cfg.rms_norm_eps);

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
};

class Qwen3MoeForCausalLM : public ARGeneration, public nn::Module {
 public:
  explicit Qwen3MoeForCausalLM(const Qwen3MoeConfig& cfg) : cfg(cfg) {
    kv_cache_ = nn::StaticCache(cfg.max_cache_length, cfg.num_hidden_layers,
                                cfg.num_attention_heads,  // q_heads
                                cfg.num_key_value_heads,  // kv_heads
                                cfg.head_dim,             // kv_dim
                                kFloat32,                 // k_dtype
                                kFloat32,                 // v_dtype
                                kCPU,                     // device_type
                                true                      // use_fa2
    );
    eos_token_id_ = cfg.end_of_text_token_id;
    max_length_ = cfg.max_cache_length;
    tie_word_embeddings_ = cfg.tie_word_embeddings;

    llm = reg<Qwen3MoeText>("model", cfg);

    if (cfg.tie_word_embeddings) {
      // NOTE:
      // model.lm_head.weight is quantization weights of model.embed_tokens.weight
      lm_head_ = reg<nn::Linear>("lm_head", cfg.hidden_size, cfg.vocab_size, false, cfg.linear_impl_type);
    }

    // Init inv freq
    auto inv = makeRoPEInvFreq(cfg.head_dim, cfg.rope_theta);
    registerBuffer("inv_freq", inv);
  }

  ARGenerationOutputPast forward(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override {
    auto sequence = input.at("sequence");

    // Generate position_ids for the current sequence
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
        for (int s = 0; s < seq_len; ++s) { position_ids_ptr[b * seq_len + s] = s; }
      }
    }

    // Generate RoPE embeddings using the inv_freq buffer
    auto [llm_embedding_sin, llm_embedding_cos] = makeRotaryPosEmbedding(position_ids, getBuffer("inv_freq"), 1.0f);

    sequence = llm(sequence, llm_embedding_sin, llm_embedding_cos, AnyValue(&kv_cache_))[0];

    // clip x to one seq length
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
  const Qwen3MoeConfig& cfg;
  Qwen3MoeText llm;
  nn::Linear lm_head_;
  bool tie_word_embeddings_;
  nn::StaticCache kv_cache_;
};

}  // namespace mllm::models::qwen3_moe
