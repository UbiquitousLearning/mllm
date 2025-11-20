// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <iostream>
#include <optional>
#include <algorithm>
#include <filesystem>
#include <nlohmann/json.hpp>

#include "mllm/mllm.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/utils/StringHelper.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/nn/lmcache/StaticCache.hpp"
#include "mllm/preprocessor/visual/ImageTransform.hpp"

#include "mllm/models/deepseek_ocr/deepencoder.hpp"
#include "mllm/models/deepseek_ocr/conversation_preprocess.hpp"
#include "mllm/models/deepseek_ocr/tokenization_deepseek_ocr.hpp"
#include "mllm/models/deepseek_ocr/configuration_deepseek_ocr.hpp"

namespace mllm::models::deepseek_ocr {

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

class DeepseekV2MLP final : public nn::Module {
  nn::Linear gate_proj_;
  nn::Linear up_proj_;
  nn::Linear down_proj_;
  nn::SiLU act_;

  int hidden_size_;
  int intermediate_size_;

 public:
  DeepseekV2MLP() = default;

  explicit DeepseekV2MLP(const std::string& name, const DpskOcrConfig& config,
                         const std::optional<int>& hidden_size = std::nullopt,
                         const std::optional<int>& intermediate_size = std::nullopt)
      : nn::Module(name) {
    hidden_size_ = hidden_size.value_or(config.hidden_size);
    intermediate_size_ = intermediate_size.value_or(config.intermediate_size);

    // clang-format off
    gate_proj_ = reg<nn::Linear>("gate_proj", hidden_size_, intermediate_size_, false, config.llm_mlp_linear_impl_type);
    up_proj_ = reg<nn::Linear>("up_proj", hidden_size_, intermediate_size_, false, config.llm_mlp_linear_impl_type);
    down_proj_ = reg<nn::Linear>("down_proj", intermediate_size_, hidden_size_, false, config.llm_mlp_linear_impl_type);
    act_ = reg<nn::SiLU>("act");
    // clang-format on
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    return {down_proj_(act_(gate_proj_(inputs[0])) * up_proj_(inputs[0]))};
  }
};

class MoEGate final : public nn::Module {
  // FIXME: We may need to support more types
  std::string scoring_func_ = "softmax";
  std::string topk_method_ = "greedy";

  int top_k_;
  int n_routed_experts_;
  float routed_scaling_factor_;
  int n_group_;
  int topk_group_;
  bool norm_topk_prob_;

  nn::Param weight_;

 public:
  MoEGate() = default;

  MoEGate(const std::string& name, const DpskOcrConfig& config) : nn::Module(name) {
    top_k_ = config.num_experts_per_tok;
    n_routed_experts_ = config.n_routed_experts;

    // FIXME: Read from config.json instead of hard-coding
    routed_scaling_factor_ = 1.f;
    norm_topk_prob_ = false;

    n_group_ = config.n_group;
    topk_group_ = config.topk_group;

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

    // FIXME: Someone may need to Norm gate to sum 1.
    // FIXME: Someone may need rescale topk_weight by routed_scaling_factor_, but here is hard-code to 1.f

    return {topk_idx, topk_weight};
  }
};

class DeepseekV2MoE final : public nn::Module {
  int num_experts_per_tok_;

  // FIXME: Should not hard-code
  int ep_size_ = 1;
  int experts_per_rank_;
  int n_shared_experts_ = 0;

  nn::ModuleList<DeepseekV2MLP> experts_;
  MoEGate gate_;
  DeepseekV2MLP shared_experts_;

 public:
  DeepseekV2MoE() = default;

  DeepseekV2MoE(const std::string& name, const DpskOcrConfig& config) : nn::Module(name) {
    num_experts_per_tok_ = config.num_experts_per_tok;
    experts_per_rank_ = config.n_routed_experts;
    n_shared_experts_ = config.n_shared_experts;

    // Init experts
    experts_ = reg<nn::ModuleList<DeepseekV2MLP>>("experts", config.n_routed_experts, config, std::nullopt,
                                                  config.moe_intermediate_size);
    gate_ = reg<MoEGate>("gate", config);

    if (n_shared_experts_ > 0) {
      auto intermediate_size = config.moe_intermediate_size * config.n_shared_experts;
      shared_experts_ = reg<DeepseekV2MLP>("shared_experts", config, std::nullopt, intermediate_size);
    }
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
    auto flat_topk_idx = topk_idx.view({-1});
    auto y = moeInfer(hidden_states, topk_idx, topk_weight).view(orig_shape);
    if (n_shared_experts_ > 0) { y = y + shared_experts_(identity)[0]; }
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

// Deepseek OCR's attention not used MLA. It's same with LlamaFlashAttention2
class DeepseekV2Attention final : public nn::Module {
  nn::Linear q_proj_;
  nn::Linear k_proj_;
  nn::Linear v_proj_;
  nn::RoPE q_rope_;
  nn::RoPE k_rope_;
  nn::Linear o_proj_;
  int hidden_size_;
  int num_head_;
  int head_dim_;
  int num_key_value_heads_;

 public:
  int layer_idx_;

  DeepseekV2Attention() = default;

  DeepseekV2Attention(const std::string& name, const DpskOcrConfig& config) : nn::Module(name) {
    hidden_size_ = config.hidden_size;
    num_head_ = config.num_attention_heads;
    head_dim_ = config.hidden_size / config.num_attention_heads;
    num_key_value_heads_ = config.num_key_value_heads;

    // clang-format off
    q_proj_ = reg<nn::Linear>("q_proj", hidden_size_, num_head_ * head_dim_, false, config.llm_mlp_linear_impl_type);
    k_proj_ = reg<nn::Linear>("k_proj", hidden_size_, num_key_value_heads_ * head_dim_, false, config.llm_mlp_linear_impl_type).redirect();
    v_proj_ = reg<nn::Linear>("v_proj", hidden_size_, num_key_value_heads_ * head_dim_, false, config.llm_mlp_linear_impl_type).redirect();
    o_proj_ = reg<nn::Linear>("o_proj", num_head_ * head_dim_, hidden_size_, false, config.llm_mlp_linear_impl_type);
    q_rope_ = reg<nn::RoPE>("q_rope", 10000.0, config.max_position_embeddings, aops::RoPEOpOptionsInputType::kBSHD).inplace();
    k_rope_ = reg<nn::RoPE>("k_rope", 10000.0, config.max_position_embeddings, aops::RoPEOpOptionsInputType::kBSHD).inplace();
    // clang-format on
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden_states = inputs[0];
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto past_kv_cache = args[0].get<nn::StaticCache*>();

    auto bsz = hidden_states.size(0);
    auto q_len = hidden_states.size(1);

    // Get KV cache for Key and Value first.
    // [B, S, H * D]
    auto [key_states_redirect, value_states_redirect] = past_kv_cache->preGetKVWriteLocation(layer_idx_, q_len);

    auto query_states = q_proj_(hidden_states);
    auto key_states = k_proj_(hidden_states, key_states_redirect);
    auto value_states = v_proj_(hidden_states, value_states_redirect);

    // [B, S, H, D]
    query_states = query_states.view({bsz, q_len, num_head_, head_dim_});
    key_states = key_states.view({bsz, q_len, num_key_value_heads_, head_dim_});

    // [B, S, H, D]
    query_states = q_rope_(query_states, llm_embedding_sin, llm_embedding_cos);
    key_states = k_rope_(key_states, llm_embedding_sin, llm_embedding_cos);

    // Get KV
    auto [K, V] = past_kv_cache->getKVCache(layer_idx_);

    // [B, S, H, D] FA2
    auto output = o_proj_(nn::functional::flashAttention2(query_states, K, V).view({bsz, q_len, num_head_ * head_dim_}));

    return {output};
  }
};

class DeepseekV2DecoderLayer final : public nn::Module {
  // Use llama2 attention impl in deepseek-ocr model
  DeepseekV2Attention self_attn_;

  // FIXME: Do not use hard-code
  int first_k_dense_replace_ = 1;
  int moe_layer_freq_ = 1;

  nn::RMSNorm input_layernorm_;
  nn::RMSNorm post_attention_layernorm_;

  std::optional<DeepseekV2MoE> mlp_opt0_ = std::nullopt;
  std::optional<DeepseekV2MLP> mlp_opt1_ = std::nullopt;

 public:
  int layer_idx_;

  DeepseekV2DecoderLayer() = default;

  DeepseekV2DecoderLayer(const std::string& name, const DpskOcrConfig& config, int layer_idx) : nn::Module(name) {
    layer_idx_ = layer_idx;
    first_k_dense_replace_ = config.first_k_dense_replace;

    self_attn_ = reg<DeepseekV2Attention>("self_attn", config);
    self_attn_.layer_idx_ = layer_idx;

    if (config.n_routed_experts > 0 && layer_idx_ >= config.first_k_dense_replace && layer_idx_ % moe_layer_freq_ == 0) {
      mlp_opt0_ = reg<DeepseekV2MoE>("mlp", config);
    } else {
      mlp_opt1_ = reg<DeepseekV2MLP>("mlp", config);
    }

    input_layernorm_ = reg<nn::RMSNorm>("input_layernorm", 1e-6);
    post_attention_layernorm_ = reg<nn::RMSNorm>("post_attention_layernorm", 1e-6);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden_states = inputs[0];
    auto rope_pos_embed_sin = inputs[1];
    auto rope_pos_embed_cos = inputs[2];
    auto kv_cache = args[0];
    auto residual = hidden_states;

    hidden_states = input_layernorm_(hidden_states);
    hidden_states = self_attn_(hidden_states, rope_pos_embed_sin, rope_pos_embed_cos, kv_cache)[0];
    hidden_states = residual + hidden_states;

    residual = hidden_states;
    hidden_states = post_attention_layernorm_(hidden_states);
    if (mlp_opt0_) {
      hidden_states = mlp_opt0_.value()(hidden_states)[0];
    } else {
      hidden_states = mlp_opt1_.value()(hidden_states)[0];
    }
    hidden_states = residual + hidden_states;

    return {hidden_states};
  }
};

class DeepSeekV2Model : public nn::Module {
 protected:
  nn::Embedding embed_tokens_;
  nn::ModuleListWithIdx<DeepseekV2DecoderLayer> layers_;
  nn::RMSNorm norm_;

 public:
  DeepSeekV2Model() = default;

  explicit DeepSeekV2Model(const std::string& name, const DpskOcrConfig& config) : nn::Module(name) {
    embed_tokens_ = reg<nn::Embedding>("embed_tokens", config.vocab_size, config.hidden_size);
    layers_ = reg<nn::ModuleListWithIdx<DeepseekV2DecoderLayer>>("layers", config.num_hidden_layers, config);
    norm_ = reg<nn::RMSNorm>("norm", 1e-6);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto& input_embeddings = inputs[0];
    auto rope_embedding_sin = inputs[1];
    auto rope_embedding_cos = inputs[2];
    auto kv_cache = args[0];

    auto hidden_states = input_embeddings;

    for (auto& layer : layers_.list()) {
      hidden_states = layer(hidden_states, rope_embedding_sin, rope_embedding_cos, kv_cache)[0];
    }

    hidden_states = norm_(hidden_states);

    return {hidden_states};
  }
};

class DeepseekOCRModel final : public DeepSeekV2Model {
  VitModel vision_model_;
  ImageEncoderViT sam_model_;
  MlpProjector projector_;
  nn::Param image_newline_;
  nn::Param view_separator_;
  int n_embed = 1280;

 public:
  DeepseekOCRModel() = default;

  explicit DeepseekOCRModel(const std::string& name, const DpskOcrConfig& config) : DeepSeekV2Model(name, config) {
    sam_model_ = reg<ImageEncoderViT>("sam_model", config);
    vision_model_ = reg<VitModel>("vision_model", config);
    projector_ = reg<MlpProjector>("projector", config);
    image_newline_ = reg<nn::Param>("image_newline", getModuleName() + ".image_newline");
    view_separator_ = reg<nn::Param>("view_seperator", getModuleName() + ".view_seperator");  ///< DeepSeek's typo.
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    // FIXME: Just support one image right now.
    // Inputs: should be [input_ids, optional[image_crop], optional[image_ori], optional[images_spatial_crop]]
    auto& input_ids = inputs[0];
    auto patches = inputs.size() > 1 ? inputs[1] : Tensor::nil();
    auto image_ori = inputs.size() > 2 ? inputs[2] : Tensor::nil();
    auto images_spatial_crop = inputs.size() > 3 ? inputs[3] : Tensor::nil();
    auto images_seq_mask = inputs.size() > 4 ? inputs[4] : Tensor::nil();
    auto rope_embedding_sin = inputs.size() > 5 ? inputs[5] : Tensor::nil();
    auto rope_embedding_cos = inputs.size() > 6 ? inputs[6] : Tensor::nil();

    // Embedding
    auto inputs_embeds = embed_tokens_(input_ids);

    // We need to process image
    auto images_in_this_batch = Tensor::nil();
    if (patches && image_ori && images_spatial_crop && images_seq_mask) {
      if (nn::functional::sum(patches).item<float>() != 0) {
        // Local features
        auto local_features_1 = sam_model_(patches)[0];
        auto local_features_2 = vision_model_(patches, local_features_1)[0];
        auto local_features = nn::functional::concat(
            {
                // FIXME: contiguous is not needed. We use contiguous because mllm has weak performance in this case.
                local_features_2[{kAll, {1, kAll}, kAll}].contiguous(),
                local_features_1.flatten(2).permute({0, 2, 1}),
            },
            -1);
        local_features = projector_(local_features)[0];

        // Global features
        auto global_features_1 = sam_model_(image_ori)[0];
        auto global_features_2 = vision_model_(image_ori, global_features_1)[0];
        auto global_features = nn::functional::concat(
            {
                // FIXME: contiguous is not needed. We use contiguous because mllm has weak performance in this case.
                global_features_2[{kAll, {1, kAll}, kAll}].contiguous(),
                global_features_1.flatten(2).permute({0, 2, 1}),
            },
            -1);
        global_features = projector_(global_features)[0];

        print("=====================");
        print("BASE: ", global_features.shape());
        print("PATCHES: ", local_features.shape());
        print("=====================");

        auto hw = global_features.size(1);
        auto n_dim = global_features.size(2);
        auto h = (int)std::sqrt(hw);
        auto w = h;

        auto hw2 = local_features.size(1);
        auto n_dim2 = local_features.size(2);
        auto h2 = (int)std::sqrt(hw2);
        auto w2 = h2;

        MLLM_RT_ASSERT_EQ(images_spatial_crop.dtype(), kInt64);
        int width_crop_num = images_spatial_crop.at<mllm_int64_t>({0, 0});
        int height_crop_num = images_spatial_crop.at<mllm_int64_t>({0, 1});

        global_features = global_features.view({h, w, n_dim});
        global_features = nn::functional::concat(
            {
                global_features,

                // FIXME: This line is in-efficient.
                // pytorch logic: self.image_newline[None, None, :].expand(h, 1, n_dim)
                //
                // Use pytorch like expand instead. Expand will only modified stride, no memory copy involved.
                // But many kernels in mllm's arm backend not use stride as loop step, but calculate itself, so we need to
                // refact it.
                image_newline_.weight().view({1, 1, -1}).repeat(h, 0),
            },
            1);

        global_features = global_features.view({-1, n_dim});

        local_features = local_features.view({height_crop_num, width_crop_num, h2, w2, n_dim2})
                             .permute({0, 2, 1, 3, 4})
                             .view({height_crop_num * h2, width_crop_num * w2, n_dim2});
        local_features = nn::functional::concat(
            {
                local_features,

                // FIXME: This line is in-efficient.
                // pytorch logic: self.image_newline[None, None, :].expand(height_crop_num * h2, 1, n_dim2)
                //
                // Use pytorch like expand instead. Expand will only modified stride, no memory copy involved.
                // But many kernels in mllm's arm backend not use stride as loop step, but calculate itself, so we need to
                // refact it.
                image_newline_.weight().view({1, 1, -1}).repeat(height_crop_num * h2, 0),
            },
            1);

        local_features = local_features.view({-1, n_dim2});
        auto global_local_features = nn::functional::concat(
            {
                local_features,
                global_features,

                // pytorch logic: self.view_seperator[None, :]
                view_separator_.weight().view({1, -1}),
            },
            0);
        images_in_this_batch = global_local_features;
      } else {
        auto global_features_1 = sam_model_(image_ori)[0];
        auto global_features_2 = vision_model_(image_ori, global_features_1)[0];
        auto global_features = nn::functional::concat(
            {
                global_features_2[{kAll, {1, kAll}, kAll}],
                global_features_1.flatten(2).permute({0, 2, 1}),
            },
            -1);

        global_features = projector_(global_features)[0];

        print("=====================");
        print("BASE: ", global_features.shape());
        print("NO PATCHES");
        print("=====================");

        auto hw = global_features.size(1);
        auto n_dim = global_features.size(2);
        auto h = (int)std::sqrt(hw);
        auto w = h;

        global_features = global_features.view({h, w, n_dim});
        global_features = nn::functional::concat(
            {
                global_features,

                // FIXME: This line is in-efficient.
                // pytorch logic: self.image_newline[None, None, :].expand(h, 1, n_dim)
                //
                // Use pytorch like expand instead. Expand will only modified stride, no memory copy involved.
                // But many kernels in mllm's arm backend not use stride as loop step, but calculate itself, so we need to
                // refact it.
                image_newline_.weight().view({1, 1, -1}).repeat(h, 0),
            },
            1);

        global_features = global_features.view({-1, n_dim});

        auto global_local_features = nn::functional::concat(
            {
                global_features,
                view_separator_.weight().view({1, -1}),
            },
            0);

        images_in_this_batch = global_local_features;
      }
    }

    // Scatter copy.
    if (images_in_this_batch) {
      nn::functional::maskedScatter(inputs_embeds, images_seq_mask.unsqueeze(-1), images_in_this_batch);
    }

    auto sequence = DeepSeekV2Model::forward({inputs_embeds, rope_embedding_sin, rope_embedding_cos}, args)[0];

    return {sequence};
  }
};

class DeepseekOCRForCausalLM final : public nn::Module, public ARGeneration {
  DeepseekOCRModel model_;
  nn::Linear lm_head_;
  nn::StaticCache kv_cache_;

 public:
  DeepseekOCRForCausalLM() = default;

  explicit DeepseekOCRForCausalLM(const DpskOcrConfig& config) {
    model_ = reg<DeepseekOCRModel>("model", config);
    lm_head_ = reg<nn::Linear>("lm_head", config.hidden_size, config.vocab_size, false, config.lm_head_linear_impl_type);

    // Init inv freq
    auto inv = makeRoPEInvFreq(config.hidden_size / config.num_attention_heads, 10000.0);
    registerBuffer("inv_freq", inv);

    // kv_cache_
    kv_cache_ = nn::StaticCache(config.max_cache_length, config.num_hidden_layers,
                                config.num_attention_heads,                       // q_heads
                                config.num_key_value_heads,                       // kv_heads
                                config.hidden_size / config.num_attention_heads,  // kv_dim
                                kFloat32,                                         // k_dtype
                                kFloat32,                                         // v_dtype
                                kCPU,                                             // device_type
                                true                                              // use_fa2
    );

    // eos
    eos_token_id_ = config.eos_token_id;
  }

  inline nn::StaticCache& kvCache() { return kv_cache_; }
  inline int64_t eosTokenId() const { return eos_token_id_; }

  ARGenerationOutputPast forward(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override {
    auto patches = input.count("patches") ? input.at("patches") : Tensor::nil();
    auto image_ori = input.count("image_ori") ? input.at("image_ori") : Tensor::nil();
    auto images_spatial_crop = input.count("images_spatial_crop") ? input.at("images_spatial_crop") : Tensor::nil();
    auto images_seq_mask = input.count("images_seq_mask") ? input.at("images_seq_mask") : Tensor::nil();

    auto sequence = input.at("sequence");

    // Generate position_ids for the current sequence
    auto batch_size = sequence.shape()[0];
    auto seq_len = sequence.shape()[1];

    auto position_ids = Tensor::nil();
    auto rope_embedding_sin = Tensor::nil();
    auto rope_embedding_cos = Tensor::nil();
    auto kv_cache = args.at("kv_cache");

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

    auto [llm_embedding_sin, llm_embedding_cos] = makeRotaryPosEmbedding(position_ids, getBuffer("inv_freq"), 1.0f);
    rope_embedding_sin = llm_embedding_sin;
    rope_embedding_cos = llm_embedding_cos;
    sequence = model_(sequence, patches, image_ori, images_spatial_crop, images_seq_mask, rope_embedding_sin,
                      rope_embedding_cos, kv_cache)[0];
    // clip x to one seq length
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

  void infer(DpskOcrTokenizer& tokenizer, const std::string& prompt, const std::string& image_fp,
             const std::string& output_path, int base_size = 1024, int image_size = 640, bool crop_mode = true) {
    // Initialize template
    initializeTemplates();

    namespace fs = std::filesystem;
    fs::path out_path(output_path);
    fs::create_directories(out_path);
    fs::create_directories(out_path / "images");

    nlohmann::json conversations;
    if (!prompt.empty() && !image_fp.empty()) {
      conversations = nlohmann::json::array();
      conversations.push_back({{"role", "<|User|>"}, {"content", prompt}, {"images", nlohmann::json::array({image_fp})}});
      conversations.push_back({{"role", "<|Assistant|>"}, {"content", ""}});
    } else if (!prompt.empty()) {
      conversations = nlohmann::json::array();
      conversations.push_back({{"role", "<|User|>"}, {"content", prompt}});
      conversations.push_back({{"role", "<|Assistant|>"}, {"content", ""}});
    } else {
      // Prompt should not be empty
      MLLM_RT_ASSERT_EQ(prompt.empty(), false);
    }

    auto processed_prompt = formatMessages(conversations, "plain", "");

    // Global constant define
    const int PATCH_SIZE = 16;
    const int DOWN_SAMPLE_RATIO = 4;
    const std::string IMAGE_TOKEN = "<image>";
    const int64_t IMAGE_TOKEN_ID = 128815;

    // Global states
    int valid_img_tokens = 0;
    float ratio = 1.f;

    // Load image
    auto images = loadImages(conversations);

    auto w = images[0].w();
    auto h = images[0].h();
    ratio = 1 - (float)((std::max(w, h) - std::min(w, h)) / (float)(std::max(w, h)));

    // Image transform infra
    auto image_transform = BasicImageTransform(std::nullopt, std::nullopt, /*mean=*/std::vector<float>{0.5, 0.5, 0.5},
                                               /*std=*/std::vector<float>{0.5, 0.5, 0.5});

    // Split text with IMAGE_TOKEN
    // Like what python does: text_splits = prompt.split(image_token)
    auto text_splits = mllm::splitString(processed_prompt, IMAGE_TOKEN);

    // Processed states
    std::vector<int64_t> tokenized_str;
    std::vector<int8_t> images_seq_mask;
    std::vector<Tensor> images_list;
    std::vector<Tensor> images_crop_list;
    std::vector<std::tuple<int, int>> images_spatial_crop;

    // text_splits's length should be greater than images' length.
    // text_splits.size() - images.size() = 1
    for (int idx = 0; idx < std::min(images.size(), text_splits.size()); ++idx) {
      auto tokenized_sep = tokenizer.encode(text_splits[idx]);
      tokenized_str.insert(tokenized_str.end(), tokenized_sep.begin(), tokenized_sep.end());
      for (int _i = 0; _i < tokenized_sep.size(); ++_i) {
        images_seq_mask.emplace_back(0);  // emplace_back(false)
      }

      // Get image in this loop
      auto image = images[idx];
      std::tuple<int, int> crop_ratio;
      std::vector<Image> images_crop_raw;

      // Processing Image
      if (crop_mode) {
        if (image.h() <= 640 && image.w() <= 640) {
          crop_ratio = {1, 1};
        } else {
          if (crop_mode) {
            auto p = dynamicPreprocess(image);
            images_crop_raw = p.first;
            crop_ratio = p.second;
          } else {
            crop_ratio = {1, 1};
          }
        }

        // color=tuple(int(x * 255) for x in image_transform.mean
        auto global_view = image.pad(base_size, base_size, (int)(255 * 0.5), (int)(255 * 0.5), (int)(255 * 0.5));

        if (base_size == 1024) {
          valid_img_tokens += (int)(256 * ratio);
        } else if (base_size == 1280) {
          valid_img_tokens += (int)(400 * ratio);
        } else {
          // Just resize. for 512 and 640
        }

        images_list.emplace_back(image_transform(global_view));

        auto [width_crop_num, height_crop_num] = crop_ratio;
        images_spatial_crop.emplace_back(width_crop_num, height_crop_num);

        // Processing crops
        if (width_crop_num > 1 || height_crop_num > 1) {
          for (const auto& _i : images_crop_raw) { images_crop_list.emplace_back(image_transform(_i)); }
        }

        // Check if image_size is 640
        valid_img_tokens += images_crop_list.size() * 100;

        // Compute query
        auto num_queries = std::ceil((image_size / PATCH_SIZE) / DOWN_SAMPLE_RATIO);
        auto num_queries_base = std::ceil((base_size / PATCH_SIZE) / DOWN_SAMPLE_RATIO);

        // Do python logic below:
        // tokenized_image = ([image_token_id] * num_queries_base + [image_token_id]) * num_queries_base
        // tokenized_image += [image_token_id]
        std::vector<int64_t> tokenized_image;
        tokenized_image.reserve((num_queries_base + 1) * num_queries_base + 1);
        for (int i = 0; i < num_queries_base; ++i) {
          tokenized_image.insert(tokenized_image.end(), num_queries_base, IMAGE_TOKEN_ID);
          tokenized_image.push_back(IMAGE_TOKEN_ID);
        }
        tokenized_image.push_back(IMAGE_TOKEN_ID);

        if (width_crop_num > 1 || height_crop_num > 1) {
          for (int h = 0; h < num_queries * height_crop_num; ++h) {
            tokenized_image.insert(tokenized_image.end(), num_queries * width_crop_num, IMAGE_TOKEN_ID);
            tokenized_image.push_back(IMAGE_TOKEN_ID);
          }
        }

        tokenized_str.insert(tokenized_str.end(), tokenized_image.begin(), tokenized_image.end());
        for (int _i = 0; _i < tokenized_image.size(); ++_i) { images_seq_mask.emplace_back(true); }
      } else {
        NYI("crop_mode = false is not supported yet.");
      }
    }

    // Processing last text split
    auto tokenized_sep = tokenizer.encode(text_splits.back());
    tokenized_str.insert(tokenized_str.end(), tokenized_sep.begin(), tokenized_sep.end());
    images_seq_mask.insert(images_seq_mask.end(), tokenized_sep.size(), false);

    // Add bos token
    // bos_id = 0
    // tokenized_str = [bos_id] + tokenized_str
    // images_seq_mask = [False] + images_seq_mask
    tokenized_str.insert(tokenized_str.begin(), 0);
    images_seq_mask.insert(images_seq_mask.begin(), false);

    // Prepare Tensor to DeepSeek-OCR Model
    auto input_ids = Tensor::fromVector(tokenized_str, {1, (int32_t)tokenized_str.size()}, kInt64);
    auto images_seq_mask_tensor = Tensor::fromVector(images_seq_mask, {1, (int32_t)images_seq_mask.size()}, kInt8);
    auto images_ori_tensor = Tensor::nil();
    auto images_spatial_crop_tensor = Tensor::nil();
    auto images_crop_tensor = Tensor::nil();
    if (images_list.empty()) {
      images_ori_tensor = Tensor::zeros({1, 3, image_size, image_size});
      images_spatial_crop_tensor = Tensor::zeros({1, 2}, kInt64);
      images_crop_tensor = Tensor::zeros({1, 3, base_size, base_size});
    } else {
      images_ori_tensor = nn::functional::stack(images_list, 0);
      images_spatial_crop_tensor = Tensor::zeros({(int32_t)images_spatial_crop.size(), 2}, kInt64);
      auto _ptr = images_spatial_crop_tensor.ptr<mllm_int64_t>();
      for (int _i = 0; _i < images_spatial_crop.size(); ++_i) {
        auto [l, h] = images_spatial_crop[_i];
        _ptr[2 * _i + 0] = l;
        _ptr[2 * _i + 1] = h;
      }
      if (!images_crop_list.empty()) {
        images_crop_tensor = nn::functional::stack(images_crop_list, 0);
      } else {
        images_crop_tensor = Tensor::zeros({1, 3, base_size, base_size});
      }
    }

    std::stringstream result;
    streamGenerate(
        {
            {"patches", images_crop_tensor},
            {"image_ori", images_ori_tensor},
            {"images_spatial_crop", images_spatial_crop_tensor},
            {"images_seq_mask", images_seq_mask_tensor},
            {"sequence", input_ids},
        },
        {
            {"kv_cache", mllm::AnyValue(&kv_cache_)},
        },
        [&](int64_t token_id) {
          auto decode = tokenizer.decode({token_id});
          result << decode;
          std::cout << decode << std::flush;
        });
    print("\n");  ///< flush
    perfSummary();

    // Post process data
    // TODO
  }
};

}  // namespace mllm::models::deepseek_ocr
