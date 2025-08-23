// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <unordered_map>

#include "mllm/mllm.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/models/qwen2_5vl/configuration_qwen2_5vl.hpp"

namespace mllm::models::qwen2_5vl {

inline auto makeVisualTokensIdBioMap(const Tensor& grid_thw, int sliding_window_size = 112, int spatial_merge_size = 2,
                                     int patch_size = 14)
    -> std::pair<std::unordered_map<mllm_int32_t, mllm_int32_t>, std::unordered_map<mllm_int32_t, mllm_int32_t>> {
  const int vit_window = 4;  // 112 / 2 / 14

  const int64_t grid_t = grid_thw.constAt<mllm_int32_t>({0, 0});
  const int64_t grid_h = grid_thw.constAt<mllm_int32_t>({0, 1});
  const int64_t grid_w = grid_thw.constAt<mllm_int32_t>({0, 2});

  const int64_t llm_grid_h = grid_h / spatial_merge_size;
  const int64_t llm_grid_w = grid_w / spatial_merge_size;

  const int64_t total_patches = grid_t * llm_grid_h * llm_grid_w;

  const int64_t pad_h = (vit_window - llm_grid_h % vit_window) % vit_window;
  const int64_t pad_w = (vit_window - llm_grid_w % vit_window) % vit_window;

  const int64_t padded_h = llm_grid_h + pad_h;
  const int64_t padded_w = llm_grid_w + pad_w;

  const int64_t num_win_h = padded_h / vit_window;
  const int64_t num_win_w = padded_w / vit_window;

  const int64_t windows_per_t = num_win_h * num_win_w;
  const int64_t total_windows = grid_t * windows_per_t;

  std::unordered_map<mllm_int32_t, mllm_int32_t> orig_2_win;
  std::unordered_map<mllm_int32_t, mllm_int32_t> win_2_orig;

  int64_t win_id = 0;
  for (int64_t t = 0; t < grid_t; ++t) {
    for (int64_t wh = 0; wh < num_win_h; ++wh) {
      for (int64_t ww = 0; ww < num_win_w; ++ww) {
        for (int64_t sh = 0; sh < vit_window; ++sh) {
          for (int64_t sw = 0; sw < vit_window; ++sw) {
            const int64_t h = wh * vit_window + sh;
            const int64_t w = ww * vit_window + sw;

            if (h < llm_grid_h && w < llm_grid_w) {
              int64_t orig_id = t * llm_grid_h * llm_grid_w + h * llm_grid_w + w;
              orig_2_win[static_cast<mllm_int32_t>(orig_id)] = static_cast<mllm_int32_t>(win_id);
              win_2_orig[static_cast<mllm_int32_t>(win_id)] = static_cast<mllm_int32_t>(orig_id);
              ++win_id;
            }
          }
        }
      }
    }
  }

  return {orig_2_win, win_2_orig};
}

class PatchEmbed final : public nn::Module {
  int32_t in_chans_;
  int32_t embed_dim_;
  int32_t patch_size_;
  int32_t temporal_patch_size_;

  nn::Conv3D proj_;

 public:
  PatchEmbed() = default;

  inline PatchEmbed(const std::string& name, const Qwen2_5VLConfig& cfg) : nn::Module(name) {
    in_chans_ = cfg.visual_in_chans;
    embed_dim_ = cfg.visual_hidden_size;
    patch_size_ = cfg.visual_patch_size;
    temporal_patch_size_ = cfg.visual_temporal_patch_size;

    proj_ = reg<nn::Conv3D>("proj", cfg.visual_in_chans, cfg.visual_hidden_size,
                            std::vector<int32_t>{cfg.visual_temporal_patch_size, cfg.visual_patch_size, cfg.visual_patch_size},
                            std::vector<int32_t>{cfg.visual_temporal_patch_size, cfg.visual_patch_size, cfg.visual_patch_size},
                            false);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden_states = inputs[0];

    // [batch_size(x), in_channel(3), temporal_patch_size(2), patch_size(14), patch_size(14)]
    hidden_states = hidden_states.view({-1, in_chans_, temporal_patch_size_, patch_size_, patch_size_});
    hidden_states = proj_(hidden_states).view({-1, embed_dim_});

    return {hidden_states};
  }
};

class PatchMerger final : public nn::Module {
  int32_t hidden_size_;
  int32_t spatial_merge_size_;
  int32_t context_dim_;

  nn::LayerNorm ln_q_;
  nn::Linear mlp_0_;
  nn::Linear mlp_2_;
  nn::GELU mlp_gelu_;

 public:
  PatchMerger() = default;

  inline PatchMerger(const std::string& name, const Qwen2_5VLConfig& cfg) : nn::Module(name) {
    context_dim_ = cfg.visual_hidden_size;
    spatial_merge_size_ = cfg.visual_spatial_merge_size;
    hidden_size_ = context_dim_ * spatial_merge_size_ * spatial_merge_size_;

    ln_q_ = reg<nn::LayerNorm>("ln_q", std::vector<int32_t>{context_dim_}, true, true, 1e-6);
    mlp_0_ = reg<nn::Linear>("mlp.0", hidden_size_, hidden_size_, true, cfg.linear_impl_type);
    mlp_gelu_ = reg<nn::GELU>("mlp.gelu");
    mlp_2_ = reg<nn::Linear>("mlp.2", hidden_size_, cfg.hidden_size, true, cfg.linear_impl_type);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto o = ln_q_(inputs[0]).view({-1, hidden_size_});
    o = mlp_0_(o);
    o = mlp_gelu_(o);
    o = mlp_2_(o);
    return {o};
  }
};

class VisionMlp final : public nn::Module {
  int32_t dim_;
  int32_t hidden_dim_;

  nn::QuickGELU act_;
  nn::Linear fc_1_;
  nn::Linear fc_2_;

 public:
  VisionMlp() = default;

  inline VisionMlp(const std::string& name, const Qwen2_5VLConfig& cfg) : nn::Module(name) {
    dim_ = cfg.visual_hidden_size;
    hidden_dim_ = cfg.visual_hidden_size * cfg.visual_mlp_ratio;

    fc_1_ = reg<nn::Linear>("fc1", dim_, hidden_dim_, true, cfg.linear_impl_type);
    fc_2_ = reg<nn::Linear>("fc2", hidden_dim_, dim_, true, cfg.linear_impl_type);
    act_ = reg<nn::QuickGELU>("act");
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    return {fc_2_(act_(fc_1_(inputs[0])))};
  }
};

class VisionAttention final : public nn::Module {
  int32_t dim_;
  int32_t num_heads_;
  int32_t head_dim_;
  int32_t num_key_value_groups = 1;
  float scaling = 0.f;

  nn::Linear qkv_;
  nn::Linear proj_;
  nn::Softmax softmax_;
  nn::VisionRoPE vision_rope_q_;
  nn::VisionRoPE vision_rope_k_;

 public:
  VisionAttention() = default;

  inline VisionAttention(const std::string& name, const Qwen2_5VLConfig& cfg) : nn::Module(name) {
    dim_ = cfg.visual_hidden_size;
    num_heads_ = cfg.visual_num_heads;
    head_dim_ = dim_ / num_heads_;
    scaling = std::sqrt(head_dim_);

    qkv_ = reg<nn::Linear>("qkv", dim_, dim_ * 3, true, cfg.linear_impl_type);
    proj_ = reg<nn::Linear>("proj", dim_, dim_, true, cfg.linear_impl_type);
    softmax_ = reg<nn::Softmax>("softmax", -1);

    vision_rope_q_ = reg<nn::VisionRoPE>("vision_rope_q", aops::VisionRoPEOpOptionsType::kQwen2VL,
                                         aops::Qwen2VLRoPEOpOptions{
                                             .dims = head_dim_,
                                             .spatial_merge_size = cfg.visual_spatial_merge_size,
                                             .theta = 10000.0,
                                         });
    vision_rope_k_ = reg<nn::VisionRoPE>("vision_rope_k", aops::VisionRoPEOpOptionsType::kQwen2VL,
                                         aops::Qwen2VLRoPEOpOptions{
                                             .dims = head_dim_,
                                             .spatial_merge_size = cfg.visual_spatial_merge_size,
                                             .theta = 10000.0,
                                         });
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    // hidden_states shape is [seq_length, dim]
    auto hidden_states = inputs[0];
    auto visual_embedding_sin = inputs[1];
    auto visual_embedding_cos = inputs[2];

    auto seq_length = hidden_states.shape()[0];

    auto [query_states, key_states, value_states] =
        nn::functional::split<3>(qkv_(hidden_states).view({seq_length, 3, num_heads_, -1}).permute({1, 0, 2, 3}), 1, 0);

    // Input to Vision ROPE must be BSHD format
    // grid_thw shape is [n, 3], n is always 1 in this case.
    query_states = vision_rope_q_(query_states, visual_embedding_sin, visual_embedding_cos);
    key_states = vision_rope_k_(key_states, visual_embedding_sin, visual_embedding_cos);

    // [B, H, S, D]
    query_states = query_states.transpose(1, 2);
    key_states = key_states.transpose(1, 2);
    value_states = value_states.transpose(1, 2);

    // attention weight
    // [B=1, H, S, S]
    auto attn = nn::functional::matmul(query_states, key_states, false, true) * (1.f / sqrtf(head_dim_));
    attn = softmax_(attn);

    // attn output
    // [B=1, H, S, S] @ [B=1, H, S, D] -> [B=1, H, S, D]
    auto attn_output = nn::functional::matmul(attn, value_states);

    // [B=1, H, S, D] -> [B=1, S, H, D] -> [S, H * D]
    attn_output = attn_output.transpose(1, 2).view({seq_length, -1});
    attn_output = proj_(attn_output);
    return {
        attn_output,
    };
  }
};

class Qwen2VLVisionBlock final : public nn::Module {
  int mlp_hidden_dim_;

  nn::LayerNorm norm1_;
  nn::LayerNorm norm2_;

  VisionAttention attn_;
  VisionMlp mlp_;

 public:
  Qwen2VLVisionBlock() = default;

  inline Qwen2VLVisionBlock(const std::string& name, const Qwen2_5VLConfig& cfg) : nn::Module(name) {
    mlp_hidden_dim_ = cfg.visual_mlp_ratio * cfg.visual_hidden_size;
    norm1_ = reg<nn::LayerNorm>("norm1", std::vector<int32_t>{cfg.visual_hidden_size}, true, true, 1e-6);
    norm2_ = reg<nn::LayerNorm>("norm2", std::vector<int32_t>{cfg.visual_hidden_size}, true, true, 1e-6);
    attn_ = reg<VisionAttention>("attn", cfg);
    mlp_ = reg<VisionMlp>("mlp", cfg);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden_states = inputs[0];
    auto visual_embedding_sin = inputs[1];
    auto visual_embedding_cos = inputs[2];

    hidden_states = hidden_states + attn_(norm1_(hidden_states), visual_embedding_sin, visual_embedding_cos)[0];
    hidden_states = hidden_states + mlp_(norm2_(hidden_states))[0];
    return {hidden_states};
  }
};

class Qwen2VisionTransformerPretrainedModel final : public nn::Module {
  PatchEmbed patch_embed_;
  PatchMerger patch_merger_;
  nn::ModuleList<Qwen2VLVisionBlock> blocks_;

 public:
  Qwen2VisionTransformerPretrainedModel() = default;

  Qwen2VisionTransformerPretrainedModel(const std::string& name, const Qwen2_5VLConfig& cfg) : nn::Module(name) {
    patch_embed_ = reg<PatchEmbed>("patch_embed", cfg);
    patch_merger_ = reg<PatchMerger>("merger", cfg);
    blocks_ = reg<nn::ModuleList<Qwen2VLVisionBlock>>("blocks", cfg.visual_depth, cfg);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden_states = inputs[0];
    auto embedding_sin = inputs[1];
    auto embedding_cos = inputs[2];

    hidden_states = patch_embed_(hidden_states)[0];

    for (auto& b : blocks_.list()) { hidden_states = b(hidden_states, embedding_sin, embedding_cos)[0]; }

    hidden_states = patch_merger_(hidden_states)[0];

    return {hidden_states};
  }
};

}  // namespace mllm::models::qwen2_5vl
