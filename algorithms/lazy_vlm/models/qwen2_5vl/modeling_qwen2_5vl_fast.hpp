// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <ranges>
#include <unordered_set>

// Custom Cache and Lazy Config
#include "lazy_vlm_cfg_fast.hpp"
#include "../../HKVCacheFast.hpp"

#include <algorithm>
#include <unordered_map>

#include <mllm/mllm.hpp>
#include <mllm/nn/Functional.hpp>
#include <mllm/core/DataTypes.hpp>

#include <mllm/models/ARGeneration.hpp>
#include <mllm/utils/AnyValue.hpp>
#include <mllm/utils/Enumerate.hpp>

#include <mllm/models/qwen2_5vl/configuration_qwen2_5vl.hpp>

using namespace mllm;  // NOLINT

std::vector<int> mappingThis2NextPos(const std::vector<int>& this_layer_pos, const std::vector<int>& next_layer_pos) {
  const std::unordered_set<int> next_set(next_layer_pos.begin(), next_layer_pos.end());
  std::vector<int> res;
  res.reserve(this_layer_pos.size());
  auto idx = std::views::iota(0, static_cast<int>(this_layer_pos.size()))
             | std::views::filter([&](int i) { return next_set.contains(this_layer_pos[i]); });
  for (int i : idx) res.push_back(i);
  return res;
}

inline auto makeWindowIndex(const Tensor& grid_thw, int window_size = 112, int spatial_merge_size = 2,
                            int patch_size = 14) -> std::pair<std::vector<int32_t>, std::vector<int32_t>> {
  int grid_t = grid_thw.constAt<mllm_int32_t>({0, 0});
  int grid_h = grid_thw.constAt<mllm_int32_t>({0, 1});
  int grid_w = grid_thw.constAt<mllm_int32_t>({0, 2});

  int vit_merger_window_size = window_size / spatial_merge_size / patch_size;
  int spatial_merge_unit = spatial_merge_size * spatial_merge_size;

  int llm_grid_h = grid_h / spatial_merge_size;
  int llm_grid_w = grid_w / spatial_merge_size;

  int pad_h = (vit_merger_window_size - llm_grid_h % vit_merger_window_size) % vit_merger_window_size;
  int pad_w = (vit_merger_window_size - llm_grid_w % vit_merger_window_size) % vit_merger_window_size;

  int num_windows_h = (llm_grid_h + pad_h) / vit_merger_window_size;
  int num_windows_w = (llm_grid_w + pad_w) / vit_merger_window_size;
  int total_windows = grid_t * num_windows_h * num_windows_w;

  std::vector<std::vector<std::vector<int32_t>>> index(
      grid_t, std::vector<std::vector<int32_t>>(llm_grid_h, std::vector<int32_t>(llm_grid_w)));

  int counter = 0;
  for (int t = 0; t < grid_t; t++) {
    for (int h = 0; h < llm_grid_h; h++) {
      for (int w = 0; w < llm_grid_w; w++) { index[t][h][w] = counter++; }
    }
  }

  std::vector<std::vector<std::vector<int32_t>>> index_padded(
      grid_t, std::vector<std::vector<int32_t>>(llm_grid_h + pad_h, std::vector<int32_t>(llm_grid_w + pad_w, -100)));

  for (int t = 0; t < grid_t; t++) {
    for (int h = 0; h < llm_grid_h; h++) {
      for (int w = 0; w < llm_grid_w; w++) { index_padded[t][h][w] = index[t][h][w]; }
    }
  }

  std::vector<int32_t> window_index;
  std::vector<int32_t> seqlens(total_windows, 0);

  for (int t = 0; t < grid_t; t++) {
    for (int wh = 0; wh < num_windows_h; wh++) {
      for (int ww = 0; ww < num_windows_w; ww++) {
        int window_idx = t * num_windows_h * num_windows_w + wh * num_windows_w + ww;

        for (int h = 0; h < vit_merger_window_size; h++) {
          for (int w = 0; w < vit_merger_window_size; w++) {
            int orig_h = wh * vit_merger_window_size + h;
            int orig_w = ww * vit_merger_window_size + w;

            if (index_padded[t][orig_h][orig_w] != -100) {
              window_index.push_back(index_padded[t][orig_h][orig_w]);
              seqlens[window_idx]++;
            }
          }
        }
      }
    }
  }

  std::vector<int32_t> cu_window_seqlens = {0};
  int cumulative = 0;
  for (int i = 0; i < total_windows; i++) {
    cumulative += seqlens[i] * spatial_merge_unit;
    cu_window_seqlens.push_back(cumulative);
  }

  return {window_index, cu_window_seqlens};
}

inline auto makeVisualRoPEInvFreq(int32_t dims, float theta) -> Tensor {
  const int half_dim = dims / (2 * 2);
  Tensor inv_freq = Tensor::empty({half_dim}, kFloat32).alloc();
  float* inv_freq_ptr = inv_freq.ptr<float>();
  const float dims_inv = 1.0f / static_cast<float>(dims / 2);
  for (int i = 0; i < half_dim; ++i) {
    const float exponent = (2.0f * i) * dims_inv;
    inv_freq_ptr[i] = 1.0f / std::pow(theta, exponent);
  }
  return inv_freq;
}

inline auto makeVisualRotaryPosEmbIds(Tensor& grid_thw, int32_t spatial_merge_size) -> Tensor {
  MLLM_RT_ASSERT_EQ(grid_thw.shape().size(), 2);

  auto img_nums = grid_thw.shape()[0];

  int total_positions = 0;
  for (int row = 0; row < img_nums; ++row) {
    const int* dims = grid_thw.offsettedPtr<int>({row, 0});
    const int t = dims[0];
    const int h = dims[1];
    const int w = dims[2];
    total_positions += t * h * w;
  }

  Tensor out = Tensor::empty({total_positions, 2}, kInt32).alloc();
  int* out_ptr = out.ptr<int>();
  int out_offset = 0;

  for (int row = 0; row < img_nums; ++row) {
    const int* dims = grid_thw.offsettedPtr<int>({row, 0});

    const int t = dims[0];
    const int h = dims[1];
    const int w = dims[2];

    const int num_h_blocks = h / spatial_merge_size;
    const int num_w_blocks = w / spatial_merge_size;
    const int total_blocks = num_h_blocks * num_w_blocks;
    const int block_area = spatial_merge_size * spatial_merge_size;
    const int grid_size = h * w;

    std::vector<int> flatten_hpos(grid_size);
    std::vector<int> flatten_wpos(grid_size);

    for (int block_idx = 0; block_idx < total_blocks; ++block_idx) {
      const int i_h = block_idx / num_w_blocks;
      const int i_w = block_idx % num_w_blocks;
      const int start_idx = block_idx * block_area;

      const int base_h = i_h * spatial_merge_size;
      const int base_w = i_w * spatial_merge_size;

      for (int j_h = 0; j_h < spatial_merge_size; ++j_h) {
        const int global_h = base_h + j_h;
        for (int j_w = 0; j_w < spatial_merge_size; ++j_w) {
          const int global_w = base_w + j_w;
          const int pos = start_idx + j_h * spatial_merge_size + j_w;
          flatten_hpos[pos] = global_h;
          flatten_wpos[pos] = global_w;
        }
      }
    }

    for (int frame = 0; frame < t; ++frame) {
      for (int pos = 0; pos < grid_size; ++pos) {
        const int out_idx = out_offset + (frame * grid_size + pos) * 2;
        out_ptr[out_idx] = flatten_hpos[pos];
        out_ptr[out_idx + 1] = flatten_wpos[pos];
      }
    }
    out_offset += t * grid_size * 2;
  }

  return out;
}

inline auto makeVisualRotaryPosEmbFull(Tensor& inv_freq, int seq_len) -> Tensor {
  MLLM_RT_ASSERT(seq_len > 0);
  const int32_t dim = inv_freq.shape()[0];
  Tensor freqs = Tensor::empty({seq_len, dim}, kFloat32, kCPU).alloc();
  float* inv_freq_ptr = inv_freq.ptr<float>();
  float* freqs_ptr = freqs.ptr<float>();
  for (int i = 0; i < seq_len; ++i) {
    const float i_val = static_cast<float>(i);
    float* row_ptr = freqs_ptr + i * dim;
    for (int j = 0; j < dim; ++j) { row_ptr[j] = i_val * inv_freq_ptr[j]; }
  }
  return freqs;
}

inline auto makeVisualRotarySinCos(Tensor& rotary_pos_emb) -> std::pair<Tensor, Tensor> {
  auto seq = rotary_pos_emb.shape()[0];
  auto dim = rotary_pos_emb.shape()[1];

  auto rotary_pos_emb_ptr = rotary_pos_emb.ptr<float>();

  Tensor sin_pos_emb = Tensor::empty({seq, dim}, kFloat32, kCPU).alloc();
  Tensor cos_pos_emb = Tensor::empty({seq, dim}, kFloat32, kCPU).alloc();

  auto sin_pos_emb_ptr = sin_pos_emb.ptr<float>();
  auto cos_pos_emb_ptr = cos_pos_emb.ptr<float>();

  for (int i = 0; i < seq; i++) {
    for (int j = 0; j < dim; j++) {
      sin_pos_emb_ptr[i * dim + j] = std::sin(rotary_pos_emb_ptr[i * dim + j]);
      cos_pos_emb_ptr[i * dim + j] = std::cos(rotary_pos_emb_ptr[i * dim + j]);
    }
  }

  return {sin_pos_emb, cos_pos_emb};
}

inline auto makeVisualRotaryPosEmb(Tensor& rotary_pos_emb_full, Tensor& pos_ids, Tensor& grid_thw) -> Tensor {
  const int* grid_dims = grid_thw.offsettedPtr<int>({0, 0});
  const int t = grid_dims[0];
  const int h = grid_dims[1];
  const int w = grid_dims[2];

  const int32_t num_positions = rotary_pos_emb_full.shape()[0];
  const int32_t dim = rotary_pos_emb_full.shape()[1];
  const int32_t batch_size = pos_ids.shape()[0];
  const int32_t seq_len = pos_ids.shape()[1];

  // [batch_size, dim]
  Tensor out = Tensor::empty({batch_size, seq_len * dim}, kFloat32, kCPU).alloc();

  auto rotary_pos_emb_full_ptr = rotary_pos_emb_full.ptr<float>();
  auto pos_ids_ptr = pos_ids.ptr<int>();
  auto out_ptr = out.ptr<float>();

  if (num_positions <= 0 || dim <= 0 || batch_size <= 0) { MLLM_ERROR_EXIT(ExitCode::kSliceOB, "Invalid tensor dimensions"); }

  if (t * h * w != batch_size) { MLLM_ERROR_EXIT(ExitCode::kSliceOB, "Grid dimensions mismatch with batch size"); }

  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < seq_len; ++j) {
      if ((*pos_ids.offsettedPtr<int>({i, j})) < 0 || (*pos_ids.offsettedPtr<int>({i, j})) >= num_positions) {
        MLLM_ERROR_EXIT(ExitCode::kSliceOB, "Position index out of bounds");
      }
    }
  }

  for (int i = 0; i < batch_size; ++i) {
    auto batch_ptr = out.offsettedPtr<float>({i, 0});
    size_t offset = 0;
    for (int j = 0; j < seq_len; ++j) {
      auto emb_ptr = rotary_pos_emb_full.offsettedPtr<float>({(*pos_ids.offsettedPtr<int>({i, j})), 0});
      std::copy(emb_ptr, emb_ptr + dim, batch_ptr + offset);
      offset += dim;
    }
  }

  return out;
}

inline auto makeMultimodalRoPEInvFreq(int output_dim, float rope_theta) -> Tensor {
  auto inv_freq = Tensor::empty({output_dim / 2}, kFloat32, kCPU).alloc();
  auto inv_freq_ptr = inv_freq.ptr<float>();
  for (int i = 0; i < output_dim / 2; i++) { inv_freq_ptr[i] = 1.0 / std::pow(rope_theta, 2.0 * i / output_dim); }
  return inv_freq;
}

inline auto makeMultimodalPositionEmbedding(Tensor& position_ids, const Tensor& inv_freq, int seq_len, int output_dim,
                                            const std::vector<int32_t>& mrope_section) -> std::pair<Tensor, Tensor> {
  // Position ids shape is [3, 1, seq]
  MLLM_RT_ASSERT_EQ(position_ids.shape().size(), 3);
  MLLM_RT_ASSERT_EQ(position_ids.shape()[1], 1);  // Batch size is always 1.

  // [3, seq, dim]
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

  // mrope is always [16, 24, 24]
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

    for (int j = 0; j < double_rope_section.size(); ++j) {
      int layer = j % 3;
      int s_j = double_rope_section[j];
      int start_col_in = start_cols[j];
      int start_col_out = start_cols[j];
      for (int row = 0; row < num_rows; ++row) {
        // Process cos
        auto in_cos_row_ptr = tmp_cos.offsettedPtr<float>({layer, row, 0});
        auto out_cos_row_ptr = cos.offsettedPtr<float>({row, 0});
        for (int c = 0; c < s_j; ++c) { out_cos_row_ptr[start_col_out + c] = in_cos_row_ptr[start_col_in + c]; }

        // Process sin
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

class PatchEmbed final : public nn::Module {
  int32_t in_chans_;
  int32_t embed_dim_;
  int32_t patch_size_;
  int32_t temporal_patch_size_;

  nn::Conv3D proj_;

 public:
  PatchEmbed() = default;

  inline PatchEmbed(const std::string& name, const models::qwen2_5vl::Qwen2_5VLConfig& cfg) : nn::Module(name) {
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

  nn::RMSNorm ln_q_;
  nn::Linear mlp_0_;
  nn::Linear mlp_2_;
  nn::GELU mlp_gelu_;

 public:
  PatchMerger() = default;

  inline PatchMerger(const std::string& name, const models::qwen2_5vl::Qwen2_5VLConfig& cfg) : nn::Module(name) {
    context_dim_ = cfg.visual_hidden_size;
    spatial_merge_size_ = cfg.visual_spatial_merge_size;
    hidden_size_ = context_dim_ * spatial_merge_size_ * spatial_merge_size_;

    ln_q_ = reg<nn::RMSNorm>("ln_q", 1e-6);
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

class Qwen2_5VLVisionMLP final : public nn::Module {
  nn::Linear gate_proj_;
  nn::Linear up_proj_;
  nn::Linear down_proj_;

 public:
  Qwen2_5VLVisionMLP() = default;
  Qwen2_5VLVisionMLP(const std::string& name, const models::qwen2_5vl::Qwen2_5VLConfig& cfg) : nn::Module(name) {
    // clang-format off
    gate_proj_ = reg<nn::Linear>("gate_proj", cfg.visual_hidden_size, cfg.visual_intermediate_size, true);
    up_proj_ = reg<nn::Linear>("up_proj", cfg.visual_hidden_size, cfg.visual_intermediate_size, true);
    down_proj_ = reg<nn::Linear>("down_proj", cfg.visual_intermediate_size, cfg.visual_hidden_size, true);
    // clang-format on
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = gate_proj_(inputs[0]);
    x = nn::functional::silu_(x);
    auto y = up_proj_(inputs[0]);
    x = x * y;
    x = down_proj_(x);
    return {x};
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

  inline VisionAttention(const std::string& name, const models::qwen2_5vl::Qwen2_5VLConfig& cfg) : nn::Module(name) {
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
    auto& mask = inputs[3];

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
    if (mask) { attn = attn + mask; }
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

class Qwen2_5VLVisionBlock final : public nn::Module {
  int mlp_hidden_dim_;

  nn::RMSNorm norm1_;
  nn::RMSNorm norm2_;

  VisionAttention attn_;
  Qwen2_5VLVisionMLP mlp_;

 public:
  Qwen2_5VLVisionBlock() = default;

  inline Qwen2_5VLVisionBlock(const std::string& name, const models::qwen2_5vl::Qwen2_5VLConfig& cfg) : nn::Module(name) {
    mlp_hidden_dim_ = cfg.visual_mlp_ratio * cfg.visual_hidden_size;
    norm1_ = reg<nn::RMSNorm>("norm1", 1e-6);
    norm2_ = reg<nn::RMSNorm>("norm2", 1e-6);
    attn_ = reg<VisionAttention>("attn", cfg);
    mlp_ = reg<Qwen2_5VLVisionMLP>("mlp", cfg);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden_states = inputs[0];
    auto visual_embedding_sin = inputs[1];
    auto visual_embedding_cos = inputs[2];
    auto mask = inputs[3];

    hidden_states = hidden_states + attn_(norm1_(hidden_states), visual_embedding_sin, visual_embedding_cos, mask)[0];
    hidden_states = hidden_states + mlp_(norm2_(hidden_states))[0];
    return {hidden_states};
  }
};

class Qwen2_5VisionTransformerPretrainedModel final : public nn::Module {
  PatchEmbed patch_embed_;
  PatchMerger patch_merger_;
  nn::ModuleList<Qwen2_5VLVisionBlock> blocks_;
  std::vector<int32_t> visual_fullatt_block_indexes_;

 public:
  Qwen2_5VisionTransformerPretrainedModel() = default;

  Qwen2_5VisionTransformerPretrainedModel(const std::string& name, const models::qwen2_5vl::Qwen2_5VLConfig& cfg)
      : nn::Module(name) {
    visual_fullatt_block_indexes_ = cfg.visual_fullatt_block_indexes;
    patch_embed_ = reg<PatchEmbed>("patch_embed", cfg);
    patch_merger_ = reg<PatchMerger>("merger", cfg);
    blocks_ = reg<nn::ModuleList<Qwen2_5VLVisionBlock>>("blocks", cfg.visual_depth, cfg);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden_states = inputs[0];
    auto embedding_sin = inputs[1];
    auto embedding_cos = inputs[2];
    auto& grid_thw = inputs[3];

    // Embedding first
    hidden_states = patch_embed_(hidden_states)[0];
    auto [window_index, cu_window_seqlens] = makeWindowIndex(grid_thw, 112, 2, 14);

    // NOTE: Transformers code:
    // seq_len, _ = hidden_states.size()
    // hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    // hidden_states = hidden_states[window_index, :, :]
    // hidden_states = hidden_states.reshape(seq_len, -1)
    // Transform image embedding and sin/cos embeddings
    auto seq_len = hidden_states.shape()[0];
    hidden_states = hidden_states.view({seq_len / 4, 4, -1});
    hidden_states = hidden_states[{window_index, {kAll}, {kAll}}];
    hidden_states = hidden_states.view({seq_len, -1});

    // NOTE: Transformers code:
    // rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    // rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    // rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    // In mllm, we have already calculate sin and cos embedding. We need to transform them separately.
    embedding_sin = embedding_sin.view({seq_len / 4, 4, -1});
    embedding_sin = embedding_sin[{window_index, {kAll}, {kAll}}];
    embedding_sin = embedding_sin.view({seq_len, -1});
    embedding_cos = embedding_cos.view({seq_len / 4, 4, -1});
    embedding_cos = embedding_cos[{window_index, {kAll}, {kAll}}];
    embedding_cos = embedding_cos.view({seq_len, -1});

    // Procsessing cu_seqlen mask
    auto mask = Tensor::empty({1, 1, seq_len, seq_len}, DataTypes::kFloat32, DeviceTypes::kCPU).alloc();
    {
      auto mask_ptr = mask.ptr<mllm_fp32_t>();
      const mllm_fp32_t neg_inf = -1e12f;
      for (int i = 0; i < seq_len * seq_len; ++i) { mask_ptr[i] = neg_inf; }
      for (int i = 1; i < cu_window_seqlens.size(); ++i) {
        int start = cu_window_seqlens[i - 1];
        int end = cu_window_seqlens[i];
        for (int r = start; r < end; ++r) {
          for (int c = start; c < end; ++c) { mask_ptr[r * seq_len + c] = 0.0f; }
        }
      }
    }

    for (auto [layer_idx, b] : enumerate(blocks_.list())) {
      if (std::find(visual_fullatt_block_indexes_.begin(), visual_fullatt_block_indexes_.end(), layer_idx)
          != visual_fullatt_block_indexes_.end()) {
        // Full Attention
        hidden_states = b(hidden_states, embedding_sin, embedding_cos, Tensor::nil())[0];
      } else {
        // Sliding Window
        hidden_states = b(hidden_states, embedding_sin, embedding_cos, mask)[0];
      }
    }

    hidden_states = patch_merger_(hidden_states)[0];

    // DeTransform image embedding embeddings.
    std::vector<int32_t> reverse_indices(window_index.size());
    std::iota(reverse_indices.begin(), reverse_indices.end(), 0);
    std::sort(reverse_indices.begin(), reverse_indices.end(),
              [&window_index](int i, int j) { return window_index[i] < window_index[j]; });
    hidden_states = hidden_states[{reverse_indices, {kAll}}];

    return {hidden_states};
  }
};

class Qwen2_5VLMLP final : public nn::Module {
  nn::Linear gate_proj_;
  nn::Linear up_proj_;
  nn::Linear down_proj_;

 public:
  Qwen2_5VLMLP() = default;
  Qwen2_5VLMLP(const std::string& name, const models::qwen2_5vl::Qwen2_5VLConfig& cfg) : nn::Module(name) {
    gate_proj_ = reg<nn::Linear>("gate_proj", cfg.hidden_size, cfg.intermediate_size, false, cfg.linear_impl_type);
    up_proj_ = reg<nn::Linear>("up_proj", cfg.hidden_size, cfg.intermediate_size, false, cfg.linear_impl_type);
    down_proj_ = reg<nn::Linear>("down_proj", cfg.intermediate_size, cfg.hidden_size, false, cfg.linear_impl_type);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = gate_proj_(inputs[0]);
    x = nn::functional::silu_(x);
    auto y = up_proj_(inputs[0]);
    x = x * y;
    x = down_proj_(x);
    return {x};
  }
};

class Qwen2_5VLAttention final : public nn::Module {
  nn::Linear q_proj_;
  nn::Linear k_proj_;
  nn::Linear v_proj_;
  nn::Linear o_proj_;
  nn::MultimodalRoPE q_rope_;
  nn::MultimodalRoPE k_rope_;
  nn::Softmax softmax_;
  nn::PagedAttn paged_attn_;

  int hidden_size_;
  int head_dim_;
  int num_attention_heads_;
  int num_key_value_heads_;
  int num_key_value_groups_;

 public:
  Qwen2_5VLAttention() = default;

  Qwen2_5VLAttention(const std::string& name, const models::qwen2_5vl::Qwen2_5VLConfig& cfg) : nn::Module(name) {
    hidden_size_ = cfg.hidden_size;
    num_attention_heads_ = cfg.num_attention_heads;
    num_key_value_heads_ = cfg.num_key_value_heads;
    head_dim_ = hidden_size_ / num_attention_heads_;
    num_key_value_groups_ = num_attention_heads_ / num_key_value_heads_;

    q_proj_ = reg<nn::Linear>("q_proj", hidden_size_, head_dim_ * num_attention_heads_, true, cfg.linear_impl_type);
    k_proj_ = reg<nn::Linear>("k_proj", hidden_size_, head_dim_ * num_key_value_heads_, true, cfg.linear_impl_type);
    v_proj_ = reg<nn::Linear>("v_proj", hidden_size_, head_dim_ * num_key_value_heads_, true, cfg.linear_impl_type);
    o_proj_ = reg<nn::Linear>("o_proj", head_dim_ * num_attention_heads_, hidden_size_, false, cfg.linear_impl_type);

    q_rope_ =
        reg<nn::MultimodalRoPE>("q_rope",
                                aops::Qwen2VLMultimodalRoPEOpOptions{.rope_theta = cfg.rope_theta,
                                                                     .max_position_embeddings = cfg.max_position_embeddings,
                                                                     .mrope_section = cfg.mrope_section},
                                aops::MultimodalRoPEOpOptionsInputType::kBSHD);
    k_rope_ =
        reg<nn::MultimodalRoPE>("k_rope",
                                aops::Qwen2VLMultimodalRoPEOpOptions{.rope_theta = cfg.rope_theta,
                                                                     .max_position_embeddings = cfg.max_position_embeddings,
                                                                     .mrope_section = cfg.mrope_section},
                                aops::MultimodalRoPEOpOptionsInputType::kBSHD);

    paged_attn_ = reg<nn::PagedAttn>("paged_attn", num_attention_heads_ / num_key_value_heads_, false, false, true);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto lazy_vlm_state = args[1].get<LazyVLMStateFast*>();
    auto lazy_vlm_cfg = args[2].get<LazyVLMConfigFast*>();

    if (lazy_vlm_state->cur_step == 0) {  // Prefill
      return forward_lazy_vlm_prefill(inputs, args);
    } else {  // Decode
      return forward_lazy_vlm_decode(inputs, args);
    }
  }

  std::vector<Tensor> forward_lazy_vlm_prefill(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) {
    auto hidden_states = inputs[0];
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto& causal_mask = inputs[3];
    auto past_kv_cache = args[0].get<HKVCacheFast*>();
    auto lazy_vlm_state = args[1].get<LazyVLMStateFast*>();
    auto lazy_vlm_cfg = args[2].get<LazyVLMConfigFast*>();

    // [B, S, H * D]
    auto query_states = q_proj_(hidden_states);
    auto key_states = k_proj_(hidden_states);
    auto value_states = v_proj_(hidden_states);

    int B = inputs[0].shape()[0];
    int S = inputs[0].shape()[1];

    // [B, S, H, D]
    query_states = query_states.view({B, S, num_attention_heads_, head_dim_});
    key_states = key_states.view({B, S, num_key_value_heads_, head_dim_});
    value_states = value_states.view({B, S, num_key_value_heads_, head_dim_});

    // [B, S, H, D]
    query_states = q_rope_(query_states, llm_embedding_sin, llm_embedding_cos);
    key_states = k_rope_(key_states, llm_embedding_sin, llm_embedding_cos);

    // Update KV Cache
    past_kv_cache->updateKVCache(layer_idx_, key_states, value_states, lazy_vlm_state->chosen_pos_in_each[layer_idx_]);

    // Get KV Cache
    std::tie(key_states, value_states) = past_kv_cache->getKVCache(layer_idx_);

    auto [output, attn] =
        paged_attn_(query_states, key_states, value_states,
                    mllm::Tensor::fromVector(lazy_vlm_state->chosen_pos_in_each[layer_idx_],
                                             {(int32_t)lazy_vlm_state->chosen_pos_in_each[layer_idx_].size()}, kInt32, kCPU),
                    causal_mask);

    // [B, S, H, D] -> [B, S, H * D]
    output = output.view({B, S, num_attention_heads_ * head_dim_});
    output = o_proj_(output);

    if (lazy_vlm_cfg->pruning_settings.count(layer_idx_)) { return {output, attn}; }
    return {output};
  }

  std::vector<Tensor> forward_lazy_vlm_decode(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) {
    auto hidden_states = inputs[0];
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto& causal_mask = inputs[3];
    auto past_kv_cache = args[0].get<HKVCacheFast*>();
    auto lazy_vlm_state = args[1].get<LazyVLMStateFast*>();
    auto lazy_vlm_cfg = args[2].get<LazyVLMConfigFast*>();

    // [B, S, H * D]
    auto query_states = q_proj_(hidden_states);
    auto key_states = k_proj_(hidden_states);
    auto value_states = v_proj_(hidden_states);

    int B = inputs[0].shape()[0];
    int S = inputs[0].shape()[1];

    // [B, S, H, D]
    query_states = query_states.view({B, S, num_attention_heads_, head_dim_});
    key_states = key_states.view({B, S, num_key_value_heads_, head_dim_});
    value_states = value_states.view({B, S, num_key_value_heads_, head_dim_});

    // [B, S, H, D]
    query_states = q_rope_(query_states, llm_embedding_sin, llm_embedding_cos);
    key_states = k_rope_(key_states, llm_embedding_sin, llm_embedding_cos);

    // Insert new generated token and delay compute token into kv cache if needed.
    {
      auto insert_pos = lazy_vlm_state->chosen_pos_to_delay_compute[layer_idx_];
      insert_pos.push_back(lazy_vlm_state->chosen_pos_in_each[layer_idx_].back());
      past_kv_cache->updateKVCache(layer_idx_, key_states, value_states, insert_pos);
    }

    // Get KV From current kv cache
    std::tie(key_states, value_states) = past_kv_cache->getKVCache(layer_idx_);

    auto [output, attn] =
        paged_attn_(query_states, key_states, value_states,
                    mllm::Tensor::fromVector(lazy_vlm_state->chosen_pos_in_each[layer_idx_],
                                             {(int32_t)lazy_vlm_state->chosen_pos_in_each[layer_idx_].size()}, kInt32, kCPU),
                    causal_mask);

    // [B, S, H, D] -> [B, S, H * D]
    output = output.view({B, S, num_attention_heads_ * head_dim_});
    output = o_proj_(output);

    if (lazy_vlm_cfg->pruning_settings.count(layer_idx_)) { return {output, attn}; }
    return {output};
  }

  int layer_idx_;
};

class Qwen2_5VLDecoder final : public nn::Module {
 public:
  int32_t layer_idx_;
  Qwen2_5VLAttention self_attn_;
  Qwen2_5VLMLP mlp_;
  nn::RMSNorm input_layer_norm_;
  nn::RMSNorm post_attention_layer_norm_;

  Qwen2_5VLDecoder() = default;

  Qwen2_5VLDecoder(const std::string& name, const models::qwen2_5vl::Qwen2_5VLConfig& cfg) : nn::Module(name) {
    self_attn_ = reg<Qwen2_5VLAttention>("self_attn", cfg);
    mlp_ = reg<Qwen2_5VLMLP>("mlp", cfg);
    input_layer_norm_ = reg<nn::RMSNorm>("input_layernorm", cfg.rms_norm_eps);
    post_attention_layer_norm_ = reg<nn::RMSNorm>("post_attention_layernorm", cfg.rms_norm_eps);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto lazy_vlm_state = args[1].get<LazyVLMStateFast*>();
    auto lazy_vlm_cfg = args[2].get<LazyVLMConfigFast*>();

    if (lazy_vlm_state->cur_step == 0) {  // Prefill
      return forward_lazy_vlm_prefill(inputs, args);
    } else {  // Decode
      return forward_lazy_vlm_decode(inputs, args);
    }
  }

  std::vector<Tensor> forward_lazy_vlm_prefill(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) {
    auto hidden_states = inputs[0];
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto causal_mask = inputs[3];
    auto& kv_cache = args[0].get<HKVCacheFast*>();
    auto lazy_vlm_state = args[1].get<LazyVLMStateFast*>();
    auto lazy_vlm_cfg = args[2].get<LazyVLMConfigFast*>();

    auto residual = hidden_states;
    hidden_states = input_layer_norm_(hidden_states);

    auto outs = self_attn_(hidden_states, llm_embedding_sin, llm_embedding_cos, causal_mask, args[0], args[1], args[2]);
    hidden_states = outs[0];
    hidden_states = residual + hidden_states;

    residual = hidden_states;
    hidden_states = post_attention_layer_norm_(hidden_states);
    hidden_states = mlp_(hidden_states)[0];
    hidden_states = residual + hidden_states;

    if (lazy_vlm_cfg->pruning_settings.count(layer_idx_)) {
      // 1. Update HS Cache for all pruning layer. HS Cache is (B, original_kv_length, D). Only pruning place need this HS
      // Cache.
      kv_cache->updateHiddenStateCache(layer_idx_, lazy_vlm_state->chosen_pos_in_each[layer_idx_], hidden_states);

      // 2. Using self_attn_weights to get the score for pruning. Update current chosen pos.
      lazy_vlm_state->chosen_pos_in_each[layer_idx_ + 1] = lazy_vlm_state->select_high_score_visual_token_prefill(
          outs[1], layer_idx_, lazy_vlm_cfg->pruning_settings[layer_idx_]);
    } else {
      // No pruning performed.
      lazy_vlm_state->chosen_pos_in_each[layer_idx_ + 1] = lazy_vlm_state->chosen_pos_in_each[layer_idx_];
    }

    // LAZY: Mark the processed visual tokens as visit
    kv_cache->visitHiddenStateCache(layer_idx_, lazy_vlm_state->chosen_pos_in_each[layer_idx_]);

    return {hidden_states};
  }

  std::vector<Tensor> forward_lazy_vlm_decode(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) {
    auto hidden_states = inputs[0];
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto causal_mask = inputs[3];
    auto& kv_cache = args[0].get<HKVCacheFast*>();
    auto lazy_vlm_state = args[1].get<LazyVLMStateFast*>();
    auto lazy_vlm_cfg = args[2].get<LazyVLMConfigFast*>();

    auto residual = hidden_states;
    hidden_states = input_layer_norm_(hidden_states);

    auto outs = self_attn_(hidden_states, llm_embedding_sin, llm_embedding_cos, causal_mask, args[0], args[1], args[2]);
    hidden_states = outs[0];
    hidden_states = residual + hidden_states;

    residual = hidden_states;
    hidden_states = post_attention_layer_norm_(hidden_states);
    hidden_states = mlp_(hidden_states)[0];
    hidden_states = residual + hidden_states;

    if (lazy_vlm_cfg->pruning_settings.count(layer_idx_)) {
      auto shape = hidden_states.shape();
      auto b = shape[0];
      auto s = shape[1];
      auto d = shape[2];
      // Update new computed HS Cache.
      // We only need to update the HS[:, :-1, :], the last HS is generated by the model not the delay computed tokens.
      // FIXME: Slice maybe has contiguous error when update.
      auto delay_computed_tokens = hidden_states[{kAll, {kAll, -1}, kAll}].view({b, s - 1, d});
      kv_cache->updateHiddenStateCache(layer_idx_, lazy_vlm_state->chosen_pos_to_delay_compute[layer_idx_],
                                       delay_computed_tokens);

      // Using self_attn_weights to get the score for pruning Update current chosen pos.
      lazy_vlm_state->chosen_pos_in_each[layer_idx_ + 1] = lazy_vlm_state->select_high_score_visual_token_decode(
          outs[1], layer_idx_, lazy_vlm_cfg->pruning_settings[layer_idx_]);
    } else {
      // No pruning performed.
      lazy_vlm_state->chosen_pos_in_each[layer_idx_ + 1] = lazy_vlm_state->chosen_pos_in_each[layer_idx_];
    }

    // LAZY: Mark the processed visual tokens as visit
    kv_cache->visitHiddenStateCache(layer_idx_, lazy_vlm_state->chosen_pos_in_each[layer_idx_]);

    return {hidden_states};
  }
};

class Qwen2_5VLText final : public nn::Module {
  nn::ModuleList<Qwen2_5VLDecoder> decode_blocks_;
  nn::RMSNorm norm_;
  nn::Linear lm_head_;
  bool kv_cache_reordered_ = false;
  bool tie_word_embeddings_;

 public:
  Qwen2_5VLText() = default;

  Qwen2_5VLText(const std::string& name, const models::qwen2_5vl::Qwen2_5VLConfig& cfg) : nn::Module(name) {
    tie_word_embeddings_ = cfg.tie_word_embeddings;

    decode_blocks_ = reg<nn::ModuleList<Qwen2_5VLDecoder>>("layers", cfg.num_hidden_layers, cfg);
    for (auto [idx, b] : enumerate(decode_blocks_.list())) {
      b.layer_idx_ = idx;
      b.self_attn_.layer_idx_ = idx;
    }

    norm_ = reg<nn::RMSNorm>("norm", cfg.rms_norm_eps);
    embedding_ = reg<nn::Embedding>("embed_tokens", cfg.vocab_size, cfg.hidden_size);
    if (cfg.tie_word_embeddings) {
      // NOTE:
      // model.lm_head.weight is quantization weights of model.embed_tokens.weight
      lm_head_ = reg<nn::Linear>("lm_head", cfg.hidden_size, cfg.vocab_size, false, cfg.linear_impl_type);
    }

    // Init inv freq
    auto inv = makeMultimodalRoPEInvFreq(cfg.hidden_size / cfg.num_attention_heads, cfg.rope_theta);
    registerBuffer("inv_freq", inv);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto lazy_vlm_state = args[1].get<LazyVLMStateFast*>();
    auto lazy_vlm_cfg = args[2].get<LazyVLMConfigFast*>();

    if (lazy_vlm_state->cur_step == 0) {  // Prefill
      return forward_lazy_vlm_prefill(inputs, args);
    } else {  // Decode
      return forward_lazy_vlm_decode(inputs, args);
    }
  }

  std::vector<Tensor> forward_lazy_vlm_prefill(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) {
    auto hidden_states = inputs[0];
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto& kv_cache = args[0];
    auto lazy_vlm_state = args[1].get<LazyVLMStateFast*>();
    auto lazy_vlm_cfg = args[2].get<LazyVLMConfigFast*>();

    // Init all states
    // 1. Which token need to be recomputed.
    // 2. Which token is selected.
    lazy_vlm_state->chosen_pos_in_each.clear();
    lazy_vlm_state->chosen_pos_to_delay_compute.clear();
    lazy_vlm_state->chosen_pos_in_each.resize(decode_blocks_.list().size() + 1, {});
    lazy_vlm_state->chosen_pos_to_delay_compute.resize(decode_blocks_.list().size() + 1, {});
    std::ranges::copy(std::views::iota(0, hidden_states.shape()[1]), std::back_inserter(lazy_vlm_state->chosen_pos_in_each[0]));

    // Init position embeddings
    lazy_vlm_state->llm_current_cos = llm_embedding_cos;
    lazy_vlm_state->llm_current_sin = llm_embedding_sin;

    // Create Causal Mask.
    Tensor causal_mask = Tensor::nil();
    // Initialize Causal Mask.
    {
      auto h_len = hidden_states.shape()[1];
      causal_mask = Tensor::zeros({1, 1, h_len, h_len}, kFloat32, kCPU);
      auto mask_data = causal_mask.ptr<mllm_fp32_t>();
      float min_value = -1e12;
      auto indices =
          std::views::iota(0LL, h_len) | std::views::transform([h_len](int64_t i) {
            return std::views::iota(i + 1, h_len) | std::views::transform([i, h_len](int64_t j) { return i * h_len + j; });
          })
          | std::views::join;
      std::ranges::for_each(indices, [mask_data, min_value](int64_t index) { mask_data[index] = min_value; });
    }

    // Original KVcache length. Before pruning length.
    auto original_kv_cache_len = hidden_states.shape()[1];

    for (auto [layer_idx, decode_layer] : enumerate(decode_blocks_.list())) {
      // The common computation
      auto outs = decode_layer(hidden_states, llm_embedding_sin, llm_embedding_cos, causal_mask, args[0], args[1], args[2]);
      hidden_states = outs[0];

      // LAZY: The Lazy part. Token Selection Here.
      if (lazy_vlm_cfg->pruning_settings.count(layer_idx)) {
        // Get the next layer's token position.
        auto& this_layer_pos = lazy_vlm_state->chosen_pos_in_each[layer_idx];
        auto& next_layer_pos = lazy_vlm_state->chosen_pos_in_each[layer_idx + 1];
        auto mapping_this_2_next_pos = mappingThis2NextPos(this_layer_pos, next_layer_pos);
        MLLM_RT_ASSERT_EQ(next_layer_pos.size(), mapping_this_2_next_pos.size());

        // Get pruned hidden_states.
        hidden_states = hidden_states[{{kAll}, mapping_this_2_next_pos, {kAll}}];

        // Re-calculate Causal Mask
        {
          auto h_len = hidden_states.shape()[1];
          causal_mask = Tensor::zeros({1, 1, h_len, h_len}, kFloat32, kCPU);
          auto mask_data = causal_mask.ptr<mllm_fp32_t>();
          float min_value = -1e12;
          auto indices =
              std::views::iota(0LL, h_len) | std::views::transform([h_len](int64_t i) {
                return std::views::iota(i + 1, h_len) | std::views::transform([i, h_len](int64_t j) { return i * h_len + j; });
              })
              | std::views::join;
          std::ranges::for_each(indices, [mask_data, min_value](int64_t index) { mask_data[index] = min_value; });
        }

        // Re-calculate sin and cos embedding.
        llm_embedding_sin = lazy_vlm_state->llm_current_sin[{next_layer_pos, {kAll}}];
        llm_embedding_cos = lazy_vlm_state->llm_current_cos[{next_layer_pos, {kAll}}];
      } else {
        auto& this_layer_pos = lazy_vlm_state->chosen_pos_in_each[layer_idx];
        auto& next_layer_pos = lazy_vlm_state->chosen_pos_in_each[layer_idx + 1];
        MLLM_RT_ASSERT_EQ(this_layer_pos.size(), next_layer_pos.size());
      }
    }

    // Due to insert method kv cache update in lazy_vlm prefill stage, we need to update the cache length manually.
    for (auto [layer_idx, _] : enumerate(decode_blocks_.list())) {
      kv_cache.get<HKVCacheFast*>()->manualCacheLengthUpdate(layer_idx, original_kv_cache_len);
    }

    hidden_states = norm_(hidden_states);
    // Clip x to one seq length
    {
      auto S = hidden_states.shape()[1];
      hidden_states = hidden_states[{kAll, {S - 1}, kAll}];
    }
    if (tie_word_embeddings_) { hidden_states = lm_head_(hidden_states); }

    return {
        hidden_states,
    };
  }

  std::vector<Tensor> forward_lazy_vlm_decode(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) {
    auto hidden_states = inputs[0];
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto& kv_cache = args[0].get<HKVCacheFast*>();
    auto lazy_vlm_state = args[1].get<LazyVLMStateFast*>();
    auto lazy_vlm_cfg = args[2].get<LazyVLMConfigFast*>();

    // Init all states
    // 1. Which token need to be recomputed.
    // 2. Which token is selected.
    lazy_vlm_state->chosen_pos_in_each.clear();
    lazy_vlm_state->chosen_pos_to_delay_compute.clear();
    lazy_vlm_state->chosen_pos_in_each.resize(decode_blocks_.list().size() + 1, {});
    lazy_vlm_state->chosen_pos_to_delay_compute.resize(decode_blocks_.list().size() + 1, {});

    std::ranges::copy(std::views::iota(0, hidden_states.shape()[1] + kv_cache->getCurrentSeqCnt(0)),
                      std::back_inserter(lazy_vlm_state->chosen_pos_in_each[0]));

    // Init position embeddings
    lazy_vlm_state->llm_current_cos = llm_embedding_cos;
    lazy_vlm_state->llm_current_sin = llm_embedding_sin;

    // Create Causal Mask.
    Tensor causal_mask = Tensor::nil();

    // Run loops
    for (auto [layer_idx, decode_layer] : enumerate(decode_blocks_.list())) {
      // The common computation
      auto outs = decode_layer(hidden_states, llm_embedding_sin, llm_embedding_cos, causal_mask, args[0], args[1], args[2]);
      hidden_states = outs[0];

      // LAZY: The Lazy part. Token Selection Here.
      if (lazy_vlm_cfg->pruning_settings.count(layer_idx)) {
        // Get the next layer's token position.
        auto& this_layer_pos = lazy_vlm_state->chosen_pos_in_each[layer_idx];
        auto& next_layer_pos = lazy_vlm_state->chosen_pos_in_each[layer_idx + 1];

        // NOTE: !!! is layer_idx + 1, we need to check the kv cache in the next layer.
        auto& next_layer_kv_cache_not_filled_pos = kv_cache->kv_not_filled_pos_[layer_idx + 1];
        auto need_to_delay_compute_in_next_layer_pos = mappingThis2NextPos(next_layer_pos, next_layer_kv_cache_not_filled_pos);
        std::ranges::sort(need_to_delay_compute_in_next_layer_pos);

        // NOTE: !!! is  layer_idx + 1.
        //  Not include the last generated token.
        lazy_vlm_state->chosen_pos_to_delay_compute[layer_idx + 1] = need_to_delay_compute_in_next_layer_pos;

        auto generated_hidden_states = Tensor::nil();
        {
          auto shape = hidden_states.shape();
          auto b = shape[0];
          auto d = shape[2];
          generated_hidden_states = hidden_states[{kAll, -1, kAll}].contiguous().view({b, 1, d});
        }

        if (need_to_delay_compute_in_next_layer_pos.size() > 0) {
          // NOTE: !!! We need to get the hidden states of this layer.
          // HS -> this_layer -> HS(This cache is what we want to get from) -> next_layer
          auto delay_compute_needed_hidden_states =
              kv_cache->getHiddenStateCache(layer_idx, need_to_delay_compute_in_next_layer_pos);

          // Update hidden states
          // [delay compute hs, generated hs]
          hidden_states = nn::functional::concat({delay_compute_needed_hidden_states, generated_hidden_states}, -2);

          // Update sin and cos embeddings.
          llm_embedding_sin =
              nn::functional::concat({lazy_vlm_state->llm_prefill_sin[{need_to_delay_compute_in_next_layer_pos, {kAll}}],
                                      lazy_vlm_state->llm_current_sin},
                                     -2);
          llm_embedding_cos =
              nn::functional::concat({lazy_vlm_state->llm_prefill_cos[{need_to_delay_compute_in_next_layer_pos, {kAll}}],
                                      lazy_vlm_state->llm_current_cos},
                                     -2);

          // Update causal mask
          {
            const auto& delay_compute = lazy_vlm_state->chosen_pos_to_delay_compute[layer_idx + 1];
            const auto& pos_in_each = lazy_vlm_state->chosen_pos_in_each[layer_idx + 1];
            int q_bound = delay_compute.size();
            int k_bound = pos_in_each.size();
            causal_mask = Tensor::zeros(
                {
                    1,
                    1,
                    q_bound + 1,
                    k_bound,
                },
                kFloat32, kCPU);
            const float min_val = -1e12;
            auto mask_ptr = causal_mask.ptr<float>();
            auto delay_range = std::views::iota(0, q_bound);
            std::ranges::for_each(delay_range, [&](int q_side_idx) {
              int target_val = delay_compute[q_side_idx];
              auto found_it = std::ranges::find(pos_in_each, target_val);
              if (found_it != pos_in_each.end()) {
                int found_index = std::distance(pos_in_each.begin(), found_it);
                size_t start_offset = q_side_idx * k_bound + (found_index + 1);
                size_t count = k_bound - (found_index + 1);
                std::fill_n(mask_ptr + start_offset, count, min_val);
              }
            });
          }
        } else {
          llm_embedding_sin = lazy_vlm_state->llm_current_sin;
          llm_embedding_cos = lazy_vlm_state->llm_current_cos;
          causal_mask = Tensor::nil();
          hidden_states = generated_hidden_states;
        }
      } else {
        lazy_vlm_state->chosen_pos_to_delay_compute[layer_idx + 1] = lazy_vlm_state->chosen_pos_to_delay_compute[layer_idx];
      }

      // NOTE!!!: Add cache length
      kv_cache->manualCacheLengthUpdate(layer_idx, 1);
    }

    hidden_states = norm_(hidden_states);
    // clip x to one seq length
    {
      auto S = hidden_states.shape()[1];
      hidden_states = hidden_states[{kAll, {S - 1}, kAll}];
    }
    if (tie_word_embeddings_) { hidden_states = lm_head_(hidden_states); }

    return {
        hidden_states,
    };
  }

  nn::Embedding embedding_;
};

class Qwen2_5VLForCausalLM : public models::ARGeneration {
 public:
  explicit Qwen2_5VLForCausalLM(const models::qwen2_5vl::Qwen2_5VLConfig& cfg, const LazyVLMConfigFast& lazy_vlm_cfg)
      : cfg(cfg), llm("model", cfg), visual("visual", cfg), lazy_vlm_cfg_(lazy_vlm_cfg) {
    kv_cache_ = HKVCacheFast(cfg.max_cache_length, cfg.num_hidden_layers,
                             cfg.num_attention_heads,                    // q_heads
                             cfg.num_key_value_heads,                    // kv_heads
                             cfg.hidden_size / cfg.num_attention_heads,  // kv_dims
                             kFloat32,                                   // k_dtype
                             kFloat32,                                   // v_dtype
                             kCPU                                        // device_type
    );
    eos_token_id_ = cfg.eos_token_id;
    max_length_ = cfg.max_cache_length;
  }

  models::ARGenerationOutputPast forward(const models::ARGenerationOutputPast& input,
                                         const models::ARGenerationArgs& args) override {
    auto sequence = input.at("sequence");

    // Calculate the text embeddings
    auto input_embeddings = llm.embedding_(sequence);

    if (input.count("img")) {
      auto img = input.at("img");
      auto grid_thw = input.at("grid_thw");

      auto v_len = img.shape()[0];
      auto inv_freq = makeVisualRoPEInvFreq(cfg.visual_hidden_size / cfg.visual_num_heads, 10000.0);
      auto pos_ids = makeVisualRotaryPosEmbIds(grid_thw, cfg.visual_spatial_merge_size);
      auto rotary_pos_emb_full = makeVisualRotaryPosEmbFull(inv_freq, v_len);
      auto pos_emb = makeVisualRotaryPosEmb(rotary_pos_emb_full, pos_ids, grid_thw);
      auto [visual_embedding_sin, visual_embedding_cos] = makeVisualRotarySinCos(pos_emb);
      customEventStartTimePoint("visual");
      auto visual_embeddings = visual(img, visual_embedding_sin, visual_embedding_cos, grid_thw)[0];
      customEventEndTimePoint("visual");

      // Insert visual embeddings into llm's embedding
      int32_t vision_pad_token_start = -1;
      {
        auto& input_ids = sequence;
        auto S = input_ids.shape()[1];
        auto input_ids_ptr = input_ids.ptr<int64_t>();
        for (int s = 0; s < S; ++s) {
          if (input_ids_ptr[s] == cfg.vision_token_id) {
            vision_pad_token_start = s;
            break;
          }
        }
        MLLM_RT_ASSERT(vision_pad_token_start != -1);
      }
      // input_embedding is [B, S, D]
      auto D = input_embeddings.shape()[2];
      auto visual_sequence = visual_embeddings.shape()[0];
      visual_embeddings.copy2(
          input_embeddings[{kAll, {vision_pad_token_start, vision_pad_token_start + visual_sequence}, kAll}]);

      // LAZY: Now, we initialize the hidden states cache in HKVCacheFast
      kv_cache_.initHiddenStateCache(1, visual_sequence, D);
      kv_cache_.kv_not_filled_pos_.reserve(cfg.num_hidden_layers);
      for (int i = 0; i < cfg.num_hidden_layers; ++i) {
        auto range = std::views::iota(vision_pad_token_start, vision_pad_token_start + visual_sequence);
        kv_cache_.kv_not_filled_pos_.emplace_back(range.begin(), range.end());
      }

      // LAZY: update the img token position
      lazy_vlm_state_.first_img_token_pos = vision_pad_token_start;
      lazy_vlm_state_.last_img_token_pos = vision_pad_token_start + visual_sequence;
    }

    auto position_ids = Tensor::nil();
    if (input.count("img")) {
      auto img = input.at("img");
      auto grid_thw = input.at("grid_thw");
      position_ids = getPositionIds(img, grid_thw, sequence, position_ids, cfg);
    } else {
      auto img = Tensor::nil();
      auto grid_thw = Tensor::nil();
      position_ids = input.at("position_ids");
      position_ids = getPositionIds(img, grid_thw, sequence, position_ids, cfg);
    }

    // Generate position ids and embedding sin and cos
    auto [llm_embedding_sin, llm_embedding_cos] =
        makeMultimodalPositionEmbedding(position_ids, llm.getBuffer("inv_freq"), cfg.max_position_embeddings,
                                        cfg.hidden_size / cfg.num_attention_heads, cfg.mrope_section);

    // LAZY: update step
    lazy_vlm_state_.cur_step++;
    if (lazy_vlm_state_.cur_step == 0) {  // Prefill Stage embedding should be saved
      lazy_vlm_state_.llm_prefill_cos = llm_embedding_cos;
      lazy_vlm_state_.llm_prefill_sin = llm_embedding_sin;
    }

    if (lazy_vlm_state_.cur_step == 0) { customEventStartTimePoint("non_visual_prefill"); }
    sequence = llm(input_embeddings, llm_embedding_sin, llm_embedding_cos, AnyValue(&kv_cache_), AnyValue(&lazy_vlm_state_),
                   AnyValue(&lazy_vlm_cfg_))[0];
    if (lazy_vlm_state_.cur_step == 0) { customEventEndTimePoint("non_visual_prefill"); }

    return {
        {"sequence", sequence},
        {"position_ids", position_ids},
    };
  }

  inline auto getPositionIds(Tensor& img, Tensor& grid_thw, Tensor& sequence, Tensor& position_ids,
                             const models::qwen2_5vl::Qwen2_5VLConfig& cfg) -> Tensor {
    // Input is [B, S, D]
    if (!img.isNil()) {  // Prefill
      return getPositionIdsPrefill(sequence, grid_thw, cfg);
    } else {  // Decode
      auto last_pos = *position_ids.offsettedPtr<int64_t>({0, 0, position_ids.shape()[2] - 1});
      auto ret_position_ids = Tensor::empty({3, 1, 1}, kInt64, kCPU).alloc();
      *ret_position_ids.offsettedPtr<int64_t>({0, 0, 0}) = last_pos + 1;
      *ret_position_ids.offsettedPtr<int64_t>({1, 0, 0}) = last_pos + 1;
      *ret_position_ids.offsettedPtr<int64_t>({2, 0, 0}) = last_pos + 1;
      return ret_position_ids;
    }
  }

  inline auto getPositionIdsPrefill(Tensor& input_ids, Tensor& image_grid_thw,
                                    const models::qwen2_5vl::Qwen2_5VLConfig& cfg) -> Tensor {
    // Input is [B, S]
    MLLM_RT_ASSERT_EQ(input_ids.shape().size(), 2);
    // image_grid_thw is [num_images, 3]
    MLLM_RT_ASSERT_EQ(image_grid_thw.shape().size(), 2);

    auto B = input_ids.shape()[0];
    MLLM_RT_ASSERT_EQ(B, 1);
    auto S = input_ids.shape()[1];

    Tensor position_ids = Tensor::empty({3, B, S}, kInt64, kCPU).alloc();

    // Process text and visual
    // 1. Find the place of the first image token
    // Only one image is supported.
    int32_t vision_pad_token_start = -1;
    {
      auto input_ids_ptr = input_ids.ptr<int64_t>();
      for (int s = 0; s < S; ++s) {
        if (input_ids_ptr[s] == cfg.vision_token_id) {
          vision_pad_token_start = s;
          break;
        }
      }
      MLLM_RT_ASSERT(vision_pad_token_start != -1);
    }

    // 2. Calculate grid dimensions
    int img_t, img_h, img_w;
    int inputs_t, inputs_h, inputs_w;
    {
      auto image_grid_thw_ptr = image_grid_thw.ptr<int32_t>();
      img_t = image_grid_thw_ptr[0];
      img_h = image_grid_thw_ptr[1];
      img_w = image_grid_thw_ptr[2];

      inputs_t = img_t;
      inputs_h = img_h / cfg.visual_spatial_merge_size;
      inputs_w = img_w / cfg.visual_spatial_merge_size;
    }

    // 3. We assume the inputs format is: T T T V V V T T T
    int64_t current_max_position_id = 0;
    // 3.1 Handle text (Sys token as usual).
    {
      int64_t start_idx = current_max_position_id;
      for (int d = 0; d < 3; ++d) {
        auto position_ids_ptr = position_ids.offsettedPtr<int64_t>({d, 0, 0});
        for (int64_t k = 0; k < vision_pad_token_start; ++k) { position_ids_ptr[k] = start_idx + k; }
      }
      current_max_position_id = vision_pad_token_start - 1;
    }
    // 3.2 Handle image
    {
      int _cnt = 0;
      int64_t vision_start_id = current_max_position_id + 1;
      for (int64_t ti = 0; ti < inputs_t; ++ti) {
        for (int64_t hi = 0; hi < inputs_h; ++hi) {
          for (int64_t wi = 0; wi < inputs_w; ++wi) {
            *position_ids.offsettedPtr<int64_t>({0, 0, vision_pad_token_start + _cnt}) = vision_start_id + ti;

            *position_ids.offsettedPtr<int64_t>({1, 0, vision_pad_token_start + _cnt}) = vision_start_id + hi;

            *position_ids.offsettedPtr<int64_t>({2, 0, vision_pad_token_start + _cnt}) = vision_start_id + wi;

            _cnt++;
          }
        }
      }
      auto dim_0_tail =
          *position_ids.offsettedPtr<int64_t>({0, 0, vision_pad_token_start + inputs_t * inputs_h * inputs_w - 1});
      auto dim_1_tail =
          *position_ids.offsettedPtr<int64_t>({1, 0, vision_pad_token_start + inputs_t * inputs_h * inputs_w - 1});
      auto dim_2_tail =
          *position_ids.offsettedPtr<int64_t>({2, 0, vision_pad_token_start + inputs_t * inputs_h * inputs_w - 1});
      current_max_position_id = std::max({dim_0_tail, dim_1_tail, dim_2_tail});
    }
    // 3.3 Handle Prompt
    {
      const int64_t vision_token_count = inputs_t * inputs_h * inputs_w;
      const int64_t trailing_text_start_seq = vision_pad_token_start + vision_token_count;
      const int64_t trailing_text_count = S - trailing_text_start_seq;

      if (trailing_text_count > 0) {
        int64_t start_id = current_max_position_id + 1;
        for (int d = 0; d < 3; ++d) {
          auto position_ids_ptr = position_ids.offsettedPtr<int64_t>({d, 0, 0});
          for (int64_t k = 0; k < trailing_text_count; ++k) {
            const int64_t seq_idx = trailing_text_start_seq + k;
            position_ids_ptr[seq_idx] = start_id + k;
          }
        }
      }
    }

    return position_ids;
  }

  void perfSummary() override {
    mllm::models::ARGeneration::perfSummary();
    if (!custom_event_time_.empty()) {
      bool has_visual = custom_event_time_.count("visual") > 0;
      bool has_non_visual_prefill = custom_event_time_.count("non_visual_prefill") > 0;

      if (has_visual || has_non_visual_prefill) {
        fmt::print(fg(fmt::color::magenta), "\n{:=^50}\n", " Qwen2.5-VL Custom Events ");

        if (has_visual) {
          const auto& time_points = custom_event_time_["visual"];
          auto duration = std::chrono::duration_cast<std::chrono::microseconds>(time_points.second - time_points.first).count();
          fmt::print(fg(fmt::color::white), "{:<20}: ", "Visual processing");
          fmt::print(fg(fmt::color::yellow), "{:>10.2f} μs\n", (double)duration);
        }

        if (has_non_visual_prefill) {
          const auto& time_points = custom_event_time_["non_visual_prefill"];
          auto duration = std::chrono::duration_cast<std::chrono::microseconds>(time_points.second - time_points.first).count();
          fmt::print(fg(fmt::color::white), "{:<20}: ", "Non-visual prefill");
          fmt::print(fg(fmt::color::yellow), "{:>10.2f} μs", (double)duration);

          if (duration > 0 && ar_prefill_tokens_ > 0) {
            double tokens_per_sec = (double)ar_prefill_tokens_ / (duration / 1000000.0);
            fmt::print(fg(fmt::color::white), " ({:>6.2f} tokens/s)", tokens_per_sec);
          }
          fmt::print("\n");
        }

        fmt::print(fg(fmt::color::magenta), "{:=^50}\n", "");
      }
    }
  }

  const models::qwen2_5vl::Qwen2_5VLConfig& cfg;
  Qwen2_5VLText llm;
  Qwen2_5VisionTransformerPretrainedModel visual;

 private:
  HKVCacheFast kv_cache_;
  LazyVLMConfigFast lazy_vlm_cfg_;
  LazyVLMStateFast lazy_vlm_state_{};
};
