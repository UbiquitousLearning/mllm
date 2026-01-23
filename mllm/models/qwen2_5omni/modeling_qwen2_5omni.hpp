// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

#include "mllm/mllm.hpp"
#include "mllm/core/SlicePrimitives.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/nn/lmcache/StaticCache.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/utils/Enumerate.hpp"

#include "mllm/models/qwen2_5omni/configuration_qwen2_5omni.hpp"

namespace mllm::models::qwen2_5omni {

inline auto makeMultimodalRoPEInvFreq(int output_dim, float rope_theta) -> Tensor {
  auto inv_freq = Tensor::empty({output_dim / 2}, kFloat32, kCPU).alloc();
  auto inv_freq_ptr = inv_freq.ptr<float>();
  for (int i = 0; i < output_dim / 2; i++) { inv_freq_ptr[i] = 1.0f / std::pow(rope_theta, 2.0f * i / output_dim); }
  return inv_freq;
}

inline auto makeMultimodalPositionEmbedding(Tensor& position_ids, const Tensor& inv_freq, int seq_len, int output_dim,
                                            const std::vector<int32_t>& mrope_section) -> std::pair<Tensor, Tensor> {
  MLLM_RT_ASSERT_EQ(position_ids.shape().size(), 3);
  MLLM_RT_ASSERT_EQ(position_ids.shape()[1], 1);

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

    for (int j = 0; j < static_cast<int>(double_rope_section.size()); ++j) {
      int layer = j % 3;
      int s_j = double_rope_section[j];
      int start_col_in = start_cols[j];
      int start_col_out = start_cols[j];
      for (int row = 0; row < num_rows; ++row) {
        auto in_cos_row_ptr = tmp_cos.offsettedPtr<float>({layer, row, 0});
        auto out_cos_row_ptr = cos.offsettedPtr<float>({row, 0});
        for (int c = 0; c < s_j; ++c) { out_cos_row_ptr[start_col_out + c] = in_cos_row_ptr[start_col_in + c]; }

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

inline auto makeWindowIndex(const Tensor& grid_thw, int window_size, int spatial_merge_size,
                            int patch_size) -> std::pair<std::vector<int32_t>, std::vector<int32_t>> {
  MLLM_RT_ASSERT_EQ(grid_thw.shape().size(), 2);
  const int grid_num = grid_thw.shape()[0];

  const int vit_merger_window_size = window_size / spatial_merge_size / patch_size;
  const int spatial_merge_unit = spatial_merge_size * spatial_merge_size;

  std::vector<int32_t> window_index;
  std::vector<int32_t> cu_window_seqlens = {0};
  int window_index_id = 0;

  for (int grid_idx = 0; grid_idx < grid_num; ++grid_idx) {
    const int grid_t = grid_thw.constAt<int>({grid_idx, 0});
    const int grid_h = grid_thw.constAt<int>({grid_idx, 1});
    const int grid_w = grid_thw.constAt<int>({grid_idx, 2});

    const int llm_grid_h = grid_h / spatial_merge_size;
    const int llm_grid_w = grid_w / spatial_merge_size;
    const int pad_h = (vit_merger_window_size - llm_grid_h % vit_merger_window_size) % vit_merger_window_size;
    const int pad_w = (vit_merger_window_size - llm_grid_w % vit_merger_window_size) % vit_merger_window_size;

    const int num_windows_h = (llm_grid_h + pad_h) / vit_merger_window_size;
    const int num_windows_w = (llm_grid_w + pad_w) / vit_merger_window_size;
    const int total_windows = grid_t * num_windows_h * num_windows_w;

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

    std::vector<int32_t> seqlens(total_windows, 0);
    for (int t = 0; t < grid_t; t++) {
      for (int wh = 0; wh < num_windows_h; wh++) {
        for (int ww = 0; ww < num_windows_w; ww++) {
          const int window_idx = t * num_windows_h * num_windows_w + wh * num_windows_w + ww;
          for (int h = 0; h < vit_merger_window_size; h++) {
            for (int w = 0; w < vit_merger_window_size; w++) {
              const int orig_h = wh * vit_merger_window_size + h;
              const int orig_w = ww * vit_merger_window_size + w;
              if (index_padded[t][orig_h][orig_w] != -100) {
                window_index.push_back(index_padded[t][orig_h][orig_w] + window_index_id);
                seqlens[window_idx]++;
              }
            }
          }
        }
      }
    }

    int cumulative = cu_window_seqlens.back();
    for (int i = 0; i < total_windows; i++) {
      cumulative += seqlens[i] * spatial_merge_unit;
      cu_window_seqlens.push_back(cumulative);
    }

    window_index_id += grid_t * llm_grid_h * llm_grid_w;
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

  const auto img_nums = grid_thw.shape()[0];
  int total_positions = 0;
  for (int row = 0; row < img_nums; ++row) {
    const int* dims = grid_thw.offsettedPtr<int>({row, 0});
    total_positions += dims[0] * dims[1] * dims[2];
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

inline auto makeVisualRotaryPosEmb(Tensor& rotary_pos_emb_full, Tensor& pos_ids, Tensor& grid_thw) -> Tensor {
  const int32_t dim = rotary_pos_emb_full.shape()[1];
  const int32_t batch_size = pos_ids.shape()[0];
  const int32_t seq_len = pos_ids.shape()[1];

  int total_positions = 0;
  for (int row = 0; row < grid_thw.shape()[0]; ++row) {
    const int* dims = grid_thw.offsettedPtr<int>({row, 0});
    total_positions += dims[0] * dims[1] * dims[2];
  }

  Tensor out = Tensor::empty({batch_size, seq_len * dim}, kFloat32, kCPU).alloc();

  auto rotary_pos_emb_full_ptr = rotary_pos_emb_full.ptr<float>();
  auto pos_ids_ptr = pos_ids.ptr<int>();

  if (rotary_pos_emb_full.shape()[0] <= 0 || dim <= 0 || batch_size <= 0) {
    MLLM_ERROR_EXIT(ExitCode::kSliceOB, "Invalid tensor dimensions");
  }

  if (total_positions != batch_size) { MLLM_ERROR_EXIT(ExitCode::kSliceOB, "Grid dimensions mismatch with batch size"); }

  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < seq_len; ++j) {
      const int idx = pos_ids_ptr[i * seq_len + j];
      if (idx < 0 || idx >= rotary_pos_emb_full.shape()[0]) {
        MLLM_ERROR_EXIT(ExitCode::kSliceOB, "Position index out of bounds");
      }
    }
  }

  for (int i = 0; i < batch_size; ++i) {
    auto batch_ptr = out.offsettedPtr<float>({i, 0});
    size_t offset = 0;
    for (int j = 0; j < seq_len; ++j) {
      const int idx = pos_ids_ptr[i * seq_len + j];
      auto emb_ptr = rotary_pos_emb_full_ptr + idx * dim;
      std::copy(emb_ptr, emb_ptr + dim, batch_ptr + offset);
      offset += dim;
    }
  }

  return out;
}

inline auto makeVisualRotarySinCos(Tensor& rotary_pos_emb) -> std::pair<Tensor, Tensor> {
  const auto seq = rotary_pos_emb.shape()[0];
  const auto dim = rotary_pos_emb.shape()[1];

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

inline auto makeAudioSinusoidalPosEmb(int32_t length, int32_t channels, float max_timescale = 10000.0f) -> Tensor {
  MLLM_RT_ASSERT(channels % 2 == 0);
  auto pos_emb = Tensor::empty({length, channels}, kFloat32, kCPU).alloc();
  auto pos_ptr = pos_emb.ptr<float>();

  const int half = channels / 2;
  const float log_timescale_increment = std::log(max_timescale) / static_cast<float>(half - 1);

  std::vector<float> inv_timescales(half);
  for (int i = 0; i < half; ++i) {
    inv_timescales[i] = std::exp(-log_timescale_increment * static_cast<float>(i));
  }

  for (int t = 0; t < length; ++t) {
    for (int i = 0; i < half; ++i) {
      const float scaled_time = static_cast<float>(t) * inv_timescales[i];
      pos_ptr[t * channels + i] = std::sin(scaled_time);
      pos_ptr[t * channels + half + i] = std::cos(scaled_time);
    }
  }

  return pos_emb;
}

class Qwen2_5OmniPatchEmbed final : public nn::Module {
  int32_t in_chans_;
  int32_t embed_dim_;
  int32_t patch_size_;
  int32_t temporal_patch_size_;

  nn::Conv3D proj_;

 public:
  Qwen2_5OmniPatchEmbed() = default;

  explicit Qwen2_5OmniPatchEmbed(const std::string& name, const Qwen2_5OmniConfig& cfg) : nn::Module(name) {
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
    hidden_states = hidden_states.view({-1, in_chans_, temporal_patch_size_, patch_size_, patch_size_});
    hidden_states = proj_(hidden_states).view({-1, embed_dim_});
    return {hidden_states};
  }
};

class Qwen2_5OmniPatchMerger final : public nn::Module {
  int32_t hidden_size_;
  int32_t spatial_merge_size_;
  int32_t context_dim_;

  nn::RMSNorm ln_q_;
  nn::Linear mlp_0_;
  nn::Linear mlp_2_;
  nn::GELU mlp_gelu_;

 public:
  Qwen2_5OmniPatchMerger() = default;

  explicit Qwen2_5OmniPatchMerger(const std::string& name, const Qwen2_5OmniConfig& cfg) : nn::Module(name) {
    context_dim_ = cfg.visual_hidden_size;
    spatial_merge_size_ = cfg.visual_spatial_merge_size;
    hidden_size_ = context_dim_ * spatial_merge_size_ * spatial_merge_size_;

    ln_q_ = reg<nn::RMSNorm>("ln_q", 1e-6);
    mlp_0_ = reg<nn::Linear>("mlp.0", hidden_size_, hidden_size_, true, cfg.linear_impl_type);
    mlp_gelu_ = reg<nn::GELU>("mlp.gelu");
    mlp_2_ = reg<nn::Linear>("mlp.2", hidden_size_, cfg.visual_out_hidden_size, true, cfg.linear_impl_type);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto o = ln_q_(inputs[0]).view({-1, hidden_size_});
    o = mlp_0_(o);
    o = mlp_gelu_(o);
    o = mlp_2_(o);
    return {o};
  }
};

class Qwen2_5OmniVisionMLP final : public nn::Module {
  nn::Linear gate_proj_;
  nn::Linear up_proj_;
  nn::Linear down_proj_;
  nn::SiLU silu_;

 public:
  Qwen2_5OmniVisionMLP() = default;
  explicit Qwen2_5OmniVisionMLP(const std::string& name, const Qwen2_5OmniConfig& cfg) : nn::Module(name) {
    gate_proj_ = reg<nn::Linear>("gate_proj", cfg.visual_hidden_size, cfg.visual_intermediate_size, true);
    silu_ = reg<nn::SiLU>("act");
    up_proj_ = reg<nn::Linear>("up_proj", cfg.visual_hidden_size, cfg.visual_intermediate_size, true);
    down_proj_ = reg<nn::Linear>("down_proj", cfg.visual_intermediate_size, cfg.visual_hidden_size, true);
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

class Qwen2_5OmniVisionAttention final : public nn::Module {
  int32_t dim_;
  int32_t num_heads_;
  int32_t head_dim_;

  nn::Linear q_;
  nn::Linear k_;
  nn::Linear v_;
  nn::Linear proj_;
  nn::Softmax softmax_;
  nn::VisionRoPE vision_rope_q_;
  nn::VisionRoPE vision_rope_k_;

 public:
  Qwen2_5OmniVisionAttention() = default;

  explicit Qwen2_5OmniVisionAttention(const std::string& name, const Qwen2_5OmniConfig& cfg) : nn::Module(name) {
    dim_ = cfg.visual_hidden_size;
    num_heads_ = cfg.visual_num_heads;
    head_dim_ = dim_ / num_heads_;

    q_ = reg<nn::Linear>("q", dim_, dim_, true, cfg.linear_impl_type);
    k_ = reg<nn::Linear>("k", dim_, dim_, true, cfg.linear_impl_type);
    v_ = reg<nn::Linear>("v", dim_, dim_, true, cfg.linear_impl_type);
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
    auto hidden_states = inputs[0];
    auto visual_embedding_sin = inputs[1];
    auto visual_embedding_cos = inputs[2];
    auto& mask = inputs[3];

    auto seq_length = hidden_states.shape()[0];

    auto query_states = q_(hidden_states).view({seq_length, num_heads_, head_dim_}).unsqueeze(0);
    auto key_states = k_(hidden_states).view({seq_length, num_heads_, head_dim_}).unsqueeze(0);
    auto value_states = v_(hidden_states).view({seq_length, num_heads_, head_dim_}).unsqueeze(0);

    query_states = vision_rope_q_(query_states, visual_embedding_sin, visual_embedding_cos);
    key_states = vision_rope_k_(key_states, visual_embedding_sin, visual_embedding_cos);

    query_states = query_states.transpose(1, 2);
    key_states = key_states.transpose(1, 2);
    value_states = value_states.transpose(1, 2);

    auto attn = nn::functional::matmul(query_states, key_states, false, true) * (1.f / sqrtf(head_dim_));
    if (mask) { attn = attn + mask; }
    attn = softmax_(attn);

    auto attn_output = nn::functional::matmul(attn, value_states);
    attn_output = attn_output.transpose(1, 2).view({seq_length, -1});
    attn_output = proj_(attn_output);
    return {attn_output};
  }
};

class Qwen2_5OmniVisionBlock final : public nn::Module {
  nn::RMSNorm norm1_;
  nn::RMSNorm norm2_;

  Qwen2_5OmniVisionAttention attn_;
  Qwen2_5OmniVisionMLP mlp_;

 public:
  Qwen2_5OmniVisionBlock() = default;

  explicit Qwen2_5OmniVisionBlock(const std::string& name, const Qwen2_5OmniConfig& cfg) : nn::Module(name) {
    norm1_ = reg<nn::RMSNorm>("norm1", 1e-6);
    norm2_ = reg<nn::RMSNorm>("norm2", 1e-6);
    attn_ = reg<Qwen2_5OmniVisionAttention>("attn", cfg);
    mlp_ = reg<Qwen2_5OmniVisionMLP>("mlp", cfg);
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

class Qwen2_5OmniVisionEncoder final : public nn::Module {
  Qwen2_5OmniPatchEmbed patch_embed_;
  Qwen2_5OmniPatchMerger patch_merger_;
  nn::ModuleList<Qwen2_5OmniVisionBlock> blocks_;
  std::vector<int32_t> visual_fullatt_block_indexes_;
  int32_t visual_window_size_ = 0;
  int32_t visual_spatial_merge_size_ = 1;
  int32_t visual_patch_size_ = 1;
  int32_t spatial_merge_unit_ = 1;

 public:
  Qwen2_5OmniVisionEncoder() = default;

  explicit Qwen2_5OmniVisionEncoder(const std::string& name, const Qwen2_5OmniConfig& cfg) : nn::Module(name) {
    visual_window_size_ = cfg.visual_window_size;
    visual_spatial_merge_size_ = cfg.visual_spatial_merge_size;
    visual_patch_size_ = cfg.visual_patch_size;
    spatial_merge_unit_ = visual_spatial_merge_size_ * visual_spatial_merge_size_;
    visual_fullatt_block_indexes_ = cfg.visual_fullatt_block_indexes;
    patch_embed_ = reg<Qwen2_5OmniPatchEmbed>("patch_embed", cfg);
    patch_merger_ = reg<Qwen2_5OmniPatchMerger>("merger", cfg);
    blocks_ = reg<nn::ModuleList<Qwen2_5OmniVisionBlock>>("blocks", cfg.visual_depth, cfg);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden_states = inputs[0];
    auto embedding_sin = inputs[1];
    auto embedding_cos = inputs[2];
    auto& grid_thw = inputs[3];

    hidden_states = patch_embed_(hidden_states)[0];
    auto [window_index, cu_window_seqlens] =
        makeWindowIndex(grid_thw, visual_window_size_, visual_spatial_merge_size_, visual_patch_size_);

    auto seq_len = hidden_states.shape()[0];
    hidden_states = hidden_states.view({seq_len / spatial_merge_unit_, spatial_merge_unit_, -1});
    hidden_states = hidden_states[{window_index, {kAll}, {kAll}}];
    hidden_states = hidden_states.view({seq_len, -1});

    embedding_sin = embedding_sin.view({seq_len / spatial_merge_unit_, spatial_merge_unit_, -1});
    embedding_sin = embedding_sin[{window_index, {kAll}, {kAll}}];
    embedding_sin = embedding_sin.view({seq_len, -1});
    embedding_cos = embedding_cos.view({seq_len / spatial_merge_unit_, spatial_merge_unit_, -1});
    embedding_cos = embedding_cos[{window_index, {kAll}, {kAll}}];
    embedding_cos = embedding_cos.view({seq_len, -1});

    auto mask = Tensor::empty({1, 1, seq_len, seq_len}, DataTypes::kFloat32, DeviceTypes::kCPU).alloc();
    {
      auto mask_ptr = mask.ptr<mllm_fp32_t>();
      const mllm_fp32_t neg_inf = -1e12f;
      for (int i = 0; i < seq_len * seq_len; ++i) { mask_ptr[i] = neg_inf; }
      for (int i = 1; i < cu_window_seqlens.size(); ++i) {
        const int start = cu_window_seqlens[i - 1];
        const int end = cu_window_seqlens[i];
        for (int r = start; r < end; ++r) {
          for (int c = start; c < end; ++c) { mask_ptr[r * seq_len + c] = 0.0f; }
        }
      }
    }

    for (auto [layer_idx, b] : enumerate(blocks_.list())) {
      if (std::find(visual_fullatt_block_indexes_.begin(), visual_fullatt_block_indexes_.end(), layer_idx)
          != visual_fullatt_block_indexes_.end()) {
        hidden_states = b(hidden_states, embedding_sin, embedding_cos, Tensor::nil())[0];
      } else {
        hidden_states = b(hidden_states, embedding_sin, embedding_cos, mask)[0];
      }
    }

    hidden_states = patch_merger_(hidden_states)[0];

    std::vector<int32_t> reverse_indices(window_index.size());
    std::iota(reverse_indices.begin(), reverse_indices.end(), 0);
    std::sort(reverse_indices.begin(), reverse_indices.end(),
              [&window_index](int i, int j) { return window_index[i] < window_index[j]; });
    hidden_states = hidden_states[{reverse_indices, {kAll}}];

    return {hidden_states};
  }
};

class Qwen2_5OmniAudioAttention final : public nn::Module {
  int32_t embed_dim_ = 0;
  int32_t num_heads_ = 0;
  int32_t head_dim_ = 0;

  nn::Linear k_proj_;
  nn::Linear v_proj_;
  nn::Linear q_proj_;
  nn::Linear out_proj_;

 public:
  Qwen2_5OmniAudioAttention() = default;

  explicit Qwen2_5OmniAudioAttention(const std::string& name, const Qwen2_5OmniConfig& cfg) : nn::Module(name) {
    embed_dim_ = cfg.audio_d_model;
    num_heads_ = cfg.audio_encoder_attention_heads;
    head_dim_ = embed_dim_ / num_heads_;

    k_proj_ = reg<nn::Linear>("k_proj", embed_dim_, embed_dim_, false);
    v_proj_ = reg<nn::Linear>("v_proj", embed_dim_, embed_dim_, true);
    q_proj_ = reg<nn::Linear>("q_proj", embed_dim_, embed_dim_, true);
    out_proj_ = reg<nn::Linear>("out_proj", embed_dim_, embed_dim_, true);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden_states = inputs[0];  // [seq_len, embed_dim]
    auto seq_len = hidden_states.shape()[0];

    auto hidden = hidden_states.unsqueeze(0);  // [1, seq_len, embed_dim]
    auto query_states = q_proj_(hidden);
    auto key_states = k_proj_(hidden);
    auto value_states = v_proj_(hidden);

    query_states = query_states.view({1, seq_len, num_heads_, head_dim_}).transpose(1, 2);
    key_states = key_states.view({1, seq_len, num_heads_, head_dim_}).transpose(1, 2);
    value_states = value_states.view({1, seq_len, num_heads_, head_dim_}).transpose(1, 2);

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    auto attn_weights = nn::functional::matmul(query_states, key_states.transpose(-2, -1)) * scale;
    attn_weights = nn::functional::softmax(attn_weights, -1);
    auto attn_output = nn::functional::matmul(attn_weights, value_states);

    attn_output = attn_output.transpose(1, 2).contiguous().view({1, seq_len, embed_dim_});
    attn_output = out_proj_(attn_output);

    return {attn_output.squeeze(0)};
  }
};

class Qwen2_5OmniAudioEncoderLayer final : public nn::Module {
  Qwen2_5OmniAudioAttention self_attn_;
  nn::LayerNorm self_attn_layer_norm_;
  nn::Linear fc1_;
  nn::Linear fc2_;
  nn::LayerNorm final_layer_norm_;
  nn::GELU activation_fn_;

 public:
  Qwen2_5OmniAudioEncoderLayer() = default;

  explicit Qwen2_5OmniAudioEncoderLayer(const std::string& name, const Qwen2_5OmniConfig& cfg) : nn::Module(name) {
    const int32_t embed_dim = cfg.audio_d_model;
    self_attn_ = reg<Qwen2_5OmniAudioAttention>("self_attn", cfg);
    self_attn_layer_norm_ =
        reg<nn::LayerNorm>("self_attn_layer_norm", std::vector<int32_t>{embed_dim}, true, true, 1e-5);
    fc1_ = reg<nn::Linear>("fc1", embed_dim, cfg.audio_encoder_ffn_dim, true);
    fc2_ = reg<nn::Linear>("fc2", cfg.audio_encoder_ffn_dim, embed_dim, true);
    final_layer_norm_ = reg<nn::LayerNorm>("final_layer_norm", std::vector<int32_t>{embed_dim}, true, true, 1e-5);
    activation_fn_ = reg<nn::GELU>("activation_fn");
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden_states = inputs[0];
    auto residual = hidden_states;

    hidden_states = self_attn_layer_norm_(hidden_states);
    hidden_states = self_attn_(hidden_states)[0];
    hidden_states = residual + hidden_states;

    residual = hidden_states;
    hidden_states = final_layer_norm_(hidden_states);
    hidden_states = fc1_(hidden_states);
    hidden_states = activation_fn_(hidden_states);
    hidden_states = fc2_(hidden_states);
    hidden_states = residual + hidden_states;

    if (hidden_states.dtype() == kFloat16) {
      const float clamp_value = 65504.0f - 1000.0f;
      hidden_states = nn::functional::clip(hidden_states, -clamp_value, clamp_value);
    }

    return {hidden_states};
  }
};

class Qwen2_5OmniAudioEncoder final : public nn::Module {
  nn::Conv1D conv1_;
  nn::Conv1D conv2_;
  nn::GELU gelu_;
  nn::ModuleList<Qwen2_5OmniAudioEncoderLayer> layers_;
  nn::LayerNorm ln_post_;
  nn::AvgPool1d avg_pooler_;
  nn::Linear proj_;
  nn::Embedding audio_bos_eos_token_;

  int32_t num_mel_bins_ = 0;
  int32_t embed_dim_ = 0;
  int32_t n_window_ = 0;
  int32_t output_dim_ = 0;

 public:
  Qwen2_5OmniAudioEncoder() = default;

  explicit Qwen2_5OmniAudioEncoder(const std::string& name, const Qwen2_5OmniConfig& cfg) : nn::Module(name) {
    num_mel_bins_ = cfg.audio_num_mel_bins;
    embed_dim_ = cfg.audio_d_model;
    n_window_ = cfg.audio_n_window;
    output_dim_ = cfg.audio_output_dim;

    conv1_ = reg<nn::Conv1D>("conv1", num_mel_bins_, embed_dim_, 3, 1, 1);
    conv2_ = reg<nn::Conv1D>("conv2", embed_dim_, embed_dim_, 3, 2, 1);
    gelu_ = reg<nn::GELU>("gelu");
    audio_bos_eos_token_ = reg<nn::Embedding>("audio_bos_eos_token", 2, cfg.audio_output_dim);
    layers_ = reg<nn::ModuleList<Qwen2_5OmniAudioEncoderLayer>>("layers", cfg.audio_encoder_layers, cfg);
    ln_post_ = reg<nn::LayerNorm>("ln_post", std::vector<int32_t>{embed_dim_}, true, true, 1e-5);
    avg_pooler_ = reg<nn::AvgPool1d>("avg_pooler", 2, 2);
    proj_ = reg<nn::Linear>("proj", embed_dim_, cfg.audio_output_dim, true);

    auto pos_emb = makeAudioSinusoidalPosEmb(cfg.audio_max_source_positions, embed_dim_);
    registerBuffer("positional_embedding", pos_emb);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto input_features = inputs[0];  // [B, n_mels, T]
    MLLM_RT_ASSERT_EQ(input_features.shape().size(), 3);

    const int32_t batch_size = input_features.shape()[0];
    MLLM_RT_ASSERT_EQ(input_features.shape()[1], num_mel_bins_);
    const int32_t feature_len = input_features.shape()[2];
    MLLM_RT_ASSERT(feature_len > 0);

    auto pos_emb = getBuffer("positional_embedding");

    std::vector<Tensor> audio_outputs;
    audio_outputs.reserve(batch_size);

    for (int32_t b = 0; b < batch_size; ++b) {
      Tensor audio_b = input_features[make_slice(b), kAll, kAll].view({1, num_mel_bins_, feature_len}).contiguous();

      const int32_t chunk_size = n_window_ * 2;
      const int32_t num_chunks = (feature_len + chunk_size - 1) / chunk_size;

      std::vector<Tensor> chunk_outputs;
      chunk_outputs.reserve(num_chunks);

      for (int32_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        const int32_t start = chunk_idx * chunk_size;
        const int32_t chunk_len = std::min(chunk_size, feature_len - start);
        auto chunk = Tensor::empty({1, num_mel_bins_, chunk_len}, kFloat32, kCPU).alloc();
        for (int32_t m = 0; m < num_mel_bins_; ++m) {
          auto src_ptr = audio_b.offsettedPtr<float>({0, m, start});
          auto dst_ptr = chunk.offsettedPtr<float>({0, m, 0});
          std::memcpy(dst_ptr, src_ptr, chunk_len * sizeof(float));
        }

        auto x = conv1_(chunk);
        x = gelu_(x);
        x = conv2_(x);
        x = gelu_(x);
        x = x.transpose(1, 2).contiguous();  // [1, T2, D]

        const int32_t t2 = x.shape()[1];
        MLLM_RT_ASSERT(t2 <= pos_emb.shape()[0]);
        auto pos_ptr = pos_emb.ptr<float>();
        auto x_ptr = x.ptr<float>();
        for (int32_t t = 0; t < t2; ++t) {
          const float* pos_row = pos_ptr + t * embed_dim_;
          float* x_row = x_ptr + t * embed_dim_;
          for (int32_t d = 0; d < embed_dim_; ++d) { x_row[d] += pos_row[d]; }
        }

        auto hidden_states = x.squeeze(0);  // [T2, D]
        for (auto& layer : layers_.list()) { hidden_states = layer(hidden_states)[0]; }
        if (hidden_states.shape()[0] < 2) { continue; }

        auto pooled = hidden_states.unsqueeze(0).transpose(1, 2);  // [1, D, T]
        pooled = avg_pooler_(pooled);
        pooled = pooled.transpose(1, 2).squeeze(0);  // [T', D]
        pooled = ln_post_(pooled);
        pooled = proj_(pooled);
        chunk_outputs.push_back(pooled);
      }

      int32_t total_len = 0;
      for (const auto& chunk : chunk_outputs) { total_len += chunk.shape()[0]; }

      auto merged = Tensor::empty({total_len, output_dim_}, kFloat32, kCPU).alloc();
      int32_t offset = 0;
      for (const auto& chunk : chunk_outputs) {
        const int32_t len = chunk.shape()[0];
        const float* src_ptr = chunk.ptr<float>();
        float* dst_ptr = merged.offsettedPtr<float>({offset, 0});
        std::memcpy(dst_ptr, src_ptr, len * output_dim_ * sizeof(float));
        offset += len;
      }

      audio_outputs.push_back(merged);
    }

    int32_t total_audio_tokens = 0;
    for (const auto& out : audio_outputs) { total_audio_tokens += out.shape()[0]; }

    auto output = Tensor::empty({total_audio_tokens, output_dim_}, kFloat32, kCPU).alloc();
    int32_t offset = 0;
    for (const auto& out : audio_outputs) {
      const int32_t len = out.shape()[0];
      const float* src_ptr = out.ptr<float>();
      float* dst_ptr = output.offsettedPtr<float>({offset, 0});
      std::memcpy(dst_ptr, src_ptr, len * output_dim_ * sizeof(float));
      offset += len;
    }

    return {output};
  }
};

class Qwen2_5OmniMLP final : public nn::Module {
  nn::Linear gate_proj_;
  nn::Linear up_proj_;
  nn::Linear down_proj_;
  nn::SiLU silu_;

 public:
  Qwen2_5OmniMLP() = default;
  Qwen2_5OmniMLP(const std::string& name, const Qwen2_5OmniConfig& cfg) : nn::Module(name) {
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

class Qwen2_5OmniAttention final : public nn::Module {
  nn::Linear q_proj_;
  nn::Linear k_proj_;
  nn::Linear v_proj_;
  nn::Linear o_proj_;
  nn::MultimodalRoPE q_rope_;
  nn::MultimodalRoPE k_rope_;
  nn::CausalMask mask_;
  nn::Softmax softmax_;

  int hidden_size_;
  int head_dim_;
  int num_attention_heads_;
  int num_key_value_heads_;
  int num_key_value_groups_;

 public:
  Qwen2_5OmniAttention() = default;

  Qwen2_5OmniAttention(const std::string& name, const Qwen2_5OmniConfig& cfg) : nn::Module(name) {
    hidden_size_ = cfg.hidden_size;
    num_attention_heads_ = cfg.num_attention_heads;
    num_key_value_heads_ = cfg.num_key_value_heads;
    head_dim_ = hidden_size_ / num_attention_heads_;
    num_key_value_groups_ = num_attention_heads_ / num_key_value_heads_;

    q_proj_ = reg<nn::Linear>("q_proj", hidden_size_, head_dim_ * num_attention_heads_, true, cfg.linear_impl_type);
    k_proj_ = reg<nn::Linear>("k_proj", hidden_size_, head_dim_ * num_key_value_heads_, true, cfg.linear_impl_type);
    v_proj_ = reg<nn::Linear>("v_proj", hidden_size_, head_dim_ * num_key_value_heads_, true, cfg.linear_impl_type);
    o_proj_ = reg<nn::Linear>("o_proj", head_dim_ * num_attention_heads_, hidden_size_, false, cfg.linear_impl_type);

    q_rope_ = reg<nn::MultimodalRoPE>(
        "q_rope", aops::Qwen2VLMultimodalRoPEOpOptions{.rope_theta = cfg.rope_theta,
                                                       .max_position_embeddings = cfg.max_position_embeddings,
                                                       .mrope_section = cfg.mrope_section});
    k_rope_ = reg<nn::MultimodalRoPE>(
        "k_rope", aops::Qwen2VLMultimodalRoPEOpOptions{.rope_theta = cfg.rope_theta,
                                                       .max_position_embeddings = cfg.max_position_embeddings,
                                                       .mrope_section = cfg.mrope_section});

    mask_ = reg<nn::CausalMask>("mask");
    softmax_ = reg<nn::Softmax>("softmax", -1);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = inputs[0];
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto past_kv_cache = args[0].get<nn::StaticCache*>();

    auto query_states = q_proj_(x);
    auto key_states = k_proj_(x);
    auto value_states = v_proj_(x);

    int B = inputs[0].shape()[0];
    int S = inputs[0].shape()[1];

    query_states = query_states.view({B, S, num_attention_heads_, head_dim_});
    key_states = key_states.view({B, S, num_key_value_heads_, head_dim_});
    value_states = value_states.view({B, S, num_key_value_heads_, head_dim_});

    query_states = query_states.transpose(1, 2);
    key_states = key_states.transpose(1, 2);
    value_states = value_states.transpose(1, 2);

    query_states = q_rope_(query_states, llm_embedding_sin, llm_embedding_cos);
    key_states = k_rope_(key_states, llm_embedding_sin, llm_embedding_cos);

    auto [k, v] = past_kv_cache->updateKVCache(layer_idx_, key_states, value_states);
    key_states = k;
    value_states = v;

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

    auto output = nn::functional::matmul(attn, value_states);
    output = output.transpose(1, 2).view({B, S, num_attention_heads_ * head_dim_});
    output = o_proj_(output);
    return {output};
  }

  int layer_idx_;
};

class Qwen2_5OmniDecoder final : public nn::Module {
 public:
  Qwen2_5OmniAttention self_attn_;
  Qwen2_5OmniMLP mlp_;
  nn::RMSNorm input_layer_norm_;
  nn::RMSNorm post_attention_layer_norm_;

  Qwen2_5OmniDecoder() = default;

  Qwen2_5OmniDecoder(const std::string& name, const Qwen2_5OmniConfig& cfg) : nn::Module(name) {
    self_attn_ = reg<Qwen2_5OmniAttention>("self_attn", cfg);
    mlp_ = reg<Qwen2_5OmniMLP>("mlp", cfg);
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
    x = mlp_(x)[0];
    x = x + tmp;
    return {x};
  }
};

class Qwen2_5OmniText final : public nn::Module {
  nn::ModuleList<Qwen2_5OmniDecoder> decode_blocks_;
  nn::RMSNorm norm_;

 public:
  Qwen2_5OmniText() = default;

  Qwen2_5OmniText(const std::string& name, const Qwen2_5OmniConfig& cfg) : nn::Module(name) {
    decode_blocks_ = reg<nn::ModuleList<Qwen2_5OmniDecoder>>("layers", cfg.num_hidden_layers, cfg);
    for (auto [idx, b] : enumerate(decode_blocks_.list())) { b.self_attn_.layer_idx_ = idx; }

    norm_ = reg<nn::RMSNorm>("norm", cfg.rms_norm_eps);
    embedding_ = reg<nn::Embedding>("embed_tokens", cfg.vocab_size, cfg.hidden_size);

    auto inv = makeMultimodalRoPEInvFreq(cfg.hidden_size / cfg.num_attention_heads, cfg.rope_theta);
    registerBuffer("inv_freq", inv);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto& blocks = decode_blocks_.list();
    auto x = inputs[0];
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto& kv_cache = args[0];

    for (auto& block : blocks) { x = block(x, llm_embedding_sin, llm_embedding_cos, kv_cache)[0]; }
    x = norm_(x);

    return {x};
  }

  nn::Embedding embedding_;
};

class Qwen2_5OmniThinker final : public nn::Module {
 public:
  Qwen2_5OmniThinker() = default;
  Qwen2_5OmniThinker(const std::string& name, const Qwen2_5OmniConfig& cfg) : nn::Module(name) {
    model_ = reg<Qwen2_5OmniText>("model", cfg);
    audio_tower_ = reg<Qwen2_5OmniAudioEncoder>("audio_tower", cfg);
    visual_ = reg<Qwen2_5OmniVisionEncoder>("visual", cfg);
    lm_head_ = reg<nn::Linear>("lm_head", cfg.hidden_size, cfg.vocab_size, false, cfg.linear_impl_type);
  }

  Qwen2_5OmniText model_;
  Qwen2_5OmniAudioEncoder audio_tower_;
  Qwen2_5OmniVisionEncoder visual_;
  nn::Linear lm_head_;
};

class Qwen2_5OmniForCausalLM : public ARGeneration {
 public:
  explicit Qwen2_5OmniForCausalLM(const Qwen2_5OmniConfig& cfg) : cfg_(cfg), thinker_("thinker", cfg) {
    kv_cache_ = nn::StaticCache(cfg.max_cache_length, cfg.num_hidden_layers,
                                cfg.num_attention_heads,
                                cfg.num_key_value_heads,
                                cfg.hidden_size / cfg.num_attention_heads,
                                kFloat32,
                                kFloat32,
                                kCPU,
                                false);
    eos_token_id_ = cfg.eos_token_id;
    max_length_ = cfg.max_cache_length;
  }

  ARGenerationOutputPast forward(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override {
    auto sequence = input.at("sequence");

    auto input_embeddings = thinker_.model_.embedding_(sequence);

    if (input.count("input_features")) {
      auto input_features = input.at("input_features");
      auto audio_embeddings = thinker_.audio_tower_(input_features)[0];
      MLLM_RT_ASSERT_EQ(audio_embeddings.shape()[1], input_embeddings.shape()[2]);
      if (audio_embeddings.dtype() != input_embeddings.dtype()) {
        audio_embeddings = audio_embeddings.to(input_embeddings.dtype());
      }

      MLLM_RT_ASSERT_EQ(sequence.shape()[0], 1);
      auto S = sequence.shape()[1];
      std::vector<int32_t> audio_positions;
      audio_positions.reserve(audio_embeddings.shape()[0]);
      auto input_ids_ptr = sequence.ptr<int64_t>();
      for (int s = 0; s < S; ++s) {
        if (input_ids_ptr[s] == cfg_.audio_token_id) { audio_positions.push_back(s); }
      }
      MLLM_RT_ASSERT_EQ(static_cast<int>(audio_positions.size()), audio_embeddings.shape()[0]);

      auto D = input_embeddings.shape()[2];
      if (input_embeddings.dtype() == kFloat32) {
        for (size_t i = 0; i < audio_positions.size(); ++i) {
          auto out_ptr = input_embeddings.offsettedPtr<mllm_fp32_t>({0, audio_positions[i], 0});
          auto in_ptr = audio_embeddings.offsettedPtr<mllm_fp32_t>({static_cast<int32_t>(i), 0});
          std::copy(in_ptr, in_ptr + D, out_ptr);
        }
      } else if (input_embeddings.dtype() == kFloat16) {
        for (size_t i = 0; i < audio_positions.size(); ++i) {
          auto out_ptr = input_embeddings.offsettedPtr<mllm_fp16_t>({0, audio_positions[i], 0});
          auto in_ptr = audio_embeddings.offsettedPtr<mllm_fp16_t>({static_cast<int32_t>(i), 0});
          std::copy(in_ptr, in_ptr + D, out_ptr);
        }
      } else {
        MLLM_ERROR_EXIT(ExitCode::kCoreError, "Unsupported embedding dtype for Qwen2.5-Omni audio input.");
      }
    }

    if (input.count("img")) {
      auto img = input.at("img");
      auto grid_thw = input.at("grid_thw");

      auto inv_freq = makeVisualRoPEInvFreq(cfg_.visual_hidden_size / cfg_.visual_num_heads, 10000.0f);
      auto pos_ids = makeVisualRotaryPosEmbIds(grid_thw, cfg_.visual_spatial_merge_size);

      int max_grid = 0;
      for (int row = 0; row < grid_thw.shape()[0]; ++row) {
        const int* dims = grid_thw.offsettedPtr<int>({row, 0});
        max_grid = std::max({max_grid, dims[1], dims[2]});
      }
      MLLM_RT_ASSERT(max_grid > 0);
      auto rotary_pos_emb_full = makeVisualRotaryPosEmbFull(inv_freq, max_grid);
      auto pos_emb = makeVisualRotaryPosEmb(rotary_pos_emb_full, pos_ids, grid_thw);
      auto [visual_embedding_sin, visual_embedding_cos] = makeVisualRotarySinCos(pos_emb);

      auto visual_embeddings = thinker_.visual_(img, visual_embedding_sin, visual_embedding_cos, grid_thw)[0];
      MLLM_RT_ASSERT_EQ(visual_embeddings.shape()[1], input_embeddings.shape()[2]);
      if (visual_embeddings.dtype() != input_embeddings.dtype()) {
        visual_embeddings = visual_embeddings.to(input_embeddings.dtype());
      }

      MLLM_RT_ASSERT_EQ(sequence.shape()[0], 1);
      auto S = sequence.shape()[1];
      std::vector<int32_t> image_positions;
      image_positions.reserve(visual_embeddings.shape()[0]);
      auto input_ids_ptr = sequence.ptr<int64_t>();
      for (int s = 0; s < S; ++s) {
        if (input_ids_ptr[s] == cfg_.image_token_id) { image_positions.push_back(s); }
      }
      MLLM_RT_ASSERT_EQ(static_cast<int>(image_positions.size()), visual_embeddings.shape()[0]);

      auto D = input_embeddings.shape()[2];
      if (input_embeddings.dtype() == kFloat32) {
        for (size_t i = 0; i < image_positions.size(); ++i) {
          auto out_ptr = input_embeddings.offsettedPtr<mllm_fp32_t>({0, image_positions[i], 0});
          auto in_ptr = visual_embeddings.offsettedPtr<mllm_fp32_t>({static_cast<int32_t>(i), 0});
          std::copy(in_ptr, in_ptr + D, out_ptr);
        }
      } else if (input_embeddings.dtype() == kFloat16) {
        for (size_t i = 0; i < image_positions.size(); ++i) {
          auto out_ptr = input_embeddings.offsettedPtr<mllm_fp16_t>({0, image_positions[i], 0});
          auto in_ptr = visual_embeddings.offsettedPtr<mllm_fp16_t>({static_cast<int32_t>(i), 0});
          std::copy(in_ptr, in_ptr + D, out_ptr);
        }
      } else {
        MLLM_ERROR_EXIT(ExitCode::kCoreError, "Unsupported embedding dtype for Qwen2.5-Omni image input.");
      }
    }

    Tensor position_ids = input.count("position_ids") ? input.at("position_ids") : Tensor::nil();
    Tensor img = input.count("img") ? input.at("img") : Tensor::nil();
    Tensor grid_thw = input.count("grid_thw") ? input.at("grid_thw") : Tensor::nil();
    position_ids = getPositionIds(img, grid_thw, sequence, position_ids);

    auto [llm_embedding_sin, llm_embedding_cos] =
        makeMultimodalPositionEmbedding(position_ids, thinker_.model_.getBuffer("inv_freq"), cfg_.max_position_embeddings,
                                        cfg_.hidden_size / cfg_.num_attention_heads, cfg_.mrope_section);

    auto hidden_states = thinker_.model_(input_embeddings, llm_embedding_sin, llm_embedding_cos, AnyValue(&kv_cache_))[0];
    auto seq_len = hidden_states.shape()[1];
    auto last_hidden = hidden_states[{kAll, {seq_len - 1}, kAll}];
    auto logits = thinker_.lm_head_(last_hidden);

    return {
        {"sequence", logits},
        {"position_ids", position_ids},
    };
  }

  Qwen2_5OmniThinker thinker_;

 private:
  Tensor getPositionIds(Tensor& img, Tensor& grid_thw, Tensor& input_ids, Tensor& position_ids) const {
    MLLM_RT_ASSERT_EQ(input_ids.shape().size(), 2);

    bool has_multimodal = false;
    auto input_ids_ptr = input_ids.ptr<int64_t>();
    auto seq_len = input_ids.shape()[1];
    for (int s = 0; s < seq_len; ++s) {
      if (input_ids_ptr[s] == cfg_.vision_start_token_id || input_ids_ptr[s] == cfg_.audio_start_token_id) {
        has_multimodal = true;
        break;
      }
    }

    if (has_multimodal) { return getPositionIdsPrefill(input_ids, grid_thw); }

    if (!position_ids.isNil()) {
      auto last_pos = *position_ids.offsettedPtr<int64_t>({0, 0, position_ids.shape()[2] - 1});
      auto ret_position_ids = Tensor::empty({3, 1, 1}, kInt64, kCPU).alloc();
      *ret_position_ids.offsettedPtr<int64_t>({0, 0, 0}) = last_pos + 1;
      *ret_position_ids.offsettedPtr<int64_t>({1, 0, 0}) = last_pos + 1;
      *ret_position_ids.offsettedPtr<int64_t>({2, 0, 0}) = last_pos + 1;
      return ret_position_ids;
    }

    auto B = input_ids.shape()[0];
    auto S = seq_len;
    MLLM_RT_ASSERT_EQ(B, 1);

    Tensor out = Tensor::empty({3, B, S}, kInt64, kCPU).alloc();
    for (int d = 0; d < 3; ++d) {
      auto out_ptr = out.offsettedPtr<int64_t>({d, 0, 0});
      for (int64_t s = 0; s < S; ++s) { out_ptr[s] = s; }
    }
    return out;
  }

  Tensor getPositionIdsPrefill(Tensor& input_ids, Tensor& image_grid_thw) const {
    MLLM_RT_ASSERT_EQ(input_ids.shape().size(), 2);

    auto B = input_ids.shape()[0];
    auto S = input_ids.shape()[1];
    MLLM_RT_ASSERT_EQ(B, 1);

    Tensor position_ids = Tensor::empty({3, B, S}, kInt64, kCPU).alloc();

    auto input_ids_ptr = input_ids.ptr<int64_t>();

    auto fill_text_positions = [&](int start_seq, int len, int64_t start_id) {
      for (int d = 0; d < 3; ++d) {
        auto out_ptr = position_ids.offsettedPtr<int64_t>({d, 0, 0});
        for (int i = 0; i < len; ++i) { out_ptr[start_seq + i] = start_id + i; }
      }
    };

    int seq_idx = 0;
    int image_idx = 0;
    int64_t current_max_position_id = -1;
    const int total_images = image_grid_thw.isNil() ? 0 : image_grid_thw.shape()[0];

    while (seq_idx < S) {
      int next_vision = -1;
      int next_audio = -1;
      for (int i = seq_idx; i < S; ++i) {
        if (input_ids_ptr[i] == cfg_.vision_start_token_id) {
          next_vision = i;
          break;
        }
      }
      for (int i = seq_idx; i < S; ++i) {
        if (input_ids_ptr[i] == cfg_.audio_start_token_id) {
          next_audio = i;
          break;
        }
      }

      if (next_vision == -1 && next_audio == -1) {
        const int text_len = S - seq_idx;
        if (text_len > 0) { fill_text_positions(seq_idx, text_len, current_max_position_id + 1); }
        break;
      }

      const bool is_vision = (next_vision != -1) && (next_audio == -1 || next_vision < next_audio);
      const int segment_start = is_vision ? next_vision : next_audio;

      const int text_len = segment_start - seq_idx;
      if (text_len > 0) {
        fill_text_positions(seq_idx, text_len, current_max_position_id + 1);
        current_max_position_id += text_len;
      }

      if (is_vision) {
        fill_text_positions(segment_start, 1, current_max_position_id + 1);
        current_max_position_id += 1;

        int vision_end = -1;
        for (int i = segment_start + 1; i < S; ++i) {
          if (input_ids_ptr[i] == cfg_.vision_end_token_id) {
            vision_end = i;
            break;
          }
        }
        MLLM_RT_ASSERT(vision_end != -1);
        MLLM_RT_ASSERT(image_idx < total_images);
        if (image_grid_thw.isNil()) {
          MLLM_ERROR_EXIT(ExitCode::kCoreError, "Missing grid_thw for Qwen2.5-Omni vision input.");
        }
        MLLM_RT_ASSERT_EQ(image_grid_thw.shape().size(), 2);

        std::vector<int32_t> image_positions;
        for (int i = segment_start + 1; i < vision_end; ++i) {
          if (input_ids_ptr[i] == cfg_.image_token_id) {
            image_positions.push_back(i);
          } else {
            MLLM_ERROR_EXIT(ExitCode::kCoreError, "Unsupported token inside vision segment.");
          }
        }

        const int* grid_dims = image_grid_thw.offsettedPtr<int>({image_idx, 0});
        const int grid_t = grid_dims[0];
        const int grid_h = grid_dims[1];
        const int grid_w = grid_dims[2];

        const int image_token_len = (grid_t * grid_h * grid_w)
                                    / (cfg_.visual_spatial_merge_size * cfg_.visual_spatial_merge_size);
        MLLM_RT_ASSERT_EQ(static_cast<int>(image_positions.size()), image_token_len);

        const int inputs_t = grid_t;
        const int inputs_h = grid_h / cfg_.visual_spatial_merge_size;
        const int inputs_w = grid_w / cfg_.visual_spatial_merge_size;

        const int64_t vision_start_id = current_max_position_id + 1;
        int pos_counter = 0;
        for (int ti = 0; ti < inputs_t; ++ti) {
          const int64_t t_id = vision_start_id + static_cast<int64_t>(ti) * cfg_.position_id_per_seconds;
          for (int hi = 0; hi < inputs_h; ++hi) {
            for (int wi = 0; wi < inputs_w; ++wi) {
              const auto seq_pos = image_positions[pos_counter++];
              *position_ids.offsettedPtr<int64_t>({0, 0, seq_pos}) = t_id;
              *position_ids.offsettedPtr<int64_t>({1, 0, seq_pos}) = vision_start_id + hi;
              *position_ids.offsettedPtr<int64_t>({2, 0, seq_pos}) = vision_start_id + wi;
            }
          }
        }

        const int64_t dim_0_tail = vision_start_id + static_cast<int64_t>(inputs_t - 1) * cfg_.position_id_per_seconds;
        const int64_t dim_1_tail = vision_start_id + inputs_h - 1;
        const int64_t dim_2_tail = vision_start_id + inputs_w - 1;
        current_max_position_id = std::max({dim_0_tail, dim_1_tail, dim_2_tail});

        fill_text_positions(vision_end, 1, current_max_position_id + 1);
        current_max_position_id += 1;

        seq_idx = vision_end + 1;
        image_idx += 1;
      } else {
        fill_text_positions(segment_start, 1, current_max_position_id + 1);
        current_max_position_id += 1;

        int audio_end = -1;
        for (int i = segment_start + 1; i < S; ++i) {
          if (input_ids_ptr[i] == cfg_.audio_end_token_id) {
            audio_end = i;
            break;
          }
        }
        MLLM_RT_ASSERT(audio_end != -1);

        std::vector<int32_t> audio_positions;
        for (int i = segment_start + 1; i < audio_end; ++i) {
          if (input_ids_ptr[i] == cfg_.audio_token_id) {
            audio_positions.push_back(i);
          } else {
            MLLM_ERROR_EXIT(ExitCode::kCoreError, "Unsupported token inside audio segment.");
          }
        }

        const int audio_len = static_cast<int>(audio_positions.size());
        if (audio_len == 0) {
          MLLM_ERROR_EXIT(ExitCode::kCoreError, "Empty audio tokens inside audio segment.");
        }
        const int64_t audio_start_id = current_max_position_id + 1;
        for (int i = 0; i < audio_len; ++i) {
          const int64_t pos_id = audio_start_id + i;
          for (int d = 0; d < 3; ++d) {
            *position_ids.offsettedPtr<int64_t>({d, 0, audio_positions[i]}) = pos_id;
          }
        }
        current_max_position_id += audio_len;

        fill_text_positions(audio_end, 1, current_max_position_id + 1);
        current_max_position_id += 1;

        seq_idx = audio_end + 1;
      }
    }

    MLLM_RT_ASSERT_EQ(image_idx, total_images);
    return position_ids;
  }

  const Qwen2_5OmniConfig& cfg_;
  nn::StaticCache kv_cache_;
};

}  // namespace mllm::models::qwen2_5omni
