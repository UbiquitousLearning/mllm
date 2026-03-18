// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
#include "mllm/models/qwen2_5omni/modeling_qwen2_5omni_talker.hpp"
#include "mllm/models/qwen2_5omni/modeling_qwen2_5omni_token2wav.hpp"

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

inline float kaiserBesselI0(float x) {
  const float ax = std::fabs(x);
  if (ax < 3.75f) {
    const float y = (x / 3.75f) * (x / 3.75f);
    return 1.0f + y * (3.5156229f + y * (3.0899424f + y * (1.2067492f + y * (0.2659732f + y * (0.0360768f + y * 0.0045813f)))));
  }
  const float y = 3.75f / ax;
  return (std::exp(ax) / std::sqrt(ax)) *
         (0.39894228f + y * (0.01328592f + y * (0.00225319f + y * (-0.00157565f + y * (0.00916281f +
                                             y * (-0.02057706f + y * (0.02635537f + y * (-0.01647633f + y * 0.00392377f))))))));
}

inline Tensor kaiserSincFilter1d(float cutoff, float half_width, int32_t kernel_size) {
  const bool is_even = (kernel_size % 2 == 0);
  const int32_t half_size = kernel_size / 2;

  const float delta_f = 4.0f * half_width;
  const float attenuation = 2.285f * (half_size - 1) * static_cast<float>(M_PI) * delta_f + 7.95f;

  float beta = 0.0f;
  if (attenuation > 50.0f) {
    beta = 0.1102f * (attenuation - 8.7f);
  } else if (attenuation >= 21.0f) {
    beta = 0.5842f * std::pow(attenuation - 21.0f, 0.4f) + 0.07886f * (attenuation - 21.0f);
  }

  std::vector<float> window(kernel_size);
  const float denom = kaiserBesselI0(beta);
  for (int32_t n = 0; n < kernel_size; ++n) {
    const float ratio = (kernel_size == 1) ? 0.0f : (2.0f * n) / (kernel_size - 1) - 1.0f;
    const float val = beta * std::sqrt(std::max(0.0f, 1.0f - ratio * ratio));
    window[n] = (denom == 0.0f) ? 0.0f : kaiserBesselI0(val) / denom;
  }

  std::vector<float> time_indices(kernel_size);
  if (is_even) {
    for (int32_t i = 0; i < kernel_size; ++i) { time_indices[i] = static_cast<float>(i - half_size) + 0.5f; }
  } else {
    for (int32_t i = 0; i < kernel_size; ++i) { time_indices[i] = static_cast<float>(i - half_size); }
  }

  Tensor filter = Tensor::empty({1, 1, kernel_size}, kFloat32, kCPU).alloc();
  auto* filter_ptr = filter.ptr<float>();

  if (cutoff == 0.0f) {
    std::fill(filter_ptr, filter_ptr + kernel_size, 0.0f);
    return filter;
  }

  float sum = 0.0f;
  for (int32_t i = 0; i < kernel_size; ++i) {
    const float x = 2.0f * cutoff * time_indices[i];
    const float sinc = (x == 0.0f) ? 1.0f : std::sin(static_cast<float>(M_PI) * x) / (static_cast<float>(M_PI) * x);
    const float value = 2.0f * cutoff * window[i] * sinc;
    filter_ptr[i] = value;
    sum += value;
  }
  if (sum != 0.0f) {
    for (int32_t i = 0; i < kernel_size; ++i) { filter_ptr[i] /= sum; }
  }

  return filter;
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

  void clearCache() { kv_cache_.clearCache(); }

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

    const bool output_hidden_states =
        args.count("output_hidden_states") ? args.at("output_hidden_states").get<bool>() : false;

    if (output_hidden_states) {
      return {
          {"sequence", logits},
          {"position_ids", position_ids},
          {"hidden_states", hidden_states},
          {"input_embeddings", input_embeddings},
      };
    }

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

struct Qwen2_5OmniAudioGenerationConfig {
  int32_t thinker_max_new_tokens = 1024;
  bool thinker_do_sample = false;
  int32_t thinker_top_k = 0;
  float thinker_top_p = 0.0f;
  float thinker_temperature = 1.0f;

  int32_t talker_max_new_tokens = 1024;
  int32_t talker_min_new_tokens = 128;
  bool talker_do_sample = true;
  int32_t talker_top_k = 40;
  float talker_top_p = 0.8f;
  float talker_temperature = 0.9f;
  float talker_repetition_penalty = 1.05f;
  std::vector<int64_t> talker_eos_token_ids = {};
  bool suppress_codec_bos = true;

  int32_t token2wav_num_steps = 10;
  float token2wav_guidance_scale = 0.5f;
  float token2wav_sway_coefficient = -1.0f;
};

struct Qwen2_5OmniAudioGenerationResult {
  Tensor sequences = Tensor::nil();
  Tensor wav = Tensor::nil();
};

class Qwen2_5OmniForConditionalGeneration {
 public:
  explicit Qwen2_5OmniForConditionalGeneration(const Qwen2_5OmniConfig& cfg)
      : cfg_(cfg),
        thinker_(cfg_),
        talker_("talker", cfg_.talker_cfg),
        token2wav_("token2wav", cfg_.token2wav_cfg) {}

  void load(const ParameterFile::ptr_t& param) {
    thinker_.thinker_.load(param);
    if (cfg_.enable_audio_output) {
      talker_.load(param);
      token2wav_.load(param);
    }
  }

  void loadSpeakers(const std::string& path) { speaker_map_ = loadSpeakerMap(path); }

  void clearCache() {
    thinker_.clearCache();
    talker_.clearCache();
  }

  Qwen2_5OmniAudioGenerationResult generateAudio(const ARGenerationOutputPast& input, const Qwen2_5OmniAudioGenerationConfig& gen_cfg,
                                                 const std::string& speaker = "") {
    if (!cfg_.enable_audio_output) {
      MLLM_ERROR_EXIT(ExitCode::kCoreError, "Audio output is disabled in Qwen2.5-Omni config.");
    }
    if (speaker_map_.speakers.empty()) {
      MLLM_ERROR_EXIT(ExitCode::kCoreError, "Speaker map is empty. Call loadSpeakers() first.");
    }

    const std::string speaker_name = speaker.empty() ? speaker_map_.default_speaker : speaker;
    auto spk_it = speaker_map_.speakers.find(speaker_name);
    if (spk_it == speaker_map_.speakers.end()) {
      MLLM_ERROR_EXIT(ExitCode::kCoreError, "Unknown speaker '{}'.", speaker_name);
    }

    auto thinker_output = runThinkerGeneration(input, gen_cfg);
    if (thinker_output.generated_ids.empty()) {
      MLLM_ERROR_EXIT(ExitCode::kCoreError, "Thinker produced no tokens; cannot run talker.");
    }

    auto talker_output = runTalkerGeneration(input, thinker_output, spk_it->second, gen_cfg);
    auto wav = token2wav_.forward(talker_output, spk_it->second.cond.to(kFloat32), spk_it->second.ref_mel.to(kFloat32),
                                  gen_cfg.token2wav_num_steps, gen_cfg.token2wav_guidance_scale, gen_cfg.token2wav_sway_coefficient);

    return {
        .sequences = thinker_output.sequences,
        .wav = wav,
    };
  }

  Tensor generateReferenceWav(const std::string& speaker = "") {
    if (speaker_map_.speakers.empty()) { return Tensor::nil(); }
    const std::string speaker_name = speaker.empty() ? speaker_map_.default_speaker : speaker;
    auto spk_it = speaker_map_.speakers.find(speaker_name);
    if (spk_it == speaker_map_.speakers.end()) { return Tensor::nil(); }
    auto ref_mel = spk_it->second.ref_mel.to(kFloat32);
    ref_mel = ref_mel.permute({0, 2, 1});
    if (!ref_mel.isContiguous()) { ref_mel = ref_mel.contiguous(); }
    return token2wav_.vocodeMel(ref_mel);
  }

 private:
  Qwen2_5OmniConfig cfg_;
  Qwen2_5OmniSpeakerMap speaker_map_{};

 public:
  Qwen2_5OmniForCausalLM thinker_;
  Qwen2_5OmniTalker talker_;
  Qwen2_5OmniToken2WavModel token2wav_;

 private:
  struct ThinkerGenerationOutput {
    Tensor sequences = Tensor::nil();
    std::vector<int64_t> generated_ids;
    std::vector<Tensor> token_embeddings;
    std::vector<Tensor> token_hidden_states;
    int32_t prompt_len = 0;
  };

  static Tensor makeTokenTensor(int64_t token_id) {
    Tensor out = Tensor::empty({1, 1}, kInt64, kCPU).alloc();
    out.at<mllm_int64_t>({0, 0}) = token_id;
    return out;
  }

  static Tensor makeTokenTensor(const std::vector<int64_t>& ids) {
    Tensor out = Tensor::empty({1, static_cast<int32_t>(ids.size())}, kInt64, kCPU).alloc();
    auto* ptr = out.ptr<mllm_int64_t>();
    std::copy(ids.begin(), ids.end(), ptr);
    return out;
  }

  static Tensor concatTokenTensors(const std::vector<Tensor>& parts) {
    MLLM_RT_ASSERT(!parts.empty());
    int32_t total_len = 0;
    for (const auto& part : parts) {
      MLLM_RT_ASSERT_EQ(part.shape().size(), 2);
      MLLM_RT_ASSERT_EQ(part.shape()[0], 1);
      MLLM_RT_ASSERT_EQ(part.dtype(), kInt64);
      MLLM_RT_ASSERT_EQ(part.device(), kCPU);
      total_len += part.shape()[1];
    }

    Tensor out = Tensor::empty({1, total_len}, kInt64, kCPU).alloc();
    auto* out_ptr = out.ptr<mllm_int64_t>();
    int32_t offset = 0;
    for (const auto& part : parts) {
      auto* in_ptr = part.ptr<mllm_int64_t>();
      int32_t len = part.shape()[1];
      std::copy(in_ptr, in_ptr + len, out_ptr + offset);
      offset += len;
    }
    return out;
  }

  static void zeroEmbeddingsByTokenId(Tensor& embeds, const Tensor& input_ids, int64_t token_id) {
    MLLM_RT_ASSERT_EQ(input_ids.shape().size(), 2);
    MLLM_RT_ASSERT_EQ(embeds.shape().size(), 3);
    MLLM_RT_ASSERT_EQ(input_ids.shape()[1], embeds.shape()[1]);

    auto seq_len = input_ids.shape()[1];
    auto dim = embeds.shape()[2];
    auto* ids = input_ids.ptr<mllm_int64_t>();

    if (embeds.dtype() == kFloat32) {
      for (int s = 0; s < seq_len; ++s) {
        if (ids[s] != token_id) continue;
        auto* out_ptr = embeds.offsettedPtr<float>({0, s, 0});
        std::fill(out_ptr, out_ptr + dim, 0.0f);
      }
    } else if (embeds.dtype() == kFloat16) {
      for (int s = 0; s < seq_len; ++s) {
        if (ids[s] != token_id) continue;
        auto* out_ptr = embeds.offsettedPtr<mllm_fp16_t>({0, s, 0});
        std::fill(out_ptr, out_ptr + dim, static_cast<mllm_fp16_t>(0));
      }
    } else {
      MLLM_ERROR_EXIT(ExitCode::kCoreError, "Unsupported embedding dtype for Qwen2.5-Omni talker preparation.");
    }
  }

  static Tensor getLastLogits(const Tensor& logits) {
    MLLM_RT_ASSERT_EQ(logits.shape().size(), 3);
    if (logits.shape()[1] == 1) { return logits; }
    return logits[{kAll, logits.shape()[1] - 1, kAll}];
  }

  static int64_t sampleFromDistribution(const std::vector<float>& probs) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(probs.begin(), probs.end());
    return dist(gen);
  }

  static int64_t categoricalSample(const Tensor& probs) {
    MLLM_RT_ASSERT_EQ(probs.dtype(), kFloat32);
    auto* prob_data = probs.ptr<float>();
    int vocab_size = probs.shape().back();

    std::vector<float> cumulative_probs(vocab_size);
    std::partial_sum(prob_data, prob_data + vocab_size, cumulative_probs.begin());

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    float r = dis(gen);

    auto it = std::lower_bound(cumulative_probs.begin(), cumulative_probs.end(), r);
    if (it == cumulative_probs.end()) { return static_cast<int64_t>(vocab_size - 1); }
    return static_cast<int64_t>(std::distance(cumulative_probs.begin(), it));
  }

  static void applyRepetitionPenalty(Tensor& logits, const std::vector<int64_t>& token_ids, float penalty) {
    if (penalty <= 1.0f || token_ids.empty()) { return; }
    if (logits.dtype() != kFloat32) { logits = logits.to(kFloat32); }

    int vocab_size = logits.shape().back();
    if (logits.shape().size() == 2) { MLLM_RT_ASSERT_EQ(logits.shape()[0], 1); }

    std::unordered_set<int64_t> unique_ids;
    unique_ids.reserve(token_ids.size());
    for (auto id : token_ids) { unique_ids.insert(id); }

    auto* logits_ptr = logits.ptr<float>();
    for (auto id : unique_ids) {
      if (id < 0 || id >= vocab_size) { continue; }
      float& v = logits_ptr[id];
      v = (v < 0.0f) ? v * penalty : v / penalty;
    }
  }

  static void applyTopKLogits(Tensor& logits, int32_t top_k) {
    if (top_k <= 0) { return; }
    if (logits.dtype() != kFloat32) { logits = logits.to(kFloat32); }
    if (logits.shape().size() == 2) { MLLM_RT_ASSERT_EQ(logits.shape()[0], 1); }

    int vocab_size = logits.shape().back();
    int k = std::min(std::max(top_k, 1), vocab_size);

    auto* logits_ptr = logits.ptr<float>();
    std::vector<int> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [&logits_ptr](int i1, int i2) { return logits_ptr[i1] > logits_ptr[i2]; });

    float threshold = logits_ptr[indices[k - 1]];
    float neg_inf = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < vocab_size; ++i) {
      if (logits_ptr[i] < threshold) { logits_ptr[i] = neg_inf; }
    }
  }

  static void applyTopPLogits(Tensor& logits, float top_p) {
    if (top_p <= 0.0f || top_p >= 1.0f) { return; }
    if (logits.dtype() != kFloat32) { logits = logits.to(kFloat32); }
    if (logits.shape().size() == 2) { MLLM_RT_ASSERT_EQ(logits.shape()[0], 1); }

    int vocab_size = logits.shape().back();
    auto* logits_ptr = logits.ptr<float>();

    std::vector<int> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&logits_ptr](int i1, int i2) { return logits_ptr[i1] > logits_ptr[i2]; });

    float max_logit = logits_ptr[indices[0]];
    std::vector<float> probs(vocab_size);
    float sum_exp = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
      float exp_val = std::exp(logits_ptr[indices[i]] - max_logit);
      probs[i] = exp_val;
      sum_exp += exp_val;
    }
    if (sum_exp <= 0.0f) { return; }
    for (auto& p : probs) { p /= sum_exp; }

    float cumulative = 0.0f;
    int keep = 0;
    for (int i = 0; i < vocab_size; ++i) {
      cumulative += probs[i];
      keep++;
      if (cumulative > top_p) { break; }
    }
    keep = std::max(keep, 1);

    float neg_inf = -std::numeric_limits<float>::infinity();
    for (int i = keep; i < vocab_size; ++i) {
      logits_ptr[indices[i]] = neg_inf;
    }
  }

  static int64_t sampleFromLogits(Tensor logits, bool do_sample) {
    if (logits.dtype() != kFloat32) { logits = logits.to(kFloat32); }
    if (!do_sample) {
      auto* logits_ptr = logits.ptr<float>();
      int vocab_size = logits.shape().back();
      auto max_it = std::max_element(logits_ptr, logits_ptr + vocab_size);
      return static_cast<int64_t>(std::distance(logits_ptr, max_it));
    }
    Tensor probs = nn::functional::softmax(logits, -1);
    if (probs.dtype() != kFloat32) { probs = probs.to(kFloat32); }
    return categoricalSample(probs);
  }

  static int64_t sampleGreedyLocal(const Tensor& logits) {
    Tensor last_logits = getLastLogits(logits);
    if (last_logits.dtype() != kFloat32) { last_logits = last_logits.to(kFloat32); }
    auto* logits_data = last_logits.ptr<float>();
    int vocab_size = last_logits.shape().back();
    auto max_it = std::max_element(logits_data, logits_data + vocab_size);
    return static_cast<int64_t>(std::distance(logits_data, max_it));
  }

  static int64_t sampleTemperatureLocal(const Tensor& logits, float temperature) {
    Tensor last_logits = getLastLogits(logits);
    if (last_logits.dtype() != kFloat32) { last_logits = last_logits.to(kFloat32); }
    if (temperature != 1.0f && temperature > 0.0f) { last_logits = last_logits * (1.f / temperature); }
    Tensor probs = nn::functional::softmax(last_logits, -1);
    if (probs.dtype() != kFloat32) { probs = probs.to(kFloat32); }
    return categoricalSample(probs);
  }

  static int64_t sampleTopKLocal(const Tensor& logits, int k, float temperature) {
    Tensor last_logits = getLastLogits(logits);
    if (last_logits.dtype() != kFloat32) { last_logits = last_logits.to(kFloat32); }
    if (temperature != 1.0f && temperature > 0.0f) { last_logits = last_logits * (1.f / temperature); }
    Tensor probs = nn::functional::softmax(last_logits, -1);
    if (probs.dtype() != kFloat32) { probs = probs.to(kFloat32); }

    auto* prob_data = probs.ptr<float>();
    int vocab_size = probs.shape().back();
    if (k <= 0 || k > vocab_size) { k = vocab_size; }

    std::vector<int> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [&prob_data](int i1, int i2) { return prob_data[i1] > prob_data[i2]; });

    std::vector<float> top_k_probs(k);
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
      top_k_probs[i] = prob_data[indices[i]];
      sum += top_k_probs[i];
    }
    if (sum <= 0.0f) { return static_cast<int64_t>(indices[0]); }
    for (int i = 0; i < k; ++i) { top_k_probs[i] *= (1.f / sum); }

    return static_cast<int64_t>(indices[sampleFromDistribution(top_k_probs)]);
  }

  static int64_t sampleTopPLocal(const Tensor& logits, float p, float temperature) {
    Tensor last_logits = getLastLogits(logits);
    if (last_logits.dtype() != kFloat32) { last_logits = last_logits.to(kFloat32); }
    if (temperature != 1.0f && temperature > 0.0f) { last_logits = last_logits * (1.f / temperature); }
    Tensor probs = nn::functional::softmax(last_logits, -1);
    if (probs.dtype() != kFloat32) { probs = probs.to(kFloat32); }

    auto* prob_data = probs.ptr<float>();
    int vocab_size = probs.shape().back();

    std::vector<int> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&prob_data](int i1, int i2) { return prob_data[i1] > prob_data[i2]; });

    std::vector<float> top_probs;
    float cumulative_prob = 0.0f;
    int i = 0;
    for (; i < vocab_size && cumulative_prob < p; ++i) {
      top_probs.push_back(prob_data[indices[i]]);
      cumulative_prob += prob_data[indices[i]];
    }

    float sum = std::accumulate(top_probs.begin(), top_probs.end(), 0.0f);
    if (sum <= 0.0f) { return static_cast<int64_t>(indices[0]); }
    for (float& prob : top_probs) { prob *= (1.f / sum); }

    return static_cast<int64_t>(indices[sampleFromDistribution(top_probs)]);
  }

  int64_t sampleToken(const Tensor& logits, bool do_sample, int32_t top_k, float top_p, float temperature) {
    bool use_sampling = do_sample || (temperature != 1.0f) || (top_k > 0) || (top_p > 0.0f);
    if (use_sampling) {
      if (top_k > 0) { return sampleTopKLocal(logits, top_k, temperature); }
      if (top_p > 0.0f) { return sampleTopPLocal(logits, top_p, temperature); }
      return sampleTemperatureLocal(logits, temperature);
    }
    return sampleGreedyLocal(logits);
  }

  ThinkerGenerationOutput runThinkerGeneration(const ARGenerationOutputPast& input, const Qwen2_5OmniAudioGenerationConfig& gen_cfg) {
    thinker_.clearCache();

    ARGenerationOutputPast past = input;
    ARGenerationArgs args;
    args.emplace("output_hidden_states", AnyValue(true));

    const auto& input_ids = input.at("sequence");
    MLLM_RT_ASSERT_EQ(input_ids.shape().size(), 2);

    std::vector<int64_t> generated_ids;
    std::vector<Tensor> token_embeddings;
    std::vector<Tensor> token_hidden_states;

    for (int32_t step = 0; step < gen_cfg.thinker_max_new_tokens; ++step) {
      auto output = thinker_.forward(past, args);
      auto logits = output.at("sequence");

      auto input_embeddings = output.at("input_embeddings");
      auto hidden_states = output.at("hidden_states");

      if (step == 0) {
        auto embeds_to_talker = input_embeddings.clone();
        if (input.count("input_features")) { zeroEmbeddingsByTokenId(embeds_to_talker, input_ids, cfg_.audio_token_id); }
        if (input.count("img")) { zeroEmbeddingsByTokenId(embeds_to_talker, input_ids, cfg_.image_token_id); }
        if (input.count("video")) { zeroEmbeddingsByTokenId(embeds_to_talker, input_ids, cfg_.video_token_id); }
        token_embeddings.emplace_back(std::move(embeds_to_talker));
      } else {
        token_embeddings.emplace_back(std::move(input_embeddings));
      }
      token_hidden_states.emplace_back(std::move(hidden_states));

      int64_t next_token_id = sampleToken(logits, gen_cfg.thinker_do_sample, gen_cfg.thinker_top_k, gen_cfg.thinker_top_p,
                                          gen_cfg.thinker_temperature);
      generated_ids.push_back(next_token_id);

      if (next_token_id == cfg_.eos_token_id) { break; }

      past = std::move(output);
      past["sequence"] = makeTokenTensor(next_token_id);
    }

    std::vector<int64_t> sequence_ids;
    sequence_ids.reserve(input_ids.shape()[1] + generated_ids.size());
    auto* input_ptr = input_ids.ptr<mllm_int64_t>();
    for (int i = 0; i < input_ids.shape()[1]; ++i) { sequence_ids.push_back(input_ptr[i]); }
    sequence_ids.insert(sequence_ids.end(), generated_ids.begin(), generated_ids.end());

    return {
        .sequences = makeTokenTensor(sequence_ids),
        .generated_ids = std::move(generated_ids),
        .token_embeddings = std::move(token_embeddings),
        .token_hidden_states = std::move(token_hidden_states),
        .prompt_len = input_ids.shape()[1],
    };
  }

  Tensor runTalkerGeneration(const ARGenerationOutputPast& input, const ThinkerGenerationOutput& thinker_output,
                             const Qwen2_5OmniSpeakerParams& speaker_params, const Qwen2_5OmniAudioGenerationConfig& gen_cfg) {
    if (thinker_output.generated_ids.empty()) { return Tensor::nil(); }

    talker_.clearCache();

    const auto& input_ids = input.at("sequence");
    const auto& token_embeddings = thinker_output.token_embeddings;
    const auto& token_hidden_states = thinker_output.token_hidden_states;

    std::vector<Tensor> reply_hidden_states(token_hidden_states.begin() + 1, token_hidden_states.end());
    std::vector<Tensor> reply_token_embeds(token_embeddings.begin() + 1, token_embeddings.end());

    auto hidden_dtype = token_hidden_states[0].dtype();
    auto hidden_device = token_hidden_states[0].device();
    auto embed_dtype = token_embeddings[0].dtype();
    auto embed_device = token_embeddings[0].device();
    Tensor reply_hidden = reply_hidden_states.empty()
                              ? Tensor::empty({1, 0, token_hidden_states[0].shape()[2]}, hidden_dtype, hidden_device).alloc()
                              : nn::functional::concat(reply_hidden_states, 1);
    Tensor reply_embeds = reply_token_embeds.empty()
                              ? Tensor::empty({1, 0, token_embeddings[0].shape()[2]}, embed_dtype, embed_device).alloc()
                              : nn::functional::concat(reply_token_embeds, 1);
    auto thinker_reply_part = reply_hidden + reply_embeds;
    if (thinker_reply_part.shape()[1] == 0) {
      MLLM_ERROR_EXIT(ExitCode::kCoreError, "Thinker response is too short for talker conditioning.");
    }

    std::vector<int64_t> talker_text_ids;
    talker_text_ids.reserve(input_ids.shape()[1] + 2);
    auto* input_ptr = input_ids.ptr<mllm_int64_t>();
    for (int i = 0; i < input_ids.shape()[1]; ++i) { talker_text_ids.push_back(input_ptr[i]); }
    talker_text_ids.push_back(speaker_params.bos_token);
    talker_text_ids.push_back(thinker_output.generated_ids.front());
    auto talker_input_text_ids = makeTokenTensor(talker_text_ids);

    std::vector<int64_t> talker_codec_ids(input_ids.shape()[1] + 2, talker_.codec_mask_token());
    talker_codec_ids[input_ids.shape()[1]] = talker_.codec_pad_token();
    talker_codec_ids[input_ids.shape()[1] + 1] = talker_.codec_bos_token();
    auto talker_input_ids = makeTokenTensor(talker_codec_ids);

    auto talker_inputs_embeds = Tensor(token_hidden_states[0]);
    talker_inputs_embeds = talker_inputs_embeds + token_embeddings[0];
    auto talker_text_bos_embed = thinker_.thinker_.model_.embedding_(makeTokenTensor(speaker_params.bos_token));
    auto first_reply = thinker_reply_part.shape()[1] > 0
                           ? thinker_reply_part[{kAll, {0, 1}, kAll}]
                           : Tensor::empty({1, 0, talker_inputs_embeds.shape()[2]}, talker_inputs_embeds.dtype(), talker_inputs_embeds.device())
                                 .alloc();
    talker_inputs_embeds = nn::functional::concat({talker_inputs_embeds, talker_text_bos_embed, first_reply}, 1);

    auto eos_embedding = thinker_.thinker_.model_.embedding_(makeTokenTensor(talker_.text_eos_token()));
    auto pad_embedding = thinker_.thinker_.model_.embedding_(makeTokenTensor(talker_.text_pad_token()));
    Tensor reply_tail =
        thinker_reply_part.shape()[1] > 1
            ? thinker_reply_part[{kAll, {1, thinker_reply_part.shape()[1]}, kAll}]
            : Tensor::empty({1, 0, talker_inputs_embeds.shape()[2]}, talker_inputs_embeds.dtype(), talker_inputs_embeds.device()).alloc();
    thinker_reply_part = nn::functional::concat({reply_tail, eos_embedding, pad_embedding}, 1);

    Tensor talker_attention_mask = Tensor::nil();
    if (input.count("attention_mask")) {
      auto mask = input.at("attention_mask");
      if (mask.dtype() != kFloat16 && mask.dtype() != kFloat32) { mask = mask.to(kFloat32); }
      auto ones = Tensor::ones({1, 2}, mask.dtype(), mask.device());
      talker_attention_mask = nn::functional::concat({mask, ones}, 1);
    }

    Tensor image_grid_thw = input.count("grid_thw") ? input.at("grid_thw") : Tensor::nil();

    std::vector<int64_t> generated_codes;
    Tensor position_ids = Tensor::nil();
    Tensor cur_input_ids = talker_input_ids;
    Tensor cur_input_text_ids = talker_input_text_ids;
    Tensor cur_inputs_embeds = talker_inputs_embeds;
    Tensor cur_reply_part = thinker_reply_part;

    std::vector<int64_t> repetition_tokens = talker_codec_ids;
    repetition_tokens.reserve(talker_codec_ids.size() + gen_cfg.talker_max_new_tokens);

    std::vector<int64_t> eos_ids = gen_cfg.talker_eos_token_ids;
    if (eos_ids.empty()) {
      eos_ids.push_back(talker_.codec_pad_token());
      eos_ids.push_back(talker_.codec_eos_token());
    }

    for (int32_t step = 0; step < gen_cfg.talker_max_new_tokens; ++step) {
      auto output = talker_.forward(cur_input_ids, cur_input_text_ids, cur_reply_part, cur_inputs_embeds, talker_attention_mask,
                                    image_grid_thw, position_ids);

      auto logits = output.logits;
      auto last_logits = getLastLogits(logits);

      const int32_t vocab_size = last_logits.shape().back();

      if (gen_cfg.suppress_codec_bos) {
        auto* logits_ptr = last_logits.ptr<float>();
        logits_ptr[talker_.codec_bos_token()] = -1e9f;
      }
      if (gen_cfg.talker_min_new_tokens > 0 && step < gen_cfg.talker_min_new_tokens) {
        auto* logits_ptr = last_logits.ptr<float>();
        for (int64_t eos_id : eos_ids) {
          if (eos_id >= 0 && eos_id < vocab_size) { logits_ptr[eos_id] = -1e9f; }
        }
      }
      applyRepetitionPenalty(last_logits, repetition_tokens, gen_cfg.talker_repetition_penalty);

      Tensor sample_logits = last_logits;
      if (gen_cfg.talker_temperature != 1.0f && gen_cfg.talker_temperature > 0.0f) {
        sample_logits = sample_logits * (1.f / gen_cfg.talker_temperature);
      }
      if (gen_cfg.talker_do_sample) {
        if (gen_cfg.talker_top_k > 0) { applyTopKLogits(sample_logits, gen_cfg.talker_top_k); }
        if (gen_cfg.talker_top_p > 0.0f) { applyTopPLogits(sample_logits, gen_cfg.talker_top_p); }
      }

      int64_t next_token_id = sampleFromLogits(sample_logits, gen_cfg.talker_do_sample);
      generated_codes.push_back(next_token_id);
      repetition_tokens.push_back(next_token_id);

      if (std::find(eos_ids.begin(), eos_ids.end(), next_token_id) != eos_ids.end()) { break; }

      position_ids = output.position_ids;
      cur_reply_part = output.thinker_reply_part;
      cur_input_ids = makeTokenTensor(next_token_id);
      cur_input_text_ids = Tensor::nil();
      cur_inputs_embeds = Tensor::nil();
    }

    if (!generated_codes.empty()) { generated_codes.pop_back(); }
    if (generated_codes.empty()) {
      MLLM_ERROR_EXIT(ExitCode::kCoreError, "Talker produced no codec tokens.");
    }
    return makeTokenTensor(generated_codes);
  }

 
};

}  // namespace mllm::models::qwen2_5omni
