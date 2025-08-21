// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <unordered_map>

#include "mllm/mllm.hpp"
#include "mllm/core/DataTypes.hpp"

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

}  // namespace mllm::models::qwen2_5vl
