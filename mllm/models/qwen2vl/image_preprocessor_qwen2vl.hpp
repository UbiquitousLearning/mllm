// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cmath>
#include <utility>
#include <algorithm>

#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/preprocessor/visual/Image.hpp"

namespace mllm::models::qwen2vl {

class Qwen2VLImagePreprocessor {
 public:
  inline explicit Qwen2VLImagePreprocessor(int min_pixels = 56 * 56, int max_pixels = 14 * 14 * 4 * 1280)
      : min_pixels_(min_pixels), max_pixels_(max_pixels) {
    OPENAI_CLIP_MEAN = Tensor::empty({3, 1, 1}, kFloat32, kCPU).alloc();
    OPENAI_CLIP_STD = Tensor::empty({3, 1, 1}, kFloat32, kCPU).alloc();

    OPENAI_CLIP_MEAN.ptr<float>()[0] = 0.48145466f;
    OPENAI_CLIP_MEAN.ptr<float>()[1] = 0.4578275f;
    OPENAI_CLIP_MEAN.ptr<float>()[2] = 0.40821073f;

    OPENAI_CLIP_STD.ptr<float>()[0] = 0.26862954f;
    OPENAI_CLIP_STD.ptr<float>()[1] = 0.26130258f;
    OPENAI_CLIP_STD.ptr<float>()[2] = 0.27577711f;
  }

  inline std::pair<int32_t, int32_t> smartResize(int height, int width, int factor = 28, int min_pixels = 56 * 56,
                                                 int max_pixels = 14 * 14 * 4 * 1280) {
    if (std::max(height, width) / static_cast<double>(std::min(height, width)) > 200.0) {
      MLLM_ERROR_EXIT(ExitCode::kIOError, "absolute aspect ratio must be smaller than 200");
    }

    int h_bar = static_cast<int>(std::round(static_cast<double>(height) / factor)) * factor;
    int w_bar = static_cast<int>(std::round(static_cast<double>(width) / factor)) * factor;
    const int current_pixels = h_bar * w_bar;

    if (current_pixels > max_pixels) {
      const double beta = std::sqrt(static_cast<double>(height) * width / max_pixels);
      const int new_height = std::max(1, static_cast<int>(std::floor(height / beta)));
      const int new_width = std::max(1, static_cast<int>(std::floor(width / beta)));

      h_bar = std::max(factor, (new_height / factor) * factor);
      w_bar = std::max(factor, (new_width / factor) * factor);
    }

    else if (current_pixels < min_pixels) {
      const double beta = std::sqrt(static_cast<double>(min_pixels) / (height * width));
      const int new_height = static_cast<int>(std::ceil(height * beta));
      const int new_width = static_cast<int>(std::ceil(width * beta));

      h_bar = (new_height + factor - 1) / factor * factor;
      w_bar = (new_width + factor - 1) / factor * factor;
    }

    return {h_bar, w_bar};
  }

  inline std::pair<Tensor, Tensor> operator()(const std::string& image_path) {
    auto img = Image::open(image_path);
    auto old_w = img.w();
    auto old_h = img.h();
    auto [new_h, new_w] = smartResize(old_h, old_w, 28, min_pixels_, max_pixels_);
    img = img.resize(new_w, new_h);

    // Process patches
    auto patches = img.tensor();           // [H, W, C]
    patches = patches.permute({2, 0, 1});  // [C, H, W]

    // Rescale
    patches = patches * (1.f / 255.f);

    // Normalize
    // FIXME: Using broadcast instead. Need to write broadcast op.
    for (int c = 0; c < 3; c++) {
      for (int h = 0; h < new_h; ++h) {
        for (int w = 0; w < new_w; ++w) {
          *patches.offsettedPtr<float>({c, h, w}) =
              (*patches.offsettedPtr<float>({c, h, w}) - OPENAI_CLIP_MEAN.ptr<float>()[c]) / (OPENAI_CLIP_STD.ptr<float>()[c]);
        }
      }
    }

    // Add time axis
    patches = patches.unsqueeze(0);  // [Time, C, H, W]

    // Repeat patches at Time axis
    patches = patches.repeat(2, 0);

    // Calculate all patches size
    // [Time, Channel, Hight, Width]
    auto channel = patches.shape()[1];
    auto grid_t = patches.shape()[0] / temporal_patch_size_;
    auto grid_h = new_h / patch_size_;
    auto grid_w = new_w / patch_size_;

    // Reshape and permute this tensor to what we want.
    patches = patches.view({grid_t, temporal_patch_size_, channel, grid_h / merge_size_, merge_size_, patch_size_,
                            grid_w / merge_size_, merge_size_, patch_size_});

    patches = patches.permute({0, 3, 6, 4, 7, 2, 1, 5, 8});
    auto flatten_patches = patches.view({grid_t * grid_h * grid_w, channel * temporal_patch_size_ * patch_size_ * patch_size_});

    Tensor grid_thw = Tensor::empty({1, 3}, kInt32, kCPU).alloc();
    grid_thw.ptr<int>()[0] = grid_t;
    grid_thw.ptr<int>()[1] = grid_h;
    grid_thw.ptr<int>()[2] = grid_w;

    return {flatten_patches, grid_thw};
  }

 private:
  Tensor OPENAI_CLIP_MEAN;
  Tensor OPENAI_CLIP_STD;
  int32_t merge_size_ = 2;
  int32_t patch_size_ = 14;
  int32_t min_pixels_ = 56 * 56;
  int32_t max_pixels_ = 14 * 14 * 4 * 1280;
  int32_t temporal_patch_size_ = 2;
};

}  // namespace mllm::models::qwen2vl
