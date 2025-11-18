// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <cmath>
#include <utility>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdio>

#include "mllm/core/DataTypes.hpp"
#include "mllm/mllm.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/preprocessor/tokenizers/BPE.hpp"
#include "mllm/preprocessor/visual/Image.hpp"

namespace mllm::models::minicpmo {

class ImageSliceProcessor {
 public:
  ImageSliceProcessor() = default;

  explicit ImageSliceProcessor(int max_slice_nums = 9, int scale_resolution = 448, int patch_size = 14)
      : max_slice_nums_(max_slice_nums), scale_resolution_(scale_resolution), patch_size_(patch_size) {}

  std::tuple<Image, std::vector<std::vector<Image>>, std::vector<int>> slice_image(Image img, bool never_split = false) {
    int original_width = img.w();
    int original_height = img.h();

    auto best_grid = get_sliced_grid(original_width, original_height, never_split);
    std::vector<std::vector<Image>> patches;
    Image source_img;
    if (best_grid.empty()) {
      // Don't need to slice, just upsample
      auto best_size = find_best_resize(original_width, original_height, true);
      source_img = img.resize(best_size.first, best_size.second);
    } else {
      // Source image: down-sampling and ensure divided by patch_size
      auto best_resize = find_best_resize(original_width, original_height);
      source_img = img.resize(best_resize.first, best_resize.second);
      // Refine image for slicing
      auto refine_size = get_refine_size(original_width, original_height, best_grid);
      auto refine_image = img.resize(refine_size.first, refine_size.second);
      patches = split_to_patches(refine_image, best_grid);
    }

    return std::make_tuple(source_img, patches, best_grid);
  }

  std::vector<std::vector<Image>> split_to_patches(Image& image, const std::vector<int>& grid) {
    std::vector<std::vector<Image>> patches;
    int width = image.w();
    int height = image.h();
    int grid_x = width / grid[0];
    int grid_y = height / grid[1];

    for (int i = 0; i < height; i += grid_y) {
      std::vector<Image> row_patches;
      for (int j = 0; j < width; j += grid_x) {
        // Calculate crop region
        int crop_width = std::min(grid_x, width - j);
        int crop_height = std::min(grid_y, height - i);

        // Create patch by cropping the region
        auto patch = crop_image(image, j, i, crop_width, crop_height);
        row_patches.push_back(patch);
      }
      patches.push_back(row_patches);
    }

    return patches;
  }

  std::vector<int> get_sliced_grid(int width, int height, bool never_split = false) {
    float log_ratio = std::log((float)width / height);
    float ratio = (float)(width * height) / (scale_resolution_ * scale_resolution_);
    int multiple = std::min((int)std::ceil(ratio), max_slice_nums_);
    if (multiple <= 1 || never_split) { return {}; }
    std::vector<int> candidate_nums;
    for (int i : {multiple - 1, multiple, multiple + 1}) {
      if (i > 1 && i <= max_slice_nums_) { candidate_nums.push_back(i); }
    }

    std::vector<std::vector<int>> candidate_grids;
    for (int split_num : candidate_nums) {
      for (int m = 1; m <= split_num; ++m) {
        if (split_num % m == 0) { candidate_grids.push_back({m, split_num / m}); }
      }
    }

    std::vector<int> best_grid = {1, 1};
    float min_error = INFINITY;
    for (auto& grid : candidate_grids) {
      float error = std::abs(log_ratio - std::log((float)grid[0] / grid[1]));
      if (error < min_error) {
        best_grid = grid;
        min_error = error;
      }
    }

    return best_grid;
  }

  std::pair<int, int> find_best_resize(int width, int height, bool allow_upscale = false) {
    if ((width * height > scale_resolution_ * scale_resolution_) || allow_upscale) {
      float r = (float)width / height;
      int new_height = (int)(scale_resolution_ / std::sqrt(r));
      int new_width = (int)(new_height * r);
      width = new_width;
      height = new_height;
    }
    int best_width = ensure_divide(width, patch_size_);
    int best_height = ensure_divide(height, patch_size_);
    return {best_width, best_height};
  }

  int ensure_divide(int length, int divisor) {
    return std::max(static_cast<int>(std::round(static_cast<float>(length) / divisor)) * divisor, divisor);
  }

  std::pair<int, int> get_refine_size(int width, int height, std::vector<int> grid) {
    int grid_x = grid[0];
    int grid_y = grid[1];
    int refine_width = ensure_divide(width, grid_x);
    int refine_height = ensure_divide(height, grid_y);
    int grid_width = refine_width / grid_x;
    int grid_height = refine_height / grid_y;
    auto best_grid_size = find_best_resize(grid_width, grid_height, true);
    return {best_grid_size.first * grid_x, best_grid_size.second * grid_y};
  }

  // crop image region (for reshape_by_patch)
  Image crop_image(Image& image, int x, int y, int crop_width, int crop_height) {
    // Get source image properties
    int src_width = image.w();
    int src_height = image.h();
    int src_channels = image.c();

    // Ensure crop bounds are valid
    x = std::max(0, std::min(x, src_width - 1));
    y = std::max(0, std::min(y, src_height - 1));
    crop_width = std::min(crop_width, src_width - x);
    crop_height = std::min(crop_height, src_height - y);

    unsigned char* src_data = image.ptr();
    unsigned char* crop_data = new unsigned char[crop_width * crop_height * src_channels];

    // Copy pixel data row by row
    for (int row = 0; row < crop_height; ++row) {
      int src_row_offset = ((y + row) * src_width + x) * src_channels;
      int crop_row_offset = row * crop_width * src_channels;
      std::memcpy(crop_data + crop_row_offset, src_data + src_row_offset, crop_width * src_channels);
    }

    // Create a temporary file to save cropped data and load as Image
    std::string temp_path = "/tmp/crop_" + std::to_string(rand()) + ".png";
    stbi_write_png(temp_path.c_str(), crop_width, crop_height, src_channels, crop_data, crop_width * src_channels);

    Image cropped_image = Image::open(temp_path);

    delete[] crop_data;
    std::remove(temp_path.c_str());

    return cropped_image;
  }

 private:
  int max_slice_nums_;
  int scale_resolution_;
  int patch_size_;
};

class MiniCPMOImageProcessor {
 public:
  explicit MiniCPMOImageProcessor(int patch_size = 14, int image_size = 980, float mean_0 = 0.5, float mean_1 = 0.5,
                                  float mean_2 = 0.5, float std_0 = 0.5, float std_1 = 0.5, float std_2 = 0.5)
      : patch_size_(patch_size),
        image_size_(image_size),
        mean_{mean_0, mean_1, mean_2},
        std_{std_0, std_1, std_2},
        image_slice_processor_(9, 448, patch_size) {}

  std::string get_slice_image_placeholder(std::pair<int, int> image_size, const std::vector<int>& grid, int image_idx = 0,
                                          bool use_image_id = true) {
    std::string image_placeholder = "<image>";
    for (int i = 0; i < image_feature_size; i++) { image_placeholder += "<unk>"; }
    image_placeholder += "</image>";

    std::string final_placeholder;
    if (use_image_id) {
      final_placeholder = "<image_id>" + std::to_string(image_idx) + "</image_id>" + image_placeholder;
    } else {
      final_placeholder = image_placeholder;
    }

    if (!grid.empty()) { final_placeholder += get_grid_placeholder(grid); }

    return final_placeholder;
  };

  std::string get_grid_placeholder(const std::vector<int>& grid) {
    std::string slice_image_placeholder = "<slice>";
    for (int i = 0; i < image_feature_size; ++i) { slice_image_placeholder += "<unk>"; }
    slice_image_placeholder += "</slice>";

    int cols = grid[0];
    int rows = grid[1];
    std::vector<std::string> slice;

    for (int i = 0; i < rows; ++i) {
      std::string line;
      for (int j = 0; j < cols; ++j) { line += slice_image_placeholder; }
      slice.push_back(line);
    }

    std::string result;
    for (size_t i = 0; i < slice.size(); ++i) {
      if (i > 0) { result += "\n"; }
      result += slice[i];
    }
    return result;
  }

  std::pair<std::vector<int64_t>, std::vector<std::pair<int, int>>> calc_bounds(const std::vector<int64_t>& input_ids,
                                                                                preprocessor::BPE& bpe, int max_length = 8192) {
    std::vector<std::pair<int, int>> image_bounds;
    // Get token IDs dynamically from BPE
    int im_start_id = bpe._lookup_vocab(L"<image>");
    int im_end_id = bpe._lookup_vocab(L"</image>");
    int slice_start_id = bpe._lookup_vocab(L"<slice>");
    int slice_end_id = bpe._lookup_vocab(L"</slice>");

    std::vector<int> image_start_positions;
    std::vector<int> image_end_positions;

    int seq_len = input_ids.size();
    for (int i = 0; i < seq_len; ++i) {
      int token_id = input_ids[i];
      if (token_id == im_start_id || token_id == slice_start_id) { image_start_positions.push_back(i); }
      if (token_id == im_end_id || token_id == slice_end_id) { image_end_positions.push_back(i); }
    }
    int valid_image_nums = std::max(image_start_positions.size(), image_end_positions.size());

    for (int i = 0; i < valid_image_nums && i < image_start_positions.size() && i < image_end_positions.size(); ++i) {
      image_bounds.emplace_back(image_start_positions[i], image_end_positions[i]);
    }

    return {input_ids, image_bounds};
  }

  std::tuple<std::vector<Tensor>, std::vector<std::pair<int, int>>, std::vector<std::pair<int, int>>, std::vector<int>> process(
      const std::string& image_path, int max_slice_nums = 9) {
    auto img = Image::open(image_path);
    std::pair<int, int> original_size = {img.w(), img.h()};
    auto [source_image, patches, grid] = image_slice_processor_.slice_image(img);
    std::vector<Image> slice_images;
    slice_images.push_back(source_image);

    if (!patches.empty()) {
      for (const auto& patch_row : patches) {
        for (const auto& patch : patch_row) { slice_images.push_back(patch); }
      }
    }
    std::vector<Tensor> processed_tensors;
    std::vector<std::pair<int, int>> tgt_sizes;

    for (auto& slice_img : slice_images) {
      auto tensor = slice_img.tensor();    // [H, W, C]
      tensor = tensor.permute({2, 0, 1});  // [C, H, W]
      tensor = tensor * (1.f / 255.f);
      normalize_tensor(tensor);  // TODO: use mllm ops
      auto reshaped_tensor = reshape_by_patch(tensor);
      processed_tensors.push_back(reshaped_tensor);

      // Calculate target size (patch dimensions)
      int patch_h = tensor.shape()[1] / patch_size_;
      int patch_w = tensor.shape()[2] / patch_size_;
      tgt_sizes.emplace_back(patch_h, patch_w);
    }
    return std::make_tuple(processed_tensors, std::vector<std::pair<int, int>>{original_size}, tgt_sizes, grid);
  }

 private:
  void normalize_tensor(Tensor& tensor) {
    auto tensor_ptr = tensor.ptr<float>();
    int channels = tensor.shape()[0];
    int height = tensor.shape()[1];
    int width = tensor.shape()[2];

    for (int c = 0; c < channels; ++c) {
      float mean = mean_[c];
      float std = std_[c];

      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          int idx = c * height * width + h * width + w;
          tensor_ptr[idx] = (tensor_ptr[idx] - mean) / std;
        }
      }
    }
  }

  Tensor reshape_by_patch(Tensor& input_tensor) {
    // Input: [C, H, W], Output: [C, patch_size, total_patches * patch_size]
    int channels = input_tensor.shape()[0];
    int height = input_tensor.shape()[1];
    int width = input_tensor.shape()[2];

    int num_patches_h = height / patch_size_;
    int num_patches_w = width / patch_size_;
    int total_patches = num_patches_h * num_patches_w;

    auto output = Tensor::empty({channels, patch_size_, total_patches * patch_size_}, kFloat32).alloc();

    // Get direct pointers for faster access
    const float* input_ptr = input_tensor.ptr<float>();
    float* output_ptr = output.ptr<float>();

    const int input_hw = height * width;
    const int output_hw = patch_size_ * total_patches * patch_size_;

    for (int c = 0; c < channels; ++c) {
      const int c_input_offset = c * input_hw;
      const int c_output_offset = c * output_hw;

      for (int ph = 0; ph < num_patches_h; ++ph) {
        for (int pw = 0; pw < num_patches_w; ++pw) {
          const int patch_idx = ph * num_patches_w + pw;
          const int start_h = ph * patch_size_;
          const int start_w = pw * patch_size_;

          // Copy each row of the patch using memcpy
          for (int kh = 0; kh < patch_size_; ++kh) {
            const int img_h = start_h + kh;
            const int input_row_offset = c_input_offset + img_h * width + start_w;
            const int output_row_offset = c_output_offset + kh * (total_patches * patch_size_) + patch_idx * patch_size_;

            // Copy entire row (patch_size elements) at once
            std::memcpy(output_ptr + output_row_offset, input_ptr + input_row_offset, patch_size_ * sizeof(float));
          }
        }
      }
    }

    return output;
  }

 private:
  int image_size_;
  int patch_size_;
  int image_feature_size = 64;  // 对应query_num
  std::array<float, 3> mean_;
  std::array<float, 3> std_;
  ImageSliceProcessor image_slice_processor_;
};

}  // namespace mllm::models::minicpmo
