// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <filesystem>
#include <nlohmann/json.hpp>

#include "mllm/mllm.hpp"
#include "mllm/utils/StringHelper.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/preprocessor/visual/ImageTransform.hpp"
#include "mllm/models/deepseek_ocr/conversation_preprocess.hpp"
#include "mllm/models/deepseek_ocr/tokenization_deepseek_ocr.hpp"
#include "mllm/models/deepseek_ocr/configuration_deepseek_ocr.hpp"

namespace mllm::models::deepseek_ocr {

class DeepseekOCRForCausalLM final : public nn::Module, public ARGeneration {
 public:
  DeepseekOCRForCausalLM() = default;

  explicit DeepseekOCRForCausalLM(const DpskOcrConfig& config) {}

  ARGenerationOutputPast forward(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override { return {}; }

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

    // Image transform infra
    auto image_transform = BasicImageTransform(std::nullopt, std::nullopt, /*mean=*/std::vector<float>{0.5, 0.5, 0.5},
                                               /*std=*/std::vector<float>{0.5, 0.5, 0.5});

    // Split text with IMAGE_TOKEN
    // Like what python does: text_splits = prompt.split(image_token)
    auto text_splits = mllm::splitString(processed_prompt, IMAGE_TOKEN);

    // Processed states
    std::vector<int64_t> tokenized_str;
    std::vector<float> images_seq_mask;
    std::vector<Tensor> images_list;
    std::vector<Tensor> images_crop_list;
    std::vector<std::tuple<int, int>> images_spatial_crop;

    // text_splits's length should be greater than images' length.
    // text_splits.size() - images.size() = 1
    for (int idx = 0; idx < std::min(images.size(), text_splits.size()); ++idx) {
      auto tokenized_sep = tokenizer.tokenize(text_splits[idx]);
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
          MLLM_RT_ASSERT(false);
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
    auto tokenized_sep = tokenizer.tokenize(text_splits.back());
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
    auto images_seq_mask_tensor = Tensor::fromVector(images_seq_mask, {1, (int32_t)images_seq_mask.size()}, kFloat32);
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

    MLLM_INFO("BRAVO! U R HERE");
    print(input_ids.shape());
    print(input_ids);
    print(images_seq_mask_tensor);
    print(images_ori_tensor);
    print(images_spatial_crop_tensor);
    print(images_crop_tensor);

    // Run model. Use generate
    // TODO

    // Post process data
    // TODO
  }
};

}  // namespace mllm::models::deepseek_ocr
