// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <filesystem>

#include <nlohmann/json.hpp>

#include "mllm/mllm.hpp"
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

    // Load image
    auto images = loadImages(conversations);

    // Image transform infra
    auto image_transform = BasicImageTransform(std::nullopt, std::nullopt, /*mean=*/std::vector<float>{0.5, 0.5, 0.5},
                                               /*std=*/std::vector<float>{0.5, 0.5, 0.5});

    // Split text with IMAGE_TOKEN
    // TODO
  }
};

}  // namespace mllm::models::deepseek_ocr
