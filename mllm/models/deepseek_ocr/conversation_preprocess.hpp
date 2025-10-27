// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>
#include <memory>
#include <limits>
#include <cassert>
#include <variant>
#include <optional>
#include <stdexcept>

#include <cmath>
#include <algorithm>

#include <nlohmann/json.hpp>

#include "mllm/preprocessor/visual/Image.hpp"

namespace mllm::models::deepseek_ocr {

/**
 * Separator styles for different conversation formats
 */
enum class SeparatorStyle {
  DeepSeek,
  DeepSeekV2,
  PLAIN,
  ALIGNMENT,
};

/**
 * A class that manages prompt templates and keeps all conversation history
 */
class Conversation {
 public:
  // Constructor with required and optional parameters
  explicit Conversation(const std::string& name, const std::string& system_template = "{system_message}",
                        const std::string& system_message = "", const std::vector<std::string>& roles = {"USER", "ASSISTANT"},
                        const std::vector<std::vector<std::string>>& messages = {}, int offset = 0,
                        SeparatorStyle sep_style = SeparatorStyle::DeepSeek, const std::string& sep = "\n",
                        const std::optional<std::string>& sep2 = std::nullopt,
                        const std::optional<std::string>& stop_str = std::nullopt,
                        const std::optional<std::vector<int>>& stop_token_ids = std::nullopt)
      : name_(name),
        system_template_(system_template),
        system_message_(system_message),
        roles_(roles),
        messages_(messages),
        offset_(offset),
        sep_style_(sep_style),
        sep_(sep),
        sep2_(sep2),
        stop_str_(stop_str),
        stop_token_ids_(stop_token_ids) {}

  /**
   * Get the prompt for generation
   */
  [[nodiscard]] std::string getPrompt() const {
    std::string system_prompt = formatSystemTemplate();

    if (sep_style_ == SeparatorStyle::DeepSeek) {
      std::vector<std::string> seps = {sep_, sep2_.value_or("")};
      std::string ret;

      if (!system_prompt.empty()) { ret = system_prompt + seps[0]; }

      for (size_t i = 0; i < messages_.size(); ++i) {
        const auto& [role, message] = std::make_pair(messages_[i][0], messages_[i][1]);
        if (!message.empty()) {
          ret += role + ": " + message + seps[i % 2];  // NOLINT
        } else {
          ret += role + ":";
        }
      }
      return ret;
    } else if (sep_style_ == SeparatorStyle::DeepSeekV2) {
      std::vector<std::string> seps = {sep_, sep2_.value_or("")};
      std::string ret;

      if (!system_prompt.empty()) { ret = system_prompt + seps[0]; }

      for (const auto& i : messages_) {
        const auto& [role, message] = std::make_pair(i[0], i[1]);
        if (!message.empty()) {
          if (role == "User") {
            ret += "<｜sft▁begin｜>\n" + message + sep_;
          } else {
            ret += message + sep2_.value_or("");
          }
        }
      }
      return ret;
    } else if (sep_style_ == SeparatorStyle::PLAIN) {
      std::vector<std::string> seps = {sep_, sep2_.value_or("")};
      std::string ret;

      for (size_t i = 0; i < messages_.size(); ++i) {
        const auto& [role, message] = std::make_pair(messages_[i][0], messages_[i][1]);
        if (!message.empty()) { ret += message + seps[i % 2]; }
      }
      return ret;
    } else if (sep_style_ == SeparatorStyle::ALIGNMENT) {
      std::vector<std::string> seps = {sep_, sep2_.value_or("")};
      std::string ret;

      for (size_t i = 0; i < messages_.size(); ++i) {
        const auto& [role, message] = std::make_pair(messages_[i][0], messages_[i][1]);
        if (!message.empty()) {
          if (i % 2 == 0) {
            ret += "<image>\n" + seps[i % 2];
          } else {
            ret += message + seps[i % 2];
          }
        }
      }
      return ret;
    } else {
      throw std::invalid_argument("Invalid separator style");
    }
  }

  /**
   * Set the system message
   */
  void setSystemMessage(const std::string& system_message) { system_message_ = system_message; }

  /**
   * Append a new message
   */
  void appendMessage(const std::string& role, const std::string& message) { messages_.push_back({role, message}); }

  /**
   * Update the last output
   * The last message is typically set to be empty when constructing the prompt,
   * so we need to update it in-place after getting the response from a model.
   */
  void updateLastMessage(const std::string& message) {
    if (!messages_.empty()) { messages_.back()[1] = message; }
  }

  /**
   * Reset messages
   */
  void resetMessages() { messages_.clear(); }

  /**
   * Convert the conversation to gradio chatbot format
   */
  [[nodiscard]] std::vector<std::vector<std::optional<std::string>>> toGradioChatbot() const {
    std::vector<std::vector<std::optional<std::string>>> ret;

    for (size_t i = offset_; i < messages_.size(); ++i) {
      const auto& [role, msg] = std::make_pair(messages_[i][0], messages_[i][1]);
      if (i % 2 == 0) {
        ret.push_back({msg, std::nullopt});
      } else if (!ret.empty()) {
        ret.back()[1] = msg;
      }
    }

    return ret;
  }

  /**
   * Convert the conversation to OpenAI chat completion format
   */
  [[nodiscard]] std::vector<std::map<std::string, std::string>> toOpenAIApiMessages() const {
    std::string system_prompt = formatSystemTemplate();
    std::vector<std::map<std::string, std::string>> ret;

    ret.push_back({{"role", "system"}, {"content", system_prompt}});

    for (size_t i = offset_; i < messages_.size(); ++i) {
      const auto& [_, msg] = std::make_pair(messages_[i][0], messages_[i][1]);
      if (i % 2 == 0) {
        ret.push_back({{"role", "user"}, {"content", msg}});
      } else if (!msg.empty()) {
        ret.push_back({{"role", "assistant"}, {"content", msg}});
      }
    }

    return ret;
  }

  /**
   * Create a copy of the conversation
   */
  [[nodiscard]] std::shared_ptr<Conversation> copy() const {
    return std::make_shared<Conversation>(name_, system_template_, system_message_, roles_, messages_, offset_, sep_style_,
                                          sep_, sep2_, stop_str_, stop_token_ids_);
  }

  /**
   * Convert the conversation to a dictionary
   */
  [[nodiscard]] std::map<std::string,
                         std::variant<std::string, std::vector<std::string>, std::vector<std::vector<std::string>>, int>>
  toDict() const {
    return {{"template_name", name_},
            {"system_message", system_message_},
            {"roles", roles_},
            {"messages", messages_},
            {"offset", offset_}};
  }

  // Getters
  [[nodiscard]] const std::string& getName() const { return name_; }
  [[nodiscard]] const std::vector<std::string>& getRoles() const { return roles_; }

 private:
  [[nodiscard]] std::string formatSystemTemplate() const {
    std::string result = system_template_;
    size_t pos = result.find("{system_message}");
    if (pos != std::string::npos) { result.replace(pos, 16, system_message_); }
    return result;
  }

 private:
  std::string name_;
  std::string system_template_;
  std::string system_message_;
  std::vector<std::string> roles_;
  std::vector<std::vector<std::string>> messages_;
  int offset_;
  SeparatorStyle sep_style_;
  std::string sep_;
  std::optional<std::string> sep2_;
  std::optional<std::string> stop_str_;
  std::optional<std::vector<int>> stop_token_ids_;
};

// A global registry for all conversation templates
static std::map<std::string, std::shared_ptr<Conversation>> conv_templates;

/**
 * Register a new conversation template
 */
void registerConvTemplate(const std::shared_ptr<Conversation>& template_ptr, bool override = false) {
  const std::string& name = template_ptr->getName();
  if (!override) { assert(conv_templates.find(name) == conv_templates.end() && (name + " has been registered.").c_str()); }
  conv_templates[name] = template_ptr;
}

/**
 * Get a conversation template
 */
std::shared_ptr<Conversation> getConvTemplate(const std::string& name) {
  auto it = conv_templates.find(name);
  if (it == conv_templates.end()) { throw std::runtime_error("Template not found: " + name); }
  return it->second->copy();
}

// Initialize templates
void initializeTemplates() {
  // DeepSeek template
  auto deepseek = std::make_shared<Conversation>(
      "deepseek", "{system_message}", "", std::vector<std::string>{"<|User|>", "<|Assistant|>"},
      std::vector<std::vector<std::string>>{}, 0, SeparatorStyle::DeepSeek, "\n\n", "<｜end▁of▁sentence｜>",
      std::optional<std::string>{"<｜end▁of▁sentence｜>"}, std::optional<std::vector<int>>{std::vector<int>{100001}});
  registerConvTemplate(deepseek);

  // DeepSeekV2 template
  auto deepseekv2 = std::make_shared<Conversation>(
      "deepseekv2", "{system_message}", "", std::vector<std::string>{"<｜User｜>", "<｜Assistant｜>"},
      std::vector<std::vector<std::string>>{}, 0, SeparatorStyle::DeepSeekV2, "", "<｜end▁of▁sentence｜>",
      std::optional<std::string>{"<｜end▁of▁sentence｜>"}, std::optional<std::vector<int>>{std::vector<int>{100001}});
  registerConvTemplate(deepseekv2);

  // Plain template
  auto plain = std::make_shared<Conversation>(
      "plain", "", "", std::vector<std::string>{"", ""}, std::vector<std::vector<std::string>>{}, 0, SeparatorStyle::PLAIN, "",
      "", std::optional<std::string>{"</s>"}, std::optional<std::vector<int>>{std::vector<int>{100001}});
  registerConvTemplate(plain);

  // Alignment template
  auto alignment = std::make_shared<Conversation>("alignment", "", "", std::vector<std::string>{"", ""},
                                                  std::vector<std::vector<std::string>>{}, 0, SeparatorStyle::ALIGNMENT, "", "",
                                                  std::optional<std::string>{"</s>"},
                                                  std::optional<std::vector<int>>{std::vector<int>{100001}});
  registerConvTemplate(alignment);
}

inline std::string formatMessages(const nlohmann::json& conversations, const std::string& sft_format = "deepseek",
                                  const std::string& system_prompt = "") {
  auto conv = getConvTemplate(sft_format);

  // Helper trim function to mimic Python's .strip()
  auto trim = [](const std::string& s) -> std::string {
    const char* ws = " \t\n\r\f\v";
    const auto start = s.find_first_not_of(ws);
    if (start == std::string::npos) return "";
    const auto end = s.find_last_not_of(ws);
    return s.substr(start, end - start + 1);
  };

  conv->setSystemMessage(system_prompt);
  for (const auto& message : conversations) {
    std::string role;
    std::string content;

    if (message.contains("role") && message["role"].is_string()) {
      role = message["role"].get<std::string>();
    } else {
      role = "";
    }

    if (message.contains("content") && message["content"].is_string()) {
      content = trim(message["content"].get<std::string>());
    } else {
      content = "";
    }

    conv->appendMessage(role, content);
  }

  return trim(conv->getPrompt());
}

//===----------------------------------------------------------------------===//
// For Image processing
//===----------------------------------------------------------------------===//

/**
 * Loads images from conversation messages
 *
 * @param conversations JSON array of conversation messages
 *        An example is:
 *        [
 *          {
 *            "role": "User",
 *            "content": "<image_placeholder>\nExtract all information from this image and convert them into markdown format.",
 *            "images": ["./examples/table_datasets.png"]
 *          },
 *          {"role": "Assistant", "content": ""}
 *        ]
 *
 * @return Vector of Image objects
 */
std::vector<Image> loadImages(const nlohmann::json& conversations) {
  std::vector<Image> ret;
  // Iterate through each conversation message
  for (const auto& message : conversations) {
    // Skip if message doesn't contain "images" field
    if (!message.contains("images")) { continue; }

    // Process each image path in the "images" array
    for (const auto& image_path : message["images"]) {
      // Load the image using Image::open
      Image img = Image::open(image_path);
      // Add to result vector
      ret.push_back(img);
    }
  }
  return ret;
}

std::pair<int, int> findClosestAspectRatio(double aspect_ratio, const std::vector<std::pair<int, int>>& target_ratios,
                                           int width, int height, int image_size) {
  double best_ratio_diff = std::numeric_limits<double>::infinity();
  std::pair<int, int> best_ratio = {1, 1};
  const double area = static_cast<double>(width) * static_cast<double>(height);

  for (const auto& ratio : target_ratios) {
    const double target_aspect_ratio = static_cast<double>(ratio.first) / static_cast<double>(ratio.second);
    const double ratio_diff = std::abs(aspect_ratio - target_aspect_ratio);

    if (ratio_diff < best_ratio_diff) {
      best_ratio_diff = ratio_diff;
      best_ratio = ratio;
    } else if (ratio_diff == best_ratio_diff) {
      if (area > 0.5 * static_cast<double>(image_size) * static_cast<double>(image_size) * static_cast<double>(ratio.first)
                     * static_cast<double>(ratio.second)) {
        best_ratio = ratio;
      }
    }
  }

  return best_ratio;
}

/**
 * Dynamic preprocess that crops and resizes an image into tiles matching
 * a target aspect ratio grid, similar to DeepSeek-OCR implementation.
 *
 * - Selects closest aspect ratio from predefined candidates.
 * - Generates non-overlapping tiles along the longer dimension.
 * - Pads out-of-bounds areas during crop and resizes tiles to target grid.
 *
 * @param image        Input RGB image
 * @param image_size   Base size used to construct target grid (e.g., 448)
 * @param max_num      Max number of tiles to generate (default 6)
 * @param use_thumbnail Whether to add a square thumbnail (image_size x image_size) as the first tile
 * @return             Vector of processed Image tiles
 */
inline std::pair<std::vector<Image>, std::pair<int, int>> dynamicPreprocess(const Image& image, int min_num = 2,
                                                                            int max_num = 9, int image_size = 640,
                                                                            bool use_thumbnail = false) {
  Image src = image;
  const int w = src.w();
  const int h = src.h();
  if (w <= 0 || h <= 0) { return {{}, {1, 1}}; }

  const double aspect_ratio = static_cast<double>(w) / static_cast<double>(h);

  // Build candidate ratios: all pairs (i, j) with min_num <= i*j <= max_num
  std::set<std::pair<int, int>> ratio_set;
  for (int n = min_num; n <= max_num; ++n) {
    for (int i = 1; i <= n; ++i) {
      for (int j = 1; j <= n; ++j) {
        const int blocks = i * j;
        if (blocks >= min_num && blocks <= max_num) { ratio_set.insert({i, j}); }
      }
    }
  }
  std::vector<std::pair<int, int>> target_ratios(ratio_set.begin(), ratio_set.end());
  std::sort(target_ratios.begin(), target_ratios.end(),
            [](const auto& a, const auto& b) { return (a.first * a.second) < (b.first * b.second); });

  const auto target_aspect_ratio = findClosestAspectRatio(aspect_ratio, target_ratios, w, h, image_size);

  const int target_width = image_size * target_aspect_ratio.first;
  const int target_height = image_size * target_aspect_ratio.second;
  const int blocks = target_aspect_ratio.first * target_aspect_ratio.second;

  Image resized_img = src.resize(target_width, target_height);
  std::vector<Image> processed_images;
  processed_images.reserve(static_cast<size_t>(blocks));

  for (int i = 0; i < blocks; ++i) {
    const int cols = target_width / image_size;  // equals target_aspect_ratio.first
    const int x0 = (i % cols) * image_size;
    const int y0 = (i / cols) * image_size;
    const int x1 = x0 + image_size;
    const int y1 = y0 + image_size;
    Image split_img = resized_img.crop(x0, y0, x1, y1);
    processed_images.push_back(split_img);
  }

  assert(static_cast<int>(processed_images.size()) == blocks);

  if (use_thumbnail && static_cast<int>(processed_images.size()) != 1) {
    processed_images.push_back(src.resize(image_size, image_size));
  }

  return {processed_images, target_aspect_ratio};
}

}  // namespace mllm::models::deepseek_ocr
