// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <cstddef>
#include <string>

#include "mllm/preprocessor/chat_template/ChatTemplate.hpp"

namespace mllm::models::qwen3_5 {

// Placeholders emitted by the official Qwen3.5 chat template for one image or
// one video content block. The runner expands them after rendering: images
// into per-patch image tokens, videos into timestamped frame markers.
inline constexpr char kQwen3_5ImagePlaceholder[] = "<|vision_start|><|image_pad|><|vision_end|>";
inline constexpr char kQwen3_5VideoPlaceholder[] = "<|vision_start|><|video_pad|><|vision_end|>";

// Builds the Transformers-style request for one Qwen3.5 runner turn. Media
// content blocks precede the text block, matching the prompt order the runner
// has always produced.
inline preprocessor::ChatTemplateRequest makeQwen3_5ChatTemplateRequest(const std::string& prompt, std::size_t image_count,
                                                                        bool has_video, bool enable_thinking = false) {
  preprocessor::ChatTemplateRequest request;
  auto content = nlohmann::ordered_json::array();
  for (std::size_t i = 0; i < image_count; ++i) { content.push_back({{"type", "image"}}); }
  if (has_video) { content.push_back({{"type", "video"}}); }
  content.push_back({{"type", "text"}, {"text", prompt}});
  request.messages.push_back({{"role", "user"}, {"content", std::move(content)}});
  request.add_generation_prompt = true;
  request.extra_context["enable_thinking"] = enable_thinking;
  return request;
}

// Migration renderer for the pre-Jinja Qwen3.5 runner prompt. It accepts only
// the single user turn produced by makeQwen3_5ChatTemplateRequest and keeps
// those bytes stable; every other shape fails closed so a missing Jinja
// template is never papered over with an approximation.
inline std::string renderLegacyQwen3_5ChatTemplate(const preprocessor::ChatTemplateRequest& request) {
  using preprocessor::ChatTemplateError;
  if (request.messages.size() != 1 || request.messages[0].value("role", "") != "user") {
    throw ChatTemplateError("legacy Qwen3.5 chat template supports exactly one user message; "
                            "select the jinja_required backend for multi-turn requests");
  }
  if (request.tools.has_value()) {
    throw ChatTemplateError("legacy Qwen3.5 chat template does not support tools; select the jinja_required backend");
  }
  const auto& message = request.messages[0];
  if (!message.contains("content")) { throw ChatTemplateError("legacy Qwen3.5 chat template requires message content"); }
  const auto& content = message.at("content");

  std::string body;
  if (content.is_string()) {
    body = content.get<std::string>();
  } else if (content.is_array()) {
    for (const auto& item : content) {
      const auto type = item.value("type", "");
      if (type == "image") {
        body += kQwen3_5ImagePlaceholder;
      } else if (type == "video") {
        body += kQwen3_5VideoPlaceholder;
      } else if (type == "text") {
        body += item.value("text", "");
      } else {
        throw ChatTemplateError("legacy Qwen3.5 chat template received an unsupported content block type '" + type + "'");
      }
    }
  } else {
    throw ChatTemplateError("legacy Qwen3.5 chat template requires string or content-block message content");
  }

  std::string output = "<|im_start|>user\n" + body + "<|im_end|>\n";
  if (request.add_generation_prompt) {
    output += "<|im_start|>assistant\n";
    output += request.extra_context.value("enable_thinking", false) ? "<think>\n" : "<think>\n\n</think>\n\n";
  }
  return output;
}

}  // namespace mllm::models::qwen3_5
