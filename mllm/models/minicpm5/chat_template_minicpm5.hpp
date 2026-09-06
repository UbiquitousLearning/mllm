// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <string>

#include "mllm/preprocessor/chat_template/ChatTemplate.hpp"

namespace mllm::models::minicpm5 {

// Builds the Transformers-style request for one MiniCPM5 runner turn.
inline preprocessor::ChatTemplateRequest makeMiniCPM5ChatTemplateRequest(const std::string& prompt, const std::string& system,
                                                                        bool enable_thinking) {
  preprocessor::ChatTemplateRequest request;
  if (!system.empty()) { request.messages.push_back({{"role", "system"}, {"content", system}}); }
  request.messages.push_back({{"role", "user"}, {"content", prompt}});
  request.add_generation_prompt = true;
  request.extra_context["enable_thinking"] = enable_thinking;
  return request;
}

// Migration renderer for the pre-Jinja MiniCPM5 runner prompt. It accepts the
// optional-system plus single-user shape produced by
// makeMiniCPM5ChatTemplateRequest and keeps those bytes stable; other shapes
// fail closed instead of approximating the official template.
inline std::string renderLegacyMiniCPM5ChatTemplate(const preprocessor::ChatTemplateRequest& request) {
  using preprocessor::ChatTemplateError;
  if (request.tools.has_value()) {
    throw ChatTemplateError("legacy MiniCPM5 chat template does not support tools; select the jinja_required backend");
  }
  const auto& messages = request.messages;
  std::size_t index = 0;
  std::string output = "<s>";
  if (index < messages.size() && messages[index].value("role", "") == "system") {
    const auto& content = messages[index].at("content");
    if (!content.is_string()) { throw ChatTemplateError("legacy MiniCPM5 chat template requires string system content"); }
    output += "<|im_start|>system\n" + content.get<std::string>() + "<|im_end|>\n";
    ++index;
  }
  if (index + 1 != messages.size() || messages[index].value("role", "") != "user") {
    throw ChatTemplateError("legacy MiniCPM5 chat template supports an optional system message followed by exactly one "
                            "user message; select the jinja_required backend for multi-turn requests");
  }
  const auto& content = messages[index].at("content");
  if (!content.is_string()) { throw ChatTemplateError("legacy MiniCPM5 chat template requires string user content"); }
  output += "<|im_start|>user\n" + content.get<std::string>() + "<|im_end|>\n";
  if (request.add_generation_prompt) {
    output += "<|im_start|>assistant\n";
    output += request.extra_context.value("enable_thinking", false) ? "<think>\n" : "<think>\n\n</think>\n\n";
  }
  return output;
}

}  // namespace mllm::models::minicpm5
