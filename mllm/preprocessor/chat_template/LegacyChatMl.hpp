// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#ifndef MLLM_PREPROCESSOR_CHAT_TEMPLATE_LEGACY_CHAT_ML_HPP
#define MLLM_PREPROCESSOR_CHAT_TEMPLATE_LEGACY_CHAT_ML_HPP

#include <optional>
#include <string>

#include "mllm/preprocessor/chat_template/ChatTemplate.hpp"

namespace mllm::preprocessor {

// Builds the single-turn request that ChatML runners (Qwen3, Qwen3-MoE, Qwen
// NPU/Ascend, MiniCPM4) send: an optional system message followed by one user
// message. `enable_thinking` is only placed in the template context when it is
// set, matching the official Qwen3-family templates where an undefined value
// means "thinking enabled".
inline ChatTemplateRequest makeSingleTurnChatTemplateRequest(const std::string& prompt, const std::string& system = "",
                                                             std::optional<bool> enable_thinking = std::nullopt) {
  ChatTemplateRequest request;
  if (!system.empty()) { request.messages.push_back({{"role", "system"}, {"content", system}}); }
  request.messages.push_back({{"role", "user"}, {"content", prompt}});
  request.add_generation_prompt = true;
  if (enable_thinking.has_value()) { request.extra_context["enable_thinking"] = *enable_thinking; }
  return request;
}

// Migration renderer for the pre-Jinja ChatML runner prompts. It accepts only
// the shape produced by makeSingleTurnChatTemplateRequest and keeps those bytes
// stable; every other shape fails closed so a missing official template is
// never approximated. The empty-thinking block follows the official Qwen3
// semantics: emitted only when enable_thinking is defined and false.
inline std::string renderLegacyChatMlSingleTurn(const ChatTemplateRequest& request) {
  if (request.tools.has_value()) {
    throw ChatTemplateError("legacy ChatML chat template does not support tools; select the jinja_required backend");
  }
  const auto& messages = request.messages;
  std::size_t index = 0;
  std::string output;
  if (index < messages.size() && messages[index].value("role", "") == "system") {
    const auto& content = messages[index].at("content");
    if (!content.is_string()) { throw ChatTemplateError("legacy ChatML chat template requires string system content"); }
    output += "<|im_start|>system\n" + content.get<std::string>() + "<|im_end|>\n";
    ++index;
  }
  if (index + 1 != messages.size() || messages[index].value("role", "") != "user") {
    throw ChatTemplateError("legacy ChatML chat template supports an optional system message followed by exactly one "
                            "user message; select the jinja_required backend for multi-turn requests");
  }
  const auto& content = messages[index].at("content");
  if (!content.is_string()) { throw ChatTemplateError("legacy ChatML chat template requires string user content"); }
  output += "<|im_start|>user\n" + content.get<std::string>() + "<|im_end|>\n";
  if (request.add_generation_prompt) {
    output += "<|im_start|>assistant\n";
    const auto thinking = request.extra_context.find("enable_thinking");
    if (thinking != request.extra_context.end() && thinking->is_boolean() && !thinking->get<bool>()) {
      output += "<think>\n\n</think>\n\n";
    }
  }
  return output;
}

}  // namespace mllm::preprocessor

#endif  // MLLM_PREPROCESSOR_CHAT_TEMPLATE_LEGACY_CHAT_ML_HPP
