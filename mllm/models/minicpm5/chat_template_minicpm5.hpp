// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <string>

#include "mllm/preprocessor/chat_template/LegacyChatMl.hpp"

namespace mllm::models::minicpm5 {

// Builds the Transformers-style request for one MiniCPM5 runner turn.
inline preprocessor::ChatTemplateRequest makeMiniCPM5ChatTemplateRequest(const std::string& prompt, const std::string& system,
                                                                        bool enable_thinking) {
  return preprocessor::makeSingleTurnChatTemplateRequest(prompt, system, enable_thinking);
}

// Migration renderer for the pre-Jinja MiniCPM5 runner prompt. It accepts the
// optional-system plus single-user shape produced by
// makeMiniCPM5ChatTemplateRequest and keeps those bytes stable; other shapes
// fail closed instead of approximating the official template.
inline std::string renderLegacyMiniCPM5ChatTemplate(const preprocessor::ChatTemplateRequest& request) {
  std::string output = "<s>";
  preprocessor::appendLegacyChatMlMessages(output, request, "MiniCPM5");
  if (request.add_generation_prompt) {
    output += "<|im_start|>assistant\n";
    output += request.extra_context.value("enable_thinking", false) ? "<think>\n" : "<think>\n\n</think>\n\n";
  }
  return output;
}

}  // namespace mllm::models::minicpm5
