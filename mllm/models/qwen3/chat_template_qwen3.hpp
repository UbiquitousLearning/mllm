// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <sstream>
#include <string>

#include "mllm/preprocessor/chat_template/ChatTemplate.hpp"

namespace mllm::models::qwen3 {
namespace detail {

inline std::string ltrimChatTemplateText(const std::string& text) {
  const auto start = text.find_first_not_of(" \n\r\t\f\v");
  return start == std::string::npos ? "" : text.substr(start);
}

inline std::string rtrimChatTemplateText(const std::string& text) {
  const auto end = text.find_last_not_of(" \n\r\t\f\v");
  return end == std::string::npos ? "" : text.substr(0, end + 1);
}

inline std::string trimChatTemplateText(const std::string& text) { return rtrimChatTemplateText(ltrimChatTemplateText(text)); }

}  // namespace detail

// Migration renderer for the pre-Jinja Qwen3 service contract. Keep this
// byte-for-byte stable while Qwen3 is configured for the Legacy backend.
inline std::string renderLegacyQwen3ChatTemplate(const preprocessor::ChatTemplateRequest& request) {
  const auto& messages = request.messages;
  const bool enable_thinking = request.extra_context.value("enable_thinking", true);
  std::ostringstream output;

  // The pre-migration Qwen3 services always passed an empty tools list to the
  // legacy formatter. Preserve that default-path behavior; JinjaRequired uses
  // request.tools through the common renderer contract.
  if (!messages.empty() && messages[0].value("role", "") == "system") {
    output << "<|im_start|>system\n" << messages[0].value("content", "") << "<|im_end|>\n";
  }

  std::size_t last_query_index = messages.empty() ? 0 : messages.size() - 1;
  bool found_last_query = false;
  if (!messages.empty()) {
    for (auto i = static_cast<std::ptrdiff_t>(messages.size()) - 1; i >= 0; --i) {
      const auto& message = messages[static_cast<std::size_t>(i)];
      if (message.value("role", "") == "user" && message.contains("content") && message["content"].is_string()) {
        const auto content = message["content"].get<std::string>();
        if (!(content.starts_with("<tool_response>")
              && content.find("</tool_response>") == content.length() - std::string("</tool_response>").length())) {
          last_query_index = static_cast<std::size_t>(i);
          found_last_query = true;
          break;
        }
      }
    }
  }

  for (std::size_t i = 0; i < messages.size(); ++i) {
    const auto& message = messages[i];
    const auto role = message.value("role", "");
    std::string content;
    if (message.contains("content") && message["content"].is_string()) { content = message["content"].get<std::string>(); }

    if (role == "user" || (role == "system" && i > 0)) {
      output << "<|im_start|>" << role << "\n" << content << "<|im_end|>\n";
    } else if (role == "assistant") {
      std::string reasoning_content;
      if (message.contains("reasoning_content") && message["reasoning_content"].is_string()) {
        reasoning_content = message["reasoning_content"].get<std::string>();
      } else {
        const auto think_end_pos = content.find("</think>");
        if (think_end_pos != std::string::npos) {
          const auto think_start_pos = content.rfind("<think>", think_end_pos);
          if (think_start_pos != std::string::npos) {
            reasoning_content = content.substr(think_start_pos + 7, think_end_pos - (think_start_pos + 7));
            content = content.substr(think_end_pos + 8);
          }
        }
      }

      output << "<|im_start|>" << role << "\n";
      if (found_last_query && i > last_query_index) {
        if (i == messages.size() - 1 || !reasoning_content.empty()) {
          output << "<think>\n"
                 << detail::trimChatTemplateText(reasoning_content) << "\n</think>\n\n"
                 << detail::ltrimChatTemplateText(content);
        } else {
          output << content;
        }
      } else {
        output << content;
      }

      if (message.contains("tool_calls")) {
        bool is_first_tool = true;
        for (const auto& tool_call_item : message["tool_calls"]) {
          if ((is_first_tool && !content.empty()) || !is_first_tool) { output << "\n"; }
          is_first_tool = false;

          const auto* tool_call = &tool_call_item;
          if (tool_call_item.contains("function")) { tool_call = &tool_call_item["function"]; }
          output << "<tool_call>\n{\"name\": \"" << tool_call->value("name", "") << R"(", "arguments": )";
          const auto& arguments = (*tool_call)["arguments"];
          if (arguments.is_string()) {
            output << arguments.get<std::string>();
          } else {
            output << arguments.dump();
          }
          output << "}\n</tool_call>";
        }
      }
      output << "<|im_end|>\n";
    } else if (role == "tool") {
      if (i == 0 || messages[i - 1].value("role", "") != "tool") { output << "<|im_start|>user"; }
      output << "\n<tool_response>\n" << content << "\n</tool_response>";
      if (i == messages.size() - 1 || messages[i + 1].value("role", "") != "tool") { output << "<|im_end|>\n"; }
    }
  }

  if (request.add_generation_prompt) {
    output << "<|im_start|>assistant\n";
    if (!enable_thinking) { output << "<think>\n\n</think>\n\n"; }
  }
  return output.str();
}

}  // namespace mllm::models::qwen3
