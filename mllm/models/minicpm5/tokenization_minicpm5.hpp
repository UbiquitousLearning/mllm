// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cwctype>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "mllm/models/ARGeneration.hpp"
#include "mllm/models/minicpm5/chat_template_minicpm5.hpp"
#include "mllm/models/minicpm5/configuration_minicpm5.hpp"
#include "mllm/preprocessor/StreamingUtf8Decoder.hpp"
#include "mllm/preprocessor/tokenizers/AutoTokenizer.hpp"
#include "mllm/preprocessor/tokenizers/BPE.hpp"
#include "mllm/preprocessor/tokenizers/Unicode.hpp"

namespace mllm::models::minicpm5 {

using MiniCPM5StreamingUtf8Decoder = preprocessor::StreamingUtf8Decoder;

// Equivalent to MiniCPM5's two Split pre-tokenizers: numeric runs are first
// isolated into chunks of at most three digits, then the Qwen-style pattern is
// applied before byte-level BPE.
inline bool miniCPM5TokenizerMatchPattern(const std::wstring& input, size_t& position, std::wstring& matched) {
  if (position >= input.size()) return false;

  static const std::wstring contractions[] = {L"'s", L"'t", L"'re", L"'ve", L"'m", L"'ll", L"'d"};
  for (const auto& contraction : contractions) {
    bool matches = position + contraction.size() <= input.size();
    for (size_t index = 0; matches && index < contraction.size(); ++index) {
      matches = std::towlower(input[position + index]) == contraction[index];
    }
    if (matches) {
      matched = input.substr(position, contraction.size());
      position += contraction.size();
      return true;
    }
  }

  {
    const size_t original_position = position;
    matched.clear();
    if (!preprocessor::isLetter(input[position]) && !preprocessor::isDigit(input[position]) && input[position] != L'\r'
        && input[position] != L'\n') {
      matched += input[position++];
    }
    if (position < input.size() && preprocessor::isLetter(input[position])) {
      do { matched += input[position++]; } while (position < input.size() && preprocessor::isLetter(input[position]));
      return true;
    }
    position = original_position;
    matched.clear();
  }

  if (preprocessor::isDigit(input[position])) {
    const size_t start = position;
    while (position < input.size() && preprocessor::isDigit(input[position]) && position - start < 3) { ++position; }
    matched = input.substr(start, position - start);
    return true;
  }

  {
    const size_t original_position = position;
    const size_t start = position;
    if (input[position] == L' ') ++position;
    if (position < input.size() && !std::iswspace(input[position]) && !preprocessor::isLetter(input[position])
        && !preprocessor::isDigit(input[position])) {
      do {
        ++position;
      } while (position < input.size() && !std::iswspace(input[position]) && !preprocessor::isLetter(input[position])
               && !preprocessor::isDigit(input[position]));
      while (position < input.size() && (input[position] == L'\r' || input[position] == L'\n')) ++position;
      matched = input.substr(start, position - start);
      return true;
    }
    position = original_position;
  }

  {
    const size_t start = position;
    size_t scan = position;
    size_t last_line_break = std::wstring::npos;
    while (scan < input.size() && std::iswspace(input[scan])) {
      if (input[scan] == L'\r' || input[scan] == L'\n') last_line_break = scan + 1;
      ++scan;
    }
    if (last_line_break != std::wstring::npos) {
      position = last_line_break;
      matched = input.substr(start, position - start);
      return true;
    }
  }

  if (std::iswspace(input[position])) {
    const size_t start = position;
    while (position < input.size() && std::iswspace(input[position])) ++position;
    if (position >= input.size()) {
      matched = input.substr(start, position - start);
      return true;
    }
    if (position - start > 1) {
      --position;
      matched = input.substr(start, position - start);
      return true;
    }
    position = start;
  }

  if (std::iswspace(input[position])) {
    const size_t start = position;
    while (position < input.size() && std::iswspace(input[position])) ++position;
    matched = input.substr(start, position - start);
    return true;
  }
  return false;
}

inline bool miniCPM5Regex(const std::string& input, std::vector<std::wstring>& pieces) {
  const auto wide_input = preprocessor::utf8string2WideString(input);
  size_t position = 0;
  while (position < wide_input.size()) {
    std::wstring matched;
    if (miniCPM5TokenizerMatchPattern(wide_input, position, matched)) {
      pieces.push_back(std::move(matched));
    } else {
      pieces.push_back(wide_input.substr(position++, 1));
    }
  }
  return true;
}

struct MiniCPM5Message {
  std::string prompt;
  std::string system;
  bool enable_thinking = false;
};

class MiniCPM5Tokenizer final : public preprocessor::AutoTokenizer {
 public:
  // The chat-template backend is selected explicitly. The default keeps the
  // migration (legacy) renderer; a JinjaRequired configuration loads the
  // official template from its model directory and fails here when it cannot.
  explicit MiniCPM5Tokenizer(const std::string& file_path, preprocessor::ChatPreprocessorConfig chat_template = {})
      : chat_preprocessor_(std::move(chat_template), renderLegacyMiniCPM5ChatTemplate) {
    preprocessor::initLocal();
    preprocessor::makeBytes2UnicodeMap(bytes_to_unicode_);
    for (const auto& [byte, codepoint] : bytes_to_unicode_) { unicode_to_bytes_.insert({codepoint, byte}); }
    if (!bpe_.initFromSentencePieceJson(file_path)) {
      throw std::invalid_argument("MiniCPM5 tokenizer JSON must contain a BPE vocabulary and merge table");
    }
    chat_preprocessor_.setControlTokens(bpe_.controlTokens());

    std::ifstream tokenizer_stream(file_path);
    if (!tokenizer_stream) { throw std::invalid_argument("Unable to read MiniCPM5 tokenizer JSON: " + file_path); }
    const auto tokenizer_json = nlohmann::json::parse(tokenizer_stream);
    if (!tokenizer_json.contains("model") || tokenizer_json["model"].value("type", "") != "BPE") {
      throw std::invalid_argument("MiniCPM5 tokenizer JSON must use the official BPE model");
    }
    if (!tokenizer_json.contains("added_tokens") || !tokenizer_json["added_tokens"].is_array()) {
      throw std::invalid_argument("MiniCPM5 tokenizer JSON is missing added_tokens");
    }
    const std::unordered_map<std::string, int64_t> required_tokens = {
        {"<s>", 0}, {"</s>", 1}, {"<think>", 8}, {"</think>", 9}, {"<|im_start|>", 130072}, {"<|im_end|>", 130073},
    };
    std::unordered_map<std::string, int64_t> observed_required_tokens;
    for (const auto& token : tokenizer_json["added_tokens"]) {
      if (token.contains("content") && token["content"].is_string()) {
        const auto content = token["content"].get<std::string>();
        added_tokens_.push_back(preprocessor::utf8string2WideString(content));
        if (required_tokens.contains(content) && token.contains("id") && token["id"].is_number_integer()) {
          observed_required_tokens[content] = token["id"].get<int64_t>();
        }
      }
    }
    for (const auto& [token, expected_id] : required_tokens) {
      if (!observed_required_tokens.contains(token) || observed_required_tokens.at(token) != expected_id) {
        throw std::invalid_argument("MiniCPM5 tokenizer JSON has an incompatible required token: " + token);
      }
    }
    std::sort(added_tokens_.begin(), added_tokens_.end(), [](const auto& lhs, const auto& rhs) {
      if (lhs.size() != rhs.size()) return lhs.size() > rhs.size();
      return lhs < rhs;
    });
  }

  // Product constructor: the chat-template backend comes from the model
  // configuration; the template is discovered next to tokenizer.json unless
  // another model directory is given.
  MiniCPM5Tokenizer(const std::string& file_path, const MiniCPM5Config& config, std::filesystem::path model_directory = {})
      : MiniCPM5Tokenizer(file_path,
                          preprocessor::ChatPreprocessorConfig{
                              .backend = config.chat_template_backend,
                              .template_options = {.model_directory = model_directory.empty()
                                                                          ? std::filesystem::path(file_path).parent_path()
                                                                          : std::move(model_directory)}}) {}

  preprocessor::ChatTemplateBackend chatTemplateBackend() const noexcept { return chat_preprocessor_.backend(); }

  // The checkpoint's control tokens; see BPE::controlTokens().
  const std::vector<std::string>& controlTokens() const { return bpe_.controlTokens(); }

  // Migration formatter: the exact pre-Jinja runner prompt. It stays available
  // as the legacy backend's renderer and as a fixed reference for tests.
  static std::string applyChatTemplate(const MiniCPM5Message& message) {
    if (message.prompt.empty()) { throw std::invalid_argument("MiniCPM5 prompt must not be empty"); }
    return renderLegacyMiniCPM5ChatTemplate(
        makeMiniCPM5ChatTemplateRequest(message.prompt, message.system, message.enable_thinking));
  }

  // Renders the prompt through the configured chat-template backend.
  std::string renderChatTemplate(const MiniCPM5Message& message) const {
    if (message.prompt.empty()) { throw std::invalid_argument("MiniCPM5 prompt must not be empty"); }
    return chat_preprocessor_.render(makeMiniCPM5ChatTemplateRequest(message.prompt, message.system, message.enable_thinking));
  }

  std::vector<std::wstring> _tokenize(const std::string& input) override {
    std::vector<std::wstring> result;
    std::vector<std::wstring> pieces;
    miniCPM5Regex(input, pieces);
    for (const auto& piece : pieces) {
      const auto bytes = preprocessor::wideString2Utf8String(piece);
      std::wstring mapped;
      for (const unsigned char byte : bytes) mapped.push_back(bytes_to_unicode_[byte]);
      const auto bpe_tokens = bpe_._bpe(mapped);
      result.insert(result.end(), bpe_tokens.begin(), bpe_tokens.end());
    }
    return result;
  }

  std::vector<std::wstring> tokenize(const std::string& input) override {
    const auto wide_input = preprocessor::utf8string2WideString(input);
    std::vector<std::wstring> result;
    size_t normal_start = 0;
    size_t position = 0;
    while (position < wide_input.size()) {
      const std::wstring* matched_token = nullptr;
      for (const auto& token : added_tokens_) {
        if (token.size() <= wide_input.size() - position && wide_input.compare(position, token.size(), token) == 0) {
          matched_token = &token;
          break;
        }
      }
      if (matched_token == nullptr) {
        ++position;
        continue;
      }
      if (normal_start < position) {
        auto bpe_tokens =
            _tokenize(preprocessor::wideString2Utf8String(wide_input.substr(normal_start, position - normal_start)));
        result.insert(result.end(), bpe_tokens.begin(), bpe_tokens.end());
      }
      result.push_back(*matched_token);
      position += matched_token->size();
      normal_start = position;
    }
    if (normal_start < wide_input.size()) {
      auto bpe_tokens = _tokenize(preprocessor::wideString2Utf8String(wide_input.substr(normal_start)));
      result.insert(result.end(), bpe_tokens.begin(), bpe_tokens.end());
    }
    return result;
  }

  std::wstring _detokenize(int64_t token_id) override { return bpe_._lookup_inverse_vocab(token_id); }

  std::string detokenizeBytes(int64_t token_id) {
    const auto token = _detokenize(token_id);
    std::string bytes;
    for (const wchar_t codepoint : token) {
      const auto iterator = unicode_to_bytes_.find(codepoint);
      if (iterator == unicode_to_bytes_.end()) {
        throw std::runtime_error("MiniCPM5 tokenizer encountered an unknown byte-unicode symbol");
      }
      bytes.push_back(static_cast<char>(iterator->second));
    }
    return bytes;
  }

  std::wstring detokenize(int64_t token_id) override { return preprocessor::utf8string2WideString(detokenizeBytes(token_id)); }

  Tensor convert2Ids(const std::vector<std::wstring>& tokens) override {
    auto sequence = Tensor::empty({1, static_cast<int32_t>(tokens.size())}, kInt64, kCPU)
                        .setMemType(kExtraInput)
                        .setName("minicpm5-tokenizer-i0")
                        .alloc();
    for (size_t index = 0; index < tokens.size(); ++index) {
      sequence.ptr<int64_t>()[index] = bpe_._lookup_vocab(tokens[index]);
    }
    return sequence;
  }

  ARGenerationOutputPast convertMessage(const MiniCPM5Message& message) {
    const auto tokens = tokenize(renderChatTemplate(message));
    auto sequence = Tensor::empty({1, static_cast<int32_t>(tokens.size())}, kInt64, kCPU)
                        .setMemType(kNormal)
                        .setName("minicpm5-tokenizer-i0")
                        .alloc();
    for (size_t index = 0; index < tokens.size(); ++index) {
      sequence.ptr<int64_t>()[index] = bpe_._lookup_vocab(tokens[index]);
    }
    return {{"sequence", sequence}};
  }

 private:
  preprocessor::BPE bpe_;
  std::unordered_map<std::wint_t, wchar_t> bytes_to_unicode_;
  std::unordered_map<wchar_t, std::wint_t> unicode_to_bytes_;
  preprocessor::ChatPreprocessor chat_preprocessor_;
  std::vector<std::wstring> added_tokens_;
};

}  // namespace mllm::models::minicpm5
