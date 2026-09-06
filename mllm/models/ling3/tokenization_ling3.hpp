// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/models/ARGeneration.hpp"
#include "mllm/preprocessor/tokenizers/AutoTokenizer.hpp"
#include "mllm/preprocessor/tokenizers/BPE.hpp"
#include "mllm/preprocessor/tokenizers/Unicode.hpp"
#include "mllm/preprocessor/tokenizers/llama_cpp_unicode/unicode.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cwctype>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mllm::models::ling3 {

inline std::string normalizeLing3NFC(const std::string& text) {
  const auto codepoints = unicode_cpts_from_utf8(text);
  std::vector<uint32_t> normalized;
  normalized.reserve(codepoints.size());
  for (size_t index = 0; index < codepoints.size();) {
    // Unicode Hangul composition is algorithmic and the bundled llama.cpp
    // table does not represent it as an ordinary combining-mark pair.
    constexpr uint32_t kLBase = 0x1100;
    constexpr uint32_t kVBase = 0x1161;
    constexpr uint32_t kTBase = 0x11A7;
    constexpr uint32_t kSBase = 0xAC00;
    constexpr uint32_t kLCount = 19;
    constexpr uint32_t kVCount = 21;
    constexpr uint32_t kTCount = 28;
    const uint32_t current = codepoints[index];
    if (current >= kLBase && current < kLBase + kLCount && index + 1 < codepoints.size() && codepoints[index + 1] >= kVBase
        && codepoints[index + 1] < kVBase + kVCount) {
      const uint32_t l_index = current - kLBase;
      const uint32_t v_index = codepoints[index + 1] - kVBase;
      uint32_t syllable = kSBase + (l_index * kVCount + v_index) * kTCount;
      index += 2;
      if (index < codepoints.size() && codepoints[index] > kTBase && codepoints[index] < kTBase + kTCount) {
        syllable += codepoints[index] - kTBase;
        ++index;
      }
      normalized.push_back(syllable);
      continue;
    }

    // Preserve code points that are already NFC. The bundled normalizer's
    // legacy NFD table stores only the base of a precomposed character, so
    // normalizing an entire string would incorrectly turn "é" into "e".
    // Restrict it to explicit base + combining-mark clusters.
    size_t cluster_end = index + 1;
    while (cluster_end < codepoints.size() && unicode_cpt_flags(codepoints[cluster_end]).is_accent_mark) { ++cluster_end; }
    if (cluster_end > index + 1) {
      const std::vector<uint32_t> cluster(codepoints.begin() + static_cast<std::ptrdiff_t>(index),
                                          codepoints.begin() + static_cast<std::ptrdiff_t>(cluster_end));
      const auto composed = unicode_cpts_normalize_nfc(cluster);
      normalized.insert(normalized.end(), composed.begin(), composed.end());
    } else {
      normalized.push_back(current);
    }
    index = cluster_end;
  }
  std::string result;
  for (const auto codepoint : normalized) { result += unicode_cpt_to_utf8(codepoint); }
  return result;
}

inline bool ling3TokenizerMatchPattern(const std::wstring& text, size_t& position, std::wstring& matched) {
  if (position >= text.size()) { return false; }

  static const std::wstring contractions[] = {L"'s", L"'d", L"'m", L"'t", L"'ll", L"'ve", L"'re"};
  for (const auto& contraction : contractions) {
    bool matches = position + contraction.size() <= text.size();
    for (size_t index = 0; matches && index < contraction.size(); ++index) {
      matches = std::towlower(text[position + index]) == contraction[index];
    }
    if (matches) {
      matched = text.substr(position, contraction.size());
      position += contraction.size();
      return true;
    }
  }

  // [^\r\n\p{L}\p{N}]?+\p{L}+
  {
    const size_t original_position = position;
    if (!preprocessor::isLetter(text[position]) && !preprocessor::isDigit(text[position]) && text[position] != L'\r'
        && text[position] != L'\n') {
      ++position;
    }
    if (position < text.size() && preprocessor::isLetter(text[position])) {
      while (position < text.size() && preprocessor::isLetter(text[position])) { ++position; }
      matched = text.substr(original_position, position - original_position);
      return true;
    }
    position = original_position;
  }

  // \p{N}
  if (preprocessor::isDigit(text[position])) {
    matched = text.substr(position, 1);
    ++position;
    return true;
  }

  // " ?[^\s\p{L}\p{N}]++[\r\n]*"
  {
    const size_t original_position = position;
    if (text[position] == L' ') { ++position; }
    if (position < text.size() && !std::iswspace(text[position]) && !preprocessor::isLetter(text[position])
        && !preprocessor::isDigit(text[position])) {
      while (position < text.size() && !std::iswspace(text[position]) && !preprocessor::isLetter(text[position])
             && !preprocessor::isDigit(text[position])) {
        ++position;
      }
      while (position < text.size() && (text[position] == L'\r' || text[position] == L'\n')) { ++position; }
      matched = text.substr(original_position, position - original_position);
      return true;
    }
    position = original_position;
  }

  // \s*[\r\n]. Greedy \s* backtracks to the last available line break.
  {
    const size_t start = position;
    size_t scan = position;
    size_t last_line_break = std::wstring::npos;
    while (scan < text.size() && std::iswspace(text[scan])) {
      if (text[scan] == L'\r' || text[scan] == L'\n') { last_line_break = scan + 1; }
      ++scan;
    }
    if (last_line_break != std::wstring::npos) {
      position = last_line_break;
      matched = text.substr(start, position - start);
      return true;
    }
  }

  // \s+(?!\S), followed by the final \s+ fallback.
  if (std::iswspace(text[position])) {
    const size_t start = position;
    while (position < text.size() && std::iswspace(text[position])) { ++position; }
    if (position >= text.size()) {
      matched = text.substr(start, position - start);
      return true;
    }
    if (position - start > 1) {
      --position;
      matched = text.substr(start, position - start);
      return true;
    }
    position = start;
    while (position < text.size() && std::iswspace(text[position])) { ++position; }
    matched = text.substr(start, position - start);
    return true;
  }
  return false;
}

inline std::vector<std::wstring> ling3RegexSplit(const std::string& text) {
  const auto wide_text = preprocessor::utf8string2WideString(text);
  std::vector<std::wstring> pieces;
  size_t position = 0;
  while (position < wide_text.size()) {
    std::wstring matched;
    if (ling3TokenizerMatchPattern(wide_text, position, matched)) {
      pieces.push_back(std::move(matched));
    } else {
      pieces.push_back(wide_text.substr(position, 1));
      ++position;
    }
  }
  return pieces;
}

struct Ling3Message {
  std::string prompt;
  std::string system_prompt;
  bool enable_thinking = true;
};

class Ling3Tokenizer final : public preprocessor::AutoTokenizer {
 public:
  explicit Ling3Tokenizer(const std::string& file_path) {
    preprocessor::initLocal();
    preprocessor::makeBytes2UnicodeMap(bytes_to_unicode_);
    for (const auto& [byte, codepoint] : bytes_to_unicode_) { unicode_to_bytes_.insert({codepoint, byte}); }
    if (!bpe_.initFromSentencePieceJson(file_path)) {
      throw std::invalid_argument("Ling-3 tokenizer could not load tokenizer.json");
    }

    std::ifstream tokenizer_file(file_path);
    if (!tokenizer_file.is_open()) { throw std::invalid_argument("Ling-3 tokenizer.json is not readable"); }
    const auto tokenizer_json = nlohmann::json::parse(tokenizer_file);
    if (!tokenizer_json.contains("normalizer") || tokenizer_json["normalizer"].value("type", "") != "NFC") {
      throw std::invalid_argument("Ling-3 tokenizer requires the official NFC normalizer");
    }
    if (!tokenizer_json.contains("added_tokens")) {
      throw std::invalid_argument("Ling-3 tokenizer.json is missing added_tokens");
    }
    for (const auto& added_token : tokenizer_json["added_tokens"]) {
      const auto content = added_token.at("content").get<std::string>();
      const auto wide_content = preprocessor::utf8string2WideString(content);
      special_tokens_trie_.add(wide_content);
      added_tokens_.insert(wide_content);
    }
  }

  std::vector<std::wstring> _tokenize(const std::string& text) override {
    std::vector<std::wstring> tokens;
    for (const auto& piece : ling3RegexSplit(normalizeLing3NFC(text))) {
      const auto utf8_piece = preprocessor::wideString2Utf8String(piece);
      std::wstring mapped;
      for (const unsigned char byte : utf8_piece) { mapped.push_back(bytes_to_unicode_.at(byte)); }
      auto bpe_tokens = bpe_._bpe(mapped);
      tokens.insert(tokens.end(), bpe_tokens.begin(), bpe_tokens.end());
    }
    return tokens;
  }

  std::vector<std::wstring> tokenize(const std::string& text) override {
    const auto split_tokens = special_tokens_trie_.split(preprocessor::utf8string2WideString(text));
    std::vector<std::wstring> tokens;
    for (const auto& token : split_tokens) {
      if (special_tokens_trie_.isSpecialToken(token)) {
        tokens.push_back(token);
      } else {
        auto ordinary_tokens = _tokenize(preprocessor::wideString2Utf8String(token));
        tokens.insert(tokens.end(), ordinary_tokens.begin(), ordinary_tokens.end());
      }
    }
    return tokens;
  }

  std::wstring _detokenize(int64_t token_id) override { return bpe_._lookup_inverse_vocab(token_id); }

  std::string detokenizeBytes(int64_t token_id) {
    const auto token = _detokenize(token_id);
    if (added_tokens_.count(token) != 0) { return preprocessor::wideString2Utf8String(token); }
    std::string bytes;
    for (const auto codepoint : token) {
      const auto iterator = unicode_to_bytes_.find(codepoint);
      if (iterator == unicode_to_bytes_.end()) {
        throw std::runtime_error("Ling-3 tokenizer encountered an unknown byte-unicode symbol");
      }
      bytes.push_back(static_cast<char>(iterator->second));
    }
    return bytes;
  }

  std::wstring detokenize(int64_t token_id) override { return preprocessor::utf8string2WideString(detokenizeBytes(token_id)); }

  Tensor convert2Ids(const std::vector<std::wstring>& tokens) override {
    auto result = Tensor::empty({1, static_cast<int32_t>(tokens.size())}, kInt64, kCPU)
                      .setMemType(kExtraInput)
                      .setName("ling3-tokenizer-i0")
                      .alloc();
    auto* ids = result.ptr<int64_t>();
    for (size_t index = 0; index < tokens.size(); ++index) { ids[index] = bpe_._lookup_vocab(tokens[index]); }
    return result;
  }

  ARGenerationOutputPast convertMessage(const Ling3Message& message) {
    if (message.prompt.empty()) { throw std::invalid_argument("Ling-3 prompt must not be empty"); }
    static constexpr std::string_view kReservedMarkers[] = {
        "<role>SYSTEM</role>",
        "<role>HUMAN</role>",
        "<role>ASSISTANT</role>",
        "<|role_end|>",
    };
    for (const auto marker : kReservedMarkers) {
      if (message.prompt.find(marker) != std::string::npos) {
        throw std::invalid_argument("Ling-3 prompt must not inject reserved chat-template markers");
      }
    }

    const std::string thinking_option = message.enable_thinking ? "on" : "off";
    std::string rendered = "<role>SYSTEM</role>";
    if (message.system_prompt.empty()) {
      rendered += "detailed thinking " + thinking_option + "<|role_end|>";
    } else if (message.system_prompt.find("detailed thinking on") != std::string::npos
               || message.system_prompt.find("detailed thinking off") != std::string::npos) {
      rendered += message.system_prompt + "<|role_end|>";
    } else {
      rendered += message.system_prompt + "\ndetailed thinking " + thinking_option + "<|role_end|>";
    }
    rendered += "<role>HUMAN</role>" + message.prompt + "<|role_end|><role>ASSISTANT</role>";
    rendered += message.enable_thinking ? "\n<think>" : "\n<think></think>";

    const auto tokens = tokenize(rendered);
    auto sequence = Tensor::empty({1, static_cast<int32_t>(tokens.size())}, kInt64, kCPU)
                        .setMemType(kNormal)
                        .setName("ling3-tokenizer-i0")
                        .alloc();
    auto* ids = sequence.ptr<int64_t>();
    for (size_t index = 0; index < tokens.size(); ++index) { ids[index] = bpe_._lookup_vocab(tokens[index]); }
    return {{"sequence", sequence}};
  }

 private:
  preprocessor::BPE bpe_;
  std::unordered_map<std::wint_t, wchar_t> bytes_to_unicode_;
  std::unordered_map<wchar_t, std::wint_t> unicode_to_bytes_;
  std::unordered_set<std::wstring> added_tokens_;
};

}  // namespace mllm::models::ling3
