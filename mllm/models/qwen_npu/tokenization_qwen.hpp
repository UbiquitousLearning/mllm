// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <vector>
#include <unordered_map>
#include <fstream>
#include <iostream>

#include "mllm/preprocessor/tokenizers/BPE.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/preprocessor/tokenizers/Unicode.hpp"
#include "mllm/preprocessor/tokenizers/AutoTokenizer.hpp"

namespace mllm::models::qwen_npu {

// we need to handle this regex pattern similar to qwen3:
// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
inline bool qwenTokenizerMatchPattern(const std::wstring& str, size_t& pos, std::wstring& matched) {
  if (pos >= str.size()) return false;

  // 1. Match contractions: "'s|'t|'re|'ve|'m|'ll|'d"
  static const std::wstring contractions[] = {L"'s", L"'t", L"'re", L"'ve", L"'m", L"'ll", L"'d"};
  for (const auto& contraction : contractions) {
    if (pos + contraction.size() <= str.size() && str.compare(pos, contraction.size(), contraction) == 0) {
      matched = contraction;
      pos += contraction.size();
      return true;
    }
  }

  // 2. Match [^\r\n\p{L}\p{N}]?\p{L}+ (non-letter/digit followed by letters)
  {
    size_t original_pos = pos;
    bool has_prefix = false;
    matched.clear();

    // Check optional non-letter/digit prefix (excluding \r\n)
    if (!preprocessor::isLetter(str[pos]) && !preprocessor::isDigit(str[pos]) && str[pos] != L'\r' && str[pos] != L'\n') {
      matched += str[pos];
      ++pos;
      has_prefix = true;
    }

    // Require at least one letter
    if (pos < str.size() && preprocessor::isLetter(str[pos])) {
      do {
        matched += str[pos];
        ++pos;
      } while (pos < str.size() && preprocessor::isLetter(str[pos]));
      return true;
    } else {
      // Rollback if no letters after prefix
      if (has_prefix) {
        pos = original_pos;
        matched.clear();
      }
    }
  }

  // 3. Match \p{N} (digits)
  if (preprocessor::isDigit(str[pos])) {
    matched = str.substr(pos, 1);
    ++pos;
    return true;
  }

  // 4. Match [^\s\p{L}\p{N}]+[\r\n]* (punctuation/symbols)
  {
    size_t original_pos = pos;
    matched.clear();

    // Require at least one non-letter/digit/whitespace
    if (pos < str.size() && !std::iswspace(str[pos]) && !preprocessor::isLetter(str[pos]) && !preprocessor::isDigit(str[pos])) {
      do {
        matched += str[pos];
        ++pos;
      } while (pos < str.size() && !std::iswspace(str[pos]) && !preprocessor::isLetter(str[pos])
               && !preprocessor::isDigit(str[pos]));

      // Capture trailing newlines
      while (pos < str.size() && (str[pos] == L'\r' || str[pos] == L'\n')) {
        matched += str[pos];
        ++pos;
      }
      return true;
    } else {
      // Rollback if no symbols found
      pos = original_pos;
    }
  }

  // 5. Match \s*[\r\n]+ (newlines with leading whitespace)
  {
    size_t start = pos;
    while (pos < str.size() && std::iswspace(str[pos])) ++pos;
    if (pos < str.size() && (str[pos] == L'\r' || str[pos] == L'\n')) {
      while (pos < str.size() && (str[pos] == L'\r' || str[pos] == L'\n')) ++pos;
      matched = str.substr(start, pos - start);
      return true;
    } else {
      pos = start;
    }
  }

  // 6. Match \s+(?!\S) (whitespace not followed by non-space)
  if (std::iswspace(str[pos])) {
    size_t start = pos;
    while (pos < str.size() && std::iswspace(str[pos])) ++pos;
    // Check if at end or followed by whitespace
    if (pos >= str.size() || std::iswspace(str[pos])) {
      matched = str.substr(start, pos - start);
      return true;
    } else {
      pos = start;
    }
  }

  // 7. Match remaining whitespace
  if (std::iswspace(str[pos])) {
    size_t start = pos;
    while (pos < str.size() && std::iswspace(str[pos])) ++pos;
    matched = str.substr(start, pos - start);
    return true;
  }

  return false;
}

inline bool qwenRegex(const std::string& str, std::vector<std::wstring>& splitted) {
  auto w_string = preprocessor::utf8string2WideString(str);
  size_t pos = 0;
  while (pos < w_string.size()) {
    std::wstring matched;
    if (qwenTokenizerMatchPattern(w_string, pos, matched)) {
      splitted.push_back(matched);
    } else {
      ++pos;
    }
  }
  return true;
}

struct QwenMessage {
  std::string prompt;
  static inline std::string message_template =
      "<|im_start|>system\nYou are a helpful "
      "assistant.<|im_end|>\n<|im_start|>user\n{{{prompt}}}<|im_end|>\n<|im_start|>assistant\n";
};

class QwenTokenizer final : public mllm::preprocessor::AutoTokenizer {
 public:
  explicit QwenTokenizer(const std::string& vocab_file, const std::string& merge_file, bool split_special_tokens = false)
      : split_special_tokens_(split_special_tokens) {
    preprocessor::initLocal();
    preprocessor::makeBytes2UnicodeMap(bytes_2_unicode_dict_);
    for (auto& kv : bytes_2_unicode_dict_) { bytes_2_unicode_dict_inverse_.insert({kv.second, kv.first}); }

    bpe_.initFromSentencePieceJson(vocab_file);
    initBpeRanks(merge_file);
    initSpecialTokens();
  }

  void initBpeRanks(const std::string& merge_file) {
    auto merge_file_stream = std::ifstream(merge_file);
    if (!merge_file_stream.good()) {
      std::cerr << "merge file is broken: " << merge_file << '\n';
      exit(1);
    }
    std::string line;
    unsigned rank = 0;
    while (std::getline(merge_file_stream, line)) {
      if (line.empty() || line[0] == '#') { continue; }
      bpe_ranks_[line] = rank;
      rank++;
    }
  }

  void initSpecialTokens() {
    special_tokens_trie_.add(L"<|endoftext|>");
    special_tokens_trie_.add(L"<|im_start|>");
    special_tokens_trie_.add(L"<|im_end|>");
    special_tokens_trie_.add(L"<unk>");
  }

  std::vector<std::wstring> _tokenize(const std::string& str) override {
    std::vector<std::wstring> ret;
    std::vector<std::wstring> splitted;
    qwenRegex(str, splitted);

    for (const auto& s : splitted) {
      auto utf_8_str = preprocessor::wideString2Utf8String(s);
      std::wstring mapped_str;
      for (unsigned char c : utf_8_str) { mapped_str.push_back(bytes_2_unicode_dict_[c]); }

      auto bpe_ts = bpe_._bpe(mapped_str);
      for (const auto& bpe_t : bpe_ts) { ret.push_back(bpe_t); }
    }

    return ret;
  }

  std::vector<std::wstring> tokenize(const std::string& str) override {
    auto tokens = special_tokens_trie_.split(preprocessor::utf8string2WideString(str));
    std::vector<std::wstring> all_tokens;
    for (const auto& token : tokens) {
      if (special_tokens_trie_.isSpecialToken(token)) {
        all_tokens.emplace_back(token);
        continue;
      }
      auto tmp_tokens = _tokenize(preprocessor::wideString2Utf8String(token));
      all_tokens.insert(all_tokens.end(), tmp_tokens.begin(), tmp_tokens.end());
    }
    return all_tokens;
  }

  std::wstring _detokenize(int64_t pos_idx) override { return bpe_._lookup_inverse_vocab(pos_idx); }

  std::wstring detokenize(int64_t pos_idx) override {
    auto str = _detokenize(pos_idx);
    std::string utf_8_str;
    for (wchar_t c : str) { utf_8_str.push_back((unsigned char)(bytes_2_unicode_dict_inverse_[c])); }
    return preprocessor::utf8string2WideString(utf_8_str);
  }

  Tensor convert2Ids(const std::vector<std::wstring>& strs) override {
    std::vector<int64_t> ids;
    ids.reserve(strs.size());
    for (const auto& str : strs) { ids.emplace_back(bpe_._lookup_vocab(str)); }

    Tensor ret = Tensor::empty({/*batch*/ 1, /*seq*/ (int32_t)ids.size()}, kInt64, kCPU)
                     .setMemType(kExtraInput)
                     .setName("qwen-tokenizer-i0")
                     .alloc();

    auto ptr = ret.ptr<int64_t>();
    for (size_t i = 0; i < ids.size(); ++i) { ptr[i] = ids[i]; }

    return ret;
  }

  std::pair<int, Tensor> tokenizeWithPadding(const std::string& text, int seqLength, int vocab_size) {
    auto tokens = tokenize(text);
    std::vector<int64_t> ids;
    ids.reserve(tokens.size());

    for (const auto& token : tokens) { ids.emplace_back(bpe_._lookup_vocab(token)); }

    auto realLength = ids.size();
    ids.resize(seqLength, vocab_size);

    Tensor ret = Tensor::empty({/*batch*/ 1, /*seq*/ (int32_t)ids.size()}, kInt64, kCPU)
                     .setMemType(kNormal)
                     .setName("qwen-tokenizer-i0")
                     .alloc();

    auto ptr = ret.ptr<int64_t>();
    for (size_t i = 0; i < ids.size(); ++i) { ptr[i] = ids[i]; }

    return std::make_pair(realLength, ret);
  }

  std::pair<int, Tensor> tokenizePaddingByChunk(const std::string& text, int chunk_size, int vocab_size) {
    auto tokens = tokenize(text);
    std::vector<int64_t> ids;
    ids.reserve(tokens.size());

    for (const auto& token : tokens) { ids.emplace_back(bpe_._lookup_vocab(token)); }

    auto realLength = ids.size();
    int paddingLength = (chunk_size - realLength % chunk_size) % chunk_size;
    ids.resize(realLength + paddingLength, vocab_size);

    Tensor ret = Tensor::empty({/*batch*/ 1, /*seq*/ (int32_t)ids.size()}, kInt64, kCPU)
                     .setMemType(kNormal)
                     .setName("qwen-tokenizer-i0")
                     .alloc();

    auto ptr = ret.ptr<int64_t>();
    for (size_t i = 0; i < ids.size(); ++i) { ptr[i] = ids[i]; }

    return std::make_pair(realLength, ret);
  }

  ARGenerationOutputPast convertMessage(const QwenMessage& message) {
    // process prompt
    auto applied_string = QwenMessage::message_template;
    size_t pos = applied_string.find("{{{prompt}}}");
    applied_string.replace(pos, 12, message.prompt);

    // process sequence
    auto sequence_str = tokenize(applied_string);
    std::vector<int64_t> ids;
    ids.reserve(sequence_str.size());
    for (const auto& str : sequence_str) { ids.emplace_back(bpe_._lookup_vocab(str)); }

    // Get sequence Tensor
    Tensor sequence = Tensor::empty({/*batch*/ 1, /*seq*/ (int32_t)ids.size()}, kInt64, kCPU)
                          .setMemType(kNormal)
                          .setName("qwen-tokenizer-i0")
                          .alloc();

    auto ptr = sequence.ptr<int64_t>();
    for (size_t i = 0; i < ids.size(); ++i) { ptr[i] = ids[i]; }

    return {
        {"sequence", sequence},
    };
  }

  std::pair<bool, std::string> postprocess(const std::string& text) {
    if (text == "<|im_end|>" || text == "<|endoftext|>") { return {false, ""}; }
    if (text == "<|im_start|>" || text == "<unk>") { return {true, ""}; }
    return {true, text};
  }

 private:
  // For text
  preprocessor::BPE bpe_;
  std::unordered_map<std::wint_t, wchar_t> bytes_2_unicode_dict_;
  std::unordered_map<wchar_t, std::wint_t> bytes_2_unicode_dict_inverse_;
  std::unordered_map<std::string, unsigned int> bpe_ranks_;

  bool split_special_tokens_ = false;

 public:
  static constexpr int64_t eos_id_ = 151645;
  static constexpr int64_t bos_id_ = 151643;
};

}  // namespace mllm::models::qwen_npu
