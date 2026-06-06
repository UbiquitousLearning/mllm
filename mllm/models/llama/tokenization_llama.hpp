// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <vector>
#include <string>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <unordered_map>

#include "mllm/preprocessor/tokenizers/BPE.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/preprocessor/tokenizers/Unicode.hpp"
#include "mllm/preprocessor/tokenizers/AutoTokenizer.hpp"

namespace mllm::models::llama {

inline bool llama3TokenizerMatchPattern(const std::wstring& str, size_t& pos, std::wstring& matched) {
  if (pos >= str.size()) return false;

  static const std::wstring contractions[] = {L"'s", L"'t", L"'re", L"'ve", L"'m", L"'ll", L"'d",
                                              L"'S", L"'T", L"'RE", L"'VE", L"'M", L"'LL", L"'D"};
  for (const auto& contraction : contractions) {
    if (pos + contraction.size() <= str.size() && str.compare(pos, contraction.size(), contraction) == 0) {
      matched = contraction;
      pos += contraction.size();
      return true;
    }
  }

  {
    size_t original_pos = pos;
    matched.clear();
    if (!preprocessor::isLetter(str[pos]) && !preprocessor::isDigit(str[pos]) && str[pos] != L'\r' && str[pos] != L'\n') {
      matched += str[pos];
      ++pos;
    }
    if (pos < str.size() && preprocessor::isLetter(str[pos])) {
      do {
        matched += str[pos];
        ++pos;
      } while (pos < str.size() && preprocessor::isLetter(str[pos]));
      return true;
    }
    pos = original_pos;
  }

  if (preprocessor::isDigit(str[pos])) {
    matched = str.substr(pos, 1);
    ++pos;
    return true;
  }

  {
    size_t start = pos;
    if (str[pos] == L' ') { ++pos; }
    if (pos < str.size() && !std::iswspace(str[pos]) && !preprocessor::isLetter(str[pos]) && !preprocessor::isDigit(str[pos])) {
      do {
        ++pos;
      } while (pos < str.size() && !std::iswspace(str[pos]) && !preprocessor::isLetter(str[pos])
               && !preprocessor::isDigit(str[pos]));
      matched = str.substr(start, pos - start);
      while (pos < str.size() && (str[pos] == L'\r' || str[pos] == L'\n')) {
        matched += str[pos];
        ++pos;
      }
      return true;
    }
    pos = start;
  }

  if (std::iswspace(str[pos])) {
    size_t start = pos;
    while (pos < str.size() && std::iswspace(str[pos])) ++pos;
    matched = str.substr(start, pos - start);
    return true;
  }

  return false;
}

inline void llama3Regex(const std::string& str, std::vector<std::wstring>& splitted) {
  auto w_string = preprocessor::utf8string2WideString(str);
  size_t pos = 0;
  while (pos < w_string.size()) {
    std::wstring matched;
    if (llama3TokenizerMatchPattern(w_string, pos, matched)) {
      splitted.push_back(matched);
    } else {
      ++pos;
    }
  }
}

struct LlamaMessage {
  std::string role;
  std::string content;
};

class LlamaTokenizer final : public mllm::preprocessor::AutoTokenizer {
 public:
  explicit LlamaTokenizer(const std::string& file_path, bool add_bos = true) : add_bos_(add_bos) {
    preprocessor::initLocal();
    preprocessor::makeBytes2UnicodeMap(bytes_2_unicode_dict_);
    for (auto& kv : bytes_2_unicode_dict_) { bytes_2_unicode_dict_inverse_.insert({kv.second, kv.first}); }

    bpe_.initFromSentencePieceJson(file_path);

    special_tokens_trie_.add(L"<|begin_of_text|>");
    special_tokens_trie_.add(L"<|end_of_text|>");
    special_tokens_trie_.add(L"<|start_header_id|>");
    special_tokens_trie_.add(L"<|end_header_id|>");
    special_tokens_trie_.add(L"<|eot_id|>");
  }

  std::string getSystemPromptPrefix() {
    std::time_t t = std::time(nullptr);
    std::tm tm_ = *std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&tm_, "%d %b %Y");
    return "Cutting Knowledge Date: December 2023\nToday Date: " + oss.str() + "\n\n";
  }

  inline std::string applyChatTemplate(const std::vector<LlamaMessage>& messages, bool add_generation_prompt = true) {
    std::string result = "";
    if (add_bos_) result += "<|begin_of_text|>";
    for (const auto& msg : messages) {
      std::string content = msg.content;
      if (msg.role == "system") content = getSystemPromptPrefix() + content;
      result += "<|start_header_id|>" + msg.role + "<|end_header_id|>\n\n" + content + "<|eot_id|>";
    }
    if (add_generation_prompt) result += "<|start_header_id|>assistant<|end_header_id|>\n\n";
    return result;
  }

  std::vector<std::wstring> _tokenize(const std::string& str) override {
    std::vector<std::wstring> ret;
    std::vector<std::wstring> splitted;
    llama3Regex(str, splitted);
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
    std::string processed_str = str;
    bool text_has_bos = (processed_str.find("<|begin_of_text|>") == 0);
    if (add_bos_ && !text_has_bos) { processed_str = "<|begin_of_text|>" + processed_str; }

    auto tokens = special_tokens_trie_.split(preprocessor::utf8string2WideString(processed_str));
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
    for (wchar_t c : str) {
      if (bytes_2_unicode_dict_inverse_.count(c)) {
        utf_8_str.push_back((unsigned char)(bytes_2_unicode_dict_inverse_[c]));
      } else {
        return str;
      }
    }
    return mllm::preprocessor::utf8string2WideString(utf_8_str);
  }

  Tensor convert2Ids(const std::vector<std::wstring>& strs) override {
    std::vector<int64_t> ids;
    for (const auto& str : strs) { ids.emplace_back(bpe_._lookup_vocab(str)); }
    Tensor ret =
        Tensor::empty({1, (int32_t)ids.size()}, kInt64, kCPU).setMemType(kExtraInput).setName("llama-tokenizer-i0").alloc();
    auto ptr = ret.ptr<int64_t>();
    for (size_t i = 0; i < ids.size(); ++i) { ptr[i] = ids[i]; }
    return ret;
  }

  std::vector<int64_t> encode(const std::string& str) {
    auto sub_tokens = tokenize(str);
    std::vector<int64_t> ret;
    for (auto& token : sub_tokens) { ret.emplace_back(bpe_._lookup_vocab(token)); }
    return ret;
  }

  std::string decode(const std::vector<int64_t>& ids) {
    std::string ret;
    for (auto& each_id : ids) {
      auto wstr = detokenize(each_id);
      ret += mllm::preprocessor::wideString2Utf8String(wstr);
    }
    return ret;
  }

  ARGenerationOutputPast convertMessage(const std::vector<LlamaMessage>& messages) {
    auto applied_string = applyChatTemplate(messages, true);
    auto sequence_str = tokenize(applied_string);
    std::vector<int64_t> ids;
    for (const auto& str : sequence_str) { ids.emplace_back(bpe_._lookup_vocab(str)); }

    Tensor sequence =
        Tensor::empty({1, (int32_t)ids.size()}, kInt64, kCPU).setMemType(kNormal).setName("llama-tokenizer-i0").alloc();
    auto ptr = sequence.ptr<int64_t>();
    for (size_t i = 0; i < ids.size(); ++i) { ptr[i] = ids[i]; }

    return {
        {"sequence", sequence},
    };
  }

 private:
  bool add_bos_ = true;
  preprocessor::BPE bpe_;
  std::unordered_map<std::wint_t, wchar_t> bytes_2_unicode_dict_;
  std::unordered_map<wchar_t, std::wint_t> bytes_2_unicode_dict_inverse_;
};

}  // namespace mllm::models::llama