// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <unordered_map>
#include <vector>

#include "mllm/models/ARGeneration.hpp"
#include "mllm/models/qwen3/tokenization_qwen3.hpp"
#include "mllm/preprocessor/tokenizers/AutoTokenizer.hpp"
#include "mllm/preprocessor/tokenizers/BPE.hpp"
#include "mllm/preprocessor/tokenizers/Unicode.hpp"

namespace mllm::models::qwen_ascend {

struct QwenAscendMessage {
  std::string prompt;
  static inline std::string message_template =
      "<|im_start|>user\n{{{prompt}}}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
};

class QwenAscendTokenizer final : public mllm::preprocessor::AutoTokenizer {
 public:
  explicit QwenAscendTokenizer(const std::string& file_path) {
    preprocessor::initLocal();
    preprocessor::makeBytes2UnicodeMap(bytes_2_unicode_dict_);
    for (auto& kv : bytes_2_unicode_dict_) { bytes_2_unicode_dict_inverse_.insert({kv.second, kv.first}); }
    bpe_.initFromSentencePieceJson(file_path);
    special_tokens_trie_.add(L"<|endoftext|>");
    special_tokens_trie_.add(L"<|im_start|>");
    special_tokens_trie_.add(L"<|im_end|>");
    special_tokens_trie_.add(L"<|object_ref_start|>");
    special_tokens_trie_.add(L"<|object_ref_end|>");
    special_tokens_trie_.add(L"<|box_start|>");
    special_tokens_trie_.add(L"<|box_end|>");
    special_tokens_trie_.add(L"<|quad_start|>");
    special_tokens_trie_.add(L"<|quad_end|>");
    special_tokens_trie_.add(L"<|vision_start|>");
    special_tokens_trie_.add(L"<|vision_end|>");
    special_tokens_trie_.add(L"<|vision_pad|>");
    special_tokens_trie_.add(L"<|image_pad|>");
    special_tokens_trie_.add(L"<|video_pad|>");
    special_tokens_trie_.add(L"<think>");
    special_tokens_trie_.add(L"</think>");
  }

  std::vector<std::wstring> _tokenize(const std::string& str) override {
    std::vector<std::wstring> ret;
    std::vector<std::wstring> splitted;
    ::mllm::models::qwen3::qwen3Regex(str, splitted);
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
    return {mllm::preprocessor::utf8string2WideString(utf_8_str)};
  }

  // Decode full id sequence as one UTF-8 string to avoid per-token mojibake.
  std::string decode(const std::vector<int64_t>& ids) {
    std::string utf_8_str;
    for (auto id : ids) {
      auto piece = _detokenize(id);
      for (wchar_t c : piece) { utf_8_str.push_back((unsigned char)(bytes_2_unicode_dict_inverse_[c])); }
    }
    return utf_8_str;
  }

  Tensor convert2Ids(const std::vector<std::wstring>& strs) override {
    std::vector<int64_t> ids;
    ids.reserve(strs.size());
    for (const auto& str : strs) { ids.emplace_back(bpe_._lookup_vocab(str)); }

    Tensor ret = Tensor::empty({1, (int32_t)ids.size()}, kInt64, kCPU).setMemType(kExtraInput).setName("qwen-ascend-tokenizer-i0").alloc();
    auto ptr = ret.ptr<int64_t>();
    for (size_t i = 0; i < ids.size(); ++i) { ptr[i] = ids[i]; }
    return ret;
  }

  ARGenerationOutputPast convertMessage(const QwenAscendMessage& message) {
    auto applied_string = QwenAscendMessage::message_template;
    size_t pos = applied_string.find("{{{prompt}}}");
    applied_string.replace(pos, 12, message.prompt);

    auto sequence_str = tokenize(applied_string);
    std::vector<int64_t> ids;
    ids.reserve(sequence_str.size());
    for (const auto& str : sequence_str) { ids.emplace_back(bpe_._lookup_vocab(str)); }

    Tensor sequence = Tensor::empty({1, (int32_t)ids.size()}, kInt64, kCPU).setMemType(kNormal).setName("qwen-ascend-tokenizer-i0").alloc();
    auto ptr = sequence.ptr<int64_t>();
    for (size_t i = 0; i < ids.size(); ++i) { ptr[i] = ids[i]; }

    return {{"sequence", sequence}};
  }

 private:
  preprocessor::BPE bpe_;
  std::unordered_map<std::wint_t, wchar_t> bytes_2_unicode_dict_;
  std::unordered_map<wchar_t, std::wint_t> bytes_2_unicode_dict_inverse_;
};

}  // namespace mllm::models::qwen_ascend
