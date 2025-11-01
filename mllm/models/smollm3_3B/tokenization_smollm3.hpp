// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <vector>
#include <unordered_map>

#include "mllm/preprocessor/tokenizers/BPE.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/preprocessor/tokenizers/Unicode.hpp"
#include "mllm/preprocessor/tokenizers/AutoTokenizer.hpp"

namespace mllm::models::smollm3 {

struct SmolLM3Message {
  std::string prompt;
  bool enable_thinking = false;

  // 使用SmolLM3模板格式
  static inline std::string thinking_template =
      "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n";
  static inline std::string direct_template =
      "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n";
};

class SmolLM3Tokenizer final : public mllm::preprocessor::AutoTokenizer {
 public:
  explicit SmolLM3Tokenizer(const std::string& file_path) {
    preprocessor::initLocal();
    
    // 初始化字节到unicode映射
    preprocessor::makeBytes2UnicodeMap(bytes_2_unicode_dict_);
    for (auto& kv : bytes_2_unicode_dict_) {
      bytes_2_unicode_dict_inverse_.insert({kv.second, kv.first});
    }

    // 加载BPE tokenizer
    bpe_.initFromSentencePieceJson(file_path);

    // 添加SmolLM3的特殊token
    special_tokens_trie_.add(L"<|im_start|>");
    special_tokens_trie_.add(L"<|im_end|>");
    special_tokens_trie_.add(L"<think>");
    special_tokens_trie_.add(L"</think>");
    special_tokens_trie_.add(L"<|system|>");
    special_tokens_trie_.add(L"<|user|>");
    special_tokens_trie_.add(L"<|assistant|>");
    special_tokens_trie_.add(L"<|endoftext|>");
  }

  std::vector<std::wstring> _tokenize(const std::string& text) override {
    if (text.empty()) {
      return {};
    }
    
    // 将UTF-8文本转换为宽字符串
    std::wstring w_text = preprocessor::utf8string2WideString(text);
    
    // 应用字节到unicode映射
    std::wstring mapped_text;
    for (wchar_t c : w_text) {
      if (c <= 255) {
        unsigned char uc = static_cast<unsigned char>(c);
        if (bytes_2_unicode_dict_.count(uc)) {
          mapped_text.push_back(bytes_2_unicode_dict_[uc]);
        } else {
          mapped_text.push_back(c);
        }
      } else {
        mapped_text.push_back(c);
      }
    }
    
    // 使用BPE分词
    auto bpe_tokens = bpe_._bpe(mapped_text);
    return bpe_tokens;
  }

  std::vector<std::wstring> tokenize(const std::string& text) override {
    if (text.empty()) {
      return {};
    }
    
    // 使用特殊token分割文本
    auto w_text = preprocessor::utf8string2WideString(text);
    auto segments = special_tokens_trie_.split(w_text);
    
    std::vector<std::wstring> all_tokens;
    for (const auto& segment : segments) {
      if (special_tokens_trie_.isSpecialToken(segment)) {
        all_tokens.push_back(segment);
      } else {
        auto segment_text = preprocessor::wideString2Utf8String(segment);
        auto segment_tokens = _tokenize(segment_text);
        all_tokens.insert(all_tokens.end(), segment_tokens.begin(), segment_tokens.end());
      }
    }
    
    return all_tokens;
  }

  std::wstring _detokenize(int64_t pos_idx) override {
    return bpe_._lookup_inverse_vocab(pos_idx);
  }

  std::wstring detokenize(int64_t pos_idx) override {
    auto token_str = _detokenize(pos_idx);
    
    // 将映射后的unicode字符转换回原始字节
    std::string result_bytes;
    for (wchar_t c : token_str) {
      if (bytes_2_unicode_dict_inverse_.count(c)) {
        result_bytes.push_back(static_cast<char>(bytes_2_unicode_dict_inverse_[c]));
      } else {
        auto utf8_str = preprocessor::wideString2Utf8String(std::wstring(1, c));
        result_bytes += utf8_str;
      }
    }
    
    return preprocessor::utf8string2WideString(result_bytes);
  }

  Tensor convert2Ids(const std::vector<std::wstring>& tokens) override {
    std::vector<int64_t> token_ids;
    token_ids.reserve(tokens.size());
    
    for (const auto& token : tokens) {
      int64_t token_id = bpe_._lookup_vocab(token);
      token_ids.push_back(token_id);
    }

    Tensor ret = Tensor::empty({1, (int32_t)token_ids.size()}, kInt64, kCPU)
                     .setMemType(kExtraInput)
                     .setName("smollm3-tokenizer-i0")
                     .alloc();

    auto ptr = ret.ptr<int64_t>();
    for (size_t i = 0; i < token_ids.size(); ++i) {
      ptr[i] = token_ids[i];
    }

    return ret;
  }

  ARGenerationOutputPast convertMessage(const SmolLM3Message& message) {
    // 选择模板
    std::string template_str = message.enable_thinking 
                               ? SmolLM3Message::thinking_template 
                               : SmolLM3Message::direct_template;

    // 替换prompt占位符
    std::string applied_string = template_str;
    size_t pos = applied_string.find("{prompt}");
    if (pos != std::string::npos) {
      applied_string.replace(pos, 8, message.prompt);
    }

    // Tokenize并转换为IDs
    auto tokens = tokenize(applied_string);
    
    std::vector<int64_t> token_ids;
    for (const auto& token : tokens) {
      token_ids.push_back(bpe_._lookup_vocab(token));
    }

    Tensor sequence = Tensor::empty({1, (int32_t)token_ids.size()}, kInt64, kCPU)
                          .setMemType(kNormal)
                          .setName("smollm3-tokenizer-i0")
                          .alloc();

    auto ptr = sequence.ptr<int64_t>();
    for (size_t i = 0; i < token_ids.size(); ++i) {
      ptr[i] = token_ids[i];
    }

    return {{"sequence", sequence}};
  }

 private:
  preprocessor::BPE bpe_;
  std::unordered_map<std::wint_t, wchar_t> bytes_2_unicode_dict_;
  std::unordered_map<wchar_t, std::wint_t> bytes_2_unicode_dict_inverse_;
};

}  // namespace mllm::models::smollm3
