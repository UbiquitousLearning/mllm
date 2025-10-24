// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/tokenization_llama.py
// and
// ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/tokenization_llama_fast.py

// LlamaTokenizerFast
#pragma once

#include <vector>
#include <vector>
#include <string>
#include <unordered_map>

#include "mllm/preprocessor/tokenizers/BPE.hpp"
#include "mllm/preprocessor/tokenizers/Unicode.hpp"
#include "mllm/preprocessor/tokenizers/AutoTokenizer.hpp"

namespace mllm::models::deepseek_ocr {

// Actually is LlamaTokenizer
class DpskOcrTokenizer final : public mllm::preprocessor::AutoTokenizer {
 public:
  explicit DpskOcrTokenizer(const std::string& file_path) {
    // Init
    preprocessor::initLocal();

    // Load bpe files
    bpe_.initFromSentencePieceJson(file_path);

    // Add special tokens to trie
    special_tokens_trie_.add(L"<|User|>");
    special_tokens_trie_.add(L"<|Assistant|>");
    special_tokens_trie_.add(L"<｜begin▁of▁sentence｜>");
    special_tokens_trie_.add(L"<｜end▁of▁sentence｜>");
    special_tokens_trie_.add(L"<｜▁pad▁｜>");
  }

  std::vector<std::wstring> _tokenize(const std::string& str) override {
    // Replace spaces with SentencePiece underline before processing
    // TODO

    auto processed_tokens = preTokenize(preprocessor::utf8string2WideString(str));
    std::vector<std::wstring> ret;

    return ret;
  }

  std::vector<std::wstring> tokenize(const std::string& str) override { return _tokenize(str); }

  std::wstring _detokenize(int64_t pos_idx) override {
    // TODO
    return L"";
  }

  std::wstring detokenize(int64_t pos_idx) override {
    // TODO
    return _detokenize(pos_idx);
  }

  std::vector<int64_t> convert2VectorIds(const std::vector<std::wstring>& strs) {
    std::vector<int64_t> ids;
    ids.reserve(strs.size());
    for (const auto& str : strs) { ids.emplace_back(bpe_._lookup_vocab(str)); }
    return ids;
  }

  Tensor convert2Ids(const std::vector<std::wstring>& strs) override {
    std::vector<int64_t> ids;
    ids.reserve(strs.size());
    for (const auto& str : strs) { ids.emplace_back(bpe_._lookup_vocab(str)); }
    Tensor ret = Tensor::empty({/*batch*/ 1, /*seq*/ (int32_t)ids.size()}, kInt64, kCPU)
                     .setMemType(kExtraInput)
                     .setName("llama-tokenizer-i0")
                     .alloc();
    auto ptr = ret.ptr<int64_t>();
    for (size_t i = 0; i < ids.size(); ++i) { ptr[i] = ids[i]; }
    return ret;
  }

 private:
  // "pre_tokenizer": {
  //   "type": "Sequence",
  //   "pretokenizers": [
  //     {
  //       "type": "Split",
  //       "pattern": {
  //         "Regex": "\\p{N}{1,3}"
  //       },
  //       "behavior": "Isolated",
  //       "invert": false
  //     },
  //     {
  //       "type": "Split",
  //       "pattern": {
  //         "Regex": "[一-龥぀-ゟ゠-ヿ]+"
  //       },
  //       "behavior": "Isolated",
  //       "invert": false
  //     },
  //     {
  //       "type": "Split",
  //       "pattern": {
  //         "Regex": "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+|
  //         ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"
  //       },
  //       "behavior": "Isolated",
  //       "invert": false
  //     },
  //     {
  //       "type": "ByteLevel",
  //       "add_prefix_space": false,
  //       "trim_offsets": true,
  //       "use_regex": false
  //     }
  //   ]
  // }
  std::vector<std::wstring> preTokenize(const std::wstring& str) {
    std::vector<std::wstring> result;
    size_t pos = 0;

    while (pos < str.size()) {
      std::wstring matched;
      bool found_match = false;

      // Pattern 1: Match 1-3 consecutive digits (\p{N}{1,3})
      if (preprocessor::isDigit(str[pos])) {
        size_t start = pos;
        size_t count = 0;
        while (pos < str.size() && preprocessor::isDigit(str[pos]) && count < 3) {
          ++pos;
          ++count;
        }
        matched = str.substr(start, count);
        found_match = true;
      }
      // Pattern 2: Match CJK characters ([一-龥぀-ゟ゠-ヿ]+)
      else if ((str[pos] >= L'一' && str[pos] <= L'龥') ||   // Chinese characters
               (str[pos] >= L'぀' && str[pos] <= L'ゟ') ||  // Hiragana
               (str[pos] >= L'゠' && str[pos] <= L'ヿ')) {   // Katakana
        size_t start = pos;
        while (pos < str.size()
               && ((str[pos] >= L'一' && str[pos] <= L'龥') || (str[pos] >= L'぀' && str[pos] <= L'ゟ')
                   || (str[pos] >= L'゠' && str[pos] <= L'ヿ'))) {
          ++pos;
        }
        matched = str.substr(start, pos - start);
        found_match = true;
      }
      // Pattern 3: Complex pattern for other characters
      // [!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+|
      // ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+
      else {
        // Handle punctuation followed by letters
        if ((str[pos] >= L'!' && str[pos] <= L'/') || (str[pos] >= L':' && str[pos] <= L'@')
            || (str[pos] >= L'[' && str[pos] <= L'`') || (str[pos] >= L'{' && str[pos] <= L'~')) {
          size_t start = pos;
          ++pos;  // consume the punctuation
          // Check if followed by letters
          if (pos < str.size() && preprocessor::isLetter(str[pos])) {
            while (pos < str.size() && preprocessor::isLetter(str[pos])) { ++pos; }
            matched = str.substr(start, pos - start);
            found_match = true;
          } else {
            pos = start + 1;  // just consume the punctuation character
            matched = str.substr(start, 1);
            found_match = true;
          }
        }
        // Handle letters with optional prefix
        else if (preprocessor::isLetter(str[pos])) {
          size_t start = pos;
          while (pos < str.size() && preprocessor::isLetter(str[pos])) { ++pos; }
          matched = str.substr(start, pos - start);
          found_match = true;
        }
        // Handle whitespace
        else if (std::iswspace(str[pos])) {
          size_t start = pos;
          while (pos < str.size() && std::iswspace(str[pos])) { ++pos; }
          matched = str.substr(start, pos - start);
          found_match = true;
        }
        // Handle any other character
        else {
          matched = str.substr(pos, 1);
          ++pos;
          found_match = true;
        }
      }

      // Add matched string to result
      if (found_match) { result.push_back(matched); }
    }

    return result;
  }

  // For text
  preprocessor::BPE bpe_;
  std::wstring SPIECE_UNDERLINE = L"▁";
};
}  // namespace mllm::models::deepseek_ocr
