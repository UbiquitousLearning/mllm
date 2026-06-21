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

#include "mllm/preprocessor/tokenizers/BPEUTF8.hpp"
#include "mllm/preprocessor/tokenizers/Unicode.hpp"
#include "mllm/preprocessor/tokenizers/AutoTokenizer.hpp"
#include "mllm/preprocessor/tokenizers/llama_cpp_unicode/unicode.h"

namespace mllm::models::deepseek_ocr {

namespace details {

// Standard GPT2 regex
// https://github.com/openai/gpt-2/blob/master/src/encoder.py#L53
constexpr char GPT2_EXPR[] = R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";

}  // namespace details

// Actually is LlamaTokenizer
class DpskOcrTokenizer final : public mllm::preprocessor::AutoTokenizerUTF8 {
 public:
  explicit DpskOcrTokenizer(const std::string& file_path) {
    // Init
    preprocessor::initLocal();

    // Load bpe files
    bpe_.initFromSentencePieceJson(file_path);

    // Add special tokens to trie
    special_tokens_trie_.add("<|User|>");
    special_tokens_trie_.add("<|Assistant|>");
    special_tokens_trie_.add("<｜begin▁of▁sentence｜>");
    special_tokens_trie_.add("<｜end▁of▁sentence｜>");
    special_tokens_trie_.add("<｜▁pad▁｜>");
    special_tokens_trie_.add("<image>");
    special_tokens_trie_.add("<|grounding|>");
    special_tokens_trie_.add("<tr>");
    special_tokens_trie_.add("</tr>");
  }

  std::vector<int64_t> encode(const std::string& str) override {
    auto sub_tokens = tokenize(str);
    auto ret = std::vector<int64_t>{};
    for (auto& token : sub_tokens) { ret.emplace_back(bpe_._lookup_vocab(token)); }
    return ret;
  }

  std::string decode(const std::vector<int64_t>& ids) override {
    std::vector<std::string> after_bpe_check;
    for (auto& each_id : ids) {
      auto each_str = bpe_._lookup_inverse_vocab(each_id);
      after_bpe_check.emplace_back(each_str);
    }
    return detokenize(after_bpe_check);
  }

  std::vector<std::string> tokenize(const std::string& str) override {
    // Replace all blank token to underscore

    std::vector<std::string> ret;
    for (auto& each_str : special_tokens_trie_.split(str)) {
      if (special_tokens_trie_.isSpecialToken(each_str)) {
        ret.emplace_back(each_str);
        continue;
      }

      // FIXME Should Regex:
      auto after_regex_process = {each_str};

      for (auto& ss : after_regex_process) {
        auto after_bytes_process = byteLevelPreTokenizer(ss);

        // Perform BPE algorithm on each sub-token
        for (auto& bbpe_str : after_bytes_process) {
          auto bbpe_str_sub_tokens = bpe_._bpe(bbpe_str);
          ret.insert(ret.end(), bbpe_str_sub_tokens.begin(), bbpe_str_sub_tokens.end());
        }
      }
    }
    return ret;
  }

  std::string detokenize(const std::vector<std::string>& tokenized_str) override {
    std::string ret;
    for (auto& each_str : tokenized_str) {
      if (special_tokens_trie_.isSpecialToken(each_str)) {
        ret += each_str;
        continue;
      }
      // Loop utf8 string
      utf8::iterator it(each_str.begin(), each_str.begin(), each_str.end());
      utf8::iterator end_it(each_str.end(), each_str.begin(), each_str.end());
      for (; it != end_it; ++it) {
        char32_t cp = *it;
        auto b = unicode_utf8_to_byte(unicode_cpt_to_utf8(cp));
        ret.push_back(b);
      }
    }
    return ret;
  }

 private:
  std::vector<std::string> byteLevelPreTokenizer(const std::string& str) {
    return unicode_regex_split(str, {std::string{details::GPT2_EXPR}});
  }

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
  std::vector<std::string> regexPreTokenizer(const std::string& token) {
    std::vector<std::string> out;
    auto it = token.begin();
    auto end = token.end();

    while (it != end) {
      auto seg_start = it;
      int digit_cnt = 0;
      auto tmp = it;
      while (digit_cnt < 3) {
        uint32_t cp = 0;
        auto next = tmp;
        utf8::next(next, end);
        if (next == tmp) break;
        cp = utf8::peek_next(tmp, end);
        if (!is_digit(cp)) break;
        tmp = next;
        ++digit_cnt;
      }
      if (digit_cnt > 0) {
        out.emplace_back(seg_start, tmp);
        it = tmp;
        continue;
      }

      uint32_t cp = utf8::peek_next(it, end);
      if (is_cjk(cp)) {
        auto tmp2 = it;
        while (tmp2 != end) {
          uint32_t nxt = utf8::peek_next(tmp2, end);
          if (!is_cjk(nxt)) break;
          utf8::next(tmp2, end);
        }
        out.emplace_back(seg_start, tmp2);
        it = tmp2;
        continue;
      }

      if (is_punct_symbol(cp)) {
        auto tmp3 = it;
        utf8::next(tmp3, end);
        if (tmp3 != end && is_letter(utf8::peek_next(tmp3, end))) {
          utf8::next(tmp3, end);
          out.emplace_back(seg_start, tmp3);
          it = tmp3;
          continue;
        }
      }

      if (!is_letter(cp) && !is_space(cp) && !is_punct_symbol(cp)) {
        auto tmp3 = it;
        utf8::next(tmp3, end);
        if (tmp3 != end && is_letter(utf8::peek_next(tmp3, end))) {
          while (tmp3 != end) {
            uint32_t nxt = utf8::peek_next(tmp3, end);
            if (!is_letter(nxt)) break;
            utf8::next(tmp3, end);
          }
          out.emplace_back(seg_start, tmp3);
          it = tmp3;
          continue;
        }
      }

      if (is_punct_symbol(cp)) {
        auto tmp3 = it;
        while (tmp3 != end) {
          uint32_t nxt = utf8::peek_next(tmp3, end);
          if (!is_punct_symbol(nxt)) break;
          utf8::next(tmp3, end);
        }

        while (tmp3 != end) {
          uint32_t nxt = utf8::peek_next(tmp3, end);
          if (nxt != 0x0A && nxt != 0x0D) break;
          utf8::next(tmp3, end);
        }
        out.emplace_back(seg_start, tmp3);
        it = tmp3;
        continue;
      }

      if (is_space(cp)) {
        auto tmp3 = it;
        bool has_nl = false;
        while (tmp3 != end) {
          uint32_t nxt = utf8::peek_next(tmp3, end);
          if (nxt == 0x0A || nxt == 0x0D) {
            has_nl = true;
            utf8::next(tmp3, end);
          } else if (is_space(nxt)) {
            utf8::next(tmp3, end);
          } else {
            break;
          }
        }
        if (has_nl) {
          out.emplace_back(seg_start, tmp3);
          it = tmp3;
          continue;
        }
        auto tmp4 = tmp3;
        while (tmp4 != end && is_space(utf8::peek_next(tmp4, end))) utf8::next(tmp4, end);
        if (tmp4 == end) {
          out.emplace_back(seg_start, tmp4);
          it = tmp4;
          continue;
        }
        while (tmp3 != end) {
          uint32_t nxt = utf8::peek_next(tmp3, end);
          if (!is_space(nxt)) break;
          utf8::next(tmp3, end);
        }
        out.emplace_back(seg_start, tmp3);
        it = tmp3;
        continue;
      }
      utf8::next(it, end);
      out.emplace_back(seg_start, it);
    }

    return out;
  }

  static inline bool is_digit(uint32_t cp) { return cp >= 0x30 && cp <= 0x39; }

  static inline bool is_cjk(uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) ||  // CJK Unified Ideographs
           (cp >= 0x3400 && cp <= 0x4DBF) ||  // CJK Extension A
           (cp >= 0xF900 && cp <= 0xFAFF) ||  // CJK Compatibility
           (cp >= 0x3040 && cp <= 0x309F) ||  // Hiragana
           (cp >= 0x30A0 && cp <= 0x30FF);    // Katakana
  }

  static inline bool is_letter(uint32_t cp) { return (cp >= 0x41 && cp <= 0x5A) || (cp >= 0x61 && cp <= 0x7A); }

  static inline bool is_punct_symbol(uint32_t cp) {
    return (cp >= 0x21 && cp <= 0x2F) || (cp >= 0x3A && cp <= 0x40) || (cp >= 0x5B && cp <= 0x60) || (cp >= 0x7B && cp <= 0x7E);
  }

  static inline bool is_space(uint32_t cp) { return cp == 0x20 || cp == 0x09 || cp == 0x0A || cp == 0x0D; }

  // For text
  preprocessor::BPEUTF8 bpe_;
  std::string SPIECE_UNDERLINE = "▁";
};
}  // namespace mllm::models::deepseek_ocr
