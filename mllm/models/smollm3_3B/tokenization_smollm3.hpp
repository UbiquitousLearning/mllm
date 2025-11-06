// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <string>
#include <ctime>
#include <iomanip>
#include <sstream>

#include "mllm/preprocessor/tokenizers/BPEUTF8.hpp"
#include "mllm/preprocessor/tokenizers/Unicode.hpp"
#include "mllm/preprocessor/tokenizers/AutoTokenizer.hpp"
#include "mllm/preprocessor/tokenizers/llama_cpp_unicode/unicode.h"

namespace mllm::models::smollm3 {

namespace details {

// Standard GPT2 regex
// https://github.com/openai/gpt-2/blob/master/src/encoder.py#L53
constexpr char GPT2_EXPR[] = R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";

}  // namespace details

struct SmolLM3Message {
  std::string prompt;
  bool enable_thinking = false;

  static inline const std::string no_think_template_str =
      "<|im_start|>system\n## Metadata\n\nKnowledge Cutoff Date: June 2025\nToday Date: "
      "{{date_in_number}} {{month}} {{year}}\nReasoning Mode: /no_think\n\n## Custom Instructions\n\nYou are a helpful AI "
      "assistant named SmolLM, trained by Hugging "
      "Face.\n\n<|im_start|>user\n{{prompt}}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n";

  static inline const std::string think_template_str =
      "<|im_start|>system\n## Metadata\n\nKnowledge Cutoff Date: June 2025\nToday Date: "
      "{{date_in_number}} {{month}} {{year}}\nReasoning Mode: /think\n\n## Custom Instructions\n\nYou are a helpful AI "
      "assistant named SmolLM, trained by Hugging Face. Your role as an assistant involves thoroughly exploring questions "
      "through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging "
      "in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracking, and iteration "
      "to develop well-considered thinking process. Please structure your response into two main sections: Thought and "
      "Solution using the specified format: <think> Thought section </think> Solution section. In the Thought section, detail "
      "your reasoning process in steps. Each step should include detailed considerations such as analysing questions, "
      "summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any "
      "errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and "
      "reflections from the Thought section, systematically present the final solution that you deem correct. The Solution "
      "section should be logical, accurate, and concise and detail necessary steps needed to reach the "
      "conclusion.\n\n<|im_start|>user\n{{prompt}}<|im_end|>\n<|im_start|>assistant\n";
};

class SmolLM3Tokenizer final : public mllm::preprocessor::AutoTokenizerUTF8 {
 public:
  explicit SmolLM3Tokenizer(const std::string& file_path) {
    // Init
    preprocessor::initLocal();

    // Load bpe files
    bpe_.initFromSentencePieceJson(file_path);

    // Add special tokens to trie
    special_tokens_trie_.add("<|im_end|>");
    special_tokens_trie_.add("<|begin_of_text|>");
    special_tokens_trie_.add("<|end_of_text|>");
    special_tokens_trie_.add("<think>");
    special_tokens_trie_.add("</think>");
    special_tokens_trie_.add("<|im_start|>");
  }

  void replaceAll(std::string& s, const std::string& from, const std::string& to) {
    if (from.empty()) return;
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
      s.replace(pos, from.size(), to);
      pos += to.size();
    }
  }

  std::string applyChatTemplate(const std::string& prompt, bool enable_thinking) {
    std::time_t t = std::time(nullptr);
    std::tm tm_ = *std::localtime(&t);

    std::ostringstream oss;
    oss << std::put_time(&tm_, "%d");
    std::string date_in_number = oss.str();

    static const char* month_names[] = {"January", "February", "March",     "April",   "May",      "June",
                                        "July",    "August",   "September", "October", "November", "December"};
    std::string month = month_names[tm_.tm_mon];
    std::string year = std::to_string(1900 + tm_.tm_year);

    std::string tpl = enable_thinking ? SmolLM3Message::think_template_str : SmolLM3Message::no_think_template_str;

    replaceAll(tpl, "{{date_in_number}}", date_in_number);
    replaceAll(tpl, "{{month}}", month);
    replaceAll(tpl, "{{year}}", year);
    replaceAll(tpl, "{{prompt}}", prompt);

    return tpl;
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

      // No need to Regex:
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

  // For text
  preprocessor::BPEUTF8 bpe_;
};

}  // namespace mllm::models::smollm3
