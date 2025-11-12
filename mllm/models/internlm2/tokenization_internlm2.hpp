// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

#include "mllm/models/ARGeneration.hpp"
#include "mllm/preprocessor/tokenizers/AutoTokenizer.hpp"
#include "mllm/preprocessor/tokenizers/BPE.hpp"
#include "mllm/preprocessor/tokenizers/Unicode.hpp"

namespace mllm::models::internlm2 {

struct InternLM2Message {
  std::string prompt;
  static inline std::string message_template = "<|im_start|>user\n{{{prompt}}}<|im_end|>\n<|im_start|>assistant\n";
  bool add_bos = true;
  bool add_eos = false;
};

class InternLM2Tokenizer final : public preprocessor::AutoTokenizer {
 public:
  explicit InternLM2Tokenizer(const std::string& file_path, bool add_bos = true, bool add_eos = false)
      : add_bos_(add_bos), add_eos_(add_eos) {
    preprocessor::initLocal();
    preprocessor::makeBytes2UnicodeMap(bytes_2_unicode_dict_);
    for (auto& kv : bytes_2_unicode_dict_) { bytes_2_unicode_dict_inverse_.insert({kv.second, kv.first}); }
    bpe_.initFromSentencePieceJson(file_path);

    special_tokens_trie_.add(L"<|plugin|>");
    special_tokens_trie_.add(L"<|interpreter|>");
    special_tokens_trie_.add(L"<|action_end|>");
    special_tokens_trie_.add(L"<|action_start|>");
    special_tokens_trie_.add(L"<|im_end|>");
    special_tokens_trie_.add(L"<|im_start|>");
  }

  std::vector<std::wstring> _tokenize(const std::string& str) override {
    std::vector<std::wstring> tokens;
    auto w_string = preprocessor::utf8string2WideString(str);
    auto normalized = normalize(w_string);

    auto bpe_tokens = bpe_._bpe(normalized);
    tokens.reserve(bpe_tokens.size());
    for (const auto& token : bpe_tokens) { tokens.push_back(token); }
    return tokens;
  }

  std::vector<std::wstring> tokenize(const std::string& str) override {
    auto tokens = special_tokens_trie_.split(preprocessor::utf8string2WideString(str));
    std::vector<std::wstring> all_tokens;

    if (add_bos_) { all_tokens.emplace_back(L"<s>"); }
    for (const auto& token : tokens) {
      if (special_tokens_trie_.isSpecialToken(token)) {
        all_tokens.emplace_back(token);
        continue;
      }
      auto tmp_tokens = _tokenize(preprocessor::wideString2Utf8String(token));
      all_tokens.insert(all_tokens.end(), tmp_tokens.begin(), tmp_tokens.end());
    }
    if (add_eos_) { all_tokens.emplace_back(L"</s>"); }
    return all_tokens;
  }

  std::wstring _detokenize(int64_t pos_idx) override {
    auto str = bpe_._lookup_inverse_vocab(pos_idx);
    return postprocess(str);
  }

  std::wstring detokenize(int64_t pos_idx) override { return _detokenize(pos_idx); }

  Tensor convert2Ids(const std::vector<std::wstring>& strs) override {
    std::vector<int64_t> ids;
    ids.reserve(strs.size());
    for (const auto& str : strs) { ids.emplace_back(bpe_._lookup_vocab(str)); }

    Tensor ret = Tensor::empty({1, static_cast<int32_t>(ids.size())}, kInt64, kCPU)
                     .setMemType(kExtraInput)
                     .setName("internlm2-tokenizer-i0")
                     .alloc();

    auto ptr = ret.ptr<int64_t>();
    for (size_t i = 0; i < ids.size(); ++i) { ptr[i] = ids[i]; }

    return ret;
  }

  ARGenerationOutputPast convertMessage(const InternLM2Message& prompt) {
    auto applied_string = InternLM2Message::message_template;
    size_t pos = applied_string.find("{{{prompt}}}");
    applied_string.replace(pos, 12, prompt.prompt);
    auto tokens = tokenize(prompt.prompt);

    if (!prompt.add_bos && !tokens.empty() && tokens.front() == L"<s>") { tokens.erase(tokens.begin()); }
    if (prompt.add_eos && (tokens.empty() || tokens.back() != L"</s>")) { tokens.emplace_back(L"</s>"); }

    std::vector<int64_t> ids;
    ids.reserve(tokens.size());
    for (const auto& token : tokens) { ids.emplace_back(bpe_._lookup_vocab(token)); }

    Tensor sequence = Tensor::empty({1, static_cast<int32_t>(ids.size())}, kInt64, kCPU)
                          .setMemType(kNormal)
                          .setName("internlm2-seq-i0")
                          .alloc();

    auto ptr = sequence.ptr<int64_t>();
    for (size_t i = 0; i < ids.size(); ++i) { ptr[i] = ids[i]; }

    return {
        {"sequence", sequence},
    };
  }

 private:
  static std::wstring normalize(const std::wstring& text) {
    if (text.empty()) { return text; }

    std::wstring normalized = text;
    std::wstring space_char = L" ";
    std::wstring underline_char = L"▁";

    size_t pos = 0;
    while ((pos = normalized.find(space_char, pos)) != std::wstring::npos) {
      normalized.replace(pos, space_char.length(), underline_char);
      pos += underline_char.length();
    }

    // if (normalized[0] != L'▁') { normalized = underline_char + normalized; }

    return normalized;
  }

  static std::wstring postprocess(const std::wstring& text) {
    if (text == L"<s>" || text == L"</s>" || text == L"") { return L""; }

    std::wstring processed = text;
    std::wregex underline_regex(L"▁");
    processed = std::regex_replace(processed, underline_regex, L" ");

    if (processed == L"<0x0A>") { return L"\n"; }
    return processed;
  }

 private:
  preprocessor::BPE bpe_;
  std::unordered_map<std::wint_t, wchar_t> bytes_2_unicode_dict_;
  std::unordered_map<wchar_t, std::wint_t> bytes_2_unicode_dict_inverse_;
  bool add_bos_ = true;
  bool add_eos_ = false;
};

}  // namespace mllm::models::internlm2