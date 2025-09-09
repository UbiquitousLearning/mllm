// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <vector>
#include <unordered_map>
#include <regex>

#include "mllm/preprocessor/tokenizers/BPE.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/preprocessor/tokenizers/Unicode.hpp"
#include "mllm/preprocessor/tokenizers/AutoTokenizer.hpp"

namespace mllm::models::llama {

struct LlamaMessage {
  std::string role;  // "user", "system", or "assistant"
  std::string content;
};

class TinyLlamaTokenizer final : public mllm::preprocessor::AutoTokenizer {
 public:
  // Configuration constants matching tokenizer_config.json
  static constexpr int MODEL_MAX_LENGTH = 2048;
  static constexpr const char* BOS_TOKEN = "<s>";
  static constexpr const char* EOS_TOKEN = "</s>";
  static constexpr const char* UNK_TOKEN = "<unk>";
  static constexpr const char* PAD_TOKEN = "</s>";  // Same as EOS in config
  static constexpr const char* TOKENIZER_CLASS = "TinyLlamaTokenizer";

  explicit TinyLlamaTokenizer(const std::string& file_path, bool add_bos = true) : add_bos_(add_bos) {
    preprocessor::initLocal();
    preprocessor::makeBytes2UnicodeMap(bytes_2_unicode_dict_);
    for (auto& kv : bytes_2_unicode_dict_) { bytes_2_unicode_dict_inverse_.insert({kv.second, kv.first}); }
    bpe_.initFromSentencePieceJson(file_path);

    // Add special tokens to trie
    special_tokens_trie_.add(L"<s>");

    // Check if this is a HuggingFace-style tokenizer
    if (file_path.find("hf") != std::string::npos) { hf_flag_ = true; }
  }

  // Chat template based on tokenizer_config.json
  inline std::string applyChatTemplate(const std::vector<LlamaMessage>& messages, bool add_generation_prompt = true) {
    std::string result;
    for (size_t i = 0; i < messages.size(); ++i) {
      const auto& message = messages[i];
      if (message.role == "user") {
        result += "<|user|>" + message.content + "</s>";
      } else if (message.role == "system") {
        result += "<|system|>" + message.content + "</s>";
      } else if (message.role == "assistant") {
        result += "<|assistant|>" + message.content + "</s>";
      }

      // Add generation prompt if this is the last message and flag is set
      if (i == messages.size() - 1 && add_generation_prompt) { result += "<|assistant|>"; }
    }
    return result;
  }

  // Normalize text according to tokenizer.json normalizer config
  std::wstring normalize(const std::wstring& text) {
    std::wstring normalized = text;

    // Step 1: Replace all spaces with ▁ (U+2581)
    std::wstring space_char = L" ";
    std::wstring underline_char = L"▁";

    size_t pos = 0;
    while ((pos = normalized.find(space_char, pos)) != std::wstring::npos) {
      normalized.replace(pos, space_char.length(), underline_char);
      pos += underline_char.length();
    }

    // Step 2: Prepend ▁ at the beginning
    if (!normalized.empty() && normalized[0] != L'▁') { normalized = underline_char + normalized; }

    return normalized;
  }

  std::vector<std::wstring> _tokenize(const std::string& str) override {
    std::vector<std::wstring> ret;
    auto w_string = preprocessor::utf8string2WideString(str);

    // Apply normalization first
    auto normalized_string = normalize(w_string);

    // For normalized text containing Unicode characters like ▁,
    // we should directly use BPE without byte-to-unicode mapping
    // since the tokenizer vocabulary already contains these Unicode tokens
    auto bpe_ts = bpe_._bpe(normalized_string);
    ret.reserve(bpe_ts.size());

    for (const auto& bpe_t : bpe_ts) { ret.push_back(bpe_t); }

    return ret;
  }

  std::vector<std::wstring> tokenize(const std::string& str) override {
    auto tokens = special_tokens_trie_.split(preprocessor::utf8string2WideString(str));
    std::vector<std::wstring> all_tokens;

    // Add BOS token if needed
    if (add_bos_) { all_tokens.emplace_back(L"<s>"); }

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

  std::wstring postprocess(std::wstring& text) {
    // Handle decoder sequence according to tokenizer.json decoder config:
    // 1. Replace ▁ with space
    // 2. Handle byte fallback
    // 3. Fuse tokens
    // 4. Strip leading space

    // Replace ▁ with space (using wide character constants)
    std::wregex underline_regex(L"▁");
    text = std::regex_replace(text, underline_regex, L" ");

    // Handle special tokens
    if (text == L"<0x0A>") return L"\n";
    if (text.empty()) return L"";
    if (text == L"</s>" || text == L"<s>") return L"";

    return text;
  }

  std::wstring _detokenize(int64_t pos_idx) override {
    auto str = bpe_._lookup_inverse_vocab(pos_idx);
    str = postprocess(str);
    return str;
  }

  std::wstring detokenize(int64_t pos_idx) override {
    // Since we're now working directly with Unicode tokens (including ▁),
    // we don't need byte-to-unicode mapping for detokenization
    return _detokenize(pos_idx);
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

  ARGenerationOutputPast convertMessage(const std::vector<LlamaMessage>& messages) {
    // Apply chat template
    auto applied_string = applyChatTemplate(messages, true);

    // Process sequence
    auto sequence_str = tokenize(applied_string);
    std::vector<int64_t> ids;
    ids.reserve(sequence_str.size());
    for (const auto& str : sequence_str) { ids.emplace_back(bpe_._lookup_vocab(str)); }

    // Get sequence Tensor
    Tensor sequence = Tensor::empty({/*batch*/ 1, /*seq*/ (int32_t)ids.size()}, kInt64, kCPU)
                          .setMemType(kNormal)
                          .setName("llama-tokenizer-i0")
                          .alloc();

    auto ptr = sequence.ptr<int64_t>();
    for (size_t i = 0; i < ids.size(); ++i) { ptr[i] = ids[i]; }

    return {
        {"sequence", sequence},
    };
  }

 private:
  bool add_bos_ = true;
  bool hf_flag_ = false;

  // Text processing
  preprocessor::BPE bpe_;
  std::unordered_map<std::wint_t, wchar_t> bytes_2_unicode_dict_;
  std::unordered_map<wchar_t, std::wint_t> bytes_2_unicode_dict_inverse_;
};

}  // namespace mllm::models::llama