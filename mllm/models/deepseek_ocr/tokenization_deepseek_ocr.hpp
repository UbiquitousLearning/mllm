// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/tokenization_llama.py
// and
// ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/tokenization_llama_fast.py

// LlamaTokenizerFast
#pragma once

#include <vector>

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
    std::wstring text = preprocessor::utf8string2WideString(str);
    std::replace(text.begin(), text.end(), L' ', SPIECE_UNDERLINE[0]);
    auto tokens = bpe_._bpe(text);

    if (tokens.size() > 1 && tokens[0] == SPIECE_UNDERLINE && special_tokens_trie_.isSpecialToken(tokens[1])) {
      tokens.erase(tokens.begin());
    }
    return tokens;
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
  // For text
  preprocessor::BPE bpe_;
  std::wstring SPIECE_UNDERLINE = L"▁";
};
}  // namespace mllm::models::deepseek_ocr
