// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/tokenization_llama.py
// and
// ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/tokenization_llama_fast.py

// LlamaTokenizerFast
#pragma once

#include <vector>
#include <unordered_map>

#include "mllm/models/ARGeneration.hpp"
#include "mllm/preprocessor/tokenizers/BPE.hpp"
#include "mllm/preprocessor/tokenizers/Unicode.hpp"
#include "mllm/preprocessor/tokenizers/AutoTokenizer.hpp"

namespace mllm::models::deepseek_ocr {

// Actually is LlamaTokenizer
class DpskOcrTokenizer final : public mllm::preprocessor::AutoTokenizer {
  explicit DpskOcrTokenizer(const std::string& file_path) { preprocessor::initLocal(); }

  std::vector<std::wstring> _tokenize(const std::string& str) override {
    // TODO
    return {};
  }

  std::vector<std::wstring> tokenize(const std::string& str) override {
    // TODO
    return {};
  }

  std::wstring _detokenize(int64_t pos_idx) override {
    // TODO
    return L"";
  }

  std::wstring detokenize(int64_t pos_idx) override {
    // TODO
    return _detokenize(pos_idx);
  }

  Tensor convert2Ids(const std::vector<std::wstring>& strs) override { return Tensor::nil(); }

 private:
  // For text
  preprocessor::BPE bpe_;
};
}  // namespace mllm::models::deepseek_ocr
