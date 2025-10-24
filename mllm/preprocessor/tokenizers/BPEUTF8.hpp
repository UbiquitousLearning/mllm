// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

// TODO
// Documents
// This Byte-Level BPE(BBPE) works as an fully correct byte-level BPE tokenizer.

#include <string>
#include <unordered_map>
#include <unordered_set>
// CPP's support of UTF-8 is weak. We use the utfcpp library to handle UTF-8 strings.
#include <utfcpp/utf8.h>
#include <nlohmann/json_fwd.hpp>

// Remember:
// utfcpp use
// std::string to represent UTF-8 strings.
// std::u16string to represent UTF-16 strings.
// std::u32string to represent UTF-32 strings.

namespace mllm::preprocessor {

struct BPEUTF8PairHash {
  std::size_t operator()(const std::pair<std::string, std::string>& key) const {
    std::size_t h1 = std::hash<std::string>{}(key.first + key.second);
    return h1;
  }
};

class BPEUTF8 {
 public:
  // BPE can accept sentence piece's json foramt.
  bool initFromSentencePieceJson(const std::string& file_path);

  std::vector<std::string> _bpe(const std::string& token);

  int64_t _lookup_vocab(const std::string& token);

  std::string _lookup_inverse_vocab(int64_t idx);

 private:
  std::unordered_set<std::pair<std::string, std::string>, BPEUTF8PairHash> _get_pairs(const std::vector<std::string>& word);

  std::unordered_map<std::string, int64_t> vocab_;
  std::unordered_map<int64_t, std::string> vocab_inverse_;
  std::unordered_map<std::pair<std::string, std::string>, int64_t, BPEUTF8PairHash> bpe_ranks_;
};

}  // namespace mllm::preprocessor
