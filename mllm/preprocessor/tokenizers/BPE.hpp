// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <nlohmann/json_fwd.hpp>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <functional>

using json = nlohmann::json;

namespace mllm::preprocessor {

struct BPEPairHash {
  std::size_t operator()(const std::pair<std::wstring, std::wstring>& key) const {
    std::size_t h1 = std::hash<std::wstring>{}(key.first + key.second);
    return h1;
  }
};

class BPE {
 public:
  // BPE can accept sentence piece's json foramt.
  bool initFromSentencePieceJson(const std::string& file_path);

  std::vector<std::wstring> _bpe(const std::wstring& token);

  int64_t _lookup_vocab(const std::wstring& token);

  std::wstring _lookup_inverse_vocab(int64_t idx);

 private:
  std::unordered_set<std::pair<std::wstring, std::wstring>, BPEPairHash> _get_pairs(const std::vector<std::wstring>& word);

  std::unordered_map<std::wstring, int64_t> vocab_;
  std::unordered_map<int64_t, std::wstring> vocab_inverse_;
  std::unordered_map<std::pair<std::wstring, std::wstring>, int64_t, BPEPairHash> bpe_ranks_;
};

}  // namespace mllm::preprocessor