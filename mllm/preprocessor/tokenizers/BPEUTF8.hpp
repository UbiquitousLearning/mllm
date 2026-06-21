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

// Use XXHash
#include <xxHash/xxhash.h>

// Remember:
// utfcpp use
// std::string to represent UTF-8 strings.
// std::u16string to represent UTF-16 strings.
// std::u32string to represent UTF-32 strings.

namespace mllm::preprocessor {

namespace details {

struct VectorUint32Hash {
  std::size_t operator()(const std::vector<uint32_t>& v) const noexcept {
    if (v.empty()) return 0;
    return static_cast<std::size_t>(XXH64(v.data(), v.size() * sizeof(uint32_t), /*seed=*/0));
  }
};

}  // namespace details

struct BPEUTF8PairHash {
  std::size_t operator()(const std::pair<std::vector<uint32_t>, std::vector<uint32_t>>& key) const noexcept {
    const auto& a = key.first;
    const auto& b = key.second;

    const std::size_t bytes_a = a.size() * sizeof(uint32_t);
    const std::size_t bytes_b = b.size() * sizeof(uint32_t);

    if (bytes_a == 0 && bytes_b == 0) return 0;

    XXH64_state_t* state = XXH64_createState();
    if (!state) return 0;
    XXH64_reset(state, /*seed=*/0);

    if (!a.empty()) XXH64_update(state, a.data(), bytes_a);
    if (!b.empty()) XXH64_update(state, b.data(), bytes_b);

    std::size_t h = static_cast<std::size_t>(XXH64_digest(state));
    XXH64_freeState(state);
    return h;
  }
};

class BPEUTF8 {
 public:
  using cpt_string_t = std::vector<uint32_t>;

  // BPE can accept sentence piece's json foramt.
  bool initFromSentencePieceJson(const std::string& file_path);

  std::vector<std::string> _bpe(const std::string& token);

  int64_t _lookup_vocab(const std::string& token);

  std::string _lookup_inverse_vocab(int64_t idx);

 private:
  inline std::vector<uint32_t> utf8String2Cpts(const std::string& str) {
    std::vector<uint32_t> word32;
    utf8::utf8to32(str.begin(), str.end(), std::back_inserter(word32));
    return word32;
  }

  inline std::string cpts2Utf8String(const std::vector<uint32_t>& cpts) {
    std::string str;
    utf8::utf32to8(cpts.begin(), cpts.end(), std::back_inserter(str));
    return str;
  }

  std::unordered_set<std::pair<cpt_string_t, cpt_string_t>, BPEUTF8PairHash> _get_pairs(const std::vector<cpt_string_t>& word);
  std::unordered_map<cpt_string_t, int64_t, details::VectorUint32Hash> vocab_;
  std::unordered_map<int64_t, cpt_string_t> vocab_inverse_;
  std::unordered_map<std::pair<cpt_string_t, cpt_string_t>, int64_t, BPEUTF8PairHash> bpe_ranks_;
};

}  // namespace mllm::preprocessor
