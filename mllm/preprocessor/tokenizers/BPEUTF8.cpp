// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <utfcpp/utf8.h>
#include <nlohmann/json.hpp>

#include "mllm/utils/Common.hpp"
#include "mllm/preprocessor/tokenizers/BPEUTF8.hpp"

namespace mllm::preprocessor {
bool BPEUTF8::initFromSentencePieceJson(const std::string& file_path) {
  std::ifstream f(file_path);
  if (!f.is_open()) {
    MLLM_ERROR("BPEUTF8 Cannot open file {}", file_path);
    return false;
  }
  auto json_data = nlohmann::json::parse(f);

  if (!json_data.contains("model") || !json_data["model"].contains("vocab") || !json_data["model"].contains("merges")) {
    MLLM_ERROR("BPEUTF8 initFromSentencePieceJson need sentence piece json, but get {}", file_path);
    return false;
  }

  for (const auto& [key, value] : json_data["model"]["vocab"].items()) {
    vocab_.insert({
        utf8String2Cpts(key),
        value,
    });
    vocab_inverse_.insert({
        value,
        utf8String2Cpts(key),
    });
  }

  for (const auto& add_token : json_data["added_tokens"].items()) {
    int64_t id = add_token.value()["id"];
    std::string content = add_token.value()["content"];
    vocab_.insert({
        utf8String2Cpts(content),
        id,
    });
    vocab_inverse_.insert({
        id,
        utf8String2Cpts(content),
    });
  }

  int64_t cnt = 0;
  for (auto& merge_item : json_data["model"]["merges"]) {
    if (merge_item.is_string()) {
      std::string wide_merge_item = merge_item;

      // 0x20 will only represent space in utf8. we can use this unsafe method to speed up.
      auto blank_pos = wide_merge_item.find(' ');
      auto first = wide_merge_item.substr(0, blank_pos);
      auto second = wide_merge_item.substr(blank_pos + 1);
      bpe_ranks_.insert({{utf8String2Cpts(first), utf8String2Cpts(second)}, cnt++});
    } else if (merge_item.is_array()) {
      bpe_ranks_.insert({{utf8String2Cpts(merge_item[0]), utf8String2Cpts(merge_item[1])}, cnt++});
    }
  }

  return true;
}

// ByteLevel BPE
std::vector<std::string> BPEUTF8::_bpe(const std::string& token) {
  // Slice all tokens to word
  std::vector<cpt_string_t> word;
  {
    auto cpts = utf8String2Cpts(token);
    for (auto cpt : cpts) { word.push_back(cpt_string_t{cpt}); }
  }

  auto pairs = _get_pairs(word);
  if (pairs.empty()) return {token};

  while (true) {
    bool has_bigram = false;
    int64_t rank_bigram = std::numeric_limits<int64_t>::max();
    std::pair<cpt_string_t, cpt_string_t> bigram;

    for (const auto& p : pairs) {
      if (bpe_ranks_.count(p)) {
        auto rank = bpe_ranks_.at(p);
        if (rank < rank_bigram) {
          rank_bigram = rank;
          bigram = p;
          has_bigram = true;
        }
      }
    }

    if (!has_bigram) { break; }

    auto [first, second] = bigram;
    std::vector<cpt_string_t> new_word;
    int i = 0;

    while (i < word.size()) {
      // Find the next occurrence of 'first' starting at i
      int j = i;
      while (j < word.size() && word[j] != first) { j++; }

      // Add elements from i to j-1 (if any)
      if (j > i) { new_word.insert(new_word.end(), word.begin() + i, word.begin() + j); }

      // Check if we can merge at position j
      if (j < word.size() - 1 && word[j] == first && word[j + 1] == second) {
        auto __merged = first;
        __merged.insert(std::end(__merged), std::begin(second), std::end(second));
        new_word.push_back(__merged);
        i = j + 2;  // Skip both merged elements
      } else if (j < word.size()) {
        new_word.push_back(word[j]);
        i = j + 1;
      } else {
        i = j;  // j == word.size()
      }
    }

    word = std::move(new_word);
    if (word.size() == 1) {
      break;
    } else {
      pairs = _get_pairs(word);
    }
  }

  std::vector<std::string> ret;
  ret.reserve(word.size());
  for (auto& cpt : word) { ret.push_back(cpts2Utf8String(cpt)); }

  return ret;
}

int64_t BPEUTF8::_lookup_vocab(const std::string& token) {
  auto cpts = utf8String2Cpts(token);
  if (vocab_.find(cpts) != vocab_.end()) {
    return vocab_[cpts];
  } else {
    MLLM_WARN("Cannot find token: {} in BPEUTF8 vocab", token);
    return 0;
  }
}

std::string BPEUTF8::_lookup_inverse_vocab(int64_t idx) {
  if (vocab_inverse_.find(idx) != vocab_inverse_.end()) {
    return cpts2Utf8String(vocab_inverse_[idx]);
  } else {
    MLLM_WARN("Cannot find token in BPEUTF8 vocab. When doing _lookup_inverse_vocab");
    return {};
  }
}

std::unordered_set<std::pair<BPEUTF8::cpt_string_t, BPEUTF8::cpt_string_t>, BPEUTF8PairHash> BPEUTF8::_get_pairs(
    const std::vector<BPEUTF8::cpt_string_t>& word) {
  std::unordered_set<std::pair<cpt_string_t, cpt_string_t>, BPEUTF8PairHash> pairs;
  if (word.size() < 2) return pairs;
  auto prev_char = word[0];
  for (size_t i = 1; i < word.size(); ++i) {
    pairs.insert({prev_char, word[i]});
    prev_char = word[i];
  }
  return pairs;
}
}  // namespace mllm::preprocessor
