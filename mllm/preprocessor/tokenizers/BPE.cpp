// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

#include "mllm/utils/Common.hpp"
#include "mllm/preprocessor/tokenizers/Unicode.hpp"
#include "mllm/preprocessor/tokenizers/BPE.hpp"

namespace mllm::preprocessor {

bool BPE::initFromSentencePieceJson(const std::string& file_path) {
  std::ifstream f(file_path);
  if (!f.is_open()) {
    MLLM_ERROR("BPE Cannot open file {}", file_path);
    return false;
  }
  auto json_data = json::parse(f);

  if (!json_data.contains("model") || !json_data["model"].contains("vocab") || !json_data["model"].contains("merges")) {
    MLLM_ERROR("BPE initFromSentencePieceJson need sentence piece json, but get {}", file_path);
    return false;
  }

  for (const auto& [key, value] : json_data["model"]["vocab"].items()) {
    auto str = utf8string2WideString(key);
    vocab_.insert({
        str,
        value,
    });
    vocab_inverse_.insert({
        value,
        str,
    });
  }

  for (const auto& add_token : json_data["added_tokens"].items()) {
    auto id = add_token.value()["id"];
    auto content = add_token.value()["content"];
    auto str = utf8string2WideString(content);
    vocab_.insert({
        str,
        id,
    });
    vocab_inverse_.insert({
        id,
        str,
    });
  }

  int64_t cnt = 0;
  for (auto& merge_item : json_data["model"]["merges"]) {
    if (merge_item.is_string()) {
      auto wide_merge_item = utf8string2WideString(merge_item);
      auto blank_pos = wide_merge_item.find(L' ');
      auto first = wide_merge_item.substr(0, blank_pos);
      auto second = wide_merge_item.substr(blank_pos + 1);
      bpe_ranks_.insert({{first, second}, cnt++});
    } else if (merge_item.is_array()) {
      auto first = utf8string2WideString(merge_item[0]);
      auto second = utf8string2WideString(merge_item[1]);
      bpe_ranks_.insert({{first, second}, cnt++});
    }
  }

  return true;
}

std::vector<std::wstring> BPE::_bpe(const std::wstring& token) {
  // TODO check cache

  std::vector<std::wstring> word;
  for (const auto& w : token) word.emplace_back(std::wstring{w});

  auto pairs = _get_pairs(word);
  if (pairs.empty()) return {token};

  while (true) {
    bool has_bigram = false;
    int64_t rank_bigram = std::numeric_limits<int64_t>::max();
    std::pair<std::wstring, std::wstring> bigram;

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
    std::vector<std::wstring> new_word;
    int i = 0;

    while (i < word.size()) {
      // Find the next occurrence of 'first' starting at i
      int j = i;
      while (j < word.size() && word[j] != first) { j++; }

      // Add elements from i to j-1 (if any)
      if (j > i) { new_word.insert(new_word.end(), word.begin() + i, word.begin() + j); }

      // Check if we can merge at position j
      if (j < word.size() - 1 && word[j] == first && word[j + 1] == second) {
        new_word.push_back(first + second);
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

  return word;
}

int64_t BPE::_lookup_vocab(const std::wstring& token) {
  if (vocab_.find(token) != vocab_.end()) {
    return vocab_[token];
  } else {
    MLLM_WARN("Cannot find token: {} in BPE vocab", wideString2Utf8String(token));
    return 0;
  }
}

std::wstring BPE::_lookup_inverse_vocab(int64_t idx) {
  if (vocab_inverse_.find(idx) != vocab_inverse_.end()) {
    return vocab_inverse_[idx];
  } else {
    MLLM_WARN("Cannot find token in BPE vocab. When doing _lookup_inverse_vocab");
    return L"";
  }
}

std::unordered_set<std::pair<std::wstring, std::wstring>, BPEPairHash> BPE::_get_pairs(const std::vector<std::wstring>& word) {
  std::unordered_set<std::pair<std::wstring, std::wstring>, BPEPairHash> pairs;
  if (word.size() < 2) return pairs;
  auto prev_char = word[0];
  for (size_t i = 1; i < word.size(); ++i) {
    pairs.insert({prev_char, word[i]});
    prev_char = word[i];
  }
  return pairs;
}

}  // namespace mllm::preprocessor
