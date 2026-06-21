// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#include "mllm/preprocessor/tokenizers/AutoTokenizer.hpp"

namespace mllm::preprocessor {

void Trie::add(const std::wstring& word) {
  if (word.empty()) return;
  special_tokens_.insert(word);

  TrieNode* current = root_.get();

  for (const auto& c : word) {
    if (!current->children.count(c)) { current->children[c] = std::make_unique<TrieNode>(); }
    current = current->children[c].get();
  }

  current->is_end = true;
}

void Trie::update(const std::vector<std::wstring>& words) {
  for (const auto& word : words) { add(word); }
}

std::vector<std::wstring> Trie::split(const std::wstring& text) {
  std::map<size_t, TrieNode*> states;
  std::vector<size_t> offsets = {0};
  size_t skip = 0;

  for (size_t current = 0; current < text.size(); ++current) {
    if (skip > current) continue;

    std::unordered_set<size_t> to_remove;
    bool reset = false;

    wchar_t current_char = text[current];

    for (auto& [_start, node] : states) {
      auto start = _start;
      if (node->is_end) {
        // trying to find the longest match
        size_t max_end = current;

        for (auto& [look_start, look_node] : states) {
          if (look_start > start) break;

          size_t lookahead = (look_start < start) ? current + 1 : current;
          size_t end = lookahead;
          TrieNode* ptr = look_node;

          while (lookahead < text.size()) {
            wchar_t ch = text[lookahead];

            if (!ptr->children.count(ch)) break;

            ptr = ptr->children[ch].get();
            lookahead++;

            if (ptr->is_end) {
              start = look_start;
              end = lookahead;
              skip = lookahead;
            }
          }

          if (ptr->is_end && end > max_end) { max_end = end; }
        }
        offsets.push_back(start);
        offsets.push_back(max_end);
        reset = true;
        break;
      }
      if (node->children.count(current_char)) {
        states[start] = node->children[current_char].get();
      } else {
        to_remove.insert(start);
      }
    }
    if (reset) {
      states.clear();
    } else {
      for (auto start : to_remove) { states.erase(start); }
    }
    if (current >= skip && root_->children.count(current_char)) { states[current] = root_->children[current_char].get(); }
  }
  for (auto& [start, node] : states) {
    if (node->is_end) {
      offsets.push_back(start);
      offsets.push_back(text.size());
      break;
    }
  }

  sort(offsets.begin(), offsets.end());
  std::vector<std::wstring> result;
  for (size_t i = 1; i < offsets.size(); ++i) {
    if (offsets[i - 1] != offsets[i]) { result.push_back(text.substr(offsets[i - 1], offsets[i] - offsets[i - 1])); }
  }
  if (offsets[offsets.size() - 1] != text.size()) { result.push_back(text.substr(offsets[offsets.size() - 1])); }
  return result;
}

bool Trie::isSpecialToken(const std::wstring& token) { return special_tokens_.count(token); }

void TrieUTF8::add(const std::string& word_utf8) {
  auto word = utf8String2Cpts(word_utf8);
  if (word.empty()) return;
  special_tokens_.insert(word);

  TrieNode* current = root_.get();

  for (const auto& c : word) {
    if (!current->children.count(c)) { current->children[c] = std::make_unique<TrieNode>(); }
    current = current->children[c].get();
  }

  current->is_end = true;
}

void TrieUTF8::update(const std::vector<std::string>& words) {
  for (const auto& word : words) { add(word); }
}

// I use FSA to implement the split function.
std::vector<std::string> TrieUTF8::split(const std::string& text_utf8) {
  auto text = utf8String2Cpts(text_utf8);

  std::map<size_t, TrieNode*> states;
  std::vector<size_t> offsets = {0};
  size_t skip = 0;

  for (size_t current = 0; current < text.size(); ++current) {
    if (skip > current) continue;

    std::unordered_set<size_t> to_remove;
    bool reset = false;

    wchar_t current_char = text[current];

    for (auto& [_start, node] : states) {
      auto start = _start;
      if (node->is_end) {
        // trying to find the longest match
        size_t max_end = current;

        for (auto& [look_start, look_node] : states) {
          if (look_start > start) break;

          size_t lookahead = (look_start < start) ? current + 1 : current;
          size_t end = lookahead;
          TrieNode* ptr = look_node;

          while (lookahead < text.size()) {
            wchar_t ch = text[lookahead];

            if (!ptr->children.count(ch)) break;

            ptr = ptr->children[ch].get();
            lookahead++;

            if (ptr->is_end) {
              start = look_start;
              end = lookahead;
              skip = lookahead;
            }
          }

          if (ptr->is_end && end > max_end) { max_end = end; }
        }
        offsets.push_back(start);
        offsets.push_back(max_end);
        reset = true;
        break;
      }
      if (node->children.count(current_char)) {
        states[start] = node->children[current_char].get();
      } else {
        to_remove.insert(start);
      }
    }
    if (reset) {
      states.clear();
    } else {
      for (auto start : to_remove) { states.erase(start); }
    }
    if (current >= skip && root_->children.count(current_char)) { states[current] = root_->children[current_char].get(); }
  }
  for (auto& [start, node] : states) {
    if (node->is_end) {
      offsets.push_back(start);
      offsets.push_back(text.size());
      break;
    }
  }

  sort(offsets.begin(), offsets.end());
  std::vector<std::string> result;

  for (size_t i = 1; i < offsets.size(); ++i) {
    if (offsets[i - 1] != offsets[i]) {
      auto cpts_str = cpts_string_t{};
      for (int __idx = offsets[i - 1]; __idx < offsets[i]; __idx++) { cpts_str.push_back(text[__idx]); }
      result.push_back(cpts2Utf8String(cpts_str));
    }
  }
  if (offsets[offsets.size() - 1] != text.size()) {
    auto cpts_str = cpts_string_t{};
    for (int __idx = offsets[offsets.size() - 1]; __idx < text.size(); __idx++) { cpts_str.push_back(text[__idx]); }
    result.push_back(cpts2Utf8String(cpts_str));
  }

  return result;
}

bool TrieUTF8::isSpecialToken(const std::string& token) { return special_tokens_.count(utf8String2Cpts(token)); }

void AutoTokenizer::addSpecialToken(const std::wstring& special_token) { special_tokens_trie_.add(special_token); }

}  // namespace mllm::preprocessor
