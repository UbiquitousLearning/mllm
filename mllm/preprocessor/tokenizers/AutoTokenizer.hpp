// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

// Ths json lib is head only. Include all functionall <nlohmann/json.hpp> will increase the compile
// time. Hence, <nlohmann/json_fwd.hpp> is provided for decrease compile time.
//
// json_fwd.hpp:
// Used for forward declaring the nlohmann::json type, suitable for scenarios where only the type
// needs to be declared, reducing compilation time and dependencies.
//
// json.hpp:
// Contains the full implementation of the JSON library, suitable for scenarios where JSON data
// needs to be manipulated.
#include <nlohmann/json_fwd.hpp>
using json = nlohmann::json;

#include <xxHash/xxhash.h>

#include <utfcpp/utf8.h>

#include "mllm/core/Tensor.hpp"

#include <vector>
#include <string>
#include <memory>
#include <unordered_set>
#include <unordered_map>

namespace mllm::preprocessor {

// split text to tokens.
// > Trie.addSpecial("<|im_start|>")
// > Trie.split("<|im_start|>Hello world!")
//
// will give: ["<|im_start|>","Hello world!"]
class Trie {
  struct TrieNode {
    std::unordered_map<wchar_t, std::unique_ptr<TrieNode>> children;
    bool is_end = false;
  };

 public:
  void add(const std::wstring& word);

  void update(const std::vector<std::wstring>& words);

  // I use FSA to implement the split function.
  std::vector<std::wstring> split(const std::wstring& text);

  bool isSpecialToken(const std::wstring& token);

 private:
  std::unique_ptr<TrieNode> root_ = std::make_unique<TrieNode>();
  std::unordered_set<std::wstring> special_tokens_;
};

class TrieUTF8 {
  using cpts_string_t = std::vector<uint32_t>;

  struct TrieNode {
    std::unordered_map<uint32_t, std::unique_ptr<TrieNode>> children;
    bool is_end = false;
  };

  struct VectorUint32Hash {
    std::size_t operator()(const std::vector<uint32_t>& v) const noexcept {
      if (v.empty()) return 0;
      return static_cast<std::size_t>(XXH64(v.data(), v.size() * sizeof(uint32_t), /*seed=*/0));
    }
  };

 public:
  void add(const std::string& word);

  void update(const std::vector<std::string>& words);

  // I use FSA to implement the split function.
  std::vector<std::string> split(const std::string& text);

  bool isSpecialToken(const std::string& token);

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

 private:
  std::unique_ptr<TrieNode> root_ = std::make_unique<TrieNode>();
  std::unordered_set<cpts_string_t, VectorUint32Hash> special_tokens_;
};

class AutoTokenizer {
 public:
  void addSpecialToken(const std::wstring& special_token);

  virtual std::vector<std::wstring> _tokenize(const std::string& str) = 0;

  virtual std::vector<std::wstring> tokenize(const std::string& str) = 0;

  virtual std::wstring _detokenize(int64_t pos_idx) = 0;

  virtual std::wstring detokenize(int64_t pos_idx) = 0;

  virtual Tensor convert2Ids(const std::vector<std::wstring>& strs) = 0;

 protected:
  Trie special_tokens_trie_;
};

class AutoTokenizerUTF8 {
 public:
  void addSpecialToken(const std::string& special_token);

  virtual std::vector<int64_t> encode(const std::string& str) = 0;

  virtual std::string decode(const std::vector<int64_t>& ids) = 0;

  virtual std::vector<std::string> tokenize(const std::string& str) = 0;

  virtual std::string detokenize(const std::vector<std::string>& tokenized_str) = 0;

 protected:
  TrieUTF8 special_tokens_trie_;
};

}  // namespace mllm::preprocessor
