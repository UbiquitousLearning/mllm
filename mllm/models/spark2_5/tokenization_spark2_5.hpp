// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once
#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>
#include "mllm/models/ARGeneration.hpp"
#include "mllm/models/spark2_5/unicode_ranges.hpp"
#include "mllm/preprocessor/StreamingUtf8Decoder.hpp"
#include "mllm/preprocessor/tokenizers/BPE.hpp"
#include "mllm/preprocessor/tokenizers/Unicode.hpp"

namespace mllm::models::spark2_5 {
// The official pipeline first isolates numeric runs and CJK runs, then applies
// its ordered word/punctuation/whitespace alternatives, then isolates digits.
inline std::vector<std::wstring> sparkPieces(const std::wstring& text) {
  auto number = [](wchar_t c) { return (unicodeFlags(c) & 16) != 0; };
  auto cjk = [](wchar_t c) { return (c >= 0x4e00 && c <= 0x9fa5) || (c >= 0x3040 && c <= 0x30ff); };
  std::vector<std::wstring> numeric, blocks, words, result;
  size_t i = 0;
  while (i < text.size()) {
    size_t b = i++;
    if (number(text[b])) {
      while (i < text.size() && i - b < 3 && number(text[i])) ++i;
    } else {
      while (i < text.size() && !number(text[i])) ++i;
    }
    numeric.push_back(text.substr(b, i - b));
  }
  for (const auto& t : numeric) {
    i = 0;
    while (i < t.size()) {
      size_t b = i++;
      bool kind = cjk(t[b]);
      while (i < t.size() && cjk(t[i]) == kind) ++i;
      blocks.push_back(t.substr(b, i - b));
    }
  }
  auto letter = [](wchar_t c) { return (unicodeFlags(c) & 1) != 0; };
  auto mark = [](wchar_t c) { return (unicodeFlags(c) & 2) != 0; };
  auto punct = [](wchar_t c) { return (unicodeFlags(c) & 12) != 0; };
  auto space = [](wchar_t c) { return (unicodeFlags(c) & 64) != 0; };
  auto asciiLetter = [](wchar_t c) { return (c >= L'A' && c <= L'Z') || (c >= L'a' && c <= L'z'); };
  const std::wstring asciiPunct = L"!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
  for (const auto& t : blocks) {
    std::wstring unmatched;
    i = 0;
    while (i < t.size()) {
      size_t b = i, e = i;
      bool matched = true;
      if (asciiPunct.find(t[i]) != std::wstring::npos && i + 1 < t.size() && asciiLetter(t[i + 1])) {
        e = i + 2;
        while (e < t.size() && asciiLetter(t[e])) ++e;
      } else {
        // [^\r\n\p{L}\p{P}\p{S}]? [\p{L}\p{M}]+
        size_t p = i;
        if (t[p] != L'\r' && t[p] != L'\n' && !letter(t[p]) && !punct(t[p])) ++p;
        if (p < t.size() && (letter(t[p]) || mark(t[p]))) {
          e = p + 1;
          while (e < t.size() && (letter(t[e]) || mark(t[e]))) ++e;
        } else if (letter(t[i]) || mark(t[i])) {
          e = i + 1;
          while (e < t.size() && (letter(t[e]) || mark(t[e]))) ++e;
        } else {
          p = i + (t[i] == L' ');
          if (p < t.size() && punct(t[p])) {
            e = p + 1;
            while (e < t.size() && punct(t[e])) ++e;
          } else if (t[i] == L'\r' || t[i] == L'\n')
            e = i + 1;
          else if (space(t[i])) {
            e = i + 1;
            while (e < t.size() && space(t[e])) ++e;
            if (e < t.size() && e - i > 1) --e;  // \s+(?!\S) precedes \s+
          } else {
            e = i + 1;
            matched = false;
          }
        }
      }
      if (!matched)
        unmatched += t.substr(b, e - b);
      else {
        if (!unmatched.empty()) {
          words.push_back(std::move(unmatched));
          unmatched.clear();
        }
        words.push_back(t.substr(b, e - b));
      }
      i = e;
    }
    if (!unmatched.empty()) words.push_back(std::move(unmatched));
  }
  for (const auto& t : words) {
    i = 0;
    while (i < t.size()) {
      size_t b = i++;
      if (!(unicodeFlags(t[b]) & 16))
        while (i < t.size() && !(unicodeFlags(t[i]) & 16)) ++i;
      result.push_back(t.substr(b, i - b));
    }
  }
  return result;
}

struct SparkMessage {
  std::string prompt;
  std::string system;
  bool enable_thinking = true;
};
class SparkTokenizer {
 public:
  explicit SparkTokenizer(const std::string& path) {
    preprocessor::makeBytes2UnicodeMap(bytes_to_unicode_);
    for (auto [b, c] : bytes_to_unicode_) unicode_to_bytes_[c] = b;
    std::ifstream f(path);
    if (!f) throw std::invalid_argument("Cannot read Spark tokenizer");
    const auto j = nlohmann::json::parse(f);
    const auto expected = nlohmann::json::parse(
        R"SPARK({"pre_tokenizer": {"type": "Sequence", "pretokenizers": [{"type": "Split", "pattern": {"Regex": "\\p{N}{1,3}"}, "behavior": "Isolated", "invert": false}, {"type": "Split", "pattern": {"Regex": "[一-龥぀-ゟ゠-ヿ]+"}, "behavior": "Isolated", "invert": false}, {"type": "Split", "pattern": {"Regex": "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+|[\r\n]|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false}, {"type": "Digits", "individual_digits": true}, {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}]}, "normalizer": {"type": "Sequence", "normalizers": []}, "post_processor": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false, "use_regex": true}, "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true}})SPARK");
    for (const auto& [key, value] : expected.items())
      if (j.at(key) != value) throw std::invalid_argument("Unsupported Spark tokenizer pipeline: " + key);
    if (j.at("model").at("type") != "BPE" || j.at("model").value("ignore_merges", false)
        || j.at("model").value("byte_fallback", false))
      throw std::invalid_argument("Unsupported Spark BPE configuration");
    if (!bpe_.initFromSentencePieceJson(path)) throw std::invalid_argument("Invalid Spark BPE vocabulary");
    for (const auto& t : j.at("added_tokens")) {
      const auto content = t.at("content").get<std::string>();
      const int64_t id = t.at("id");
      if (t.value("lstrip", false) || t.value("rstrip", false) || t.value("single_word", false) || t.value("normalized", false))
        throw std::invalid_argument("Unsupported Spark added-token attributes");
      added_.push_back(preprocessor::utf8string2WideString(content));
      added_bytes_[id] = content;
    }
    for (const auto& [id, s] : std::vector<std::pair<int64_t, std::string>>{{0, "<｜start▁of▁sentence｜>"},
                                                                            {1, "<｜end▁of▁sentence｜>"},
                                                                            {3, "<think>"},
                                                                            {4, "</think>"},
                                                                            {130976, "<|Bot|>"}})
      if (!added_bytes_.contains(id) || added_bytes_.at(id) != s)
        throw std::invalid_argument("Incompatible Spark special tokens");
    std::sort(added_.begin(), added_.end(), [](const auto& a, const auto& b) { return a.size() > b.size(); });
  }
  static std::string applyChatTemplate(const SparkMessage& m) {
    if (m.prompt.empty()) throw std::invalid_argument("Spark prompt must not be empty");
    std::string s = "<｜start▁of▁sentence｜><|System|>\nyou are a helpful assistant.";
    if (!m.system.empty()) s += "\n\n" + m.system;
    return s + "<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|User|>" + m.prompt
           + "<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|Bot|>" + (m.enable_thinking ? "<think>" : "</think>");
  }
  std::vector<int64_t> encode(const std::string& text) {
    const auto w = preprocessor::utf8string2WideString(text);
    std::vector<int64_t> ids;
    auto normal = [&](size_t b, size_t e) {
      for (const auto& p : sparkPieces(w.substr(b, e - b))) {
        std::wstring mapped;
        for (unsigned char c : preprocessor::wideString2Utf8String(p)) mapped += bytes_to_unicode_.at(c);
        for (const auto& t : bpe_._bpe(mapped)) ids.push_back(bpe_._lookup_vocab(t));
      }
    };
    size_t b = 0, i = 0;
    while (i < w.size()) {
      const std::wstring* match = nullptr;
      for (const auto& t : added_)
        if (w.compare(i, t.size(), t) == 0) {
          match = &t;
          break;
        }
      if (!match) {
        ++i;
        continue;
      }
      normal(b, i);
      ids.push_back(bpe_._lookup_vocab(*match));
      i += match->size();
      b = i;
    }
    normal(b, w.size());
    return ids;
  }
  std::string detokenizeBytes(int64_t id) {
    if (added_bytes_.contains(id)) return added_bytes_.at(id);
    std::string s;
    for (auto c : bpe_._lookup_inverse_vocab(id)) s += static_cast<char>(unicode_to_bytes_.at(c));
    return s;
  }
  ARGenerationOutputPast convertMessage(const SparkMessage& m) {
    const auto ids = encode(applyChatTemplate(m));
    auto t = Tensor::empty({1, static_cast<int32_t>(ids.size())}, kInt64, kCPU).alloc();
    std::copy(ids.begin(), ids.end(), t.ptr<int64_t>());
    return {{"sequence", t}};
  }

 private:
  preprocessor::BPE bpe_;
  std::unordered_map<std::wint_t, wchar_t> bytes_to_unicode_;
  std::unordered_map<wchar_t, std::wint_t> unicode_to_bytes_;
  std::vector<std::wstring> added_;
  std::unordered_map<int64_t, std::string> added_bytes_;
};
}  // namespace mllm::models::spark2_5
