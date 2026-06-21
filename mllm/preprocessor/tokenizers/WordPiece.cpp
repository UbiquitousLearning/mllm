// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "WordPiece.hpp"
#include <fstream>
#include <cctype>
#include <utfcpp/utf8.h>
#include <nlohmann/json.hpp>
#include "mllm/mllm.hpp"

using json = nlohmann::json;

namespace mllm::preprocessor {

bool WordPiece::initFromVocabFile(const std::string& vocab_file, bool do_lower_case, bool strip_accents,
                                  const std::string& unk_token, int max_input_chars_per_word) {
  do_lower_case_ = do_lower_case;
  strip_accents_ = strip_accents;
  unk_token_ = unk_token;
  max_input_chars_per_word_ = max_input_chars_per_word;

  // Load vocabulary from file
  std::ifstream file(vocab_file);
  if (!file.is_open()) { return false; }

  std::string line;
  int64_t idx = 0;
  while (std::getline(file, line)) {
    // Remove trailing newline/carriage return
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) { line.pop_back(); }

    vocab_[line] = idx;
    vocab_inverse_[idx] = line;
    idx++;
  }

  // Find unk token ID
  auto it = vocab_.find(unk_token_);
  if (it != vocab_.end()) {
    unk_token_id_ = it->second;
  } else {
    unk_token_id_ = 0;  // Default to 0 if not found
  }

  // Find pad token ID
  auto pad_it = vocab_.find(pad_token_);
  if (pad_it != vocab_.end()) {
    pad_token_id_ = pad_it->second;
  } else {
    pad_token_id_ = 0;  // Default to 0 if not found
  }

  return true;
}

bool WordPiece::initFromTokenizerJson(const std::string& tokenizer_json_file) {
  std::ifstream file(tokenizer_json_file);
  if (!file.is_open()) { return false; }

  json j;
  file >> j;

  // Parse model vocab
  if (j.contains("model") && j["model"].contains("vocab")) {
    for (auto& [token, idx] : j["model"]["vocab"].items()) {
      vocab_[token] = idx;
      vocab_inverse_[idx] = token;
    }
  }

  // Parse added_tokens (these may override or extend the base vocabulary)
  // Added tokens are special tokens like [PAD], [UNK], [CLS], [SEP], [MASK], etc.
  if (j.contains("added_tokens")) {
    for (const auto& added_token : j["added_tokens"]) {
      if (added_token.contains("id") && added_token.contains("content")) {
        int64_t token_id = added_token["id"];
        std::string token_content = added_token["content"];

        // Add or update the token in vocabulary
        vocab_[token_content] = token_id;
        vocab_inverse_[token_id] = token_content;

        // Store added token information
        AddedTokenInfo info;
        info.id = token_id;
        info.content = token_content;

        // Parse optional fields
        if (added_token.contains("single_word")) { info.single_word = added_token["single_word"].get<bool>(); }
        if (added_token.contains("lstrip")) { info.lstrip = added_token["lstrip"].get<bool>(); }
        if (added_token.contains("rstrip")) { info.rstrip = added_token["rstrip"].get<bool>(); }
        if (added_token.contains("normalized")) { info.normalized = added_token["normalized"].get<bool>(); }
        if (added_token.contains("special")) { info.special = added_token["special"].get<bool>(); }

        added_tokens_[token_content] = info;
      }
    }
  }

  // Parse normalizer settings
  if (j.contains("normalizer")) {
    auto& normalizer = j["normalizer"];
    if (normalizer.contains("lowercase") && !normalizer["lowercase"].is_null()) {
      do_lower_case_ = normalizer["lowercase"].get<bool>();
    }
    if (normalizer.contains("strip_accents") && !normalizer["strip_accents"].is_null()) {
      strip_accents_ = normalizer["strip_accents"].get<bool>();
    }
    if (normalizer.contains("handle_chinese_chars") && !normalizer["handle_chinese_chars"].is_null()) {
      tokenize_chinese_chars_ = normalizer["handle_chinese_chars"].get<bool>();
    }
  }

  // Find unk token
  if (j.contains("model") && j["model"].contains("unk_token")) { unk_token_ = j["model"]["unk_token"]; }

  auto it = vocab_.find(unk_token_);
  if (it != vocab_.end()) {
    unk_token_id_ = it->second;
  } else {
    unk_token_id_ = 0;
  }

  // Find pad token from added_tokens
  // First try to find [PAD] in added_tokens
  bool pad_token_found = false;
  for (const auto& [token_content, token_info] : added_tokens_) {
    if (token_info.content == "[PAD]" || token_content == "[PAD]") {
      pad_token_ = token_info.content;
      pad_token_id_ = token_info.id;
      pad_token_found = true;
      break;
    }
  }

  // If not found in added_tokens, try to find in vocabulary
  if (!pad_token_found) {
    auto pad_it = vocab_.find(pad_token_);
    if (pad_it != vocab_.end()) {
      pad_token_id_ = pad_it->second;
      pad_token_found = true;
    }
  }

  // If still not found, default to 0 (same as unk_token by default)
  if (!pad_token_found) { pad_token_id_ = 0; }

  return true;
}

std::vector<std::string> WordPiece::tokenize(const std::string& text) {
  std::vector<std::string> result;

  // Basic tokenization
  auto basic_tokens = basic_tokenize(text);

  // WordPiece tokenization
  for (const auto& token : basic_tokens) {
    auto wp_tokens = wordpiece_tokenize(token);
    result.insert(result.end(), wp_tokens.begin(), wp_tokens.end());
  }

  return result;
}

int64_t WordPiece::token2id(const std::string& token) {
  auto it = vocab_.find(token);
  return it != vocab_.end() ? it->second : unk_token_id_;
}

std::string WordPiece::id2token(int64_t idx) {
  auto it = vocab_inverse_.find(idx);
  return it != vocab_inverse_.end() ? it->second : unk_token_;
}

std::vector<int64_t> WordPiece::convert_tokens_to_ids(const std::vector<std::string>& tokens) {
  std::vector<int64_t> ids;
  ids.reserve(tokens.size());
  for (const auto& token : tokens) { ids.push_back(token2id(token)); }
  return ids;
}

std::vector<std::string> WordPiece::basic_tokenize(const std::string& text) {
  // Clean text
  std::string cleaned = clean_text(text);

  // Tokenize Chinese characters
  if (tokenize_chinese_chars_) { cleaned = tokenize_chinese_chars(cleaned); }

  // IMPORTANT: Split by special tokens FIRST before whitespace tokenization
  // This handles cases like "[CLS]Hello[SEP]" where special tokens are not space-separated
  auto special_split = split_special_tokens(cleaned);

  std::vector<std::string> result;
  for (const auto& segment : special_split) {
    // If this segment is a special token, keep it as-is (no normalization)
    if (is_special_token(segment)) {
      result.push_back(segment);
      continue;
    }

    // For non-special segments, do whitespace tokenization
    auto tokens = whitespace_tokenize(segment);

    for (auto& token : tokens) {
      // Lowercase if needed
      if (do_lower_case_) { token = lowercase(token); }

      // Strip accents if needed
      if (strip_accents_) { token = strip_accents_fn(token); }

      // Split on punctuation
      auto split_tokens = split_on_punctuation(token);
      result.insert(result.end(), split_tokens.begin(), split_tokens.end());
    }
  }

  return result;
}

std::vector<std::string> WordPiece::wordpiece_tokenize(const std::string& text) {
  std::vector<std::string> output_tokens;

  if (text.empty()) { return output_tokens; }

  // Check if text is too long
  if (static_cast<int>(text.length()) > max_input_chars_per_word_) {
    output_tokens.push_back(unk_token_);
    return output_tokens;
  }

  bool is_bad = false;
  int start = 0;
  std::vector<std::string> sub_tokens;

  while (start < static_cast<int>(text.length())) {
    int end = static_cast<int>(text.length());
    std::string cur_substr;
    bool found = false;

    while (start < end) {
      std::string substr = text.substr(start, end - start);

      // Add prefix if not at start
      if (start > 0) { substr.insert(0, wordpieces_prefix_); }

      // Check if in vocabulary
      if (vocab_.find(substr) != vocab_.end()) {
        cur_substr = substr;
        found = true;
        break;
      }
      end--;
    }

    if (!found) {
      is_bad = true;
      break;
    }

    sub_tokens.push_back(cur_substr);
    start = end;
  }

  if (is_bad) {
    output_tokens.push_back(unk_token_);
  } else {
    output_tokens = sub_tokens;
  }

  return output_tokens;
}

std::string WordPiece::clean_text(const std::string& text) {
  std::string result;

  for (auto it = text.begin(); it != text.end();) {
    uint32_t cp = utf8::next(it, text.end());

    // Skip control characters except whitespace
    if (is_control(cp)) {
      if (cp == 0x09 || cp == 0x0A || cp == 0x0D) {      // tab, newline, carriage return
        utf8::append(0x20, std::back_inserter(result));  // Convert to space
      }
      continue;
    }

    // Keep valid characters
    utf8::append(cp, std::back_inserter(result));
  }

  return result;
}

std::string WordPiece::lowercase(const std::string& text) {
  std::string result;

  for (auto it = text.begin(); it != text.end();) {
    uint32_t cp = utf8::next(it, text.end());
    // Simple ASCII lowercase (for full Unicode support, use ICU library)
    if (cp >= 'A' && cp <= 'Z') { cp = cp + 32; }
    utf8::append(cp, std::back_inserter(result));
  }

  return result;
}

std::string WordPiece::strip_accents_fn(const std::string& text) {
  // Simplified accent stripping
  // For full Unicode normalization, use ICU library
  // This is a placeholder implementation
  return text;
}

std::string WordPiece::tokenize_chinese_chars(const std::string& text) {
  std::string result;

  for (auto it = text.begin(); it != text.end();) {
    uint32_t cp = utf8::next(it, text.end());

    if (is_chinese_char(cp)) {
      // Add spaces around Chinese characters
      utf8::append(0x20, std::back_inserter(result));  // space
      utf8::append(cp, std::back_inserter(result));
      utf8::append(0x20, std::back_inserter(result));  // space
    } else {
      utf8::append(cp, std::back_inserter(result));
    }
  }

  return result;
}

bool WordPiece::is_punctuation(uint32_t cp) {
  // ASCII punctuation
  if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) { return true; }

  // Unicode categories (simplified check)
  // For full support, use ICU library
  return false;
}

bool WordPiece::is_whitespace(uint32_t cp) {
  // Space, tab, newline, carriage return
  if (cp == 0x20 || cp == 0x09 || cp == 0x0A || cp == 0x0D) { return true; }

  // Unicode whitespace (simplified)
  if (cp == 0x00A0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x202F || cp == 0x205F || cp == 0x3000) {
    return true;
  }

  return false;
}

bool WordPiece::is_control(uint32_t cp) {
  // C0 and C1 control characters
  if (cp < 0x20 || (cp >= 0x7F && cp <= 0x9F)) { return true; }

  return false;
}

bool WordPiece::is_chinese_char(uint32_t cp) {
  // CJK Unified Ideographs ranges
  return (cp >= 0x4E00 && cp <= 0x9FFF) ||    // CJK Unified Ideographs
         (cp >= 0x3400 && cp <= 0x4DBF) ||    // CJK Unified Ideographs Extension A
         (cp >= 0x20000 && cp <= 0x2A6DF) ||  // CJK Unified Ideographs Extension B
         (cp >= 0x2A700 && cp <= 0x2B73F) ||  // CJK Unified Ideographs Extension C
         (cp >= 0x2B740 && cp <= 0x2B81F) ||  // CJK Unified Ideographs Extension D
         (cp >= 0x2B820 && cp <= 0x2CEAF) ||  // CJK Unified Ideographs Extension E
         (cp >= 0xF900 && cp <= 0xFAFF) ||    // CJK Compatibility Ideographs
         (cp >= 0x2F800 && cp <= 0x2FA1F);    // CJK Compatibility Ideographs Supplement
}

std::vector<std::string> WordPiece::split_on_punctuation(const std::string& text) {
  std::vector<std::string> result;
  std::string current;

  for (auto it = text.begin(); it != text.end();) {
    auto start_it = it;
    uint32_t cp = utf8::next(it, text.end());

    if (is_punctuation(cp)) {
      // Save current token if any
      if (!current.empty()) {
        result.push_back(current);
        current.clear();
      }

      // Add punctuation as separate token
      std::string punct_str(start_it, it);
      result.push_back(punct_str);
    } else {
      // Accumulate into current token
      std::string char_str(start_it, it);
      current += char_str;
    }
  }

  if (!current.empty()) { result.push_back(current); }

  return result;
}

std::vector<std::string> WordPiece::whitespace_tokenize(const std::string& text) {
  std::vector<std::string> result;
  std::string current;

  for (auto it = text.begin(); it != text.end();) {
    auto start_it = it;
    uint32_t cp = utf8::next(it, text.end());

    if (is_whitespace(cp)) {
      // Save current token if any
      if (!current.empty()) {
        result.push_back(current);
        current.clear();
      }
    } else {
      // Accumulate into current token
      std::string char_str(start_it, it);
      current += char_str;
    }
  }

  if (!current.empty()) { result.push_back(current); }

  return result;
}

std::vector<std::string> WordPiece::split_special_tokens(const std::string& text) {
  if (added_tokens_.empty()) {
    // No special tokens to split, return the whole text
    return {text};
  }

  std::vector<std::string> result;
  size_t pos = 0;

  while (pos < text.length()) {
    // Try to find the earliest special token starting from current position
    size_t earliest_pos = std::string::npos;
    std::string earliest_token;

    for (const auto& [token_content, token_info] : added_tokens_) {
      if (!token_info.special) continue;  // Only split on special tokens

      size_t found_pos = text.find(token_content, pos);
      if (found_pos != std::string::npos) {
        if (earliest_pos == std::string::npos || found_pos < earliest_pos) {
          earliest_pos = found_pos;
          earliest_token = token_content;
        }
      }
    }

    if (earliest_pos == std::string::npos) {
      // No more special tokens found, add remaining text
      if (pos < text.length()) {
        std::string remaining = text.substr(pos);
        if (!remaining.empty()) { result.push_back(remaining); }
      }
      break;
    }

    // Add text before the special token (if any)
    if (earliest_pos > pos) {
      std::string before = text.substr(pos, earliest_pos - pos);
      if (!before.empty()) { result.push_back(before); }
    }

    // Add the special token itself
    result.push_back(earliest_token);

    // Move position forward
    pos = earliest_pos + earliest_token.length();
  }

  return result;
}

bool WordPiece::is_special_token(const std::string& token) const {
  auto it = added_tokens_.find(token);
  return it != added_tokens_.end() && it->second.special;
}

const AddedTokenInfo* WordPiece::get_added_token_info(const std::string& token) const {
  auto it = added_tokens_.find(token);
  return it != added_tokens_.end() ? &it->second : nullptr;
}

}  // namespace mllm::preprocessor
