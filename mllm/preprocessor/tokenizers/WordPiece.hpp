// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <nlohmann/json_fwd.hpp>

namespace mllm::preprocessor {

/**
 * @brief Information about an added token (from tokenizer.json)
 */
struct AddedTokenInfo {
  int64_t id;
  std::string content;
  bool single_word = false;
  bool lstrip = false;
  bool rstrip = false;
  bool normalized = false;
  bool special = false;
};

/**
 * @brief WordPiece tokenizer implementation for BERT-style models
 *
 * This class implements the WordPiece algorithm used in BERT and similar models.
 * It supports:
 * - Basic tokenization (whitespace + punctuation splitting)
 * - WordPiece subword tokenization
 * - Special tokens handling (including added_tokens from tokenizer.json)
 * - Optional lowercasing and accent stripping
 */
class WordPiece {
 public:
  /**
   * @brief Initialize WordPiece from a vocabulary file
   *
   * @param vocab_file Path to vocab.txt file (one token per line)
   * @param do_lower_case Whether to lowercase input text
   * @param strip_accents Whether to strip accents (auto-determined if not set)
   * @param unk_token Unknown token (default: "[UNK]")
   * @param max_input_chars_per_word Maximum characters per word (default: 200)
   * @return true if initialization successful
   */
  bool initFromVocabFile(const std::string& vocab_file, bool do_lower_case = true, bool strip_accents = false,
                         const std::string& unk_token = "[UNK]", int max_input_chars_per_word = 200);

  /**
   * @brief Initialize WordPiece from HuggingFace tokenizer.json format
   *
   * @param tokenizer_json_file Path to tokenizer.json file
   * @return true if initialization successful
   */
  bool initFromTokenizerJson(const std::string& tokenizer_json_file);

  /**
   * @brief Tokenize a string into WordPiece tokens
   *
   * @param text Input text to tokenize
   * @return Vector of token strings
   */
  std::vector<std::string> tokenize(const std::string& text);

  /**
   * @brief Convert token to ID
   *
   * @param token Token string
   * @return Token ID (unk_token_id if not found)
   */
  int64_t token2id(const std::string& token);

  /**
   * @brief Convert ID to token
   *
   * @param idx Token ID
   * @return Token string (unk_token if not found)
   */
  std::string id2token(int64_t idx);

  /**
   * @brief Convert tokens to IDs
   *
   * @param tokens Vector of token strings
   * @return Vector of token IDs
   */
  std::vector<int64_t> convert_tokens_to_ids(const std::vector<std::string>& tokens);

  /**
   * @brief Get vocabulary size
   */
  [[nodiscard]] size_t vocab_size() const { return vocab_.size(); }

  /**
   * @brief Get unknown token ID
   */
  [[nodiscard]] int64_t unk_token_id() const { return unk_token_id_; }

  /**
   * @brief Get unknown token string
   */
  [[nodiscard]] const std::string& unk_token() const { return unk_token_; }

  /**
   * @brief Get padding token ID
   */
  [[nodiscard]] int64_t pad_token_id() const { return pad_token_id_; }

  /**
   * @brief Get padding token string
   */
  [[nodiscard]] const std::string& pad_token() const { return pad_token_; }

  /**
   * @brief Check if a token is a special token (added_token with special=true)
   *
   * @param token Token content to check
   * @return true if token is marked as special in added_tokens
   */
  [[nodiscard]] bool is_special_token(const std::string& token) const;

  /**
   * @brief Get added token information
   *
   * @param token Token content
   * @return Pointer to AddedTokenInfo if found, nullptr otherwise
   */
  [[nodiscard]] const AddedTokenInfo* get_added_token_info(const std::string& token) const;

 private:
  /**
   * @brief Basic tokenization (whitespace + punctuation)
   */
  std::vector<std::string> basic_tokenize(const std::string& text);

  /**
   * @brief WordPiece tokenization on a single word
   */
  std::vector<std::string> wordpiece_tokenize(const std::string& text);

  /**
   * @brief Clean text (remove control characters, normalize whitespace)
   */
  std::string clean_text(const std::string& text);

  /**
   * @brief Apply lowercasing if enabled
   */
  std::string lowercase(const std::string& text);

  /**
   * @brief Strip accents from text
   */
  std::string strip_accents_fn(const std::string& text);

  /**
   * @brief Tokenize Chinese characters
   */
  std::string tokenize_chinese_chars(const std::string& text);

  /**
   * @brief Check if character is punctuation
   */
  bool is_punctuation(uint32_t cp);

  /**
   * @brief Check if character is whitespace
   */
  bool is_whitespace(uint32_t cp);

  /**
   * @brief Check if character is control
   */
  bool is_control(uint32_t cp);

  /**
   * @brief Check if character is Chinese
   */
  bool is_chinese_char(uint32_t cp);

  /**
   * @brief Split on punctuation
   */
  std::vector<std::string> split_on_punctuation(const std::string& text);

  /**
   * @brief Whitespace tokenize
   */
  std::vector<std::string> whitespace_tokenize(const std::string& text);

  /**
   * @brief Split text by special tokens, preserving them as separate tokens
   *
   * This is crucial for handling inputs like "[CLS]Hello[SEP]" where special
   * tokens are not separated by whitespace.
   */
  std::vector<std::string> split_special_tokens(const std::string& text);

 private:
  // Vocabulary mappings
  std::unordered_map<std::string, int64_t> vocab_;
  std::unordered_map<int64_t, std::string> vocab_inverse_;

  // Added tokens information (from tokenizer.json)
  std::unordered_map<std::string, AddedTokenInfo> added_tokens_;

  // Configuration
  bool do_lower_case_ = false;
  bool strip_accents_ = false;
  bool tokenize_chinese_chars_ = true;
  int max_input_chars_per_word_ = 200;
  std::string wordpieces_prefix_ = "##";

  // Special tokens
  std::string unk_token_ = "[UNK]";
  int64_t unk_token_id_ = 0;
  std::string pad_token_ = "[PAD]";
  int64_t pad_token_id_ = 0;
};

}  // namespace mllm::preprocessor
