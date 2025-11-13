// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>
#include <memory>
#include "mllm/preprocessor/tokenizers/WordPiece.hpp"
#include "mllm/preprocessor/tokenizers/AutoTokenizer.hpp"
#include "mllm/models/ARGeneration.hpp"

namespace mllm::models::chattts {

/**
 * @brief ChatTTS tokenizer based on BERT WordPiece
 *
 * This tokenizer implements BertTokenizerFast functionality for ChatTTS models.
 * It supports:
 * - WordPiece tokenization
 * - Special tokens (CLS, SEP, PAD, MASK, UNK)
 * - Building sequences with special tokens
 * - Token type IDs generation
 */
class ChatTTSTokenizer final : public mllm::preprocessor::AutoTokenizerUTF8 {
 public:
  /**
   * @brief Construct tokenizer from HuggingFace tokenizer.json
   *
   * @param tokenizer_file Path to tokenizer.json
   * @param do_lower_case Whether to lowercase input
   * @param unk_token Unknown token
   * @param sep_token Separator token
   * @param pad_token Padding token
   * @param cls_token Classifier token
   * @param mask_token Mask token
   * @param tokenize_chinese_chars Whether to tokenize Chinese characters
   * @param strip_accents Whether to strip accents
   */
  explicit ChatTTSTokenizer(const std::string& tokenizer_file, bool do_lower_case = true,
                            const std::string& unk_token = "[UNK]", const std::string& sep_token = "[SEP]",
                            const std::string& pad_token = "[PAD]", const std::string& cls_token = "[CLS]",
                            const std::string& mask_token = "[MASK]", bool tokenize_chinese_chars = true,
                            bool strip_accents = false);

  /**
   * @brief Encode text to token IDs
   *
   * @param str Input text
   * @return Vector of token IDs
   */
  std::vector<int64_t> encode(const std::string& str) override;

  /**
   * @brief Decode token IDs to text
   *
   * @param ids Vector of token IDs
   * @return Decoded text
   */
  std::string decode(const std::vector<int64_t>& ids) override;

  /**
   * @brief Tokenize text to tokens
   *
   * @param str Input text
   * @return Vector of token strings
   */
  std::vector<std::string> tokenize(const std::string& str) override;

  /**
   * @brief Detokenize tokens to text
   *
   * @param tokenized_str Vector of token strings
   * @return Detokenized text
   */
  std::string detokenize(const std::vector<std::string>& tokenized_str) override;

  /**
   * @brief Build model inputs with special tokens for single sequence
   *
   * Format: [CLS] X [SEP]
   *
   * @param token_ids_0 Token IDs for first sequence
   * @return Token IDs with special tokens added
   */
  std::vector<int64_t> build_inputs_with_special_tokens(const std::vector<int64_t>& token_ids_0);

  /**
   * @brief Build model inputs with special tokens for sequence pair
   *
   * Format: [CLS] A [SEP] B [SEP]
   *
   * @param token_ids_0 Token IDs for first sequence
   * @param token_ids_1 Token IDs for second sequence
   * @return Token IDs with special tokens added
   */
  std::vector<int64_t> build_inputs_with_special_tokens(const std::vector<int64_t>& token_ids_0,
                                                        const std::vector<int64_t>& token_ids_1);

  /**
   * @brief Create token type IDs from sequences
   *
   * Format for single sequence: [0 0 0 ...]
   * Format for pair: [0 0 0 ... 0 1 1 ... 1]
   *
   * @param token_ids_0 Token IDs for first sequence
   * @param token_ids_1 Optional token IDs for second sequence
   * @return Token type IDs
   */
  std::vector<int64_t> create_token_type_ids_from_sequences(const std::vector<int64_t>& token_ids_0,
                                                            const std::vector<int64_t>& token_ids_1 = {});

  /**
   * @brief Convert text to input tensors
   *
   * @param text Input text
   * @return ARGenerationOutputPast with input tensors
   */
  ARGenerationOutputPast convertText(const std::string& text);

  /**
   * @brief Get vocabulary size
   */
  [[nodiscard]] size_t vocab_size() const { return wordpiece_->vocab_size(); }

  /**
   * @brief Get special token IDs
   */
  [[nodiscard]] int64_t cls_token_id() const { return cls_token_id_; }
  [[nodiscard]] int64_t sep_token_id() const { return sep_token_id_; }
  [[nodiscard]] int64_t pad_token_id() const { return pad_token_id_; }
  [[nodiscard]] int64_t mask_token_id() const { return mask_token_id_; }
  [[nodiscard]] int64_t unk_token_id() const { return wordpiece_->unk_token_id(); }

 private:
  std::unique_ptr<mllm::preprocessor::WordPiece> wordpiece_;

  // Special tokens
  std::string cls_token_;
  std::string sep_token_;
  std::string pad_token_;
  std::string mask_token_;

  // Special token IDs
  int64_t cls_token_id_;
  int64_t sep_token_id_;
  int64_t pad_token_id_;
  int64_t mask_token_id_;

  bool do_lower_case_;
};

}  // namespace mllm::models::chattts
