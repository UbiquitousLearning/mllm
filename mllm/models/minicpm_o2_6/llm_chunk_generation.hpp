// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/mllm.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/models/minicpm_o2_6/modeling_minicpmo.hpp"
#include "mllm/models/minicpm_o2_6/tokenization_minicpmo.hpp"

namespace mllm::models {

/**
 * @brief Output structure for chunk-based generation
 */
struct ChunkGenerationOutput {
  std::string text;          // Decoded text for this chunk
  bool finished = false;     // Whether generation is complete
  int64_t tokens_generated;  // Total tokens generated so far
};

/**
 * @brief Configuration for chunk-based generation
 */
struct ChunkGenerationConfig {
  int32_t max_new_tokens = 25;
  int32_t chunk_size = 3;  // Tokens to generate per chunk
  float temperature = 1.0f;
  int32_t top_k = 0;
  float top_p = 0.0f;
  bool do_sample = false;
  bool save_first_chunk_hidden_states = false;  // Whether to save hidden states after first chunk, used in MiniCPM-o2.6
  std::vector<int64_t> eos_token_ids;           // Support multiple terminators
};

/**
 * @brief Chunk-based iterator for LLM generation
 *
 * This iterator generates tokens in chunks (multiple tokens per iteration)
 * and handles incomplete UTF-8 sequences to avoid returning malformed text.
 *
 * @note It is specialized for MiniCPM-o currently as it saves hidden states after the first chunk, which is
 * architecture-specific.
 *
 * Key features:
 * 1. Generates `chunk_size` tokens per iteration
 * 2. Buffers incomplete tokens to avoid UTF-8 decoding issues
 * 3. Maintains past_key_values for efficient generation
 * 4. Provides iterator interface compatible with range-based for loops
 *
 * Example usage:
 * ```cpp
 * ChunkGenerationConfig config;
 * config.chunk_size = 3;
 * config.max_new_tokens = 100;
 *
 * auto chunk_gen = ChunkGenerator(model, tokenizer, config);
 * chunk_gen.initialize(input_ids, attention_mask);
 *
 * for (auto& output : chunk_gen) {
 *     std::cout << output.text << std::flush;
 *     if (output.finished) break;
 * }
 * ```
 */
class ChunkGenerator {
 public:
  /**
   * @brief Tokenizer interface for decoding tokens to text
   */
  struct TokenizerInterface {
    virtual ~TokenizerInterface() = default;

    /**
     * @brief Decode token IDs to text string
     * @param token_ids Vector of token IDs to decode
     * @return Decoded text string
     */
    virtual std::string decode(const std::vector<int64_t>& token_ids) = 0;
  };

  ChunkGenerator() = default;

  ChunkGenerator(minicpmo::MiniCPMOForCausalLM* model, minicpmo::MiniCPMOTokenizer* tokenizer,
                 const ChunkGenerationConfig& config)
      : model_(model), tokenizer_(tokenizer), config_(config) {}

  /**
   * @brief Initialize chunk generation
   *
   * @param input_ids Initial input token IDs [B, S]
   * @param attention_mask Optional attention mask (will be auto-generated if not provided)
   */
  ChunkGenerator& initialize(const Tensor& input_ids, const Tensor& position_ids = Tensor::nil(),
                             const Tensor& attention_mask = Tensor::nil()) {
    current_input_["input_ids"] = input_ids;
    if (!position_ids.isNil()) { current_input_["position_ids"] = position_ids; }
    if (!attention_mask.isNil()) { current_input_["attention_mask"] = attention_mask; }

    new_token_count_ = 0;
    first_chunk_ = true;
    finished_ = false;
    left_token_ids_.clear();
    return *this;
  }

  /**
   * @brief Iterator class for chunk-based generation
   */
  class Iterator {
   public:
    using iterator_category = std::input_iterator_tag;
    using value_type = ChunkGenerationOutput;
    using difference_type = std::ptrdiff_t;
    using pointer = const ChunkGenerationOutput*;
    using reference = const ChunkGenerationOutput&;

    Iterator() = default;

    Iterator(ChunkGenerator* generator, bool is_end) : generator_(generator), is_end_(is_end) {
      if (!is_end_) { MLLM_EMPTY_SCOPE }
    }

    reference operator*() const { return current_output_; }
    pointer operator->() const { return &current_output_; }

    Iterator& operator++() {
      advance();
      return *this;
    }

    bool operator!=(const Iterator& other) const { return is_end_ != other.is_end_; }

    bool operator==(const Iterator& other) const { return is_end_ == other.is_end_; }

    bool operator==(std::nullptr_t) const { return is_end_; }

    [[nodiscard]] Tensor getLastHiddenStates() const { return generator_->last_hidden_states_; }

   private:
    void advance() {
      if (is_end_) return;

      bool has_next = generator_->generate_next_chunk(current_output_);
      if (!has_next || current_output_.finished) { is_end_ = true; }
    }

    ChunkGenerator* generator_;
    bool is_end_;
    ChunkGenerationOutput current_output_;
  };

  Iterator begin() { return {this, false}; }
  Iterator end() { return {this, true}; }

 private:
  /**
   * @brief Decode token IDs to UTF-8 string
   * @param token_ids Vector of token IDs to decode
   * @return Decoded UTF-8 string
   */
  std::string decodeTokens(const std::vector<int64_t>& token_ids) {
    std::string result;
    for (auto id : token_ids) {
      std::wstring wide_char = tokenizer_->detokenize(id);
      result += preprocessor::wideString2Utf8String(wide_char);
    }
    return result;
  }

  /**
   * @brief Check if token sequence has incomplete UTF-8 characters
   *
   * Decodes tokens and checks for replacement character (�) at the end,
   * which indicates incomplete UTF-8 sequence.
   *
   * @param token_ids Token IDs to check
   * @return Number of valid tokens (tokens before incomplete sequence)
   */
  int32_t check_incomplete_tokens(const std::vector<int64_t>& token_ids) {
    if (token_ids.empty()) return 0;

    // Decode all tokens first
    std::string decoded_text = decodeTokens(token_ids);
    int32_t end = token_ids.size();

    // Check for UTF-8 replacement character (�) at the end
    // The UTF-8 encoding of U+FFFD is: 0xEF 0xBF 0xBD
    while (!decoded_text.empty() && decoded_text.size() >= 3
           && static_cast<unsigned char>(decoded_text[decoded_text.size() - 3]) == 0xEF
           && static_cast<unsigned char>(decoded_text[decoded_text.size() - 2]) == 0xBF
           && static_cast<unsigned char>(decoded_text[decoded_text.size() - 1]) == 0xBD) {
      end--;
      if (end == 0) break;

      std::vector<int64_t> partial_ids(token_ids.begin(), token_ids.begin() + end);
      decoded_text = decodeTokens(partial_ids);
    }

    return end;
  }

  /**
   * @brief Check if token ID is an EOS (end-of-sequence) token
   */
  [[nodiscard]] bool is_eos_token(int64_t token_id) const {
    for (auto eos_id : config_.eos_token_ids) {
      if (token_id == eos_id) return true;
    }
    return false;
  }

  /**
   * @brief Generate next chunk of tokens
   */
  bool generate_next_chunk(ChunkGenerationOutput& output) {
    if (finished_) {
      output.finished = true;
      return false;
    }

    // Check max token limit
    if (new_token_count_ >= config_.max_new_tokens) {
      finished_ = true;
      output.finished = true;

      // Output any remaining buffered tokens
      if (!left_token_ids_.empty()) { output.text = decodeTokens(left_token_ids_); }
      return false;
    }

    // Determine chunk size for this iteration
    // First chunk may be smaller to reduce TTFT (time to first token)
    int32_t chunk_size = first_chunk_ ? std::min(config_.chunk_size, 3) : config_.chunk_size;

    // Don't exceed max_new_tokens
    chunk_size = std::min(chunk_size, config_.max_new_tokens - new_token_count_);

    // Prepare generation arguments

    // Generate chunk
    std::vector<int64_t> chunk_ids;
    bool eos_encountered = false;

    for (int32_t i = 0; i < chunk_size; ++i) {
      ARGenerationOutputPast forward_output = model_->forward(current_input_, {});

      Tensor logits = forward_output["sequence"];
      int64_t next_token_id;

      // Sample next token based on configuration
      if (config_.do_sample || config_.temperature != 1.0f || config_.top_k > 0 || config_.top_p > 0.0f) {
        if (config_.top_k > 0) {
          next_token_id = model_->sampleTopK(logits, config_.top_k, config_.temperature);
        } else if (config_.top_p > 0.0f) {
          next_token_id = model_->sampleTopP(logits, config_.top_p, config_.temperature);
        } else {
          next_token_id = model_->sampleTemperature(logits, config_.temperature);
        }
      } else {
        next_token_id = model_->sampleGreedy(logits);
      }

      chunk_ids.push_back(next_token_id);

      // Check for EOS token
      if (is_eos_token(next_token_id)) {
        eos_encountered = true;
        break;
      }

      // Update input for next iteration
      current_input_ = std::move(forward_output);
      current_input_["sequence"] = Tensor::empty({1, 1}, kInt64, kCPU).alloc();
      current_input_["sequence"].at<int64_t>({0, 0}) = next_token_id;

      if (first_chunk_ && i == 0 && config_.save_first_chunk_hidden_states) {
        last_hidden_states_ = model_->llm_.llm.getBuffer("last_hidden_states").clone();
      }
      first_chunk_ = false;
    }

    // Update token count
    int32_t input_length = chunk_ids.size();
    new_token_count_ += input_length;

    // Combine with leftover tokens from previous iteration
    std::vector<int64_t> combined_ids;
    combined_ids.insert(combined_ids.end(), left_token_ids_.begin(), left_token_ids_.end());
    combined_ids.insert(combined_ids.end(), chunk_ids.begin(), chunk_ids.end());

    // Check for incomplete UTF-8 sequences
    int32_t valid_end = check_incomplete_tokens(combined_ids);

    // Store incomplete tokens for next iteration
    left_token_ids_.clear();
    if (valid_end < static_cast<int32_t>(combined_ids.size())) {
      left_token_ids_.insert(left_token_ids_.end(), combined_ids.begin() + valid_end, combined_ids.end());
    }

    // Decode valid tokens
    std::vector<int64_t> valid_ids(combined_ids.begin(), combined_ids.begin() + valid_end);
    output.text = valid_end > 0 ? decodeTokens(valid_ids) : "";
    output.tokens_generated = new_token_count_;
    output.finished = eos_encountered || (new_token_count_ >= config_.max_new_tokens);

    if (output.finished) {
      finished_ = true;
      // Include any remaining buffered tokens in final output
      if (!left_token_ids_.empty() && !eos_encountered) { output.text += decodeTokens(left_token_ids_); }
    }

    return true;
  }

 private:
  minicpmo::MiniCPMOForCausalLM* model_ = nullptr;
  minicpmo::MiniCPMOTokenizer* tokenizer_ = nullptr;
  ChunkGenerationConfig config_;

  // Generation state
  ARGenerationOutputPast current_input_;
  std::vector<int64_t> left_token_ids_;  // Buffered incomplete tokens
  int32_t new_token_count_ = 0;
  bool first_chunk_ = true;
  bool finished_ = false;
  Tensor last_hidden_states_;
};

}  // namespace mllm::models
