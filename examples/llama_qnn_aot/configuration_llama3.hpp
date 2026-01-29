// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <cstdint>
#include "mllm/core/aops/LinearOp.hpp"
#include "mllm/engine/ConfigFile.hpp"

namespace mllm::models::llama3 {

/**
 * @brief Configuration for Llama 3.x models used in QNN AOT compilation.
 *
 * This configuration is designed to support Llama 3.2 3B Instruct and similar models.
 * It includes all necessary fields for QNN AOT compilation, including:
 * - head_dim: Dimension of each attention head
 * - max_cache_length: Maximum KV cache length
 * - end_of_text_token_id: Token ID for end of text
 */
struct Llama3Config : protected ConfigFile {
  Llama3Config() = default;

  explicit Llama3Config(const std::string& file_path) : ConfigFile(file_path) {
    // Init all
    vocab_size = data()["vocab_size"];
    hidden_size = data()["hidden_size"];
    intermediate_size = data()["intermediate_size"];
    num_hidden_layers = data()["num_hidden_layers"];
    num_attention_heads = data()["num_attention_heads"];
    num_key_value_heads = data()["num_key_value_heads"];
    hidden_act = data()["hidden_act"];
    max_position_embeddings = data()["max_position_embeddings"];
    rms_norm_eps = data()["rms_norm_eps"];
    rope_theta = data()["rope_theta"];
    attention_bias = data()["attention_bias"];

    // Handle head_dim - compute from hidden_size/num_attention_heads if not provided
    if (data().contains("head_dim")) {
      head_dim = data()["head_dim"];
    } else {
      head_dim = hidden_size / num_attention_heads;
    }

    // Handle default values for optional parameters
    if (num_key_value_heads == 0) { num_key_value_heads = num_attention_heads; }

    // Token IDs
    bos_token_id = data()["bos_token_id"];
    eos_token_id = data()["eos_token_id"];

    // End of text token - use eos_token_id as default
    if (data().contains("end_of_text_token_id")) {
      end_of_text_token_id = data()["end_of_text_token_id"];
    } else {
      end_of_text_token_id = eos_token_id;
    }

    tie_word_embeddings = data()["tie_word_embeddings"];
    max_cache_length = data()["max_cache_length"];

    linear_impl_type = aops::str2LinearImplTypes(data()["linear_impl_type"]);
  }

  // Model architecture parameters
  int32_t vocab_size = 128256;               // Llama 3.2 vocabulary size
  int32_t hidden_size = 3072;                // Llama 3.2 3B hidden size
  int32_t head_dim = 128;                    // Head dimension
  int32_t intermediate_size = 8192;          // FFN intermediate size
  int32_t num_hidden_layers = 28;            // Number of transformer layers
  int32_t num_attention_heads = 24;          // Number of attention heads
  int32_t num_key_value_heads = 8;           // Number of KV heads (GQA)
  std::string hidden_act = "silu";           // Activation function
  int32_t max_position_embeddings = 131072;  // Max sequence length

  // Normalization and RoPE
  float rms_norm_eps = 1e-5;    // RMSNorm epsilon
  float rope_theta = 500000.0;  // RoPE base frequency

  // Attention
  bool attention_bias = false;  // Whether to use bias in attention

  // Token IDs
  int64_t bos_token_id = 128000;          // Begin of sequence token
  int64_t eos_token_id = 128009;          // End of sequence token
  int32_t end_of_text_token_id = 128009;  // End of text token for generation

  // Word embedding
  bool tie_word_embeddings = true;  // Tie input/output embeddings

  // Cache configuration
  int32_t max_cache_length = 2048;  // Maximum KV cache length

  // Linear implementation type for quantization
  aops::LinearImplTypes linear_impl_type = aops::LinearImplTypes::kDefault;
};

}  // namespace mllm::models::llama3
