// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include "mllm/core/aops/LinearOp.hpp"
#include "mllm/engine/ConfigFile.hpp"

namespace mllm::models::smollm3 {

struct Smollm3Config : protected ConfigFile {
  Smollm3Config() = default;

  explicit Smollm3Config(const std::string& file_path) : ConfigFile(file_path) {
    // Init all fields from actual config.json
    attention_bias = data()["attention_bias"];
    hidden_size = data()["hidden_size"];
    intermediate_size = data()["intermediate_size"];
    num_attention_heads = data()["num_attention_heads"];
    num_key_value_heads = data()["num_key_value_heads"];
    num_hidden_layers = data()["num_hidden_layers"];
    max_position_embeddings = data()["max_position_embeddings"];
    rms_norm_eps = data()["rms_norm_eps"];
    vocab_size = data()["vocab_size"];
    
    // Calculate head_dim based on hidden_size and num_attention_heads
    head_dim = hidden_size / num_attention_heads;

    bos_token_id = data()["bos_token_id"];
    eos_token_id = data()["eos_token_id"];
    pad_token_id = data()["pad_token_id"];
    rope_theta = data()["rope_theta"];

    tie_word_embeddings = data()["tie_word_embeddings"];
    max_cache_length = 2048; // Use reasonable default

    // Read linear_impl_type from config, use default if not present
    if (data().contains("linear_impl_type")) {
        linear_impl_type = aops::str2LinearImplTypes(data()["linear_impl_type"]);
    } else {
        linear_impl_type = aops::LinearImplTypes::kDefault;
    }
  }

  bool attention_bias = false;
  int32_t hidden_size = 2048;
  int32_t head_dim = 128;  // 2048 / 16 = 128
  int32_t intermediate_size = 11008;
  int32_t num_attention_heads = 16;
  int32_t num_key_value_heads = 4;
  int32_t num_hidden_layers = 36;
  int32_t max_position_embeddings = 65536;
  float rms_norm_eps = 1e-06;
  int32_t vocab_size = 128256;

  int64_t bos_token_id = 128000;
  int64_t eos_token_id = 128012;
  int64_t pad_token_id = 128004;
  float rope_theta = 5000000.0;

  bool tie_word_embeddings = true;
  int32_t max_cache_length = 2048;

  aops::LinearImplTypes linear_impl_type = aops::LinearImplTypes::kDefault;
};

}  // namespace mllm::models::smollm3
