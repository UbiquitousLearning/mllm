// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <cstdint>
#include "mllm/core/aops/LinearOp.hpp"
#include "mllm/engine/ConfigFile.hpp"

namespace mllm::models::llama {

struct LLaMAConfig : protected ConfigFile {
  LLaMAConfig() = default;

  explicit LLaMAConfig(const std::string& file_path) : ConfigFile(file_path) {
    // Init all
    vocab_size = data()["vocab_size"];
    hidden_size = data()["hidden_size"];
    intermediate_size = data()["intermediate_size"];
    num_hidden_layers = data()["num_hidden_layers"];
    num_attention_heads = data()["num_attention_heads"];
    num_key_value_heads = data()["num_key_value_heads"];
    hidden_act = data()["hidden_act"];
    max_position_embeddings = data()["max_position_embeddings"];
    initializer_range = data()["initializer_range"];
    rms_norm_eps = data()["rms_norm_eps"];
    rope_theta = data()["rope_theta"];
    attention_bias = data()["attention_bias"];
    eos_token_id = data()["eos_token_id"];

    // Handle default values for optional parameters
    if (num_key_value_heads == 0) { num_key_value_heads = num_attention_heads; }

    tie_word_embeddings = data()["tie_word_embeddings"];
    linear_impl_type = aops::str2LinearImplTypes(data()["linear_impl_type"]);
  }

  int32_t vocab_size = 32000;
  int32_t hidden_size = 4096;
  int32_t intermediate_size = 11008;
  int32_t num_hidden_layers = 32;
  int32_t num_attention_heads = 32;
  int32_t num_key_value_heads = 32;  // Defaults to num_attention_heads
  std::string hidden_act = "silu";
  int32_t max_position_embeddings = 2048;
  float initializer_range = 0.02;
  float rms_norm_eps = 1e-6;
  float rope_theta = 10000.0;
  bool attention_bias = false;
  int32_t eos_token_id = 2;

  bool tie_word_embeddings = false;

  aops::LinearImplTypes linear_impl_type = aops::LinearImplTypes::kDefault;
};

}  // namespace mllm::models::llama