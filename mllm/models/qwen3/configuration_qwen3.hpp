// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include "mllm/core/aops/LinearOp.hpp"
#include "mllm/engine/ConfigFile.hpp"

namespace mllm::models::qwen3 {

struct Qwen3Config : protected ConfigFile {
  Qwen3Config() = default;

  explicit Qwen3Config(const std::string& file_path) : ConfigFile(file_path) {
    // Init all
    attention_bias = data()["attention_bias"];
    hidden_size = data()["hidden_size"];
    intermediate_size = data()["intermediate_size"];
    num_attention_heads = data()["num_attention_heads"];
    num_key_value_heads = data()["num_key_value_heads"];
    num_hidden_layers = data()["num_hidden_layers"];
    max_position_embeddings = data()["max_position_embeddings"];
    rms_norm_eps = data()["rms_norm_eps"];
    vocab_size = data()["vocab_size"];
    head_dim = data()["head_dim"];

    bos_token_id = data()["bos_token_id"];
    eos_token_id = data()["eos_token_id"];
    rope_theta = data()["rope_theta"];

    tie_word_embeddings = data()["tie_word_embeddings"];
    max_cache_length = data()["max_cache_length"];

    linear_impl_type = aops::str2LinearImplTypes(data()["linear_impl_type"]);
  }

  bool attention_bias = false;
  int32_t hidden_size = 1024;
  int32_t head_dim = 128;
  int32_t intermediate_size = 3072;
  int32_t num_attention_heads = 16;
  int32_t num_key_value_heads = 8;
  int32_t num_hidden_layers = 28;
  int32_t max_position_embeddings = 40960;
  float rms_norm_eps = 1e-06;
  int32_t vocab_size = 151936;

  int64_t bos_token_id = 151643;
  int64_t eos_token_id = 151645;
  float rope_theta = 1000000.0;

  bool tie_word_embeddings = true;
  int32_t max_cache_length = 2048;
  int32_t end_of_text_token_id = 151645;
  int32_t thinking_start_token_id = 151667;
  int32_t thinking_end_token_id = 151668;

  aops::LinearImplTypes linear_impl_type = aops::LinearImplTypes::kDefault;
};

}  // namespace mllm::models::qwen3
