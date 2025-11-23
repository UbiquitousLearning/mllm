// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include "mllm/core/aops/LinearOp.hpp"
#include "mllm/engine/ConfigFile.hpp"
#include <vector>

namespace mllm::models::minicpm4 {

struct MiniCPM4Config : protected ConfigFile {
  MiniCPM4Config() = default;

  explicit MiniCPM4Config(const std::string& file_path) : ConfigFile(file_path) {
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

    bos_token_id = data()["bos_token_id"];

    if (data()["eos_token_id"].is_array()) {
      eos_token_ids = data()["eos_token_id"].get<std::vector<int64_t>>();
      eos_token_id = eos_token_ids[0];
    } else {
      eos_token_id = data()["eos_token_id"];
      eos_token_ids.push_back(eos_token_id);
    }

    rope_theta = data()["rope_theta"];

    tie_word_embeddings = data()["tie_word_embeddings"];
    max_cache_length = data()["max_cache_length"];

    // MiniCPM4 specific parameters
    scale_emb = data().value("scale_emb", 1.0);
    dim_model_base = data().value("dim_model_base", 1.0);
    scale_depth = data().value("scale_depth", 1.0);

    // RoPE scaling configuration
    if (data().contains("rope_scaling") && !data()["rope_scaling"].is_null()) {
      auto rope_scaling = data()["rope_scaling"];
      rope_scaling_type = rope_scaling["rope_type"];
      rope_scaling_factor = rope_scaling.value("factor", 1.0);

      if (rope_scaling_type == "longrope") {
        original_max_position_embeddings = rope_scaling["original_max_position_embeddings"];

        // Load short_factor and long_factor arrays
        if (rope_scaling.contains("short_factor")) { short_factor = rope_scaling["short_factor"].get<std::vector<float>>(); }
        if (rope_scaling.contains("long_factor")) { long_factor = rope_scaling["long_factor"].get<std::vector<float>>(); }
      }
    }

    linear_impl_type = aops::str2LinearImplTypes(data()["linear_impl_type"]);
  }

  bool attention_bias = false;
  int32_t hidden_size = 1024;
  int32_t intermediate_size = 4096;
  int32_t num_attention_heads = 16;
  int32_t num_key_value_heads = 2;
  int32_t num_hidden_layers = 24;
  int32_t max_position_embeddings = 32768;
  float rms_norm_eps = 1e-05;
  int32_t vocab_size = 73448;

  int64_t bos_token_id = 1;
  int64_t eos_token_id = 2;
  std::vector<int64_t> eos_token_ids = {2};
  float rope_theta = 10000.0;

  bool tie_word_embeddings = true;
  int32_t max_cache_length = 8192;

  // MiniCPM4 specific parameters
  float scale_emb = 12.0;
  float dim_model_base = 256.0;
  float scale_depth = 1.4;

  // RoPE scaling
  std::string rope_scaling_type = "";
  float rope_scaling_factor = 1.0;
  int32_t original_max_position_embeddings = 32768;
  std::vector<float> short_factor;
  std::vector<float> long_factor;

  aops::LinearImplTypes linear_impl_type = aops::LinearImplTypes::kDefault;
};

}  // namespace mllm::models::minicpm4
