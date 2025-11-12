// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "mllm/core/aops/LinearOp.hpp"
#include "mllm/engine/ConfigFile.hpp"

namespace mllm::models::internlm2 {

struct InternLM2Config : protected ConfigFile {
  InternLM2Config() = default;

  explicit InternLM2Config(const std::string& file_path) : ConfigFile(file_path) {
    auto& json = data();

    if (json.contains("bias")) { bias = json["bias"].get<bool>(); }
    if (json.contains("hidden_size")) { hidden_size = json["hidden_size"].get<int32_t>(); }
    if (json.contains("intermediate_size")) { intermediate_size = json["intermediate_size"].get<int32_t>(); }
    if (json.contains("num_hidden_layers")) { num_hidden_layers = json["num_hidden_layers"].get<int32_t>(); }
    if (json.contains("num_attention_heads")) { num_attention_heads = json["num_attention_heads"].get<int32_t>(); }
    if (json.contains("num_key_value_heads")) {
      num_key_value_heads = json["num_key_value_heads"].get<int32_t>();
    } else {
      num_key_value_heads = num_attention_heads;
    }
    if (json.contains("max_position_embeddings")) { max_position_embeddings = json["max_position_embeddings"].get<int32_t>(); }
    if (json.contains("rms_norm_eps")) { rms_norm_eps = json["rms_norm_eps"].get<float>(); }
    if (json.contains("vocab_size")) { vocab_size = json["vocab_size"].get<int32_t>(); }
    if (json.contains("rope_theta")) { rope_theta = json["rope_theta"].get<float>(); }
    if (json.contains("tie_word_embeddings")) { tie_word_embeddings = json["tie_word_embeddings"].get<bool>(); }
    if (json.contains("use_cache")) { use_cache = json["use_cache"].get<bool>(); }
    if (json.contains("pad_token_id")) { pad_token_id = json["pad_token_id"].get<int32_t>(); }
    if (json.contains("bos_token_id")) { bos_token_id = json["bos_token_id"].get<int32_t>(); }
    if (json.contains("eos_token_id")) { eos_token_id = json["eos_token_id"].get<int32_t>(); }
    if (json.contains("initializer_range")) { initializer_range = json["initializer_range"].get<float>(); }

    if (json.contains("rope_scaling")) {
      const auto& scaling = json["rope_scaling"];
      if (scaling.contains("type")) { rope_scaling_type = scaling["type"].get<std::string>(); }
      if (scaling.contains("factor")) { rope_scaling_factor = scaling["factor"].get<float>(); }
    }

    if (json.contains("linear_impl_type")) {
      linear_impl_type = aops::str2LinearImplTypes(json["linear_impl_type"].get<std::string>());
    }

    head_dim = hidden_size / num_attention_heads;
    max_cache_length = max_position_embeddings;
    end_of_text_token_id = static_cast<int32_t>(eos_token_id);
  }

  bool bias = false;
  int32_t hidden_size = 4096;
  int32_t intermediate_size = 11008;
  int32_t num_hidden_layers = 32;
  int32_t num_attention_heads = 32;
  int32_t num_key_value_heads = 32;
  int32_t max_position_embeddings = 2048;
  int32_t max_cache_length = 2048;
  int32_t head_dim = 128;
  int32_t vocab_size = 32000;
  float rms_norm_eps = 1e-6f;
  float rope_theta = 10000.0f;
  float rope_scaling_factor = 1.0f;
  std::string rope_scaling_type;

  float initializer_range = 0.02f;
  bool tie_word_embeddings = false;
  bool use_cache = true;

  int32_t pad_token_id = 0;
  int32_t bos_token_id = 1;
  int32_t eos_token_id = 2;
  int32_t end_of_text_token_id = 2;

  aops::LinearImplTypes linear_impl_type = aops::LinearImplTypes::kDefault;
};

}  // namespace mllm::models::internlm2
