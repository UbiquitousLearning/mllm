// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <string>
#include <vector>

#include "mllm/core/aops/LinearOp.hpp"
#include "mllm/engine/ConfigFile.hpp"

namespace mllm::models::qwen3_5 {

struct Qwen3_5Config : protected ConfigFile {
  Qwen3_5Config() = default;

  explicit Qwen3_5Config(const std::string& file_path) : ConfigFile(file_path) {
    // The Qwen3.5 config nests text params under "text_config"
    auto& tc = data().contains("text_config") ? data()["text_config"] : data();

    attention_bias = tc["attention_bias"];
    hidden_size = tc["hidden_size"];
    intermediate_size = tc["intermediate_size"];
    num_attention_heads = tc["num_attention_heads"];
    num_key_value_heads = tc["num_key_value_heads"];
    num_hidden_layers = tc["num_hidden_layers"];
    max_position_embeddings = tc["max_position_embeddings"];
    rms_norm_eps = tc["rms_norm_eps"];
    vocab_size = tc["vocab_size"];
    head_dim = tc["head_dim"];
    tie_word_embeddings = tc["tie_word_embeddings"];

    // Qwen3.5 hybrid attention
    attn_output_gate = tc.value("attn_output_gate", true);
    full_attention_interval = tc.value("full_attention_interval", 4);

    // GDN (Gated Delta Network) parameters
    linear_num_key_heads = tc.value("linear_num_key_heads", 16);
    linear_num_value_heads = tc.value("linear_num_value_heads", 16);
    linear_key_head_dim = tc.value("linear_key_head_dim", 128);
    linear_value_head_dim = tc.value("linear_value_head_dim", 128);
    linear_conv_kernel_dim = tc.value("linear_conv_kernel_dim", 4);

    // RoPE parameters (nested under rope_parameters)
    if (tc.contains("rope_parameters")) {
      auto& rp = tc["rope_parameters"];
      rope_theta = rp.value("rope_theta", 10000000.0f);
      partial_rotary_factor = rp.value("partial_rotary_factor", 0.25f);
    }

    // Layer types: explicit list or computed from full_attention_interval
    if (tc.contains("layer_types")) {
      for (auto& lt : tc["layer_types"]) { layer_types.push_back(lt.get<std::string>()); }
    } else {
      for (int i = 0; i < num_hidden_layers; ++i) {
        if ((i + 1) % full_attention_interval == 0) {
          layer_types.push_back("full_attention");
        } else {
          layer_types.push_back("linear_attention");
        }
      }
    }

    // Token IDs — Qwen3.5 uses different IDs than Qwen3
    if (tc.contains("eos_token_id")) { eos_token_id = tc["eos_token_id"]; }

    if (data().contains("max_cache_length")) { max_cache_length = data()["max_cache_length"]; }
    if (tc.contains("linear_impl_type")) {
      linear_impl_type = aops::str2LinearImplTypes(tc["linear_impl_type"]);
    }
  }

  // Standard transformer params
  bool attention_bias = false;
  int32_t hidden_size = 1024;
  int32_t head_dim = 256;
  int32_t intermediate_size = 3584;
  int32_t num_attention_heads = 8;
  int32_t num_key_value_heads = 2;
  int32_t num_hidden_layers = 24;
  int32_t max_position_embeddings = 262144;
  float rms_norm_eps = 1e-06;
  int32_t vocab_size = 248320;

  // Qwen3.5-specific: hybrid attention
  bool attn_output_gate = true;
  int32_t full_attention_interval = 4;
  std::vector<std::string> layer_types;  // "full_attention" or "linear_attention"

  // Qwen3.5-specific: partial RoPE
  float partial_rotary_factor = 0.25;
  float rope_theta = 10000000.0;
  int32_t rotary_dim() const { return static_cast<int32_t>(head_dim * partial_rotary_factor); }

  // Qwen3.5-specific: GDN (Gated Delta Network) params
  int32_t linear_num_key_heads = 16;
  int32_t linear_num_value_heads = 16;
  int32_t linear_key_head_dim = 128;
  int32_t linear_value_head_dim = 128;
  int32_t linear_conv_kernel_dim = 4;

  // Token IDs
  int64_t eos_token_id = 248044;
  int64_t end_of_text_token_id = 248044;
  int64_t im_start_token_id = 248045;
  int64_t im_end_token_id = 248046;
  int64_t thinking_start_token_id = 248068;
  int64_t thinking_end_token_id = 248069;

  bool tie_word_embeddings = true;
  int32_t max_cache_length = 2048;

  aops::LinearImplTypes linear_impl_type = aops::LinearImplTypes::kDefault;

  // Helpers
  bool isFullAttentionLayer(int layer_idx) const {
    return layer_types[layer_idx] == "full_attention";
  }
  int32_t numFullAttentionLayers() const {
    int32_t count = 0;
    for (auto& lt : layer_types) {
      if (lt == "full_attention") ++count;
    }
    return count;
  }
  int32_t numGDNLayers() const { return num_hidden_layers - numFullAttentionLayers(); }
};

}  // namespace mllm::models::qwen3_5
