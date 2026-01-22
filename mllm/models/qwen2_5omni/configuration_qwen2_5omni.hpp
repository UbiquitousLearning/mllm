// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <cstdint>
#include <vector>

#include "mllm/core/aops/LinearOp.hpp"
#include "mllm/engine/ConfigFile.hpp"

namespace mllm::models::qwen2_5omni {

struct Qwen2_5OmniConfig : protected ConfigFile {
  Qwen2_5OmniConfig() = default;

  explicit Qwen2_5OmniConfig(const std::string& file_path) : ConfigFile(file_path) {
    auto& root = data();

    if (root.contains("thinker_config")) {
      auto& thinker_cfg = root["thinker_config"];
      auto& text_cfg = thinker_cfg["text_config"];

      hidden_size = text_cfg["hidden_size"];
      intermediate_size = text_cfg["intermediate_size"];
      num_attention_heads = text_cfg["num_attention_heads"];
      num_key_value_heads = text_cfg["num_key_value_heads"];
      num_hidden_layers = text_cfg["num_hidden_layers"];
      max_position_embeddings = text_cfg["max_position_embeddings"];
      rms_norm_eps = text_cfg["rms_norm_eps"];
      vocab_size = text_cfg["vocab_size"];
      rope_theta = text_cfg["rope_theta"];
      tie_word_embeddings = text_cfg.value("tie_word_embeddings", false);

      if (text_cfg.contains("rope_scaling") && text_cfg["rope_scaling"].contains("mrope_section")) {
        mrope_section = text_cfg["rope_scaling"]["mrope_section"].get<std::vector<int32_t>>();
      }

      bos_token_id = thinker_cfg.value("bos_token_id", bos_token_id);
      eos_token_id = thinker_cfg.value("eos_token_id", eos_token_id);
      pad_token_id = thinker_cfg.value("pad_token_id", pad_token_id);
      image_token_id = thinker_cfg.value("image_token_index", image_token_id);
      audio_token_id = thinker_cfg.value("audio_token_index", audio_token_id);
      video_token_id = thinker_cfg.value("video_token_index", video_token_id);
    } else {
      hidden_size = root["hidden_size"];
      intermediate_size = root["intermediate_size"];
      num_attention_heads = root["num_attention_heads"];
      num_key_value_heads = root["num_key_value_heads"];
      num_hidden_layers = root["num_hidden_layers"];
      max_position_embeddings = root["max_position_embeddings"];
      rms_norm_eps = root["rms_norm_eps"];
      vocab_size = root["vocab_size"];
      rope_theta = root["rope_theta"];
      tie_word_embeddings = root.value("tie_word_embeddings", tie_word_embeddings);
      if (root.contains("mrope_section")) {
        mrope_section = root["mrope_section"].get<std::vector<int32_t>>();
      }
      bos_token_id = root.value("bos_token_id", bos_token_id);
      eos_token_id = root.value("eos_token_id", eos_token_id);
      pad_token_id = root.value("pad_token_id", pad_token_id);
      image_token_id = root.value("image_token_id", image_token_id);
      audio_token_id = root.value("audio_token_id", audio_token_id);
      video_token_id = root.value("video_token_id", video_token_id);
    }

    max_cache_length = root.value("max_cache_length", max_position_embeddings);

    if (root.contains("linear_impl_type")) {
      linear_impl_type = aops::str2LinearImplTypes(root["linear_impl_type"]);
    }
  }

  int32_t hidden_size = 3584;
  int32_t intermediate_size = 18944;
  int32_t num_attention_heads = 28;
  int32_t num_key_value_heads = 4;
  int32_t num_hidden_layers = 28;
  int32_t max_position_embeddings = 32768;
  float rms_norm_eps = 1e-06f;
  int32_t vocab_size = 152064;
  std::vector<int32_t> mrope_section = {16, 24, 24};
  float rope_theta = 1000000.0f;
  bool tie_word_embeddings = false;

  int32_t max_cache_length = 32768;

  int64_t bos_token_id = 151644;
  int64_t eos_token_id = 151645;
  int64_t pad_token_id = 151643;
  int64_t image_token_id = 151655;
  int64_t audio_token_id = 151646;
  int64_t video_token_id = 151656;

  aops::LinearImplTypes linear_impl_type = aops::LinearImplTypes::kDefault;
};

}  // namespace mllm::models::qwen2_5omni
