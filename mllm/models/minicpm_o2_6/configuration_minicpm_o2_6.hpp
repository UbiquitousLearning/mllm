// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include "mllm/core/aops/LinearOp.hpp"
#include "mllm/engine/ConfigFile.hpp"

namespace mllm::models::minicpm_o2_6 {

struct MiniCPMO2_6Config : protected ConfigFile {
  MiniCPMO2_6Config() = default;

  explicit MiniCPMO2_6Config(const std::string& file_path) : ConfigFile(file_path) {
    // Init all
    hidden_size = data()["hidden_size"];
    intermediate_size = data()["intermediate_size"];
    num_attention_heads = data()["num_attention_heads"];
    num_key_value_heads = data()["num_key_value_heads"];
    num_hidden_layers = data()["num_hidden_layers"];
    max_position_embeddings = data()["max_position_embeddings"];
    rms_norm_eps = data()["rms_norm_eps"];
    vocab_size = data()["vocab_size"];
    tie_word_embeddings = data()["tie_word_embeddings"];

    // RoPE parameters
    rope_theta = data()["rope_theta"];

    // Vision-related parameters
    image_size = data()["image_size"];
    patch_size = data()["patch_size"];
    vision_hidden_size = data()["vision_config"]["hidden_size"];
    vision_intermediate_size = data()["vision_config"]["intermediate_size"];
    vision_num_attention_heads = data()["vision_config"]["num_attention_heads"];
    vision_num_hidden_layers = data()["vision_config"]["num_hidden_layers"];
    vision_patch_size = data()["vision_config"]["patch_size"];

    // Audio-related parameters
    audio_d_model = data()["audio_config"]["d_model"];
    audio_num_attention_heads = data()["audio_config"]["decoder_attention_heads"];
    audio_ffn_dim = data()["audio_config"]["decoder_ffn_dim"];
    audio_num_layers = data()["audio_config"]["decoder_layers"];

    // TTS parameters
    tts_llm_dim = data()["tts_config"]["llm_dim"];

    // Other parameters
    bos_token_id = data()["bos_token_id"];
    eos_token_id = data()["eos_token_id"];
    sliding_window = data()["sliding_window"];
    attention_dropout = data()["attention_dropout"];

    linear_impl_type = aops::str2LinearImplTypes(data()["linear_impl_type"]);
  }

  // Text model parameters
  int32_t hidden_size = 3584;
  int32_t intermediate_size = 18944;
  int32_t num_attention_heads = 28;
  int32_t num_key_value_heads = 4;
  int32_t num_hidden_layers = 28;
  int32_t max_position_embeddings = 32768;
  float rms_norm_eps = 1e-06;
  int32_t vocab_size = 151700;
  bool tie_word_embeddings = false;
  float rope_theta = 1000000.0;
  int32_t sliding_window = 131072;
  float attention_dropout = 0.0;
  int32_t bos_token_id = 151643;
  int32_t eos_token_id = 151645;

  // Vision model parameters
  int32_t image_size = 448;
  int32_t patch_size = 14;
  int32_t vision_hidden_size = 1152;
  int32_t vision_intermediate_size = 4304;
  int32_t vision_num_attention_heads = 16;
  int32_t vision_num_hidden_layers = 27;
  int32_t vision_patch_size = 14;

  // Audio model parameters
  int32_t audio_d_model = 1024;
  int32_t audio_num_attention_heads = 16;
  int32_t audio_ffn_dim = 4096;
  int32_t audio_num_layers = 24;

  // TTS parameters
  int32_t tts_llm_dim = 3584;

  // Linear implementation
  aops::LinearImplTypes linear_impl_type = aops::LinearImplTypes::kDefault;
};

}  // namespace mllm::models::minicpm_o2_6