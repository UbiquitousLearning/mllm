// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include "mllm/core/aops/LinearOp.hpp"
#include "mllm/engine/ConfigFile.hpp"

namespace mllm::models::minicpmo {

struct MiniCPMOConfig : protected ConfigFile {
  MiniCPMOConfig() = default;

  explicit MiniCPMOConfig(const std::string& file_path) : ConfigFile(file_path) {
    // Vision Config
    vision_hidden_size = data()["vision_hidden_size"];
    vision_intermediate_size = data()["vision_intermediate_size"];
    vision_num_hidden_layers = data()["vision_num_hidden_layers"];
    vision_num_attention_heads = data()["vision_num_attention_heads"];
    vision_num_channels = data()["vision_num_channels"];
    vision_image_size = data()["vision_image_size"];
    vision_patch_size = data()["vision_patch_size"];

    // LLM Config (Qwen2 based)
    hidden_size = data()["hidden_size"];
    intermediate_size = data()["intermediate_size"];
    num_attention_heads = data()["num_attention_heads"];
    num_key_value_heads = data()["num_key_value_heads"];
    num_hidden_layers = data()["num_hidden_layers"];
    max_position_embeddings = data()["max_position_embeddings"];
    rms_norm_eps = data()["rms_norm_eps"];
    vocab_size = data()["vocab_size"];

    // Resampler Config
    query_num = data()["query_num"];

    // Audio Config (Whisper based)
    audio_hidden_size = data()["audio_hidden_size"];
    audio_num_hidden_layers = data()["audio_num_hidden_layers"];
    audio_num_attention_heads = data()["audio_num_attention_heads"];
    audio_max_position_embeddings = data()["audio_max_position_embeddings"];
    audio_chunk_length = data()["audio_chunk_length"];
    audio_pool_step = data()["audio_pool_step"];

    // TTS Config
    tts_llm_dim = data()["tts_llm_dim"];

    // Common Config
    max_cache_length = data()["max_cache_length"];
    eos_token_id = data()["eos_token_id"];
    rope_theta = data()["rope_theta"];
    tie_word_embeddings = data()["tie_word_embeddings"];

    linear_impl_type = aops::str2LinearImplTypes(data()["linear_impl_type"]);
  }

  // Vision Config (SigLIP)
  int32_t vision_hidden_size = 1152;
  int32_t vision_intermediate_size = 4304;
  int32_t vision_num_hidden_layers = 27;
  int32_t vision_num_attention_heads = 16;
  int32_t vision_num_channels = 3;
  int32_t vision_image_size = 980;
  int32_t vision_patch_size = 14;

  // LLM Config (Qwen2.5-7B)
  int32_t hidden_size = 3584;
  int32_t intermediate_size = 18944;
  int32_t num_attention_heads = 28;
  int32_t num_key_value_heads = 4;
  int32_t num_hidden_layers = 28;
  int32_t max_position_embeddings = 32768;
  float rms_norm_eps = 1e-06;
  int32_t vocab_size = 151700;

  // Resampler Config
  int32_t query_num = 64;

  // Audio Config (Whisper)
  int32_t audio_hidden_size = 1024;
  int32_t audio_num_hidden_layers = 24;
  int32_t audio_num_attention_heads = 16;
  int32_t audio_max_position_embeddings = 1500;
  float audio_chunk_length = 1.0;
  int32_t audio_pool_step = 2;

  // TTS Config (按实际添加更改)
  int32_t tts_llm_dim = 3584;

  // Common Config
  int32_t max_cache_length = 8192;
  int64_t eos_token_id = 151645;
  int64_t bos_token_id = 151643;
  float rope_theta = 1000000.0;
  bool tie_word_embeddings = false;

  aops::LinearImplTypes linear_impl_type = aops::LinearImplTypes::kDefault;
};

}  // namespace mllm::models::minicpmo
