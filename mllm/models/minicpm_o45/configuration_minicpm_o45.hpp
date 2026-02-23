// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <string>

#include "mllm/core/aops/LinearOp.hpp"
#include "mllm/engine/ConfigFile.hpp"

namespace mllm::models::minicpm_o45 {

struct MiniCPMO45Config : protected ConfigFile {
  MiniCPMO45Config() = default;

  explicit MiniCPMO45Config(const std::string& file_path) : ConfigFile(file_path) {
    auto& cfg = data();
    auto get_or = [&](const std::string& key, auto fallback) {
      using T = decltype(fallback);
      return cfg.contains(key) ? cfg[key].get<T>() : fallback;
    };

    auto vision_cfg = cfg.contains("vision_config") ? cfg["vision_config"] : nlohmann::json::object();
    auto audio_cfg = cfg.contains("audio_config") ? cfg["audio_config"] : nlohmann::json::object();
    auto tts_cfg = cfg.contains("tts_config") ? cfg["tts_config"] : nlohmann::json::object();

    auto get_vision = [&](const std::string& key, auto fallback) {
      using T = decltype(fallback);
      if (vision_cfg.contains(key)) { return vision_cfg[key].get<T>(); }
      if (cfg.contains(key)) { return cfg[key].get<T>(); }
      return fallback;
    };

    auto get_audio = [&](const std::string& key, auto fallback) {
      using T = decltype(fallback);
      if (audio_cfg.contains(key)) { return audio_cfg[key].get<T>(); }
      if (cfg.contains(key)) { return cfg[key].get<T>(); }
      return fallback;
    };

    auto get_tts = [&](const std::string& key, auto fallback) {
      using T = decltype(fallback);
      if (tts_cfg.contains(key)) { return tts_cfg[key].get<T>(); }
      if (cfg.contains(key)) { return cfg[key].get<T>(); }
      return fallback;
    };

    // Vision config.
    vision_hidden_size = get_vision("vision_hidden_size", get_vision("hidden_size", vision_hidden_size));
    vision_intermediate_size = get_vision("vision_intermediate_size", get_vision("intermediate_size", vision_intermediate_size));
    vision_num_hidden_layers = get_vision("vision_num_hidden_layers", get_vision("num_hidden_layers", vision_num_hidden_layers));
    vision_num_attention_heads = get_vision("vision_num_attention_heads", get_vision("num_attention_heads", vision_num_attention_heads));
    vision_num_channels = get_vision("vision_num_channels", get_vision("num_channels", vision_num_channels));
    vision_image_size = get_vision("vision_image_size", get_vision("image_size", vision_image_size));
    vision_patch_size = get_vision("vision_patch_size", get_vision("patch_size", vision_patch_size));

    // LLM config (Qwen3).
    attention_bias = get_or("attention_bias", attention_bias);
    hidden_size = get_or("hidden_size", hidden_size);
    num_attention_heads = get_or("num_attention_heads", num_attention_heads);
    num_key_value_heads = get_or("num_key_value_heads", num_key_value_heads);
    head_dim = get_or("head_dim", hidden_size / std::max(num_attention_heads, 1));
    intermediate_size = get_or("intermediate_size", intermediate_size);
    num_hidden_layers = get_or("num_hidden_layers", num_hidden_layers);
    max_position_embeddings = get_or("max_position_embeddings", max_position_embeddings);
    rms_norm_eps = get_or("rms_norm_eps", rms_norm_eps);
    vocab_size = get_or("vocab_size", vocab_size);

    // Resampler config.
    query_num = get_or("query_num", query_num);

    // Audio config (Whisper encoder).
    audio_hidden_size = get_audio("audio_hidden_size", get_audio("d_model", audio_hidden_size));
    audio_num_hidden_layers = get_audio("audio_num_hidden_layers", get_audio("num_hidden_layers", audio_num_hidden_layers));
    audio_num_attention_heads = get_audio("audio_num_attention_heads", get_audio("encoder_attention_heads", audio_num_attention_heads));
    audio_max_position_embeddings =
        get_audio("audio_max_position_embeddings", get_audio("max_source_positions", audio_max_position_embeddings));
    audio_chunk_length = get_audio("audio_chunk_length", audio_chunk_length);
    audio_pool_step = get_or("audio_pool_step", audio_pool_step);

    // TTS config (token generation stage).
    tts_llm_dim = get_tts("tts_llm_dim", get_tts("llm_dim", tts_llm_dim));
    tts_llm_intermediate_size = get_tts("tts_llm_intermediate_size", get_tts("llm_intermediate_size", tts_llm_intermediate_size));
    tts_hidden_size = get_tts("tts_hidden_size", get_tts("hidden_size", tts_hidden_size));
    tts_intermediate_size = get_tts("tts_intermediate_size", get_tts("intermediate_size", tts_intermediate_size));
    tts_num_attention_heads = get_tts("tts_num_attention_heads", get_tts("num_attention_heads", tts_num_attention_heads));
    tts_num_key_value_heads = get_tts("tts_num_key_value_heads", get_tts("num_key_value_heads", tts_num_key_value_heads));
    tts_num_hidden_layers = get_tts("tts_num_hidden_layers", get_tts("num_hidden_layers", tts_num_hidden_layers));
    tts_max_position_embeddings = get_tts("tts_max_position_embeddings", get_tts("max_position_embeddings", tts_max_position_embeddings));
    tts_num_audio_tokens = get_tts("tts_num_audio_tokens", get_tts("num_audio_tokens", tts_num_audio_tokens));
    tts_num_text_tokens = get_tts("tts_num_text_tokens", get_tts("num_text_tokens", tts_num_text_tokens));
    tts_num_vq = get_tts("tts_num_vq", get_tts("num_vq", tts_num_vq));
    tts_audio_bos_token_id = get_tts("tts_audio_bos_token_id", get_tts("audio_bos_token_id", tts_audio_bos_token_id));
    tts_text_eos_token_id = get_tts("tts_text_eos_token_id", get_tts("text_eos_token_id", tts_text_eos_token_id));
    tts_backbone_vocab_size = tts_cfg.contains("vocab_size") ? tts_cfg["vocab_size"].get<int32_t>() : tts_backbone_vocab_size;
    tts_rms_norm_eps = get_tts("tts_rms_norm_eps", get_tts("rms_norm_eps", tts_rms_norm_eps));
    tts_rope_theta = get_tts("tts_rope_theta", get_tts("rope_theta", tts_rope_theta));
    tts_hidden_act = get_tts("tts_hidden_act", get_tts("hidden_act", tts_hidden_act));
    tts_projector_type = get_tts("tts_projector_type", get_tts("projector_type", tts_projector_type));
    tts_condition_type = get_tts("tts_condition_type", get_tts("condition_type", tts_condition_type));
    tts_normalize_projected_hidden = get_tts("tts_normalize_projected_hidden", get_tts("normalize_projected_hidden", tts_normalize_projected_hidden));

    // Common config.
    max_cache_length = get_or("max_cache_length", max_cache_length);
    eos_token_id = get_or("eos_token_id", eos_token_id);
    bos_token_id = get_or("bos_token_id", bos_token_id);
    rope_theta = get_or("rope_theta", rope_theta);
    tie_word_embeddings = get_or("tie_word_embeddings", tie_word_embeddings);

    linear_impl_type = cfg.contains("linear_impl_type") ? aops::str2LinearImplTypes(cfg["linear_impl_type"]) : linear_impl_type;
  }

  // Vision config (SigLIP).
  int32_t vision_hidden_size = 1152;
  int32_t vision_intermediate_size = 4304;
  int32_t vision_num_hidden_layers = 27;
  int32_t vision_num_attention_heads = 16;
  int32_t vision_num_channels = 3;
  int32_t vision_image_size = 980;
  int32_t vision_patch_size = 14;

  // LLM config (Qwen3-8B).
  bool attention_bias = false;
  int32_t hidden_size = 4096;
  int32_t head_dim = 128;
  int32_t intermediate_size = 12288;
  int32_t num_attention_heads = 32;
  int32_t num_key_value_heads = 8;
  int32_t num_hidden_layers = 36;
  int32_t max_position_embeddings = 40960;
  float rms_norm_eps = 1e-06f;
  int32_t vocab_size = 151748;

  // Resampler config.
  int32_t query_num = 64;

  // Audio config (Whisper-medium).
  int32_t audio_hidden_size = 1024;
  int32_t audio_num_hidden_layers = 24;
  int32_t audio_num_attention_heads = 16;
  int32_t audio_max_position_embeddings = 1500;
  float audio_chunk_length = 1.0f;
  int32_t audio_pool_step = 5;

  // TTS config (MiniCPMTTS in MiniCPM-o-4_5).
  int32_t tts_llm_dim = 4096;
  int32_t tts_llm_intermediate_size = 768;
  int32_t tts_hidden_size = 768;
  int32_t tts_intermediate_size = 3072;
  int32_t tts_num_attention_heads = 12;
  int32_t tts_num_key_value_heads = 12;
  int32_t tts_num_hidden_layers = 20;
  int32_t tts_max_position_embeddings = 4096;
  int32_t tts_num_audio_tokens = 6562;
  int32_t tts_num_text_tokens = 152064;
  int32_t tts_num_vq = 1;
  int32_t tts_audio_bos_token_id = 151687;
  int32_t tts_text_eos_token_id = 151692;
  int32_t tts_backbone_vocab_size = 32000;
  float tts_rms_norm_eps = 1e-06f;
  float tts_rope_theta = 10000.0f;
  std::string tts_hidden_act = "silu";
  std::string tts_projector_type = "mlp";
  std::string tts_condition_type = "hidden_text_merge";
  bool tts_normalize_projected_hidden = true;

  // Common config.
  int32_t max_cache_length = 4096;
  int64_t eos_token_id = 151645;
  int64_t bos_token_id = 151643;
  float rope_theta = 1000000.0f;
  bool tie_word_embeddings = false;

  aops::LinearImplTypes linear_impl_type = aops::LinearImplTypes::kDefault;
};

}  // namespace mllm::models::minicpm_o45
