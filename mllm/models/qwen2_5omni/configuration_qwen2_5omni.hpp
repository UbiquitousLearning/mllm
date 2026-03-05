// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "mllm/core/aops/LinearOp.hpp"
#include "mllm/engine/ConfigFile.hpp"

namespace mllm::models::qwen2_5omni {

struct Qwen2_5OmniTalkerConfig {
  Qwen2_5OmniTalkerConfig() = default;

  explicit Qwen2_5OmniTalkerConfig(const nlohmann::json& root) { parse(root); }

  void parse(const nlohmann::json& root) {
    audio_token_id = root.value("audio_token_index", audio_token_id);
    image_token_id = root.value("image_token_index", image_token_id);
    video_token_id = root.value("video_token_index", video_token_id);

    vocab_size = root.value("vocab_size", vocab_size);
    tts_text_start_token_id = root.value("tts_text_start_token_id", tts_text_start_token_id);
    tts_text_end_token_id = root.value("tts_text_end_token_id", tts_text_end_token_id);
    tts_text_pad_token_id = root.value("tts_text_pad_token_id", tts_text_pad_token_id);
    tts_codec_start_token_id = root.value("tts_codec_start_token_id", tts_codec_start_token_id);
    tts_codec_end_token_id = root.value("tts_codec_end_token_id", tts_codec_end_token_id);
    tts_codec_pad_token_id = root.value("tts_codec_pad_token_id", tts_codec_pad_token_id);
    tts_codec_mask_token_id = root.value("tts_codec_mask_token_id", tts_codec_mask_token_id);

    vision_start_token_id = root.value("vision_start_token_id", vision_start_token_id);
    vision_end_token_id = root.value("vision_end_token_id", vision_end_token_id);
    audio_start_token_id = root.value("audio_start_token_id", audio_start_token_id);
    audio_end_token_id = root.value("audio_end_token_id", audio_end_token_id);

    embedding_size = root.value("embedding_size", embedding_size);
    hidden_size = root.value("hidden_size", hidden_size);
    intermediate_size = root.value("intermediate_size", intermediate_size);
    num_hidden_layers = root.value("num_hidden_layers", num_hidden_layers);
    num_attention_heads = root.value("num_attention_heads", num_attention_heads);
    num_key_value_heads = root.value("num_key_value_heads", num_key_value_heads);
    head_dim = root.value("head_dim", head_dim);
    max_position_embeddings = root.value("max_position_embeddings", max_position_embeddings);
    rms_norm_eps = root.value("rms_norm_eps", rms_norm_eps);
    rope_theta = root.value("rope_theta", rope_theta);
    use_sliding_window = root.value("use_sliding_window", use_sliding_window);
    sliding_window = root.value("sliding_window", sliding_window);
    max_window_layers = root.value("max_window_layers", max_window_layers);
    attention_dropout = root.value("attention_dropout", attention_dropout);
    position_id_per_seconds = root.value("position_id_per_seconds", position_id_per_seconds);
    seconds_per_chunk = root.value("seconds_per_chunk", seconds_per_chunk);
    spatial_merge_size = root.value("spatial_merge_size", spatial_merge_size);

    if (root.contains("rope_scaling") && root["rope_scaling"].contains("mrope_section")) {
      mrope_section = root["rope_scaling"]["mrope_section"].get<std::vector<int32_t>>();
    }
  }

  int64_t audio_token_id = 151646;
  int64_t image_token_id = 151655;
  int64_t video_token_id = 151656;

  int32_t vocab_size = 8448;
  int64_t tts_text_start_token_id = 151860;
  int64_t tts_text_end_token_id = 151861;
  int64_t tts_text_pad_token_id = 151859;
  int64_t tts_codec_start_token_id = 8293;
  int64_t tts_codec_end_token_id = 8294;
  int64_t tts_codec_pad_token_id = 8292;
  int64_t tts_codec_mask_token_id = 8296;

  int64_t vision_start_token_id = 151652;
  int64_t vision_end_token_id = 151653;
  int64_t audio_start_token_id = 151647;
  int64_t audio_end_token_id = 151648;

  int32_t embedding_size = 3584;
  int32_t hidden_size = 896;
  int32_t intermediate_size = 18944;
  int32_t num_hidden_layers = 24;
  int32_t num_attention_heads = 12;
  int32_t num_key_value_heads = 4;
  int32_t head_dim = 128;
  int32_t max_position_embeddings = 32768;
  float rms_norm_eps = 1e-06f;
  float rope_theta = 1000000.0f;
  bool use_sliding_window = false;
  int32_t sliding_window = 32768;
  int32_t max_window_layers = 28;
  float attention_dropout = 0.0f;
  int32_t position_id_per_seconds = 25;
  int32_t seconds_per_chunk = 2;
  int32_t spatial_merge_size = 2;
  std::vector<int32_t> mrope_section = {16, 24, 24};
};

struct Qwen2_5OmniDiTConfig {
  Qwen2_5OmniDiTConfig() = default;

  explicit Qwen2_5OmniDiTConfig(const nlohmann::json& root) { parse(root); }

  void parse(const nlohmann::json& root) {
    hidden_size = root.value("dim", hidden_size);
    num_hidden_layers = root.value("depth", num_hidden_layers);
    num_attention_heads = root.value("heads", num_attention_heads);
    ff_mult = root.value("ff_mult", ff_mult);
    emb_dim = root.value("emb_dim", emb_dim);
    head_dim = root.value("head_dim", head_dim);
    repeats = root.value("repeats", repeats);
    num_embeds = root.value("num_embeds", num_embeds);
    mel_dim = root.value("mel_dim", mel_dim);
    dropout = root.value("dropout", dropout);

    max_position_embeddings = root.value("max_position_embeddings", max_position_embeddings);
    block_size = root.value("block_size", block_size);
    if (root.contains("look_ahead_layers")) { look_ahead_layers = root["look_ahead_layers"].get<std::vector<int32_t>>(); }
    if (root.contains("look_backward_layers")) { look_backward_layers = root["look_backward_layers"].get<std::vector<int32_t>>(); }
    rope_theta = root.value("rope_theta", rope_theta);
    rope_type = root.value("rope_type", rope_type);
    if (root.contains("rope_parameters")) {
      const auto& rope_params = root["rope_parameters"];
      rope_theta = rope_params.value("rope_theta", rope_theta);
      rope_type = rope_params.value("rope_type", rope_type);
    }

    enc_emb_dim = root.value("enc_emb_dim", enc_emb_dim);
    enc_dim = root.value("enc_dim", enc_dim);
    if (root.contains("enc_channels")) { enc_channels = root["enc_channels"].get<std::vector<int32_t>>(); }
    if (root.contains("enc_kernel_sizes")) { enc_kernel_sizes = root["enc_kernel_sizes"].get<std::vector<int32_t>>(); }
    if (root.contains("enc_dilations")) { enc_dilations = root["enc_dilations"].get<std::vector<int32_t>>(); }
    enc_attention_channels = root.value("enc_attention_channels", enc_attention_channels);
    enc_res2net_scale = root.value("enc_res2net_scale", enc_res2net_scale);
    enc_se_channels = root.value("enc_se_channels", enc_se_channels);
  }

  int32_t hidden_size = 1024;
  int32_t num_hidden_layers = 22;
  int32_t num_attention_heads = 16;
  int32_t ff_mult = 2;
  int32_t emb_dim = 512;
  int32_t head_dim = 64;
  int32_t max_position_embeddings = 32768;
  int32_t block_size = 24;
  std::vector<int32_t> look_ahead_layers = {10};
  std::vector<int32_t> look_backward_layers = {0, 20};
  int32_t repeats = 2;
  int32_t num_embeds = 8193;
  int32_t mel_dim = 80;
  float dropout = 0.1f;

  int32_t enc_emb_dim = 192;
  int32_t enc_dim = 128;
  std::vector<int32_t> enc_channels = {256, 256, 256, 256, 768};
  std::vector<int32_t> enc_kernel_sizes = {5, 3, 3, 3, 1};
  std::vector<int32_t> enc_dilations = {1, 2, 3, 4, 1};
  int32_t enc_attention_channels = 64;
  int32_t enc_res2net_scale = 2;
  int32_t enc_se_channels = 64;

  float rope_theta = 10000.0f;
  std::string rope_type = "default";
};

struct Qwen2_5OmniBigVGANConfig {
  Qwen2_5OmniBigVGANConfig() = default;

  explicit Qwen2_5OmniBigVGANConfig(const nlohmann::json& root) { parse(root); }

  void parse(const nlohmann::json& root) {
    mel_dim = root.value("mel_dim", mel_dim);
    upsample_initial_channel = root.value("upsample_initial_channel", upsample_initial_channel);
    if (root.contains("resblock_kernel_sizes")) {
      resblock_kernel_sizes = root["resblock_kernel_sizes"].get<std::vector<int32_t>>();
    }
    if (root.contains("resblock_dilation_sizes")) {
      resblock_dilation_sizes = root["resblock_dilation_sizes"].get<std::vector<std::vector<int32_t>>>();
    }
    if (root.contains("upsample_rates")) { upsample_rates = root["upsample_rates"].get<std::vector<int32_t>>(); }
    if (root.contains("upsample_kernel_sizes")) {
      upsample_kernel_sizes = root["upsample_kernel_sizes"].get<std::vector<int32_t>>();
    }
  }

  int32_t mel_dim = 80;
  int32_t upsample_initial_channel = 1536;
  std::vector<int32_t> resblock_kernel_sizes = {3, 7, 11};
  std::vector<std::vector<int32_t>> resblock_dilation_sizes = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
  std::vector<int32_t> upsample_rates = {5, 3, 2, 2, 2, 2};
  std::vector<int32_t> upsample_kernel_sizes = {11, 7, 4, 4, 4, 4};
};

struct Qwen2_5OmniToken2WavConfig {
  Qwen2_5OmniToken2WavConfig() = default;

  explicit Qwen2_5OmniToken2WavConfig(const nlohmann::json& root) { parse(root); }

  void parse(const nlohmann::json& root) {
    if (root.contains("dit_config")) { dit_config.parse(root["dit_config"]); }
    if (root.contains("bigvgan_config")) { bigvgan_config.parse(root["bigvgan_config"]); }
  }

  Qwen2_5OmniDiTConfig dit_config{};
  Qwen2_5OmniBigVGANConfig bigvgan_config{};
};

struct Qwen2_5OmniConfig : protected ConfigFile {
  Qwen2_5OmniConfig() = default;

  explicit Qwen2_5OmniConfig(const std::string& file_path) : ConfigFile(file_path) {
    auto& root = data();
    enable_audio_output = root.value("enable_audio_output", root.value("enable_talker", enable_audio_output));

    if (root.contains("talker_config")) { talker_cfg.parse(root["talker_config"]); }
    if (root.contains("token2wav_config")) { token2wav_cfg.parse(root["token2wav_config"]); }

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

      if (thinker_cfg.contains("vision_config")) {
        auto& vision_cfg = thinker_cfg["vision_config"];
        visual_in_chans = vision_cfg.value("in_channels", vision_cfg.value("in_chans", visual_in_chans));
        visual_hidden_size = vision_cfg.value("hidden_size", vision_cfg.value("embed_dim", visual_hidden_size));
        visual_patch_size = vision_cfg.value("patch_size", vision_cfg.value("spatial_patch_size", visual_patch_size));
        visual_temporal_patch_size = vision_cfg.value("temporal_patch_size", visual_temporal_patch_size);
        visual_spatial_merge_size = vision_cfg.value("spatial_merge_size", visual_spatial_merge_size);
        visual_out_hidden_size = vision_cfg.value("out_hidden_size", visual_out_hidden_size);
        visual_num_heads = vision_cfg.value("num_heads", visual_num_heads);
        visual_depth = vision_cfg.value("depth", visual_depth);
        visual_intermediate_size = vision_cfg.value("intermediate_size", visual_intermediate_size);
        if (vision_cfg.contains("fullatt_block_indexes")) {
          visual_fullatt_block_indexes = vision_cfg["fullatt_block_indexes"].get<std::vector<int32_t>>();
        }
        visual_window_size = vision_cfg.value("window_size", visual_window_size);
      }

      if (thinker_cfg.contains("audio_config")) {
        auto& audio_cfg = thinker_cfg["audio_config"];
        audio_d_model = audio_cfg.value("d_model", audio_d_model);
        audio_num_mel_bins = audio_cfg.value("num_mel_bins", audio_num_mel_bins);
        audio_encoder_layers = audio_cfg.value("encoder_layers", audio_encoder_layers);
        audio_encoder_attention_heads = audio_cfg.value("encoder_attention_heads", audio_encoder_attention_heads);
        audio_encoder_ffn_dim = audio_cfg.value("encoder_ffn_dim", audio_encoder_ffn_dim);
        audio_max_source_positions = audio_cfg.value("max_source_positions", audio_max_source_positions);
        audio_n_window = audio_cfg.value("n_window", audio_n_window);
        audio_output_dim = audio_cfg.value("output_dim", audio_output_dim);
      }

      bos_token_id = thinker_cfg.value("bos_token_id", bos_token_id);
      eos_token_id = thinker_cfg.value("eos_token_id", eos_token_id);
      pad_token_id = thinker_cfg.value("pad_token_id", pad_token_id);
      image_token_id = thinker_cfg.value("image_token_index", image_token_id);
      audio_token_id = thinker_cfg.value("audio_token_index", audio_token_id);
      video_token_id = thinker_cfg.value("video_token_index", video_token_id);
      audio_start_token_id = thinker_cfg.value("audio_start_token_id", audio_start_token_id);
      audio_end_token_id = thinker_cfg.value("audio_end_token_id", audio_end_token_id);
      vision_start_token_id = thinker_cfg.value("vision_start_token_id", vision_start_token_id);
      vision_end_token_id = thinker_cfg.value("vision_end_token_id", vision_end_token_id);
      vision_token_id = thinker_cfg.value("vision_token_id", vision_token_id);
      position_id_per_seconds = thinker_cfg.value("position_id_per_seconds", position_id_per_seconds);
      seconds_per_chunk = thinker_cfg.value("seconds_per_chunk", seconds_per_chunk);
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
      if (root.contains("audio_config")) {
        auto& audio_cfg = root["audio_config"];
        audio_d_model = audio_cfg.value("d_model", audio_d_model);
        audio_num_mel_bins = audio_cfg.value("num_mel_bins", audio_num_mel_bins);
        audio_encoder_layers = audio_cfg.value("encoder_layers", audio_encoder_layers);
        audio_encoder_attention_heads = audio_cfg.value("encoder_attention_heads", audio_encoder_attention_heads);
        audio_encoder_ffn_dim = audio_cfg.value("encoder_ffn_dim", audio_encoder_ffn_dim);
        audio_max_source_positions = audio_cfg.value("max_source_positions", audio_max_source_positions);
        audio_n_window = audio_cfg.value("n_window", audio_n_window);
        audio_output_dim = audio_cfg.value("output_dim", audio_output_dim);
      }
      bos_token_id = root.value("bos_token_id", bos_token_id);
      eos_token_id = root.value("eos_token_id", eos_token_id);
      pad_token_id = root.value("pad_token_id", pad_token_id);
      image_token_id = root.value("image_token_id", image_token_id);
      audio_token_id = root.value("audio_token_id", audio_token_id);
      video_token_id = root.value("video_token_id", video_token_id);
      audio_start_token_id = root.value("audio_start_token_id", audio_start_token_id);
      audio_end_token_id = root.value("audio_end_token_id", audio_end_token_id);
      vision_start_token_id = root.value("vision_start_token_id", vision_start_token_id);
      vision_end_token_id = root.value("vision_end_token_id", vision_end_token_id);
      vision_token_id = root.value("vision_token_id", vision_token_id);
      position_id_per_seconds = root.value("position_id_per_seconds", position_id_per_seconds);
      seconds_per_chunk = root.value("seconds_per_chunk", seconds_per_chunk);
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

  int32_t visual_in_chans = 3;
  int32_t visual_hidden_size = 1280;
  int32_t visual_patch_size = 14;
  int32_t visual_temporal_patch_size = 2;
  int32_t visual_spatial_merge_size = 2;
  int32_t visual_out_hidden_size = 3584;
  int32_t visual_num_heads = 16;
  int32_t visual_depth = 32;
  int32_t visual_intermediate_size = 3420;
  std::vector<int32_t> visual_fullatt_block_indexes = {7, 15, 23, 31};
  int32_t visual_window_size = 112;

  int32_t audio_d_model = 1280;
  int32_t audio_num_mel_bins = 128;
  int32_t audio_encoder_layers = 32;
  int32_t audio_encoder_attention_heads = 20;
  int32_t audio_encoder_ffn_dim = 5120;
  int32_t audio_max_source_positions = 1500;
  int32_t audio_n_window = 100;
  int32_t audio_output_dim = 3584;

  int32_t max_cache_length = 32768;

  int64_t bos_token_id = 151644;
  int64_t eos_token_id = 151645;
  int64_t pad_token_id = 151643;
  int64_t image_token_id = 151655;
  int64_t audio_token_id = 151646;
  int64_t video_token_id = 151656;
  int64_t audio_start_token_id = 151647;
  int64_t audio_end_token_id = 151648;
  int64_t vision_start_token_id = 151652;
  int64_t vision_end_token_id = 151653;
  int64_t vision_token_id = 151654;
  int32_t position_id_per_seconds = 25;
  int32_t seconds_per_chunk = 2;

  bool enable_audio_output = true;
  Qwen2_5OmniTalkerConfig talker_cfg{};
  Qwen2_5OmniToken2WavConfig token2wav_cfg{};

  aops::LinearImplTypes linear_impl_type = aops::LinearImplTypes::kDefault;
};

}  // namespace mllm::models::qwen2_5omni
