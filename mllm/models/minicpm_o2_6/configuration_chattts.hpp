// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include "mllm/core/aops/LinearOp.hpp"
#include "mllm/engine/ConfigFile.hpp"

namespace mllm::models::chattts {

struct ChatTTSConfig : protected ConfigFile {
  ChatTTSConfig() = default;

  explicit ChatTTSConfig(const std::string& file_path) : ConfigFile(file_path) {
    // Init all
    llm_dim = data()["llm_dim"];
    hidden_size = data()["hidden_size"];
    intermediate_size = data()["intermediate_size"];
    num_attention_heads = data()["num_attention_heads"];
    num_key_value_heads = data()["num_key_value_heads"];
    num_hidden_layers = data()["num_hidden_layers"];
    max_position_embeddings = data()["max_position_embeddings"];
    num_audio_tokens = data()["num_audio_tokens"];
    num_text_tokens = data()["num_text_tokens"];
    num_mel_bins = data()["num_mel_bins"];
    num_vq = data()["num_vq"];
    use_speaker_embedding = data()["use_speaker_embedding"];
    use_llm_hidden_state = data()["use_llm_hidden_state"];
    spk_emb_token_id = data()["spk_emb_token_id"];
    num_spk_embs = data()["num_spk_embs"];
    audio_bos_token_id = data()["audio_bos_token_id"];
    text_eos_token_id = data()["text_eos_token_id"];
    use_text = data()["use_text"];
    streaming = data()["streaming"];
    streaming_text_chunk_size = data()["streaming_text_chunk_size"];
    streaming_text_reserved_len = data()["streaming_text_reserved_len"];
    streaming_audio_chunk_size = data()["streaming_audio_chunk_size"];
    use_mlp = data()["use_mlp"];
    attention_bias = data()["attention_bias"];
    vocab_size = data()["vocab_size"];
    hidden_act = data()["hidden_act"];
    initializer_range = data()["initializer_range"];
    rms_norm_eps = data()["rms_norm_eps"];
    rope_theta = data()["rope_theta"];
    tie_word_embeddings = data()["tie_word_embeddings"];
    eos_token_id = data()["eos_token_id"];
    linear_impl_type = aops::str2LinearImplTypes(data()["linear_impl_type"]);
  }

  // Model parameters
  int32_t llm_dim = 2560;
  int32_t hidden_size = 768;
  int32_t intermediate_size = 3072;
  int32_t num_attention_heads = 12;
  int32_t num_hidden_layers = 20;
  int32_t max_position_embeddings = 4096;
  int32_t num_audio_tokens = 626;
  int32_t num_text_tokens = 21178;
  int32_t num_mel_bins = 100;
  int32_t num_vq = 4;
  bool use_speaker_embedding = true;
  bool use_llm_hidden_state = false;
  int32_t spk_emb_token_id = 21143;
  int32_t num_spk_embs = 1;
  int32_t audio_bos_token_id = 21132;
  int32_t text_eos_token_id = 21133;
  bool use_text = true;
  bool streaming = true;
  int32_t streaming_text_chunk_size = 10;
  int32_t streaming_text_reserved_len = 300;
  int32_t streaming_audio_chunk_size = 50;
  bool use_mlp = true;
  bool attention_bias = false;
  int32_t vocab_size = 32000;
  int32_t num_key_value_heads = 12;
  std::string hidden_act = "silu";
  float initializer_range = 0.02;
  float rms_norm_eps = 1e-6;
  float rope_theta = 10000.0;
  bool tie_word_embeddings = false;
  int32_t eos_token_id = 2;

  // Linear implementation
  aops::LinearImplTypes linear_impl_type = aops::LinearImplTypes::kDefault;
};

}  // namespace mllm::models::chattts