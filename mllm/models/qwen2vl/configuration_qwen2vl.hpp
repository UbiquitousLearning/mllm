// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include "mllm/core/aops/LinearOp.hpp"
#include "mllm/engine/ConfigFile.hpp"

namespace mllm::models::qwen2vl {

struct Qwen2VLConfig : protected ConfigFile {
  Qwen2VLConfig() = default;

  explicit Qwen2VLConfig(const std::string& file_path) : ConfigFile(file_path) {
    // Init all
    visual_in_chans = data()["visual_in_chans"];
    visual_embed_dim = data()["visual_embed_dim"];
    visual_patch_size = data()["visual_patch_size"];
    visual_temporal_patch_size = data()["visual_temporal_patch_size"];
    visual_spatial_merge_size = data()["visual_spatial_merge_size"];
    visual_mlp_ratio = data()["visual_mlp_ratio"];
    visual_num_heads = data()["visual_num_heads"];
    visual_depth = data()["visual_depth"];

    hidden_size = data()["hidden_size"];
    intermediate_size = data()["intermediate_size"];
    num_attention_heads = data()["num_attention_heads"];
    num_key_value_heads = data()["num_key_value_heads"];
    num_hidden_layers = data()["num_hidden_layers"];
    max_position_embeddings = data()["max_position_embeddings"];
    rms_norm_eps = data()["rms_norm_eps"];
    vocab_size = data()["vocab_size"];

    max_cache_length = data()["max_cache_length"];
    mrope_section = data()["mrope_section"].get<std::vector<int32_t>>();
    vision_token_id = data()["vision_token_id"];
    eos_token_id = data()["eos_token_id"];
    end_of_text_token_id = data()["end_of_text_token_id"];
    rope_theta = data()["rope_theta"];

    tie_word_embeddings = data()["tie_word_embeddings"];

    linear_impl_type = aops::str2LinearImplTypes(data()["linear_impl_type"]);
  }

  int32_t visual_in_chans = 3;
  int32_t visual_embed_dim = 1280;
  int32_t visual_patch_size = 14;
  int32_t visual_temporal_patch_size = 2;
  int32_t visual_spatial_merge_size = 2;
  int32_t visual_mlp_ratio = 4;
  int32_t visual_num_heads = 16;
  int32_t visual_depth = 32;

  int32_t hidden_size = 1536;
  int32_t intermediate_size = 8960;
  int32_t num_attention_heads = 12;
  int32_t num_key_value_heads = 2;
  int32_t num_hidden_layers = 28;
  int32_t max_position_embeddings = 32786;
  float rms_norm_eps = 1e-06;
  int32_t vocab_size = 151936;

  int32_t max_cache_length = 2048;
  std::vector<int32_t> mrope_section = {16, 24, 24};
  int64_t vision_token_id = 151654;
  int64_t eos_token_id = 151645;
  int32_t end_of_text_token_id = 151643;
  float rope_theta = 1000000.0;

  bool tie_word_embeddings = true;

  aops::LinearImplTypes linear_impl_type = aops::LinearImplTypes::kDefault;
};

}  // namespace mllm::models::qwen2vl
