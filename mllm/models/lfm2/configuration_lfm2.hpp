// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "mllm/core/ParameterFile.hpp"
#include "mllm/core/aops/LinearOp.hpp"
#include "mllm/engine/ConfigFile.hpp"

namespace mllm::models::lfm2 {

struct Lfm2Config : protected ConfigFile {
  Lfm2Config() = default;

  explicit Lfm2Config(const std::string& file_path) : ConfigFile(file_path) {
    const auto& cfg = data();
    use_tiled_fa2_prefill = cfg.value("use_tiled_fa2_prefill", false);
    hidden_size = cfg.at("hidden_size");
    intermediate_size = cfg.at("intermediate_size");
    num_hidden_layers = cfg.at("num_hidden_layers");
    num_attention_heads = cfg.at("num_attention_heads");
    // head_dim's default divides by this below, which runs before validate().
    if (num_attention_heads <= 0) { throw std::invalid_argument("LFM2 num_attention_heads must be positive"); }
    num_key_value_heads = cfg.at("num_key_value_heads");
    max_position_embeddings = cfg.at("max_position_embeddings");
    vocab_size = cfg.at("vocab_size");
    conv_L_cache = cfg.at("conv_L_cache");
    conv_bias = cfg.at("conv_bias");
    norm_eps = cfg.at("norm_eps");
    tie_word_embeddings = cfg.at("tie_word_embeddings");
    bos_token_id = cfg.at("bos_token_id");
    eos_token_id = cfg.at("eos_token_id");
    pad_token_id = cfg.at("pad_token_id");
    block_auto_adjust_ff_dim = cfg.value("block_auto_adjust_ff_dim", false);
    block_ffn_dim_multiplier = cfg.value("block_ffn_dim_multiplier", 1.0F);
    block_multiple_of = cfg.value("block_multiple_of", 256);
    head_dim = cfg.value("head_dim", hidden_size / num_attention_heads);
    max_cache_length = cfg.value("max_cache_length", 2048);

    const auto& rope = cfg.at("rope_parameters");
    rope_theta = rope.at("rope_theta");
    rope_type = rope.value("rope_type", std::string("default"));
    for (const auto& layer_type : cfg.at("layer_types")) { layer_types.push_back(layer_type.get<std::string>()); }

    const auto linear_impl_name = cfg.value("linear_impl_type", std::string("Default"));
    linear_impl_type = aops::str2LinearImplTypes(linear_impl_name);
    if (linear_impl_type == aops::LinearImplTypes::kDefault && linear_impl_name != "Default") {
      throw std::invalid_argument("LFM2 contains an unsupported linear_impl_type: " + linear_impl_name);
    }
    validate();
  }

  bool use_tiled_fa2_prefill = false;
  int32_t hidden_size = 2048;
  int32_t intermediate_size = 10752;
  int32_t num_hidden_layers = 30;
  int32_t num_attention_heads = 32;
  int32_t num_key_value_heads = 8;
  int32_t head_dim = 64;
  int32_t max_position_embeddings = 131072;
  int32_t vocab_size = 128000;
  int32_t conv_L_cache = 3;
  bool conv_bias = false;
  float norm_eps = 1.0e-5F;
  float rope_theta = 10000000.0F;
  std::string rope_type = "default";
  bool block_auto_adjust_ff_dim = false;
  float block_ffn_dim_multiplier = 1.0F;
  int32_t block_multiple_of = 256;
  bool tie_word_embeddings = true;
  int64_t bos_token_id = 124894;
  int64_t eos_token_id = 124900;
  int64_t pad_token_id = 124893;
  int32_t max_cache_length = 2048;
  std::vector<std::string> layer_types;
  aops::LinearImplTypes linear_impl_type = aops::LinearImplTypes::kDefault;

  [[nodiscard]] bool isAttentionLayer(int32_t layer_idx) const {
    return layer_types.at(static_cast<size_t>(layer_idx)) == "full_attention";
  }
  [[nodiscard]] int32_t numAttentionLayers() const {
    int32_t count = 0;
    for (const auto& type : layer_types) count += type == "full_attention";
    return count;
  }
  [[nodiscard]] int32_t numConvLayers() const { return num_hidden_layers - numAttentionLayers(); }
  [[nodiscard]] int32_t attentionSlotForPhysicalLayer(int32_t physical_layer) const {
    if (!isAttentionLayer(physical_layer)) { throw std::invalid_argument("LFM2 physical layer is not attention"); }
    int32_t slot = 0;
    for (int32_t layer = 0; layer < physical_layer; ++layer) slot += isAttentionLayer(layer);
    return slot;
  }

 private:
  void validate() const {
    if (hidden_size <= 0 || intermediate_size <= 0 || num_hidden_layers <= 0 || num_attention_heads <= 0
        || num_key_value_heads <= 0 || head_dim <= 0 || max_position_embeddings <= 0 || vocab_size <= 0 || max_cache_length <= 0
        || conv_L_cache <= 1 || num_attention_heads % num_key_value_heads != 0
        || hidden_size != num_attention_heads * head_dim) {
      throw std::invalid_argument("LFM2 contains invalid model dimensions");
    }
    if (layer_types.size() != static_cast<size_t>(num_hidden_layers)) {
      throw std::invalid_argument("LFM2 layer_types size must equal num_hidden_layers");
    }
    for (const auto& type : layer_types) {
      if (type != "conv" && type != "full_attention") {
        throw std::invalid_argument("LFM2 layer_types contains unsupported value: " + type);
      }
    }
    if (!std::isfinite(norm_eps) || norm_eps <= 0.0F || !std::isfinite(rope_theta) || rope_theta <= 0.0F) {
      throw std::invalid_argument("LFM2 norm_eps and rope_theta must be finite and positive");
    }
    if (rope_type != "default") { throw std::invalid_argument("LFM2 CPU currently supports default RoPE only"); }
    if (conv_bias) { throw std::invalid_argument("LFM2 CPU currently supports conv_bias=false only"); }
    if (!tie_word_embeddings) { throw std::invalid_argument("LFM2 CPU requires tied embeddings"); }
    if (block_auto_adjust_ff_dim) { throw std::invalid_argument("LFM2 CPU does not support adjusted FFN dimensions"); }
  }
};

inline auto officialLayerTypes() -> const std::vector<std::string>& {
  static const std::vector<std::string> schedule = {
      "conv",           "conv",           "full_attention", "conv",           "conv", "full_attention",
      "conv",           "conv",           "conv",           "full_attention", "conv", "conv",
      "conv",           "full_attention", "conv",           "conv",           "conv", "full_attention",
      "conv",           "conv",           "conv",           "full_attention", "conv", "conv",
      "full_attention", "conv",           "conv",           "full_attention", "conv", "conv"};
  return schedule;
}

inline auto matchesOfficialRuntimeContract(const Lfm2Config& cfg) -> bool {
  constexpr auto kKai = aops::LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p8x8_1x8x32;
  return cfg.hidden_size == 2048 && cfg.intermediate_size == 10752 && cfg.num_hidden_layers == 30
         && cfg.num_attention_heads == 32 && cfg.num_key_value_heads == 8 && cfg.head_dim == 64
         && cfg.max_position_embeddings == 131072 && cfg.vocab_size == 128000 && cfg.conv_L_cache == 3 && !cfg.conv_bias
         && cfg.norm_eps == 1.0e-5F && cfg.rope_theta == 10000000.0F && cfg.rope_type == "default"
         && !cfg.block_auto_adjust_ff_dim && cfg.tie_word_embeddings && cfg.bos_token_id == 124894 && cfg.eos_token_id == 124900
         && cfg.pad_token_id == 124893 && cfg.max_cache_length == 2048 && cfg.layer_types == officialLayerTypes()
         && cfg.numAttentionLayers() == 8 && cfg.linear_impl_type == kKai;
}

inline void validateModelConfigMatch(const Lfm2Config& cfg, const ParameterFile::ptr_t& parameter_file) {
  constexpr auto kEmbedding = "model.embed_tokens.weight";
  if (!matchesOfficialRuntimeContract(cfg)) {
    throw std::invalid_argument("LFM2 model/config mismatch: CPU runner supports only official LFM2.5-2.6B W4A32");
  }
  if (parameter_file == nullptr || !parameter_file->has(kEmbedding)) {
    throw std::invalid_argument(std::string("LFM2 model/config mismatch: missing ") + kEmbedding);
  }
  const auto embedding = parameter_file->pull(kEmbedding);
  if (embedding.dtype() != kFloat32) {
    throw std::invalid_argument("LFM2 model/config mismatch: embedding must remain float32");
  }
  const auto expected_numel = static_cast<size_t>(cfg.vocab_size) * static_cast<size_t>(cfg.hidden_size);
  if (parameter_file->version() == ModelFileVersion::kV1) {
    if (embedding.numel() != expected_numel) { throw std::invalid_argument("LFM2 model/config embedding element mismatch"); }
  } else {
    const auto shape = embedding.shape();
    if (shape.size() != 2 || shape[0] != cfg.vocab_size || shape[1] != cfg.hidden_size) {
      throw std::invalid_argument("LFM2 model/config embedding shape mismatch");
    }
  }
  if (!parameter_file->has("lm_head_out.weight")) {
    throw std::invalid_argument("LFM2 model/config mismatch: converted W4A32 model is missing tied lm_head_out.weight alias");
  }
}

}  // namespace mllm::models::lfm2
