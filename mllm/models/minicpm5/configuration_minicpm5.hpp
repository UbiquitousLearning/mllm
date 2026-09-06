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
#include "mllm/preprocessor/chat_template/ChatTemplate.hpp"

namespace mllm::models::minicpm5 {

struct MiniCPM5Config : protected ConfigFile {
  MiniCPM5Config() = default;

  explicit MiniCPM5Config(const std::string& file_path) : ConfigFile(file_path) {
    const auto& config = data();
    if (!config.contains("head_dim") || !config.contains("eos_token_id")) {
      throw std::invalid_argument("MiniCPM5 config must explicitly define head_dim and eos_token_id");
    }
    if (config.contains("rope_scaling") && !config["rope_scaling"].is_null()) {
      throw std::invalid_argument("MiniCPM5 CPU currently supports the official unscaled RoPE contract only");
    }
    vocab_size = config.value("vocab_size", vocab_size);
    hidden_size = config.value("hidden_size", hidden_size);
    intermediate_size = config.value("intermediate_size", intermediate_size);
    num_hidden_layers = config.value("num_hidden_layers", num_hidden_layers);
    num_attention_heads = config.value("num_attention_heads", num_attention_heads);
    num_key_value_heads = config.value("num_key_value_heads", num_key_value_heads);
    head_dim = config.value("head_dim", head_dim);
    hidden_act = config.value("hidden_act", hidden_act);
    max_position_embeddings = config.value("max_position_embeddings", max_position_embeddings);
    rms_norm_eps = config.value("rms_norm_eps", rms_norm_eps);
    rope_theta = config.value("rope_theta", rope_theta);
    attention_bias = config.value("attention_bias", attention_bias);
    tie_word_embeddings = config.value("tie_word_embeddings", tie_word_embeddings);
    bos_token_id = config.value("bos_token_id", bos_token_id);
    pad_token_id = config.value("pad_token_id", pad_token_id);
    max_cache_length = config.value("max_cache_length", max_cache_length);

    if (config.contains("eos_token_id")) {
      if (config["eos_token_id"].is_array()) {
        eos_token_ids = config["eos_token_id"].get<std::vector<int64_t>>();
      } else {
        eos_token_ids = {config["eos_token_id"].get<int64_t>()};
      }
    }

    chat_template_backend = preprocessor::parseChatTemplateBackend(config.value("chat_template_backend", "legacy"));

    if (config.contains("linear_impl_type")) {
      const auto linear_impl_name = config["linear_impl_type"].get<std::string>();
      linear_impl_type = aops::str2LinearImplTypes(linear_impl_name);
      if (linear_impl_type == aops::LinearImplTypes::kDefault && linear_impl_name != "Default") {
        throw std::invalid_argument("MiniCPM5 contains an unsupported linear_impl_type: " + linear_impl_name);
      }
    }

    if (vocab_size <= 0 || hidden_size <= 0 || intermediate_size <= 0 || num_hidden_layers <= 0 || num_attention_heads <= 0
        || num_key_value_heads <= 0 || head_dim <= 0 || num_attention_heads % num_key_value_heads != 0) {
      throw std::invalid_argument("MiniCPM5 contains invalid transformer dimensions");
    }
    if (max_position_embeddings <= 0 || max_cache_length <= 0 || max_cache_length > max_position_embeddings) {
      throw std::invalid_argument("MiniCPM5 contains an invalid cache or position limit");
    }
    if (!std::isfinite(rms_norm_eps) || rms_norm_eps <= 0.0F || !std::isfinite(rope_theta) || rope_theta <= 0.0F) {
      throw std::invalid_argument("MiniCPM5 rms_norm_eps and rope_theta must be finite and positive");
    }
    if (eos_token_ids.empty()) { throw std::invalid_argument("MiniCPM5 must define at least one EOS token id"); }
    for (const auto token_id : eos_token_ids) {
      if (token_id < 0 || token_id >= vocab_size) { throw std::invalid_argument("MiniCPM5 contains an invalid EOS token id"); }
    }
  }

  int32_t vocab_size = 130560;
  int32_t hidden_size = 1536;
  int32_t intermediate_size = 4608;
  int32_t num_hidden_layers = 24;
  int32_t num_attention_heads = 16;
  int32_t num_key_value_heads = 2;
  int32_t head_dim = 128;
  std::string hidden_act = "silu";
  int32_t max_position_embeddings = 131072;
  float rms_norm_eps = 1.0e-6F;
  float rope_theta = 5000000.0F;
  bool attention_bias = false;
  bool tie_word_embeddings = false;
  int64_t bos_token_id = 0;
  int64_t pad_token_id = 1;
  std::vector<int64_t> eos_token_ids = {1, 130073};
  int32_t max_cache_length = 2048;
  aops::LinearImplTypes linear_impl_type = aops::LinearImplTypes::kDefault;
  preprocessor::ChatTemplateBackend chat_template_backend = preprocessor::ChatTemplateBackend::Legacy;
};

inline auto matchesOfficialMiniCPM5_1BRuntimeContract(const MiniCPM5Config& config) -> bool {
  constexpr auto kKaiLinearImpl = aops::LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p8x8_1x8x32;
  return config.vocab_size == 130560 && config.hidden_size == 1536 && config.intermediate_size == 4608
         && config.num_hidden_layers == 24 && config.num_attention_heads == 16 && config.num_key_value_heads == 2
         && config.head_dim == 128 && config.hidden_act == "silu" && config.max_position_embeddings == 131072
         && config.rms_norm_eps == 1.0e-6F && config.rope_theta == 5000000.0F && !config.attention_bias
         && !config.tie_word_embeddings && config.bos_token_id == 0 && config.pad_token_id == 1
         && config.eos_token_ids == std::vector<int64_t>({1, 130073}) && config.max_cache_length == 2048
         && config.linear_impl_type == kKaiLinearImpl;
}

inline void validateModelConfigMatch(const MiniCPM5Config& config, const ParameterFile::ptr_t& parameter_file) {
  constexpr auto kEmbeddingWeight = "model.embed_tokens.weight";
  constexpr auto kLmHeadWeight = "lm_head.weight";
  if (!matchesOfficialMiniCPM5_1BRuntimeContract(config)) {
    throw std::invalid_argument(
        "MiniCPM5 model/config mismatch: CPU runner supports only the official MiniCPM5-1B runtime contract");
  }
  if (parameter_file == nullptr || !parameter_file->has(kEmbeddingWeight) || !parameter_file->has(kLmHeadWeight)) {
    throw std::invalid_argument("MiniCPM5 model/config mismatch: checkpoint requires model.embed_tokens.weight and "
                                "lm_head.weight");
  }

  const auto embedding = parameter_file->pull(kEmbeddingWeight);
  if (embedding.dtype() != kFloat32) {
    throw std::invalid_argument("MiniCPM5 model/config mismatch: embedding tensor must remain float32");
  }
  const auto expected_numel = static_cast<size_t>(config.vocab_size) * static_cast<size_t>(config.hidden_size);
  if (parameter_file->version() == ModelFileVersion::kV1) {
    if (embedding.numel() != expected_numel) {
      throw std::invalid_argument("MiniCPM5 model/config mismatch: embedding element count does not match config");
    }
    return;
  }

  const auto shape = embedding.shape();
  if (shape.size() != 2 || shape[0] != config.vocab_size || shape[1] != config.hidden_size) {
    throw std::invalid_argument("MiniCPM5 model/config mismatch: embedding shape must be [130560, 1536]");
  }
}

}  // namespace mllm::models::minicpm5
