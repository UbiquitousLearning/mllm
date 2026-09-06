// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <stdexcept>
#include <vector>

#include "mllm/core/ParameterFile.hpp"
#include "mllm/core/aops/LinearOp.hpp"
#include "mllm/engine/ConfigFile.hpp"
#include "mllm/preprocessor/chat_template/ChatTemplate.hpp"

namespace mllm::models::qwen3_5 {

struct Qwen3_5Config : protected ConfigFile {
  Qwen3_5Config() { populateLayerTypes(); }

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
    tie_word_embeddings = tc.value("tie_word_embeddings", data().value("tie_word_embeddings", tie_word_embeddings));
    hidden_act = tc.value("hidden_act", hidden_act);
    mamba_ssm_dtype = tc.value("mamba_ssm_dtype", mamba_ssm_dtype);

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
      mrope_interleaved = rp.value("mrope_interleaved", false);
      if (rp.contains("mrope_section")) { mrope_section = rp["mrope_section"].get<std::vector<int32_t>>(); }
    }

    // Layer types: explicit list or computed from full_attention_interval
    if (tc.contains("layer_types")) {
      for (auto& lt : tc["layer_types"]) { layer_types.push_back(lt.get<std::string>()); }
    } else {
      populateLayerTypes();
    }

    // Token IDs — Qwen3.5 uses different IDs than Qwen3
    if (tc.contains("eos_token_id")) { eos_token_id = tc["eos_token_id"]; }

    if (data().contains("max_cache_length")) {
      max_cache_length = data()["max_cache_length"];
    } else if (tc.contains("max_cache_length")) {
      max_cache_length = tc["max_cache_length"];
    }
    std::string linear_impl_name;
    bool has_linear_impl = false;
    if (tc.contains("linear_impl_type")) {
      linear_impl_name = tc["linear_impl_type"].get<std::string>();
      has_linear_impl = true;
    } else if (data().contains("linear_impl_type")) {
      linear_impl_name = data()["linear_impl_type"].get<std::string>();
      has_linear_impl = true;
    }
    if (has_linear_impl) {
      linear_impl_type = aops::str2LinearImplTypes(linear_impl_name);
      if (linear_impl_type == aops::LinearImplTypes::kDefault && linear_impl_name != "Default") {
        throw std::invalid_argument("Qwen3.5 contains an unsupported linear_impl_type: " + linear_impl_name);
      }
    }

    chat_template_backend = preprocessor::parseChatTemplateBackend(data().value("chat_template_backend", "legacy"));

    vision_enabled = data().contains("vision_config");
    if (vision_enabled) {
      const auto& vc = data()["vision_config"];
      vision_depth = vc["depth"];
      vision_hidden_size = vc["hidden_size"];
      vision_intermediate_size = vc["intermediate_size"];
      vision_in_channels = vc["in_channels"];
      vision_num_heads = vc["num_heads"];
      vision_num_position_embeddings = vc["num_position_embeddings"];
      vision_out_hidden_size = vc["out_hidden_size"];
      vision_patch_size = vc["patch_size"];
      vision_spatial_merge_size = vc["spatial_merge_size"];
      vision_temporal_patch_size = vc["temporal_patch_size"];
      vision_hidden_act = vc.value("hidden_act", vision_hidden_act);
      if (vc.contains("deepstack_visual_indexes")) {
        vision_deepstack_visual_indexes = vc["deepstack_visual_indexes"].get<std::vector<int32_t>>();
      }

      image_token_id = data().at("image_token_id");
      video_token_id = data().at("video_token_id");
      vision_start_token_id = data().at("vision_start_token_id");
      vision_end_token_id = data().at("vision_end_token_id");
      image_min_pixels = data().value("image_min_pixels", image_min_pixels);
      image_max_pixels = data().value("image_max_pixels", image_max_pixels);
    }

    if (num_hidden_layers <= 0 || hidden_size <= 0 || head_dim <= 0 || intermediate_size <= 0 || vocab_size <= 0
        || max_position_embeddings <= 0 || num_attention_heads <= 0 || num_key_value_heads <= 0
        || num_attention_heads % num_key_value_heads != 0) {
      throw std::invalid_argument("Qwen3.5 contains invalid transformer dimensions");
    }
    if (linear_num_key_heads <= 0 || linear_num_value_heads <= 0 || linear_num_value_heads % linear_num_key_heads != 0
        || linear_key_head_dim <= 0 || linear_value_head_dim <= 0 || linear_conv_kernel_dim <= 1) {
      throw std::invalid_argument("Qwen3.5 contains invalid GDN dimensions");
    }
    if (layer_types.size() != static_cast<size_t>(num_hidden_layers)) {
      throw std::invalid_argument("Qwen3.5 layer_types size must equal num_hidden_layers");
    }
    for (const auto& layer_type : layer_types) {
      if (layer_type != "full_attention" && layer_type != "linear_attention") {
        throw std::invalid_argument("Qwen3.5 layer_types contains an unsupported value: " + layer_type);
      }
    }
    if (rotary_dim() <= 0 || rotary_dim() > head_dim || rotary_dim() % 2 != 0) {
      throw std::invalid_argument("Qwen3.5 partial rotary dimension must be positive, even, and no larger than head_dim");
    }
    if (!std::isfinite(rope_theta) || rope_theta <= 0.0F || !std::isfinite(rms_norm_eps) || rms_norm_eps <= 0.0F) {
      throw std::invalid_argument("Qwen3.5 rope_theta and rms_norm_eps must be finite and positive");
    }
    if (max_cache_length <= 0) { throw std::invalid_argument("Qwen3.5 max_cache_length must be positive"); }
    if (!tie_word_embeddings) { throw std::invalid_argument("Qwen3.5 CPU currently requires tie_word_embeddings=true"); }
    if (vision_enabled) {
      if (vision_depth <= 0 || vision_hidden_size <= 0 || vision_intermediate_size <= 0 || vision_in_channels != 3
          || vision_num_heads <= 0 || vision_hidden_size % vision_num_heads != 0 || vision_num_position_embeddings <= 0
          || vision_out_hidden_size != hidden_size || vision_patch_size <= 0 || vision_spatial_merge_size <= 0
          || vision_temporal_patch_size <= 0 || (vision_hidden_size / vision_num_heads) % 4 != 0) {
        throw std::invalid_argument("Qwen3.5 contains invalid vision dimensions");
      }
      const auto position_grid = static_cast<int32_t>(std::sqrt(vision_num_position_embeddings));
      if (position_grid * position_grid != vision_num_position_embeddings) {
        throw std::invalid_argument("Qwen3.5 vision position embeddings must form a square grid");
      }
      if (vision_hidden_act != "gelu_pytorch_tanh") {
        throw std::invalid_argument("Qwen3.5 CPU only supports gelu_pytorch_tanh in vision blocks");
      }
      if (!vision_deepstack_visual_indexes.empty()) {
        throw std::invalid_argument("Qwen3.5 CPU multimodal support does not implement deep-stack visual features");
      }
      if (!mrope_interleaved || mrope_section.size() != 3
          || std::any_of(mrope_section.begin(), mrope_section.end(), [](int32_t section) { return section <= 0; })
          || mrope_section[0] + mrope_section[1] + mrope_section[2] != rotary_dim() / 2) {
        throw std::invalid_argument("Qwen3.5 multimodal support requires a valid interleaved three-axis MRoPE layout");
      }
      if (image_token_id < 0 || video_token_id < 0 || vision_start_token_id < 0 || vision_end_token_id < 0) {
        throw std::invalid_argument("Qwen3.5 multimodal token IDs must be non-negative");
      }
      const int64_t image_factor = static_cast<int64_t>(vision_patch_size) * vision_spatial_merge_size;
      if (image_min_pixels <= 0 || image_max_pixels < image_min_pixels || image_max_pixels < image_factor * image_factor) {
        throw std::invalid_argument("Qwen3.5 image pixel limits are invalid");
      }
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
  std::string hidden_act = "silu";
  std::string mamba_ssm_dtype = "float32";

  // Qwen3.5-specific: hybrid attention
  bool attn_output_gate = true;
  int32_t full_attention_interval = 4;
  std::vector<std::string> layer_types;  // "full_attention" or "linear_attention"

  // Qwen3.5-specific: partial RoPE
  float partial_rotary_factor = 0.25;
  float rope_theta = 10000000.0;
  bool mrope_interleaved = false;
  std::vector<int32_t> mrope_section = {11, 11, 10};
  [[nodiscard]] int32_t rotary_dim() const { return static_cast<int32_t>(head_dim * partial_rotary_factor); }

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
  int64_t vision_start_token_id = 248053;
  int64_t vision_end_token_id = 248054;
  int64_t image_token_id = 248056;
  int64_t video_token_id = 248057;

  // Optional vision tower. Text-only configs intentionally leave this off.
  bool vision_enabled = false;
  int32_t vision_depth = 12;
  int32_t vision_hidden_size = 768;
  int32_t vision_intermediate_size = 3072;
  int32_t vision_in_channels = 3;
  int32_t vision_num_heads = 12;
  int32_t vision_num_position_embeddings = 2304;
  int32_t vision_out_hidden_size = 1024;
  int32_t vision_patch_size = 16;
  int32_t vision_spatial_merge_size = 2;
  int32_t vision_temporal_patch_size = 2;
  std::string vision_hidden_act = "gelu_pytorch_tanh";
  std::vector<int32_t> vision_deepstack_visual_indexes;

  // Deliberately bounded mobile deployment envelope. The official processor
  // allows larger images, but dense vision attention needs a 512x512 cap to
  // keep prefill/RSS predictable on mobile.
  int32_t image_min_pixels = 256 * 256;
  int32_t image_max_pixels = 512 * 512;

  bool tie_word_embeddings = true;
  int32_t max_cache_length = 2048;

  aops::LinearImplTypes linear_impl_type = aops::LinearImplTypes::kDefault;
  preprocessor::ChatTemplateBackend chat_template_backend = preprocessor::ChatTemplateBackend::Legacy;

  // Helpers
  [[nodiscard]] bool isFullAttentionLayer(int layer_idx) const {
    return layer_types.at(static_cast<size_t>(layer_idx)) == "full_attention";
  }
  [[nodiscard]] int32_t numFullAttentionLayers() const {
    int32_t count = 0;
    for (auto& lt : layer_types) {
      if (lt == "full_attention") ++count;
    }
    return count;
  }
  [[nodiscard]] int32_t numGDNLayers() const { return num_hidden_layers - numFullAttentionLayers(); }

 private:
  void populateLayerTypes() {
    if (full_attention_interval <= 0) { throw std::invalid_argument("Qwen3.5 full_attention_interval must be positive"); }
    if (num_hidden_layers <= 0) { throw std::invalid_argument("Qwen3.5 num_hidden_layers must be positive"); }
    layer_types.clear();
    layer_types.reserve(static_cast<size_t>(num_hidden_layers));
    for (int32_t layer_idx = 0; layer_idx < num_hidden_layers; ++layer_idx) {
      layer_types.push_back((layer_idx + 1) % full_attention_interval == 0 ? "full_attention" : "linear_attention");
    }
  }
};

/// Checks that \p config uses the official Qwen3.5 hybrid layer schedule, i.e. a
/// full-attention interval of 4 with every 4th layer marked "full_attention" and all
/// other layers marked "linear_attention".
/// \param config Qwen3.5 text-tower configuration to inspect.
/// \return true when the layer-type list exactly matches the official schedule.
inline auto hasOfficialLayerSchedule(const Qwen3_5Config& config) -> bool {
  if (config.full_attention_interval != 4 || config.layer_types.size() != static_cast<size_t>(config.num_hidden_layers)) {
    return false;
  }
  for (int32_t layer_idx = 0; layer_idx < config.num_hidden_layers; ++layer_idx) {
    const auto expected_type = (layer_idx + 1) % 4 == 0 ? "full_attention" : "linear_attention";
    if (config.layer_types[static_cast<size_t>(layer_idx)] != expected_type) { return false; }
  }
  return true;
}

/// Checks the size-independent half of the supported mobile CPU runtime contract:
/// attention/GDN flags, head and vocabulary geometry, activation and norm settings,
/// RoPE parameters, special-token ids, tied embeddings, cache length, the official
/// layer schedule, and the KleidiAI W4A32 Linear implementation.
/// \param config Qwen3.5 text-tower configuration to inspect.
/// \return true when every shared runtime-contract field matches the official value.
inline auto hasOfficialCommonRuntimeContract(const Qwen3_5Config& config) -> bool {
  constexpr auto kKaiLinearImpl = aops::LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p8x8_1x8x32;
  return !config.attention_bias && config.attn_output_gate && config.head_dim == 256 && config.max_position_embeddings == 262144
         && config.rms_norm_eps == 1e-6F && config.vocab_size == 248320 && config.hidden_act == "silu"
         && config.mamba_ssm_dtype == "float32" && hasOfficialLayerSchedule(config) && config.linear_num_key_heads == 16
         && config.linear_key_head_dim == 128 && config.linear_value_head_dim == 128 && config.linear_conv_kernel_dim == 4
         && config.rope_theta == 10000000.0F && config.partial_rotary_factor == 0.25F && config.mrope_interleaved
         && config.mrope_section == std::vector<int32_t>({11, 11, 10}) && config.eos_token_id == 248044
         && config.im_end_token_id == 248046 && config.tie_word_embeddings && config.max_cache_length == 2048
         && config.linear_impl_type == kKaiLinearImpl;
}

inline auto hasOfficialQwen35_08BTextGeometry(const Qwen3_5Config& config) -> bool {
  return hasOfficialCommonRuntimeContract(config) && config.hidden_size == 1024 && config.intermediate_size == 3584
         && config.num_hidden_layers == 24 && config.num_attention_heads == 8 && config.num_key_value_heads == 2
         && config.linear_num_value_heads == 16;
}

/// Checks whether \p config is the official text-only Qwen3.5-0.8B runtime configuration.
inline auto isOfficialQwen35_08BTextRuntimeConfig(const Qwen3_5Config& config) -> bool {
  return hasOfficialQwen35_08BTextGeometry(config) && !config.vision_enabled;
}

/// Checks the Qwen3.5-0.8B vision and bounded mobile deployment contract.
inline auto isOfficialQwen35_08BMultimodalRuntimeConfig(const Qwen3_5Config& config) -> bool {
  return hasOfficialQwen35_08BTextGeometry(config) && config.vision_enabled && config.vision_depth == 12
         && config.vision_hidden_size == 768 && config.vision_intermediate_size == 3072 && config.vision_in_channels == 3
         && config.vision_num_heads == 12 && config.vision_num_position_embeddings == 2304
         && config.vision_out_hidden_size == 1024 && config.vision_patch_size == 16 && config.vision_spatial_merge_size == 2
         && config.vision_temporal_patch_size == 2 && config.vision_hidden_act == "gelu_pytorch_tanh"
         && config.vision_deepstack_visual_indexes.empty() && config.image_token_id == 248056 && config.video_token_id == 248057
         && config.vision_start_token_id == 248053 && config.vision_end_token_id == 248054
         && config.image_min_pixels == 256 * 256 && config.image_max_pixels == 512 * 512;
}

/// Checks whether \p config is either supported Qwen3.5-0.8B runtime contract.
inline auto isOfficialQwen35_08BRuntimeConfig(const Qwen3_5Config& config) -> bool {
  return isOfficialQwen35_08BTextRuntimeConfig(config) || isOfficialQwen35_08BMultimodalRuntimeConfig(config);
}

inline auto hasOfficialQwen35_4BTextGeometry(const Qwen3_5Config& config) -> bool {
  return hasOfficialCommonRuntimeContract(config) && config.hidden_size == 2560 && config.intermediate_size == 9216
         && config.num_hidden_layers == 32 && config.num_attention_heads == 16 && config.num_key_value_heads == 4
         && config.linear_num_value_heads == 32;
}

/// Checks whether \p config is the official text-only Qwen3.5-4B runtime configuration.
/// \param config Qwen3.5 text-tower configuration to inspect.
/// \return true only for the supported text-only 4B variant.
inline auto isOfficialQwen35_4BTextRuntimeConfig(const Qwen3_5Config& config) -> bool {
  return hasOfficialQwen35_4BTextGeometry(config) && !config.vision_enabled;
}

/// Checks the Qwen3.5-4B vision and bounded mobile deployment contract.
inline auto isOfficialQwen35_4BMultimodalRuntimeConfig(const Qwen3_5Config& config) -> bool {
  return hasOfficialQwen35_4BTextGeometry(config) && config.vision_enabled && config.vision_depth == 24
         && config.vision_hidden_size == 1024 && config.vision_intermediate_size == 4096 && config.vision_in_channels == 3
         && config.vision_num_heads == 16 && config.vision_num_position_embeddings == 2304
         && config.vision_out_hidden_size == 2560 && config.vision_patch_size == 16 && config.vision_spatial_merge_size == 2
         && config.vision_temporal_patch_size == 2 && config.vision_hidden_act == "gelu_pytorch_tanh"
         && config.vision_deepstack_visual_indexes.empty() && config.image_token_id == 248056 && config.video_token_id == 248057
         && config.vision_start_token_id == 248053 && config.vision_end_token_id == 248054
         && config.image_min_pixels == 256 * 256 && config.image_max_pixels == 512 * 512;
}

/// Checks whether \p config is either supported Qwen3.5-4B runtime contract.
inline auto isOfficialQwen35_4BRuntimeConfig(const Qwen3_5Config& config) -> bool {
  return isOfficialQwen35_4BTextRuntimeConfig(config) || isOfficialQwen35_4BMultimodalRuntimeConfig(config);
}

/// Checks whether \p config is either supported multimodal runtime contract.
inline auto isOfficialQwen35MultimodalRuntimeConfig(const Qwen3_5Config& config) -> bool {
  return isOfficialQwen35_08BMultimodalRuntimeConfig(config) || isOfficialQwen35_4BMultimodalRuntimeConfig(config);
}

/// Checks whether \p config is one of the runtime contracts this CPU runner supports.
/// \param config Qwen3.5 text-tower configuration to inspect.
/// \return true when the configuration is the official 0.8B or 4B variant.
inline auto matchesOfficialRuntimeContract(const Qwen3_5Config& config) -> bool {
  return isOfficialQwen35_08BRuntimeConfig(config) || isOfficialQwen35_4BRuntimeConfig(config);
}

/// Resolves a human-readable model name for diagnostics and error messages.
/// \param config Qwen3.5 text-tower configuration to inspect.
/// \return A size- and modality-specific name for a supported variant, otherwise the
/// generic fallback "Qwen3.5 text model".
inline auto modelNameForConfig(const Qwen3_5Config& config) -> std::string {
  if (isOfficialQwen35_08BMultimodalRuntimeConfig(config)) { return "Qwen3.5-0.8B Multimodal"; }
  if (isOfficialQwen35_4BMultimodalRuntimeConfig(config)) { return "Qwen3.5-4B Multimodal"; }
  if (isOfficialQwen35_08BTextRuntimeConfig(config)) { return "Qwen3.5-0.8B"; }
  if (isOfficialQwen35_4BTextRuntimeConfig(config)) { return "Qwen3.5-4B"; }
  return "Qwen3.5 text model";
}

/// Verifies that a loaded parameter file matches the requested runtime configuration
/// before the model is built, so a mismatched checkpoint fails early with a descriptive
/// message instead of later during inference.
/// \param config Qwen3.5 text-tower configuration to validate against.
/// \param parameter_file Loaded V1 or V2 parameter file providing the embedding tensor.
/// \throws std::invalid_argument when the runtime configuration is not a supported
/// official contract; when \p parameter_file is null or lacks
/// model.language_model.embed_tokens.weight; when the embedding dtype is not float32;
/// when a V1 embedding has an element count other than vocab_size * hidden_size; or when
/// a V2 embedding shape is not exactly [vocab_size, hidden_size]. Multimodal configs
/// additionally require V2 and matching patch/position embedding descriptors.
inline void validateModelConfigMatch(const Qwen3_5Config& config, const ParameterFile::ptr_t& parameter_file) {
  constexpr auto kEmbeddingWeight = "model.language_model.embed_tokens.weight";
  const auto model_name = modelNameForConfig(config);
  if (!matchesOfficialRuntimeContract(config)) {
    throw std::invalid_argument("Qwen3.5 model/config mismatch: CPU runner supports only the official Qwen3.5-0.8B and "
                                "Qwen3.5-4B runtime contracts");
  }
  if (parameter_file == nullptr || !parameter_file->has(kEmbeddingWeight)) {
    throw std::invalid_argument("Qwen3.5 model/config mismatch: " + model_name + " requires " + kEmbeddingWeight);
  }

  const auto embedding = parameter_file->pull(kEmbeddingWeight);
  if (embedding.dtype() != kFloat32) {
    throw std::invalid_argument("Qwen3.5 model/config mismatch: " + model_name + " requires a float32 embedding tensor");
  }
  const bool multimodal = isOfficialQwen35MultimodalRuntimeConfig(config);
  if (multimodal && parameter_file->version() != ModelFileVersion::kV2) {
    throw std::invalid_argument("Qwen3.5 model/config mismatch: multimodal inference requires model-file V2");
  }
  const auto expected_numel = static_cast<size_t>(config.vocab_size) * static_cast<size_t>(config.hidden_size);
  if (parameter_file->version() == ModelFileVersion::kV1) {
    if (embedding.numel() != expected_numel) {
      throw std::invalid_argument("Qwen3.5 model/config mismatch: " + model_name + " expects " + std::to_string(expected_numel)
                                  + " embedding elements, got " + std::to_string(embedding.numel()));
    }
    return;
  }

  const auto shape = embedding.shape();
  if (shape.size() != 2 || shape[0] != config.vocab_size || shape[1] != config.hidden_size) {
    throw std::invalid_argument("Qwen3.5 model/config mismatch: " + model_name + " expects embedding shape ["
                                + std::to_string(config.vocab_size) + ", " + std::to_string(config.hidden_size) + "]");
  }

  if (multimodal) {
    const auto require_float32_shape = [&](const std::string& name, const std::vector<int32_t>& expected_shape) {
      if (!parameter_file->has(name)) {
        throw std::invalid_argument("Qwen3.5 model/config mismatch: " + model_name + " requires " + name);
      }
      const auto tensor = parameter_file->pull(name);
      if (tensor.dtype() != kFloat32 || tensor.shape() != expected_shape) {
        throw std::invalid_argument("Qwen3.5 model/config mismatch: " + name + " has an incompatible descriptor");
      }
    };
    require_float32_shape("model.visual.patch_embed.proj.weight",
                          {config.vision_hidden_size, config.vision_in_channels, config.vision_temporal_patch_size,
                           config.vision_patch_size, config.vision_patch_size});
    require_float32_shape("model.visual.pos_embed.weight", {config.vision_num_position_embeddings, config.vision_hidden_size});
  }
}

}  // namespace mllm::models::qwen3_5
