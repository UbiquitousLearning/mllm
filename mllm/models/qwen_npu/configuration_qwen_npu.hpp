// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "mllm/core/aops/LinearOp.hpp"
#include "mllm/engine/ConfigFile.hpp"

namespace mllm::models::qwen_npu {

struct QwenNPUConfig : protected ConfigFile {
  QwenNPUConfig() = default;

  explicit QwenNPUConfig(const std::string& file_path) : ConfigFile(file_path) {
    // Init all
    attention_bias = data()["attention_bias"];
    hidden_size = data()["hidden_size"];
    intermediate_size = data()["intermediate_size"];
    num_attention_heads = data()["num_attention_heads"];
    num_key_value_heads = data()["num_key_value_heads"];
    num_hidden_layers = data()["num_hidden_layers"];
    max_position_embeddings = data()["max_position_embeddings"];
    rms_norm_eps = data()["rms_norm_eps"];
    vocab_size = data()["vocab_size"];
    head_dim = data()["head_dim"];

    bos_token_id = data()["bos_token_id"];
    eos_token_id = data()["eos_token_id"];
    rope_theta = data()["rope_theta"];

    tie_word_embeddings = data()["tie_word_embeddings"];
    max_cache_length = data()["max_cache_length"];

    linear_impl_type = aops::str2LinearImplTypes(data()["linear_impl_type"]);
    qnn_down_proj_spatial_tiles = data().value("qnn_down_proj_spatial_tiles", false);
    qnn_mlp_spatial_tiles = data().value("qnn_mlp_spatial_tiles", false);
    shadow_down_proj_layers = data().value("shadow_down_proj_layers", shadow_down_proj_layers);
    qnn_shadow_activation_fp16 = data().value("qnn_shadow_activation_fp16", false);
    qnn_shadow_correction_packet = data().value("qnn_shadow_correction_packet", false);
    qnn_shadow_correction_int16_guard = data().value("qnn_shadow_correction_int16_guard", false);
    if (qnn_shadow_correction_int16_guard && !qnn_shadow_correction_packet) {
      throw std::invalid_argument("Shadow correction INT16 guard requires the correction packet");
    }
    if (qnn_shadow_correction_packet && qnn_shadow_activation_fp16) {
      throw std::invalid_argument("Shadow correction packets require the original FP32 activation");
    }
    qnn_shadow_int8_input = data().value("qnn_shadow_int8_input", false);
    qnn_cross_layer_graphs = data().value("qnn_cross_layer_graphs", false);
    if (qnn_cross_layer_graphs && num_hidden_layers < 2) {
      throw std::invalid_argument("Qwen cross-layer QNN graphs require at least two layers");
    }
    auto sorted_shadow_layers = shadow_down_proj_layers;
    std::sort(sorted_shadow_layers.begin(), sorted_shadow_layers.end());
    if (std::adjacent_find(sorted_shadow_layers.begin(), sorted_shadow_layers.end()) != sorted_shadow_layers.end()) {
      throw std::invalid_argument("Qwen shadow down-projection layers must be unique");
    }
    for (const int32_t layer_idx : sorted_shadow_layers) {
      if (layer_idx < 0 || layer_idx >= num_hidden_layers) {
        throw std::invalid_argument("Qwen shadow down-projection layer index is outside the model");
      }
    }

    if (data().contains("shadownpu")) {
      const auto& shadownpu = data()["shadownpu"];
      prefill_attention_sparsity = shadownpu.value("sparsity", prefill_attention_sparsity);
      prefill_attention_pattern_sparse = shadownpu.value("pattern_sparse", prefill_attention_pattern_sparse);
      prefill_attention_hmx_selector = shadownpu.value("hmx_selector", prefill_attention_hmx_selector);
      prefill_attention_keep_dense_edge_layers =
          shadownpu.value("keep_dense_edge_layers", prefill_attention_keep_dense_edge_layers);
      prefill_attention_local_ratio = shadownpu.value("local_ratio", prefill_attention_local_ratio);
      prefill_attention_prefix_tokens = shadownpu.value("prefix_tokens", prefill_attention_prefix_tokens);
      prefill_attention_dense_tokens = shadownpu.value("dense_tokens", prefill_attention_dense_tokens);
      prefill_attention_random_pattern = shadownpu.value("random_pattern", prefill_attention_random_pattern);
      prefill_attention_random_seed = shadownpu.value("random_seed", prefill_attention_random_seed);
      prefill_attention_pack_reserve_tokens = shadownpu.value("pack_reserve_tokens", prefill_attention_pack_reserve_tokens);
      prefill_attention_layer_sparsities = shadownpu.value("layer_sparsities", prefill_attention_layer_sparsities);
      prefill_attention_head_retentions = shadownpu.value("head_retentions", prefill_attention_head_retentions);
    }
  }

  bool attention_bias = false;
  int32_t hidden_size = 1024;
  int32_t head_dim = 128;
  int32_t intermediate_size = 3072;
  int32_t num_attention_heads = 16;
  int32_t num_key_value_heads = 8;
  int32_t num_hidden_layers = 28;
  int32_t max_position_embeddings = 40960;
  float rms_norm_eps = 1e-06;
  int32_t vocab_size = 151936;

  int64_t bos_token_id = 151643;
  int64_t eos_token_id = 151645;
  float rope_theta = 1000000.0;

  bool tie_word_embeddings = true;
  int32_t max_cache_length = 2048;
  int32_t end_of_text_token_id = 151645;

  aops::LinearImplTypes linear_impl_type = aops::LinearImplTypes::kDefault;
  bool qnn_down_proj_spatial_tiles = false;
  bool qnn_mlp_spatial_tiles = false;
  std::vector<int32_t> shadow_down_proj_layers;
  bool qnn_shadow_activation_fp16 = false;
  bool qnn_shadow_correction_packet = false;
  bool qnn_shadow_correction_int16_guard = false;
  bool qnn_shadow_int8_input = false;
  bool qnn_cross_layer_graphs = false;
  // Rotated checkpoints may carry Q/K/V ranges calibrated before adding their Float32 biases.
  bool qnn_qkv_bias_after_dequant = false;

  // ShadowNPU is opt-in and applies to prefill only. Decode remains dense.
  float prefill_attention_sparsity = 0.0F;
  bool prefill_attention_pattern_sparse = false;
  bool prefill_attention_hmx_selector = false;
  bool prefill_attention_keep_dense_edge_layers = true;
  float prefill_attention_local_ratio = 0.5F;
  int32_t prefill_attention_prefix_tokens = 0;
  int32_t prefill_attention_dense_tokens = 0;
  bool prefill_attention_random_pattern = false;
  uint32_t prefill_attention_random_seed = 1;
  int32_t prefill_attention_pack_reserve_tokens = 0;
  std::vector<float> prefill_attention_layer_sparsities;
  std::vector<std::vector<float>> prefill_attention_head_retentions;

  [[nodiscard]] bool useShadowDownProjection(int32_t layer_idx) const {
    return std::find(shadow_down_proj_layers.begin(), shadow_down_proj_layers.end(), layer_idx)
           != shadow_down_proj_layers.end();
  }

  [[nodiscard]] std::vector<float> prefillAttentionHeadRetentionsForLayer(int32_t layer_idx) const {
    if (prefill_attention_head_retentions.empty()) { return {}; }
    if (prefill_attention_head_retentions.size() != static_cast<size_t>(num_hidden_layers)) {
      throw std::invalid_argument("Qwen per-head retention layer count must match the model");
    }
    if (layer_idx < 0 || layer_idx >= num_hidden_layers) { return {}; }
    const auto& retentions = prefill_attention_head_retentions[layer_idx];
    if (retentions.size() != static_cast<size_t>(num_attention_heads)) {
      throw std::invalid_argument("Qwen per-head retention count must match the model");
    }
    for (const float retention : retentions) {
      if (!(retention > 0.0F && retention <= 1.0F)) {
        throw std::invalid_argument("Qwen per-head retention must be in (0, 1]");
      }
    }
    return retentions;
  }

  [[nodiscard]] float prefillAttentionSparsityForLayer(int32_t layer_idx) const {
    if (layer_idx < 0 || layer_idx >= num_hidden_layers) { return 0.0F; }
    if (!prefill_attention_layer_sparsities.empty()) {
      if (prefill_attention_layer_sparsities.size() != static_cast<size_t>(num_hidden_layers)) {
        throw std::invalid_argument("Qwen per-layer sparsity count must match the model");
      }
      const float sparsity = prefill_attention_layer_sparsities[layer_idx];
      if (!(sparsity >= 0.0F && sparsity < 1.0F)) { throw std::invalid_argument("Qwen per-layer sparsity must be in [0, 1)"); }
      return sparsity;
    }
    if (!(prefill_attention_sparsity > 0.0F && prefill_attention_sparsity < 1.0F)) { return 0.0F; }
    if (!prefill_attention_keep_dense_edge_layers) { return prefill_attention_sparsity; }
    return layer_idx >= 2 && layer_idx < num_hidden_layers - 1 ? prefill_attention_sparsity : 0.0F;
  }

  [[nodiscard]] bool useSparsePrefillAttention(int32_t layer_idx) const {
    const auto retentions = prefillAttentionHeadRetentionsForLayer(layer_idx);
    if (!retentions.empty()) {
      return std::any_of(retentions.begin(), retentions.end(), [](float retention) { return retention < 1.0F; });
    }
    return prefillAttentionSparsityForLayer(layer_idx) > 0.0F;
  }
};

}  // namespace mllm::models::qwen_npu
