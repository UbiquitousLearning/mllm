// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/aops/LinearOp.hpp"
#include "mllm/engine/ConfigFile.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace mllm::models::ling3 {

struct Ling3Config : protected ConfigFile {
  Ling3Config() = default;

  explicit Ling3Config(const std::string& file_path) : ConfigFile(file_path) {
    const auto& config = data();
    hidden_size = config.at("hidden_size");
    intermediate_size = config.at("intermediate_size");
    num_hidden_layers = config.at("num_hidden_layers");
    num_attention_heads = config.at("num_attention_heads");
    num_key_value_heads = config.at("num_key_value_heads");
    head_dim = config.at("head_dim");
    vocab_size = config.at("vocab_size");
    max_position_embeddings = config.at("max_position_embeddings");
    rms_norm_eps = config.at("rms_norm_eps");
    rope_theta = config.at("rope_theta");
    hidden_act = config.at("hidden_act");
    tie_word_embeddings = config.at("tie_word_embeddings");
    use_qkv_bias = config.at("use_qkv_bias");

    layer_group_size = config.at("layer_group_size");
    short_conv_kernel_size = config.at("short_conv_kernel_size");
    no_kda_lora = config.at("no_kda_lora");
    kda_safe_gate = config.at("kda_safe_gate");
    kda_lower_bound = config.at("kda_lower_bound");

    q_lora_rank = config.at("q_lora_rank");
    kv_lora_rank = config.at("kv_lora_rank");
    qk_rope_head_dim = config.at("qk_rope_head_dim");
    qk_nope_head_dim = config.at("qk_nope_head_dim");
    qk_head_dim = config.at("qk_head_dim");
    v_head_dim = config.at("v_head_dim");
    rope_interleave = config.at("rope_interleave");
    gated_attention_proj_granularity_type = config.at("gated_attention_proj_granularity_type");

    num_experts = config.at("num_experts");
    num_shared_experts = config.at("num_shared_experts");
    num_experts_per_tok = config.at("num_experts_per_tok");
    n_group = config.at("n_group");
    topk_group = config.at("topk_group");
    moe_intermediate_size = config.at("moe_intermediate_size");
    moe_shared_expert_intermediate_size = config.at("moe_shared_expert_intermediate_size");
    first_k_dense_replace = config.at("first_k_dense_replace");
    routed_scaling_factor = config.at("routed_scaling_factor");
    scoring_func = config.at("scoring_func");
    topk_method = config.at("topk_method");
    moe_router_enable_expert_bias = config.at("moe_router_enable_expert_bias");

    pad_token_id = config.at("pad_token_id");
    eos_token_id = config.at("eos_token_id");
    max_cache_length = config.value("max_cache_length", max_cache_length);
    if (config.contains("linear_impl_type")) {
      const auto impl_name = config.at("linear_impl_type").get<std::string>();
      linear_impl_type = aops::str2LinearImplTypes(impl_name);
      if (linear_impl_type == aops::LinearImplTypes::kDefault && impl_name != "Default") {
        throw std::invalid_argument("Ling-3 contains an unsupported linear_impl_type: " + impl_name);
      }
    }

    validate();
  }

  int32_t hidden_size = 1536;
  int32_t intermediate_size = 4608;
  int32_t num_hidden_layers = 24;
  int32_t num_attention_heads = 16;
  int32_t num_key_value_heads = 16;
  int32_t head_dim = 128;
  int32_t vocab_size = 157184;
  int32_t max_position_embeddings = 131072;
  float rms_norm_eps = 1.0e-6F;
  float rope_theta = 6000000.0F;
  std::string hidden_act = "silu";
  bool tie_word_embeddings = false;
  bool use_qkv_bias = false;

  int32_t layer_group_size = 4;
  int32_t short_conv_kernel_size = 4;
  bool no_kda_lora = true;
  bool kda_safe_gate = true;
  float kda_lower_bound = -5.0F;

  int32_t q_lora_rank = 256;
  int32_t kv_lora_rank = 512;
  int32_t qk_rope_head_dim = 64;
  int32_t qk_nope_head_dim = 128;
  int32_t qk_head_dim = 192;
  int32_t v_head_dim = 128;
  bool rope_interleave = true;
  std::string gated_attention_proj_granularity_type = "head_wise";

  int32_t num_experts = 128;
  int32_t num_shared_experts = 1;
  int32_t num_experts_per_tok = 8;
  int32_t n_group = 8;
  int32_t topk_group = 4;
  int32_t moe_intermediate_size = 512;
  int32_t moe_shared_expert_intermediate_size = 512;
  int32_t first_k_dense_replace = 1;
  float routed_scaling_factor = 2.5F;
  std::string scoring_func = "sigmoid";
  std::string topk_method = "noaux_tc";
  bool moe_router_enable_expert_bias = true;

  int64_t pad_token_id = 156892;
  int64_t eos_token_id = 156895;
  int64_t end_of_text_token_id = 156892;
  int64_t bos_token_id = 156891;
  int32_t max_cache_length = 2048;

  aops::LinearImplTypes linear_impl_type = aops::LinearImplTypes::kDefault;

  [[nodiscard]] bool isFullAttentionLayer(int32_t layer_index) const {
    return (layer_index + 1) % layer_group_size == 0 || layer_index >= num_hidden_layers / layer_group_size * layer_group_size;
  }

  [[nodiscard]] int32_t numFullAttentionLayers() const {
    int32_t count = 0;
    for (int32_t layer = 0; layer < num_hidden_layers; ++layer) {
      if (isFullAttentionLayer(layer)) { ++count; }
    }
    return count;
  }

  [[nodiscard]] int32_t numKDALayers() const { return num_hidden_layers - numFullAttentionLayers(); }

 private:
  void validate() const {
    if (hidden_size <= 0 || intermediate_size <= 0 || num_hidden_layers <= 0 || num_attention_heads <= 0
        || num_key_value_heads <= 0 || head_dim <= 0 || vocab_size <= 0 || max_position_embeddings <= 0
        || max_cache_length <= 0) {
      throw std::invalid_argument("Ling-3 contains invalid model dimensions");
    }
    if (layer_group_size <= 0 || short_conv_kernel_size <= 1 || q_lora_rank <= 0 || kv_lora_rank <= 0 || qk_rope_head_dim <= 0
        || qk_nope_head_dim <= 0 || qk_head_dim != qk_nope_head_dim + qk_rope_head_dim || v_head_dim <= 0) {
      throw std::invalid_argument("Ling-3 contains invalid hybrid attention dimensions");
    }
    if (num_experts <= 0 || num_shared_experts != 1 || num_experts_per_tok <= 0 || num_experts_per_tok > num_experts
        || n_group <= 0 || num_experts % n_group != 0 || topk_group <= 0 || topk_group > n_group || moe_intermediate_size <= 0
        || moe_shared_expert_intermediate_size <= 0 || first_k_dense_replace < 0 || first_k_dense_replace > num_hidden_layers) {
      throw std::invalid_argument("Ling-3 contains invalid MoE dimensions");
    }
    if (!std::isfinite(rms_norm_eps) || rms_norm_eps <= 0.0F || !std::isfinite(rope_theta) || rope_theta <= 0.0F
        || !std::isfinite(kda_lower_bound) || kda_lower_bound >= 0.0F || !std::isfinite(routed_scaling_factor)
        || routed_scaling_factor <= 0.0F) {
      throw std::invalid_argument("Ling-3 contains invalid numerical parameters");
    }
    if (hidden_act != "silu" || scoring_func != "sigmoid" || topk_method != "noaux_tc" || !no_kda_lora || !kda_safe_gate
        || !rope_interleave || gated_attention_proj_granularity_type != "head_wise" || tie_word_embeddings || use_qkv_bias
        || !moe_router_enable_expert_bias) {
      throw std::invalid_argument("Ling-3 CPU received an unsupported architecture variant");
    }
  }
};

inline bool hasOfficialLing3TinyArchitecture(const Ling3Config& config) {
  return config.hidden_size == 1536 && config.intermediate_size == 4608 && config.num_hidden_layers == 24
         && config.num_attention_heads == 16 && config.num_key_value_heads == 16 && config.head_dim == 128
         && config.vocab_size == 157184 && config.max_position_embeddings == 131072 && config.rms_norm_eps == 1.0e-6F
         && config.rope_theta == 6000000.0F && config.layer_group_size == 4 && config.numFullAttentionLayers() == 6
         && config.numKDALayers() == 18 && config.short_conv_kernel_size == 4 && config.q_lora_rank == 256
         && config.kv_lora_rank == 512 && config.qk_rope_head_dim == 64 && config.qk_nope_head_dim == 128
         && config.qk_head_dim == 192 && config.v_head_dim == 128 && config.num_experts == 128
         && config.num_experts_per_tok == 8 && config.n_group == 8 && config.topk_group == 4
         && config.moe_intermediate_size == 512 && config.moe_shared_expert_intermediate_size == 512
         && config.first_k_dense_replace == 1 && config.routed_scaling_factor == 2.5F && config.pad_token_id == 156892
         && config.eos_token_id == 156895;
}

inline void validateLing3ModelConfigMatch(const Ling3Config& config, const ParameterFile::ptr_t& parameter_file) {
  constexpr auto kEmbeddingWeight = "model.word_embeddings.weight";
  constexpr auto kLmHeadWeight = "lm_head.weight";
  if (!hasOfficialLing3TinyArchitecture(config)) {
    throw std::invalid_argument("Ling-3 model/config mismatch: expected inclusionAI/Ling-3.0-tiny");
  }
  if (parameter_file == nullptr || parameter_file->version() != ModelFileVersion::kV2) {
    throw std::invalid_argument("Ling-3 mobile CPU inference requires an MLLM V2 model file");
  }
  if (!parameter_file->has(kEmbeddingWeight) || !parameter_file->has(kLmHeadWeight)) {
    throw std::invalid_argument("Ling-3 model file is missing untied embedding or LM-head weights");
  }
  const auto embedding = parameter_file->pull(kEmbeddingWeight);
  if (embedding.dtype() != kFloat32 || embedding.shape() != Tensor::shape_t({config.vocab_size, config.hidden_size})) {
    throw std::invalid_argument("Ling-3 model file has an incompatible word embedding descriptor");
  }
  const auto lm_head = parameter_file->pull(kLmHeadWeight);
  if (lm_head.dtype() != kByte || lm_head.shape().size() != 1) {
    throw std::invalid_argument("Ling-3 mobile model requires a packed KAI W4A32 LM head");
  }
}

}  // namespace mllm::models::ling3
