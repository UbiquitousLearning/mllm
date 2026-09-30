// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once
#include <cmath>
#include <stdexcept>
#include "mllm/engine/ConfigFile.hpp"
#include <vector>
#include <string>
#include "mllm/core/aops/LinearOp.hpp"
#include "mllm/core/ParameterFile.hpp"
#include <map>

namespace mllm::models::spark2_5 {
struct SparkConfig : protected ConfigFile {
  SparkConfig() = default;
  explicit SparkConfig(const std::string& path) : ConfigFile(path) {
    const auto& config = data();
    vocab_size = config.at("vocab_size");
    hidden_size = config.at("hidden_size");
    intermediate_size = config.at("intermediate_size");
    num_hidden_layers = config.at("num_hidden_layers");
    num_attention_heads = config.at("num_attention_heads");
    num_key_value_heads = config.at("num_key_value_heads");
    head_dim = config.at("head_dim");
    sliding_window = config.at("sliding_window");
    max_position_embeddings = config.at("max_position_embeddings");
    rms_norm_eps = config.at("rms_norm_eps");
    max_cache_length = config.value("max_cache_length", 4096);
    layer_types = config.at("layer_types").get<std::vector<std::string>>();
    const auto& rope = config.at("rope_parameters");
    for (const auto& type : {"full_attention", "sliding_attention"})
      if (rope.at(type).value("rope_type", std::string("default")) != "default")
        throw std::invalid_argument("Only default Spark RoPE is supported");
    full_theta = rope.at("full_attention").at("rope_theta");
    sliding_theta = rope.at("sliding_attention").at("rope_theta");
    const double full_factor = rope.at("full_attention").at("partial_rotary_factor");
    const double sliding_factor = rope.at("sliding_attention").at("partial_rotary_factor");
    if (!std::isfinite(full_factor) || !std::isfinite(sliding_factor) || full_factor <= 0 || full_factor > 1
        || sliding_factor != 1.0 || head_dim <= 0 || full_factor * head_dim != std::floor(full_factor * head_dim))
      throw std::invalid_argument("Unsupported Spark RoPE geometry");
    rotary_dim = static_cast<int32_t>(head_dim * full_factor);
    if (config.at("model_type") != "spark2_5" || config.at("hidden_act") != "gelu"
        || config.at("gate_attn_act_mode") != "sigmoid" || !config.at("headwise_attn_output_gate").get<bool>()
        || config.at("attention_bias").get<bool>() || config.at("mlp_bias").get<bool>()
        || !config.at("tie_word_embeddings").get<bool>() || config.at("eos_token_id") != 1 || config.at("bos_token_id") != 0
        || config.at("pad_token_id") != 2)
      throw std::invalid_argument("Unsupported Spark architecture or special tokens");
    const auto impl = config.value("linear_impl_type", std::string("Default"));
    linear_impl_type = aops::str2LinearImplTypes(impl);
    constexpr auto kai = aops::LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p8x8_1x8x32;
    if ((linear_impl_type != aops::LinearImplTypes::kDefault && linear_impl_type != kai)
        || (linear_impl_type == aops::LinearImplTypes::kDefault && impl != "Default"))
      throw std::invalid_argument("Unsupported Spark linear implementation");
    if (vocab_size < 3 || hidden_size <= 0 || intermediate_size <= 0 || num_hidden_layers <= 0 || num_attention_heads <= 0
        || num_key_value_heads <= 0 || num_attention_heads % num_key_value_heads || head_dim % 2 || rotary_dim < 2
        || rotary_dim % 2 || sliding_window < 2 || max_cache_length <= 0 || max_cache_length > max_position_embeddings
        || !std::isfinite(rms_norm_eps) || rms_norm_eps <= 0 || !std::isfinite(full_theta) || full_theta <= 0
        || !std::isfinite(sliding_theta) || sliding_theta <= 0 || layer_types.size() != static_cast<size_t>(num_hidden_layers))
      throw std::invalid_argument("Invalid Spark dimensions, cache or normalization");
    for (const auto& type : layer_types) {
      if (type == "full_attention")
        ++full_layers;
      else if (type != "sliding_attention")
        throw std::invalid_argument("Unknown Spark layer type");
    }
    if (full_layers == 0) throw std::invalid_argument("Spark requires at least one full attention layer");
  }
  int32_t vocab_size, hidden_size, intermediate_size, num_hidden_layers, num_attention_heads, num_key_value_heads;
  int32_t head_dim, sliding_window, rotary_dim, max_position_embeddings, max_cache_length, full_layers = 0;
  float rms_norm_eps, full_theta, sliding_theta;
  std::vector<std::string> layer_types;
  aops::LinearImplTypes linear_impl_type;
};

inline void validateModelConfigMatch(const SparkConfig& c, const ParameterFile::ptr_t& params) {
  if (c.vocab_size != 131072 || c.hidden_size != 2048 || c.intermediate_size != 6656 || c.num_hidden_layers != 28
      || c.num_attention_heads != 8 || c.num_key_value_heads != 2 || c.head_dim != 256 || c.rotary_dim != 64
      || c.sliding_window != 512 || c.full_theta != 5000000.0F || c.sliding_theta != 10000.0F || c.rms_norm_eps != 1e-6F
      || c.max_position_embeddings != 1048576 || c.max_cache_length > 8192)
    throw std::invalid_argument("Runner supports Spark-X2.5-1.7B with at most 8192 context tokens");
  for (int i = 0; i < 28; ++i)
    if (c.layer_types[i] != (i % 4 == 3 ? "full_attention" : "sliding_attention"))
      throw std::invalid_argument("Spark layer schedule differs from the official checkpoint");
  if (!params || params->version() != ModelFileVersion::kV2) throw std::invalid_argument("Spark requires ModelFileV2");
  const bool quantized = c.linear_impl_type != aops::LinearImplTypes::kDefault;
  std::map<std::string, Tensor::shape_t> shapes;
  auto add = [&](const std::string& name, int n, int k, bool packed) {
    if (packed && quantized)
      shapes[name] = {n * (k / 32 * 18 + 8)};
    else
      shapes[name] = k ? Tensor::shape_t{n, k} : Tensor::shape_t{n};
  };
  add("model.embedding.weight", 131072, 2048, false);
  add("lm_head.weight", 131072, 2048, true);
  add("model.norm.weight", 2048, 0, false);
  for (int i = 0; i < 28; ++i) {
    auto p = "model.layers." + std::to_string(i) + ".";
    add(p + "input_layernorm.weight", 2048, 0, false);
    add(p + "post_attention_layernorm.weight", 2048, 0, false);
    add(p + "self_attn.q_k_v_proj.weight", 3072, 2048, true);
    add(p + "self_attn.out_proj.weight", 2048, 2048, true);
    add(p + "self_attn.g_proj.weight", 8, 2048, false);
    add(p + "mlp.gate_proj.weight", 6656, 2048, true);
    add(p + "mlp.up_proj.weight", 6656, 2048, true);
    add(p + "mlp.down_proj.weight", 2048, 6656, true);
  }
  size_t count = 0;
  for (const auto& entry : *params) {
    ++count;
    if (!shapes.contains(entry.first)) throw std::invalid_argument("Unexpected Spark parameter: " + entry.first);
  }
  if (count != shapes.size()) throw std::invalid_argument("Missing Spark parameters");
  for (const auto& [name, shape] : shapes) {
    if (!params->has(name)) throw std::invalid_argument("Missing Spark parameter: " + name);
    const auto t = params->pull(name);
    const bool packed = quantized && shape.size() == 1 && shape[0] != 2048;
    if (t.shape() != shape || t.dtype() != (packed ? kByte : kFloat32))
      throw std::invalid_argument("Spark model/config mismatch: " + name);
  }
}
}  // namespace mllm::models::spark2_5
