// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include "mllm/core/aops/LinearOp.hpp"
#include "mllm/engine/ConfigFile.hpp"
#include <vector>

namespace mllm::models::deepseek_ocr {

struct DpskOcrConfig : protected ConfigFile {
  DpskOcrConfig() = default;

  explicit DpskOcrConfig(const std::string& file_path) : ConfigFile(file_path) {
    // Init all
    _name_or_path = data()["_name_or_path"];

    // Parse candidate_resolutions
    if (data().contains("candidate_resolutions") && data()["candidate_resolutions"].is_array()) {
      for (const auto& res : data()["candidate_resolutions"]) {
        if (res.is_array() && res.size() == 2) { candidate_resolutions.push_back({res[0], res[1]}); }
      }
    }

    global_view_pos = data()["global_view_pos"];
    model_type = data()["model_type"];
    tile_tag = data()["tile_tag"];
    transformers_version = data()["transformers_version"];

    // Language config
    language_config.bos_token_id = data()["language_config"]["bos_token_id"];
    language_config.eos_token_id = data()["language_config"]["eos_token_id"];
    language_config.first_k_dense_replace = data()["language_config"]["first_k_dense_replace"];
    language_config.hidden_size = data()["language_config"]["hidden_size"];
    language_config.intermediate_size = data()["language_config"]["intermediate_size"];
    language_config.kv_lora_rank = data()["language_config"]["kv_lora_rank"].is_null()
                                       ? -1
                                       : static_cast<int32_t>(data()["language_config"]["kv_lora_rank"]);
    language_config.lm_head = data()["language_config"]["lm_head"];
    language_config.max_position_embeddings = data()["language_config"]["max_position_embeddings"];
    language_config.moe_intermediate_size = data()["language_config"]["moe_intermediate_size"];
    language_config.n_group = data()["language_config"]["n_group"];
    language_config.n_routed_experts = data()["language_config"]["n_routed_experts"];
    language_config.n_shared_experts = data()["language_config"]["n_shared_experts"];
    language_config.num_attention_heads = data()["language_config"]["num_attention_heads"];
    language_config.num_experts_per_tok = data()["language_config"]["num_experts_per_tok"];
    language_config.num_hidden_layers = data()["language_config"]["num_hidden_layers"];
    language_config.num_key_value_heads = data()["language_config"]["num_key_value_heads"];
    language_config.q_lora_rank = data()["language_config"]["q_lora_rank"].is_null()
                                      ? -1
                                      : static_cast<int32_t>(data()["language_config"]["q_lora_rank"]);
    language_config.qk_nope_head_dim = data()["language_config"]["qk_nope_head_dim"];
    language_config.qk_rope_head_dim = data()["language_config"]["qk_rope_head_dim"];
    language_config.rm_head = data()["language_config"]["rm_head"];
    language_config.topk_group = data()["language_config"]["topk_group"];
    language_config.topk_method = data()["language_config"]["topk_method"];
    language_config.use_mla = data()["language_config"]["use_mla"];
    language_config.v_head_dim = data()["language_config"]["v_head_dim"];
    language_config.vocab_size = data()["language_config"]["vocab_size"];

    // Projector config
    projector_config.input_dim = data()["projector_config"]["input_dim"];
    projector_config.model_type = data()["projector_config"]["model_type"];
    projector_config.n_embed = data()["projector_config"]["n_embed"];
    projector_config.projector_type = data()["projector_config"]["projector_type"];

    // Vision config
    vision_config.image_size = data()["vision_config"]["image_size"];
    vision_config.mlp_ratio = data()["vision_config"]["mlp_ratio"];
    vision_config.model_name = data()["vision_config"]["model_name"];
    vision_config.model_type = data()["vision_config"]["model_type"];

    // Main config values
    bos_token_id = data()["bos_token_id"];
    eos_token_id = data()["eos_token_id"];
    first_k_dense_replace = data()["first_k_dense_replace"];
    hidden_size = data()["hidden_size"];
    intermediate_size = data()["intermediate_size"];
    kv_lora_rank = data()["kv_lora_rank"].is_null() ? -1 : static_cast<int32_t>(data()["kv_lora_rank"]);
    lm_head = data()["lm_head"];
    max_position_embeddings = data()["max_position_embeddings"];
    moe_intermediate_size = data()["moe_intermediate_size"];
    n_group = data()["n_group"];
    n_routed_experts = data()["n_routed_experts"];
    n_shared_experts = data()["n_shared_experts"];
    num_attention_heads = data()["num_attention_heads"];
    num_experts_per_tok = data()["num_experts_per_tok"];
    num_hidden_layers = data()["num_hidden_layers"];
    num_key_value_heads = data()["num_key_value_heads"];
    q_lora_rank = data()["q_lora_rank"].is_null() ? -1 : static_cast<int32_t>(data()["q_lora_rank"]);
    qk_nope_head_dim = data()["qk_nope_head_dim"];
    qk_rope_head_dim = data()["qk_rope_head_dim"];
    rm_head = data()["rm_head"];
    topk_group = data()["topk_group"];
    topk_method = data()["topk_method"];
    use_mla = data()["use_mla"];
    v_head_dim = data()["v_head_dim"];
    vocab_size = data()["vocab_size"];
    clip_linear_impl_type = aops::str2LinearImplTypes(data()["clip_linear_impl_type"]);
    llm_mlp_linear_impl_type = aops::str2LinearImplTypes(data()["llm_mlp_linear_impl_type"]);
    lm_head_linear_impl_type = aops::str2LinearImplTypes(data()["lm_head_linear_impl_type"]);
    mlp_projector_linear_impl_type = aops::str2LinearImplTypes(data()["mlp_projector_linear_impl_type"]);
    sam_linear_impl_type = aops::str2LinearImplTypes(data()["sam_linear_impl_type"]);
  }

  // Nested structs for complex configuration
  struct LanguageConfig {
    int64_t bos_token_id = 0;
    int64_t eos_token_id = 1;
    int32_t first_k_dense_replace = 1;
    int32_t hidden_size = 1280;
    int32_t intermediate_size = 6848;
    int32_t kv_lora_rank = -1;  // null in JSON
    bool lm_head = true;
    int32_t max_position_embeddings = 8192;
    int32_t moe_intermediate_size = 896;
    int32_t n_group = 1;
    int32_t n_routed_experts = 64;
    int32_t n_shared_experts = 2;
    int32_t num_attention_heads = 10;
    int32_t num_experts_per_tok = 6;
    int32_t num_hidden_layers = 12;
    int32_t num_key_value_heads = 10;
    int32_t q_lora_rank = -1;  // null in JSON
    int32_t qk_nope_head_dim = 0;
    int32_t qk_rope_head_dim = 0;
    bool rm_head = false;
    int32_t topk_group = 1;
    std::string topk_method = "greedy";
    bool use_mla = false;
    int32_t v_head_dim = 0;
    int32_t vocab_size = 129280;
  };

  struct ProjectorConfig {
    int32_t input_dim = 2048;
    std::string model_type = "mlp_projector";
    int32_t n_embed = 1280;
    std::string projector_type = "linear";
  };

  struct VisionConfig {
    int32_t image_size = 1024;
    float mlp_ratio = 3.7362;
    std::string model_name = "deeplip_b_l";
    std::string model_type = "vision";
  };

  std::string _name_or_path = "deepseek-ai/DeepSeek-OCR";
  std::vector<std::vector<int32_t>> candidate_resolutions = {{1024, 1024}};
  std::string global_view_pos = "head";
  std::string model_type = "deepseek_vl_v2";
  std::string tile_tag = "2D";
  std::string transformers_version = "4.46.3";

  LanguageConfig language_config;
  ProjectorConfig projector_config;
  VisionConfig vision_config;

  // Main config values
  int64_t bos_token_id = 0;
  int64_t eos_token_id = 1;
  int32_t first_k_dense_replace = 1;
  int32_t hidden_size = 1280;
  int32_t intermediate_size = 6848;
  int32_t kv_lora_rank = -1;  // null in JSON
  bool lm_head = true;
  int32_t max_position_embeddings = 8192;
  int32_t moe_intermediate_size = 896;
  int32_t n_group = 1;
  int32_t n_routed_experts = 64;
  int32_t n_shared_experts = 2;
  int32_t num_attention_heads = 10;
  int32_t num_experts_per_tok = 6;
  int32_t num_hidden_layers = 12;
  int32_t num_key_value_heads = 10;
  int32_t q_lora_rank = -1;  // null in JSON
  int32_t qk_nope_head_dim = 0;
  int32_t qk_rope_head_dim = 0;
  bool rm_head = false;
  int32_t topk_group = 1;
  std::string topk_method = "greedy";
  bool use_mla = false;
  int32_t v_head_dim = 0;
  int32_t vocab_size = 129280;

  // MLLM Related Stuff
  int32_t max_cache_length = 2048;
  aops::LinearImplTypes clip_linear_impl_type = aops::LinearImplTypes::kDefault;
  aops::LinearImplTypes sam_linear_impl_type = aops::LinearImplTypes::kDefault;
  aops::LinearImplTypes mlp_projector_linear_impl_type = aops::LinearImplTypes::kDefault;
  aops::LinearImplTypes lm_head_linear_impl_type = aops::LinearImplTypes::kDefault;
  aops::LinearImplTypes llm_mlp_linear_impl_type = aops::LinearImplTypes::kDefault;
};

}  // namespace mllm::models::deepseek_ocr
