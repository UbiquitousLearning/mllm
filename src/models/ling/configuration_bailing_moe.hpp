#pragma once
#include "models/transformer/configuration_transformer.hpp"

using namespace mllm;

class BailingMoeNameConfig : public TransformerNameConfig {
public:
    std::string blk_name;
    std::string token_embd_name;
    std::string post_norm_name;
    std::string lm_head_name;
    std::string _gate_proj_name;

    void init() {
        blk_name = "model.layers.";
        _attn_base_name = "attention.";
        _ffn_base_name = "mlp.";
        _qkv_proj_name = "query_key_value";
        _o_proj_name = "dense";
        _gate_proj_name = "gate_proj";
        _up_proj_name = "up_proj";
        _down_proj_name = "down_proj";
        _attn_norm_name = "input_layernorm";
        _ffn_norm_name = "post_attention_layernorm";
        token_embd_name = "model.word_embeddings";
        post_norm_name = "model.norm";
        lm_head_name = "lm_head";
    }
};

struct BailingMoeConfig : public TransformerConfig {
    explicit BailingMoeConfig(int token_limit, string type = "A2.75B") : //"A1.3B"
        cache_limit(token_limit) {
        names_config.init();
    }

    int num_experts = 64;             // 64
    int num_experts_per_tok = 6;      // 6
    int num_shared_experts = 2;       // 2
    bool norm_topk_prob = true;       // true
    bool use_cache = true;            // true
    bool use_bias = false;            // false
    bool use_qkv_bias = false;        // false
    bool tie_word_embeddings = false; // false

    float attention_dropout = 0.0;
    int bos_token_id = 1;
    int eos_token_id = 126081; // 126081
    std::string hidden_act = "silu";
    int hidden_size = 2048;                           // 2048
    float initializer_range = 0.006;                  // 0.006
    int intermediate_size = 1408;                     // 1408
    int moe_intermediate_size = 1408;                 // 1408
    int max_position_embeddings = 32768;              // 32768
    std::string model_type = "ling_moe";              // "ling_moe"
    int num_attention_heads = 16;                     // 16
    int num_hidden_layers = 28;                       // 28
    int num_key_value_heads = 4;                      // 4
    double rms_norm_eps = 1e-06;                      // 1e-06
    float rope_theta = 600000.0;                      // 600000
    int vocab_size = 126464;                          // 126464
    int head_dim = hidden_size / num_attention_heads; // 2048/16= 128

    int cache_limit;
    RoPEType RoPE_type = RoPEType::HFHUBROPE;
    BailingMoeNameConfig names_config;
};