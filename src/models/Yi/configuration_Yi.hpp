/**
 * @file configuration_Yi.hpp
 * @author Chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-07-02
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef CONFIG_YI_HPP
#define CONFIG_YI_HPP
#include "models/transformer/configuration_transformer.hpp"

using namespace mllm;

class YiNameConfig : public TransformerNameConfig {
public:
    std::string blk_name;
    std::string token_embd_name;
    std::string post_norm_name;
    std::string lm_head_name;
    std::string _gate_proj_name;

    void init(RoPEType type = LLAMAROPE) {
        switch (type) {
        case LLAMAROPE: {
            blk_name = "layers.";
            _attn_base_name = "attention.";
            _ffn_base_name = "feed_forward.";
            _q_proj_name = "wq";
            _k_proj_name = "wk";
            _v_proj_name = "wv";
            _o_proj_name = "wo";
            _gate_proj_name = "w1";
            _up_proj_name = "w3";
            _down_proj_name = "w2";
            _attn_norm_name = "attention_norm";
            _ffn_norm_name = "ffn_norm";
            token_embd_name = "tok_embeddings";
            post_norm_name = "norm";
            lm_head_name = "output";
            break;
        }
        case HFHUBROPE: {
            blk_name = "model.layers.";
            _attn_base_name = "self_attn.";
            _ffn_base_name = "mlp.";
            _q_proj_name = "q_proj";
            _k_proj_name = "k_proj";
            _v_proj_name = "v_proj";
            _o_proj_name = "o_proj";
            _gate_proj_name = "gate_proj";
            _up_proj_name = "up_proj";
            _down_proj_name = "down_proj";
            _attn_norm_name = "input_layernorm";
            _ffn_norm_name = "post_attention_layernorm";
            token_embd_name = "model.embed_tokens";
            post_norm_name = "model.norm";
            lm_head_name = "lm_head";
            break;
        }
        default: {
            throw std::runtime_error("Unsupported llama type");
        }
        }
    }
};

class YiConfig {
public:
    explicit YiConfig(int token_limit, string billions = "6B", RoPEType type = LLAMAROPE, int vocab = 64000) {
        names_config.init(type);
        vocab_size = vocab;
        if (!(billions == "6B" || billions == "6b")) {
            throw std::runtime_error("Unsupported model size");
        }
        RoPE_type = type;
        cache_limit = token_limit;
    }

public:
    bool attention_bias = false;
    float attention_drop = 0.0;
    int pad_token_id = 0;
    int bos_token_id = 1;
    int eos_token_id = 2;
    int hidden_size = 4096;
    float initializer_range = 0.02;
    int intermediate_size = 11008;
    int max_position_embeddings = 4096;
    int num_attention_heads = 32;
    int num_hidden_layers = 32;
    int num_key_value_heads = 4;
    int pretraining_tp = 1;
    float rms_norm_eps = 1e-6;
    float rope_theta = 5000000.0;
    int vocab_size = 64000;
    int cache_limit;
    RoPEType RoPE_type;
    YiNameConfig names_config;
};

#endif //! CONFIG_YI_HPP