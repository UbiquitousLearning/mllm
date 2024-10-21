/**
 * @file configuration_smollm.hpp
 * @author Chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2024-09-25
 *
 * @copyright Copyright (c) 2024
 *
 */
#pragma once
#ifndef CONFIG_SMOLLM_HPP
#define CONFIG_SMOLLM_HPP
#include "Types.hpp"
#include "models/transformer/configuration_transformer.hpp"

using namespace mllm;

class SmolLMNameConfig : public TransformerNameConfig {
public:
    std::string blk_name;
    std::string token_embd_name;
    std::string post_norm_name;
    std::string lm_head_name;
    std::string _gate_proj_name;

    void init(RoPEType type = HFHUBROPE) {
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

class SmolLMConfig : public TransformerConfig {
public:
    int vocab_size{};
    int hidden_dim{};
    int head_size{};
    int num_key_value_heads{};
    int ffn_hidden{};
    int block_num{};
    RoPEType RoPE_type;
    int cache_limit{};
    SmolLMNameConfig names_config;
    float rope_theta;
    int max_position_embeddings;

    explicit SmolLMConfig(int token_limit, string billions = "1.7B", RoPEType type = HFHUBROPE, int vocab = 49152) {
        names_config.init(type);
        vocab_size = vocab;
        if (billions == "1.7B" || billions == "1.7b") {
            hidden_dim = 2048;
            head_size = 32;
            num_key_value_heads = 32;
            ffn_hidden = 8192;
            block_num = 24;
            max_position_embeddings = 2048;
            rope_theta = 10000;
        } else if (billions == "360M" || billions == "360m") {
            hidden_dim = 960;
            head_size = 15;
            num_key_value_heads = 5;
            ffn_hidden = 2560;
            block_num = 32;
            max_position_embeddings = 2048;
            rope_theta = 10000;
        } else {
            throw std::runtime_error("Unsupported model size");
        }
        RoPE_type = type;
        cache_limit = token_limit;
    }
};

#endif // CONFIG_SMOLLM_HPP