//
// Created by Guo Xiaoqiang on 2024/8/12 .
//
#ifndef CONFIG_PHI3_HPP
#define CONFIG_PHI3_HPP
#include "models/transformer/configuration_transformer.hpp"

using namespace mllm;

class Phi3NameConfig : public TransformerNameConfig {
public:
    std::string blk_name;
    std::string token_embd_name;
    std::string post_norm_name;
    std::string lm_head_name;
    std::string _gate_up_proj_name;

    void init(RoPEType type = HFHUBROPE) {
        switch (type) {
        case HFHUBROPE: {
            blk_name = "model.layers.";
            _attn_base_name = "self_attn.";
            _ffn_base_name = "mlp.";
            _qkv_proj_name = "qkv_proj";
            _o_proj_name = "o_proj";
            _gate_up_proj_name = "gate_up_proj";
            _down_proj_name = "down_proj";
            _attn_norm_name = "input_layernorm";
            _ffn_norm_name = "post_attention_layernorm";
            token_embd_name = "model.embed_tokens";
            post_norm_name = "model.norm";
            lm_head_name = "lm_head";
            break;
        }
        default: {
            throw std::runtime_error("Unsupported phi3 type");
        }
        }
    }
};

class Phi3Config : public TransformerConfig {
public:
    int vocab_size{};
    int hidden_dim{};
    int head_size{};
    int num_key_value_heads{};
    int ffn_hidden{};
    int block_num{};
    RoPEType RoPE_type;
    int cache_limit{};
    Phi3NameConfig names_config;
    float rope_theta;
    int max_position_embeddings;

    explicit Phi3Config(int token_limit, string billions = "3.8B", RoPEType type = HFHUBROPE, int vocab = 32064) {
        names_config.init(type);
        vocab_size = vocab;
        if (billions == "3.8B" || billions == "3.8b") {
            hidden_dim = 3072;
            head_size = 32;
            num_key_value_heads = 32;
            ffn_hidden = 8192;
            block_num = 32;
            max_position_embeddings = 4096;
            rope_theta = 10000.0;
        } else {
            throw std::runtime_error("Unsupported model size");
        }
        RoPE_type = type;
        cache_limit = token_limit;
    }
};

#endif // CONFIG_PHI3_HPP
