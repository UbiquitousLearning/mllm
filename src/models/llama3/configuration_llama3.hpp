//
// Created by xwk on 25-1-10.
//

#ifndef MLLM_CONFIGURATION_LLAMA3_H
#define MLLM_CONFIGURATION_LLAMA3_H

#include "Types.hpp"
#include "models/transformer/configuration_transformer.hpp"

using namespace mllm;

class Llama3NameConfig : public TransformerNameConfig {
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

class Llama3Config : public TransformerConfig {
public:
    int vocab_size{};
    int hidden_dim{};
    int head_size{};
    int num_key_value_heads{};
    int ffn_hidden{};
    int block_num{};
    RoPEType RoPE_type;
    int cache_limit{};
    Llama3NameConfig names_config;
    float rope_theta;
    int max_position_embeddings;

    bool tie_word_embeddings = false;
    map<string, std::any> rope_scaling;

    explicit Llama3Config(int token_limit, string billions = "1B", RoPEType type = HFHUBROPE) {
        names_config.init(type);
        RoPE_type = type;
        cache_limit = token_limit;
        if (billions == "1B" || billions == "1b") {
            vocab_size = 128256;
            hidden_dim = 2048;
            head_size = 32;
            num_key_value_heads = 8;
            ffn_hidden = 8192;
            block_num = 16;
            max_position_embeddings = 131072;
            rope_theta = 500000.0;
            tie_word_embeddings = true;

            rope_scaling = {
                {"factor", 32.0f},
                {"high_freq_factor", 4.0f},
                {"low_freq_factor", 1.0f},
                {"original_max_position_embeddings", 8192},
                {"rope_type", std::string("llama3")}};
        } else if (billions == "3B" || billions == "3b") {
            vocab_size = 128256;
            hidden_dim = 3072;
            head_size = 24;
            num_key_value_heads = 8;
            ffn_hidden = 8192;
            block_num = 28;
            max_position_embeddings = 131072;
            rope_theta = 500000.0;
            tie_word_embeddings = true;

            rope_scaling = {
                {"factor", 32.0f},
                {"high_freq_factor", 4.0f},
                {"low_freq_factor", 1.0f},
                {"original_max_position_embeddings", 8192},
                {"rope_type", std::string("llama3")}};
        }
    }
};

#endif // MLLM_CONFIGURATION_LLAMA3_H