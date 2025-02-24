/**
 * @file configuration_ds_qwen2.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-24
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "Types.hpp"
#include "models/transformer/configuration_transformer.hpp"
#include <cctype>
#include <iterator>

using namespace mllm;

class QWenNameConfig : public TransformerNameConfig {
public:
    /**
     * @brief QWen2 following the hugging face naming method
     *
     * @param type RoPEType
     */
    void init(RoPEType type = RoPEType::HFHUBROPE) {
        switch (type) {
        case RoPEType::HFHUBROPE: {
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
            throw std::runtime_error("Unsupported gemma RoPE type");
        }
        }
    }

    std::string blk_name;
    std::string token_embd_name;
    std::string post_norm_name;
    std::string lm_head_name;
    std::string _gate_proj_name;
};

struct QWenConfig : public TransformerConfig {
    explicit QWenConfig(int token_limit, string billions = "1.5B", RoPEType type = RoPEType::HFHUBROPE) :
        cache_limit(token_limit) {
        names_config.init(type);
        string billionsType;
        std::transform(billions.begin(), billions.end(), std::back_inserter(billionsType),
                       ::tolower);
        if (billionsType == "1.5b") {
            attention_dropout = 0.0;
            std::string hidden_act = "silu";
            hidden_size = 1536;
            intermediate_size = 8960;
            max_position_embeddings = 131072;
            max_window_layers = 28;
            num_attention_heads = 12;
            num_hidden_layers = 28;
            num_key_value_heads = 2;
            rms_norm_eps = 1e-6;
            rope_theta = 10000.f;
            sliding_window = 131072;
            vocab_size = 151936;
            tie_embedding_words = false;
        } else {
            throw std::runtime_error("Unsupported model size");
        }
        RoPE_type = type;
    };

    float attention_dropout = 0.0;
    int bos_token_id = 151643;
    int eos_token_id = 151643;
    std::string hidden_act = "silu";
    int hidden_size = 1024;
    float initializer_range = 0.02;
    int intermediate_size = 2816;
    int max_position_embeddings = 32768;
    int max_window_layers = 21;
    std::string model_type = "qwen2";
    int num_attention_heads = 16;
    int num_hidden_layers = 24;
    int num_key_value_heads = 16;
    double rms_norm_eps = 1e-6;
    float rope_theta = 1000000.0;
    int sliding_window = 32768;
    int vocab_size = 151936;
    bool tie_embedding_words = false;

    int cache_limit;
    RoPEType RoPE_type = RoPEType::HFHUBROPE;
    QWenNameConfig names_config;
};
