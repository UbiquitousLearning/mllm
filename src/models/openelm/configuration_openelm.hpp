/**
 * @file configuration_openelm.hpp
 * @author Chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-09-25
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef CONFIG_OPENELM_HPP
#define CONFIG_OPENELM_HPP
#include "Types.hpp"
#include "models/transformer/configuration_transformer.hpp"
#include <cctype>
#include <iterator>

using namespace mllm;

class OpenELMNameConfig : public TransformerNameConfig {
public:
    /**
     * @brief OpenELM2 following the hugging face naming method
     *
     * @param type RoPEType
     */
    void init(RoPEType type = RoPEType::HFHUBROPE) {
        switch (type) {
        case RoPEType::HFHUBROPE: {
            blk_name = "transformer.layers.";
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

struct OpenELMConfig : public TransformerConfig {
    explicit OpenELMConfig(int token_limit, string billions = "1.1B", RoPEType type = RoPEType::HFHUBROPE) :
        cache_limit(token_limit) {
        names_config.init(type);
        string billionsType;
        std::transform(billions.begin(), billions.end(), std::back_inserter(billionsType),
                       ::tolower);
        if (billionsType == "1.1b") {
            // Do nothing.
        } else if (billionsType == "450m") {
            // TODO
        } else {
            throw std::runtime_error("Unsupported model size");
        }
        RoPE_type = type;
    };

    int bos_token_id = 1;
    int eos_token_id = 2;
    int ffn_dim_divisor = 256;
    std::vector<float> ffn_multipliers{
        0.5,
        0.63,
        0.76,
        0.89,
        1.02,
        1.15,
        1.28,
        1.41,
        1.54,
        1.67,
        1.8,
        1.93,
        2.06,
        2.19,
        2.31,
        2.44,
        2.57,
        2.7,
        2.83,
        2.96,
        3.09,
        3.22,
        3.35,
        3.48,
        3.61,
        3.74,
        3.87,
        4.0,
    };
    int head_dim = 64;
    float initializer_range = 0.02;
    int max_context_length = 2048;
    int model_dim = 2048;
    int num_gqa_groups = 4;
    std::vector<int> num_kv_heads{
        4,
        4,
        4,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        7,
        7,
        7,
        7,
        7,
        7,
        8,
        8,
        8,
        8,
    };
    std::vector<int> num_query_heads{
        16,
        16,
        16,
        20,
        20,
        20,
        20,
        20,
        20,
        20,
        24,
        24,
        24,
        24,
        24,
        24,
        24,
        24,
        28,
        28,
        28,
        28,
        28,
        28,
        32,
        32,
        32,
        32,
    };
    int num_transformer_layers = 28;
    std::vector<float> qkv_multipliers{
        0.5,
        1.0,
    };
    float rope_freq_constant = 10000;
    int rope_max_length = 4096;
    int vocab_size = 32000;
    int cache_limit;
    RoPEType RoPE_type = RoPEType::HFHUBROPE;
    OpenELMNameConfig names_config;
};

#endif //! CONFIG_OPENELM_HPP
