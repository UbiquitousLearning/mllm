/**
 * @file configuration_mistral.hpp
 * @author Chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief Mistral 7B instruction 0.2V in Huggingface
 * @version 0.1
 * @date 2024-05-25
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef CONFIG_MISTRAL_HPP
#define CONFIG_MISTRAL_HPP

#include "Types.hpp"
#include "models/transformer/configuration_transformer.hpp"
#include <cctype>
#include <iterator>

using namespace mllm;

class MistralNameConfig : public TransformerNameConfig {
public:
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
        case RoPEType::LLAMAROPE: /*the mistral is same to llama*/ {
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

struct MistralConfig : public TransformerConfig {
    explicit MistralConfig(int token_limit, string billions = "7B", RoPEType type = RoPEType::HFHUBROPE) :
        cache_limit(token_limit) {
        names_config.init(type);
        string billionsType;
        std::transform(billions.begin(), billions.end(), std::back_inserter(billionsType),
                       ::tolower);
        if (billionsType == "7b") {
            attention_dropout = 0.0;
            bos_token_id = 1;
            eos_token_id = 2;
            hidden_act = "silu";
            hidden_size = 4096;
            initializer_range = 0.02;
            intermediate_size = 14336;
            max_position_embeddings = 32768;
            model_type = "mistral";
            num_attention_heads = 32;
            num_hidden_layers = 32;
            num_key_value_heads = 8;
            rms_norm_eps = 1e-05;
            rope_theta = 1000000.0;
            vocab_size = 32000;
        } else {
            throw std::runtime_error("Unsupported model size");
        }
        RoPE_type = type;
    }

    float attention_dropout = 0.0;
    int bos_token_id = 1;
    int eos_token_id = 2;
    std::string hidden_act = "silu";
    int hidden_size = 4096;
    float initializer_range = 0.02;
    int intermediate_size = 14336;
    int max_position_embeddings = 32768;
    std::string model_type = "mistral";
    int num_attention_heads = 32;
    int num_hidden_layers = 32;
    int num_key_value_heads = 8;
    double rms_norm_eps = 1e-05;
    float rope_theta = 1000000.0;
    int vocab_size = 32000;

    int cache_limit;
    RoPEType RoPE_type = RoPEType::HFHUBROPE;
    MistralNameConfig names_config;
};

#endif //! CONFIG_MISTRAL_HPP