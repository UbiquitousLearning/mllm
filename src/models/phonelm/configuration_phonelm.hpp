/**
 * @file configuration_gemma.hpp
 * @author Chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief configuration file of qwen llm.
 * @version 0.1
 * @date 2024-04-03
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef CONFIG_PHONELM_HPP
#define CONFIG_PHONELM_HPP
#include "Types.hpp"
#include "models/transformer/configuration_transformer.hpp"
#include <cctype>
#include <iterator>

using namespace mllm;

class PhoneLMNameConfig : public TransformerNameConfig {
public:
    /**
     * @brief PhoneLM2 following the hugging face naming method
     *
     */
    void init() {
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
    }

    std::string blk_name;
    std::string token_embd_name;
    std::string post_norm_name;
    std::string lm_head_name;
    std::string _gate_proj_name;
};

struct PhoneLMdimConfig {
    int hidden_size = 1024;
    int intermediate_size = 2816;
    int num_attention_heads = 16;
    int num_key_value_heads = 16;
    int num_hidden_layers = 24;
    string activation = "ReLU";
};

struct PhoneLMConfig : public TransformerConfig {
    explicit PhoneLMConfig(int token_limit, string billions = "1.5B") :
        cache_limit(token_limit) {
        names_config.init();
        string billionsType;
        std::transform(billions.begin(), billions.end(), std::back_inserter(billionsType),
                       ::tolower);
        if (billionsType == "1.5b") {
            hidden_size = 2560;
            intermediate_size = 6816; // 6912; // 6816;
            num_attention_heads = 16;
            num_key_value_heads = 16;
            num_hidden_layers = 19;
        } else if (billionsType == "0.5b") {
            hidden_size = 1024;
            intermediate_size = 4864;
            num_attention_heads = 16;
            num_key_value_heads = 16;
            num_hidden_layers = 24;
        } else {
            throw std::runtime_error("Unsupported PhoneLM model size");
        }
    };
    explicit PhoneLMConfig(int token_limit, PhoneLMdimConfig dim_config) :
        cache_limit(token_limit) {
        names_config.init();
        hidden_size = dim_config.hidden_size;
        intermediate_size = dim_config.intermediate_size;
        num_attention_heads = dim_config.num_attention_heads;
        num_key_value_heads = dim_config.num_key_value_heads;
        num_hidden_layers = dim_config.num_hidden_layers;
        hidden_act = dim_config.activation;
    };

    float attention_dropout = 0.0;
    int bos_token_id = 151643;
    int eos_token_id = 151643;
    std::string hidden_act = "ReLU";
    int hidden_size = 1024;
    float initializer_range = 0.02;
    int intermediate_size = 2816;
    int max_position_embeddings = 32768;
    int max_window_layers = 21;
    int num_attention_heads = 16;
    int num_hidden_layers = 24;
    int num_key_value_heads = 16;
    double rms_norm_eps = 1e-6;
    float rope_theta = 10000.0;
    int vocab_size = 49152;
    bool tie_embedding_words = true;

    int cache_limit;
    RoPEType RoPE_type = RoPEType::HFHUBROPE;
    PhoneLMNameConfig names_config;
};

#endif //! CONFIG_PHONELM_HPP
