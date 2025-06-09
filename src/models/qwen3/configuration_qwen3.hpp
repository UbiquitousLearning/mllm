/**
 * @file configuration_qwen3.hpp
 * @author hyh
 * @brief configuration file of qwen3 llm.
 * @version 0.1
 * @date 2025-05-11
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef CONFIGURATION_QWEN3_HPP
#define CONFIGURATION_QWEN3_HPP
#include "Types.hpp"
#include "models/transformer/configuration_transformer.hpp"
#include <cctype>
#include <iterator>

using namespace mllm;

class QWen3NameConfig : public TransformerNameConfig {
public:
    /**
     * @brief QWen3 following the hugging face naming method
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
        case RoPEType::LLAMAROPE: {
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

struct QWen3Config : public TransformerConfig {
    explicit QWen3Config(int token_limit, string billions = "0.6B", RoPEType type = RoPEType::HFHUBROPE) :
        cache_limit(token_limit) {
        names_config.init(type);
        string billionsType;
        std::transform(billions.begin(), billions.end(), std::back_inserter(billionsType),
                       ::tolower);
        if (billionsType == "0.6b") {
            attention_bias = false;
            attention_dropout = 0.0;
            bos_token_id = 151643;
            eos_token_id = 151645;
            head_dim = 128;
            hidden_act = "silu";
            hidden_size = 1024;
            initializer_range = 0.02;
            intermediate_size = 3072;
            max_position_embeddings = 40960;
            max_window_layers = 28;
            model_type = "qwen3";
            num_attention_heads = 16;
            num_hidden_layers = 28;
            num_key_value_heads = 8;
            rms_norm_eps = 1e-6;
            rope_theta = 1000000.0;
            vocab_size = 151936;
            tie_embedding_words = true;
        } 
        else if(billionsType == "4b"){
            attention_bias = false;
            attention_dropout = 0.0;
            bos_token_id = 151643;
            eos_token_id = 151645;
            head_dim = 128;
            hidden_act = "silu";
            hidden_size = 2560;
            initializer_range = 0.02;
            intermediate_size = 9728;
            max_position_embeddings = 40960;
            max_window_layers = 36;
            model_type = "qwen3";
            num_attention_heads = 32;
            num_hidden_layers = 36;
            num_key_value_heads = 8;
            rms_norm_eps = 1e-6;
            rope_theta = 1000000.0;
            vocab_size = 151936;
            tie_embedding_words = true;
        }
        else {
            throw std::runtime_error("Unsupported model size");
        }
        RoPE_type = type;
    };

    //这下面是赋初始默认值，上面是构造函数，构造函数中的值会覆盖掉初始默认值
    
    bool attention_bias = false;
    float attention_dropout = 0.0;
    int bos_token_id = 151643;
    int eos_token_id = 151645;
    int head_dim = 128;
    std::string hidden_act = "silu";
    int hidden_size = 1024;
    float initializer_range = 0.02;
    int intermediate_size = 3072;
    int max_position_embeddings = 40960;
    int max_window_layers = 28;
    std::string model_type = "qwen3";
    int num_attention_heads = 16;
    int num_hidden_layers = 28;
    int num_key_value_heads = 8;
    double rms_norm_eps = 1e-6;
    float rope_theta = 1000000.0;
    int vocab_size = 151936;
    bool tie_embedding_words = true;    
    

    int cache_limit;
    RoPEType RoPE_type = RoPEType::HFHUBROPE;
    QWen3NameConfig names_config;
};

#endif 
