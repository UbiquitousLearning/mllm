/**
 * @file configuration_gemma.hpp
 * @author Chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief configuration file of gemma llm.
 * @version 0.1
 * @date 2024-04-03
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef CONFIG_GEMMA_HPP
#define CONFIG_GEMMA_HPP
#include "Types.hpp"
#include "models/transformer/configuration_transformer.hpp"

using namespace mllm;

class GemmaNameConfig : public TransformerNameConfig {
public:
    /**
     * @brief Gemma following the hugging face naming method
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
            lm_head_name = "model.embed_tokens";
            break;
        }
        case RoPEType::LLAMAROPE: /*the gemma is same to llama*/ {
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
            lm_head_name = "tok_embeddings";
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

struct GemmaConfig : public TransformerConfig {
    explicit GemmaConfig(int token_limit, const string billions = "2B", RoPEType type = RoPEType::HFHUBROPE) :
        cache_limit(token_limit) {
        names_config.init(type);
        if (!(billions == "2B" || billions == "2b")) {
            throw std::runtime_error("Unsupported model size");
        }
        RoPE_type = type;
    };

    int vocab_size = 256000;
    int max_position_embeddings = 8192;
    int num_hidden_layers = 18;
    int num_attention_heads = 8;
    int num_key_value_heads = 1;
    int hidden_size = 2048;
    int intermediate_size = 16384;
    int head_dim = 256;
    float rms_norm_eps = 1e-6;
    float rope_theta = 10000;

    int cache_limit;
    RoPEType RoPE_type = RoPEType::HFHUBROPE;
    GemmaNameConfig names_config;
};

#endif //! CONFIG_GEMMA_HPP
