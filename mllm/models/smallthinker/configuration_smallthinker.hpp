#ifndef CONFIG_SMOLTHINKER_HPP
#define CONFIG_SMOLTHINKER_HPP
#include "models/transformer/configuration_transformer.hpp"

using namespace mllm;

class SmallThinkerNameConfig : public TransformerNameConfig {
public:
    std::string blk_name;
    std::string token_embd_name;
    std::string post_norm_name;
    std::string lm_head_name;
    std::string _gate_proj_name;

    void init() {
        blk_name = "model.layers.";
        _attn_base_name = "self_attn.";
        _ffn_base_name = "block_sparse_moe.";
        _q_proj_name = "q_proj";
        _k_proj_name = "k_proj";
        _v_proj_name = "v_proj";
        _o_proj_name = "o_proj";
        _gate_proj_name = "gate";
        _up_proj_name = "up";
        _down_proj_name = "down";
        _attn_norm_name = "input_layernorm";
        _ffn_norm_name = "post_attention_layernorm";
        token_embd_name = "model.embed_tokens";
        post_norm_name = "model.norm";
        lm_head_name = "lm_head";
    }
};

struct SmallThinkerConfig : public TransformerConfig {
    explicit SmallThinkerConfig(int token_limit, string billions = "4BA0.6B") :
        cache_limit(token_limit) {
        names_config.init();
        string billionsType;
        std::transform(billions.begin(), billions.end(), std::back_inserter(billionsType),
                       ::tolower);
        if (billionsType == "4ba0.6b") {
        }
        if (billionsType == "4ba0.6b-lm") {
            tie_embedding_words = false;
        } else {
            throw std::runtime_error("Unsupported model size");
        }
    }

    int num_experts = 32;
    int num_experts_per_tok = 4;

    // std::string hidden_act = "relu";
    int hidden_size = 1536;
    int intermediate_size = 768;
    int max_position_embeddings = 32768;
    int num_hidden_layers = 32;
    int num_attention_heads = 12;
    int num_key_value_heads = 2;
    double rms_norm_eps = 1e-06;
    float rope_theta = 1.5e6;
    int vocab_size = 151936;
    int head_dim = 128; // hidden_size/num_attention_heads

    int cache_limit;
    RoPEType RoPE_type = RoPEType::HFHUBROPE;
    SmallThinkerNameConfig names_config;
    bool tie_embedding_words = true; // false;
};

#endif // CONFIG_SMOLTHINKER_HPP
