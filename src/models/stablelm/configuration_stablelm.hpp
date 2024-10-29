#ifndef CONFIG_STABLELM_HPP
#define CONFIG_STABLELM_HPP
#include "models/transformer/configuration_transformer.hpp"

using namespace mllm;

class stablelmNameConfig : public TransformerNameConfig {
public:
    std::string blk_name;
    std::string token_embd_name;
    std::string post_norm_name;
    std::string lm_head_name;
    std::string _gate_proj_name;

    void init(RoPEType type = HFHUBROPE) {
        switch (type) {
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

class StableLMConfig : public TransformerConfig {
public:
    int vocab_size{};
    int hidden_dim{};
    int head_size{};
    int ffn_hidden{};
    int block_num{};
    RoPEType RoPE_type;
    int cache_limit{};
    stablelmNameConfig names_config;

    explicit StableLMConfig(int token_limit, string billions = "1.6B", RoPEType type = HFHUBROPE, int vocab = 100352) {
        names_config.init(type);
        vocab_size = vocab;
        if (billions == "1.6B" || billions == "1.6b") {
            hidden_dim = 2048;
            head_size = 32;
            ffn_hidden = 5632;
            block_num = 24;
        } else {
            throw std::runtime_error("Unsupported model size");
        }
        RoPE_type = type;
        cache_limit = token_limit;
    }
};

#endif //