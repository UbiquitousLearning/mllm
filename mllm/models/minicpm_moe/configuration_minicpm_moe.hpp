#ifndef CONFIG_MINICPMMOE_HPP
#define CONFIG_MINICPMMOE_HPP
#include "models/transformer/configuration_transformer.hpp"

using namespace mllm;

class MiniCPMNameConfig : public TransformerNameConfig {
public:
    std::string blk_name;
    std::string token_embd_name;
    std::string post_norm_name;
    std::string lm_head_name;
    std::string _gate_proj_name;

    void init() {
        blk_name = "model.layers.";
        _attn_base_name = "self_attn.";
        _ffn_base_name = "mlp.";
        _q_proj_name = "q_proj";
        _k_proj_name = "k_proj";
        _v_proj_name = "v_proj";
        _o_proj_name = "o_proj";
        _gate_proj_name = "w1";
        _up_proj_name = "w3";
        _down_proj_name = "w2";
        _attn_norm_name = "input_layernorm";
        _ffn_norm_name = "post_attention_layernorm";
        token_embd_name = "model.embed_tokens";
        post_norm_name = "model.norm";
        lm_head_name = "lm_head";
    }
};

struct MiniCPMConfig : public TransformerConfig {
    explicit MiniCPMConfig(int token_limit, string billions = "2B") :
        cache_limit(token_limit) {
        names_config.init();
        string billionsType;
        std::transform(billions.begin(), billions.end(), std::back_inserter(billionsType),
                       ::tolower);
        if (billionsType == "2b") {
            attention_dropout = 0.0;
            bos_token_id = 1;
            eos_token_id = 2;
            hidden_act = "silu";
            hidden_size = 2304;
            initializer_range = 0.1;
            intermediate_size = 5760;
            max_position_embeddings = 4096;
            model_type = "minicpm";
            num_attention_heads = 36;
            num_hidden_layers = 40;
            num_key_value_heads = 36;
            rms_norm_eps = 1e-05;
            rope_theta = 10000.0;
            vocab_size = 122753;
            head_dim = 64;
            scale_depth = 1.4;
            scale_emb = 12;
            dim_model_base = 256;
            num_experts = 8;
            num_experts_per_tok = 2;
        } else {
            throw std::runtime_error("Unsupported model size");
        }
    }

    int num_experts = 8;
    int num_experts_per_tok = 2;

    float attention_dropout = 0.0;
    int bos_token_id = 1;
    int eos_token_id = 2;
    std::string hidden_act = "silu";
    int hidden_size = 2304;
    float initializer_range = 0.1;
    int intermediate_size = 5760;
    int max_position_embeddings = 4096;
    std::string model_type = "minicpm";
    int num_attention_heads = 36;
    int num_hidden_layers = 40;
    int num_key_value_heads = 36;
    double rms_norm_eps = 1e-05;
    float rope_theta = 10000.0;
    int vocab_size = 122753;
    int head_dim = 64; // self.hidden_size // self.num_heads
    float scale_depth = 1.4;
    float scale_emb = 12;
    float dim_model_base = 256;

    int cache_limit;
    RoPEType RoPE_type = RoPEType::HFHUBROPE;
    MiniCPMNameConfig names_config;
};

#endif // CONFIG_MINICPMMOE_HPP
