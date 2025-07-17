//
// Created by Lu Yiwen on 2025/6/3 .
//
#ifndef CONFIG_PHI4_HPP
#define CONFIG_PHI4_HPP
#include "models/transformer/configuration_transformer.hpp"

using namespace mllm;

class Phi4NameConfig : public TransformerNameConfig {
public:
    std::string blk_name;
    std::string token_embd_name;
    std::string post_norm_name;
    std::string lm_head_name;
    std::string _gate_up_proj_name;

    void init(RoPEType = HFHUBROPE) {
        blk_name = "model.layers.";
        _attn_base_name = "self_attn.";
        _ffn_base_name = "mlp.";
        _qkv_proj_name = "qkv_proj";
        _o_proj_name = "o_proj";
        _gate_up_proj_name = "gate_up_proj";
        _down_proj_name = "down_proj";
        _attn_norm_name = "input_layernorm";
        _ffn_norm_name = "post_attention_layernorm";
        token_embd_name = "model.embed_tokens";
        post_norm_name = "model.norm";
        lm_head_name = token_embd_name; 
    }
};

class Phi4Config : public TransformerConfig {
public:
    
    int vocab_size{};
    int hidden_dim{};
    int head_size{};
    int num_key_value_heads{};
    int ffn_hidden{};
    int block_num{};
    int max_position_embeddings;
    // RoPE
    RoPEType RoPE_type;
    float rope_theta;
    int rope_original_max_position_embeddings; 
    std::vector<float> rope_long_factor;       
    std::vector<float> rope_short_factor;      
    
    float attention_dropout; 
    float rms_norm_eps;      
    int num_attention_heads; 
    
    int cache_limit{};
    Phi4NameConfig names_config;
    bool tie_embedding_words;
    bool attention_bias;
    float partial_rotary_factor;

    explicit Phi4Config(int token_limit, string billions = "4-mini", RoPEType type = HFHUBROPE, int vocab = 200064) {
        names_config.init(type);

        if (billions == "4-mini" || billions == "phi4-mini") {
            vocab_size = 200064;
            hidden_dim = 3072;                // config.hidden_size
            head_size = 3072 / 24;            // hidden_size/num_attention_heads
            num_key_value_heads = 8;          // config.num_key_value_heads
            ffn_hidden = 8192;                // config.intermediate_size
            block_num = 32;                   // config.num_hidden_layers
            max_position_embeddings = 131072; // config.original_max_position_embeddings
            rope_theta = 10000.0f;            // config.rope_theta

            // NEW
            num_attention_heads = 24; // config.json.num_attention_heads
            attention_dropout = 0.0f; // config.json.attention_dropout
            rms_norm_eps = 1e-5f;     // config.json.rms_norm_eps
            tie_embedding_words = true;
            attention_bias = false;
            partial_rotary_factor = 0.75;
        } else {
            throw std::runtime_error("Unsupported model size");
        }
        RoPE_type = type;
        
        rope_original_max_position_embeddings = 4096;

        rope_long_factor = {
            1.0f, 1.118320672f, 1.250641126f, 1.398617824f,
            1.564103225f, 1.74916897f, 1.956131817f, 2.187582649f,
            2.446418898f, 2.735880826f, 3.059592084f, 3.421605075f,
            3.826451687f, 4.279200023f, 4.785517845f, 5.351743533f,
            5.984965424f, 6.693110555f, 7.485043894f, 8.370679318f,
            9.36110372f, 10.4687158f, 11.70738129f, 13.09260651f,
            14.64173252f, 16.37415215f, 18.31155283f, 20.47818807f,
            22.90118105f, 25.61086418f, 28.64115884f, 32.03f,
            32.1f, 32.13f, 32.23f, 32.6f,
            32.61f, 32.64f, 32.66f, 32.7f,
            32.71f, 32.93f, 32.97f, 33.28f,
            33.49f, 33.5f, 44.16f, 47.77f};

        rope_short_factor = rope_long_factor;

        cache_limit = token_limit;
    }
    
    void validate_rope_scaling() const {
        int head_dim = hidden_dim / num_attention_heads;     // 3072 / 24 = 128
        int rotary_ndims = head_dim * partial_rotary_factor; // 96
        int expect_len = rotary_ndims / 2;                   // 48
        if ((int)rope_long_factor.size() != expect_len) {
            throw std::runtime_error(
                "`rope_long_factor` length must be " + std::to_string(expect_len) + ", but got " + std::to_string(rope_long_factor.size()));
        }
        if ((int)rope_short_factor.size() != expect_len) {
            throw std::runtime_error(
                "`rope_short_factor` length must be " + std::to_string(expect_len) + ", but got " + std::to_string(rope_short_factor.size()));
        }
    }
};

#endif // CONFIG_PHI4_HPP
