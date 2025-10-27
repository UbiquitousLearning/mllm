//
// Created by Rongjie Yi on 2024/2/4 0004.
//

#ifndef CONFIG_FUYU_HPP
#define CONFIG_FUYU_HPP
#include <models/transformer/configuration_transformer.hpp>

using namespace mllm;

class FuyuNameConfig : public TransformerNameConfig {
public:
    string blk_name = "language_model.model.layers.";
    string vision_embed_tokens_name = "vision_embed_tokens";
    string token_embd_name = "language_model.model.embed_tokens";
    string host_name = "language_model.model.";
    string post_norm_name = "language_model.model.final_layernorm";
    string lm_head_name = "language_model.lm_head";
    void init() {
        _attn_base_name = "self_attn.";
        _ffn_base_name = "mlp.";
        _qkv_proj_name = "query_key_value";
        _q_norm_name = "q_layernorm";
        _k_norm_name = "k_layernorm";
        _o_proj_name = "dense";
        _up_proj_name = "dense_h_to_4h";
        _down_proj_name = "dense_4h_to_h";
        _attn_norm_name = "input_layernorm";
        _ffn_norm_name = "post_attention_layernorm";
    }
};

class FuyuConfig {
public:
    int vocab_size{};
    int hidden_dim{};
    int head_size{};
    int ffn_hidden{};
    int block_num{};
    int patch_size{};
    int chl_size{};
    int cache_limit{};
    float rope_theta;
    int max_position_embeddings;

    FuyuNameConfig name_config;

    explicit FuyuConfig(int token_limit, const string &billions = "8B") {
        name_config.init();
        vocab_size = 262144;
        if (billions == "8B" || billions == "8b") {
            hidden_dim = 4096;
            head_size = 64;
            ffn_hidden = 4096 * 4;
            block_num = 36;
            patch_size = 30;
            chl_size = 3;
            max_position_embeddings = 16384;
            rope_theta = 25000;
        } else {
            throw std::runtime_error("Unsupported model size");
        }
        cache_limit = token_limit;
    }
    string attn_implementation = "flash_attention_2"; // Options: "flash_attention_2", "eager"
};

#endif // CONFIG_FUYU_HPP
