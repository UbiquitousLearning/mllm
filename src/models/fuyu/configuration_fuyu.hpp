//
// Created by Rongjie Yi on 2024/2/4 0004.
//

#ifndef CONFIG_FUYU_HPP
#define CONFIG_FUYU_HPP

using namespace mllm;

class FuyuNameConfig {
public:
    string blk_name = "language_model.model.layers.";
    string vision_embed_tokens_name = "vision_embed_tokens";
    string token_embd_name = "language_model.model.embed_tokens";
    string host_name = "language_model.model.";
    string _attn_base_name = "self_attn.";
    string _ffn_base_name = "mlp.";
    string _qkv_proj_name = "query_key_value";
    string _q_norm_name = "q_layernorm";
    string _k_norm_name = "k_layernorm";
    string _o_proj_name = "dense";
    string _up_proj_name = "dense_h_to_4h";
    string _down_proj_name = "dense_4h_to_h";
    string _attn_norm_name = "input_layernorm";
    string _ffn_norm_name = "post_attention_layernorm";
    string post_norm_name = "language_model.model.final_layernorm";
    string lm_head_name = "language_model.lm_head";
};

class FuyuConfig {
public:
    int vocab_size{};
    int hidden_dim{};
    int head_size{};
    int mlp_hidden{};
    int block_num{};
    int patch_size{};
    int chl_size{};
    int cache_limit{};

    FuyuNameConfig name_config;

    void init(int token_limit, const string &billions = "8B") {
        vocab_size = 262144;
        if (billions == "8B" || billions == "8b") {
            hidden_dim = 4096;
            head_size = 64;
            mlp_hidden = 4096 * 4;
            block_num = 36;
            patch_size = 30;
            chl_size = 3;
        } else {
            throw std::runtime_error("Unsupported model size");
        }
        cache_limit = token_limit;
    }
};

#endif // CONFIG_FUYU_HPP
