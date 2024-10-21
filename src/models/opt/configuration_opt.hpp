#ifndef CONFIG_OPT_HPP
#define CONFIG_OPT_HPP
#include "models/transformer/configuration_transformer.hpp"

using namespace mllm;

class optNameConfig : public TransformerNameConfig {
public:
    std::string blk_name;
    std::string token_embd_name;
    std::string post_norm_name;
    std::string lm_head_name;
    std::string pos_name;

    void init() {
        blk_name = "model.decoder.layers.";
        _attn_base_name = "self_attn.";
        _ffn_base_name = "fc";
        _q_proj_name = "q_proj";
        _k_proj_name = "k_proj";
        _v_proj_name = "v_proj";
        _o_proj_name = "out_proj";
        _up_proj_name = "1";
        _down_proj_name = "2";
        _attn_norm_name = "self_attn_layer_norm";
        _ffn_norm_name = "final_layer_norm";
        token_embd_name = "model.decoder.embed_tokens";
        post_norm_name = "model.decoder.final_layer_norm";
        lm_head_name = "lm_head";
        pos_name = "model.decoder.embed_positions";
    }
};

class OPTConfig : public TransformerConfig {
public:
    optNameConfig names_config;
    int vocab_size{};
    int hidden_dim{};
    int head_size{};
    int block_num{};
    int ffn_hidden{};
    int max_position_embeddings{};
    float init_std{};
    float dropout{};
    float attention_dropout{};
    float activation_dropout{};
    bool do_layer_norm_before{};
    std::string prefix{};
    std::string torch_dtype{};
    bool use_cache{};
    int cache_limit{};

    explicit OPTConfig(int token_limit, string billions = "1.3B") {
        if (billions == "1.3B" || billions == "1.3b") {
            vocab_size = 50272;
            hidden_dim = 2048;
            head_size = 32;
            ffn_hidden = 8192;
            block_num = 24;
            max_position_embeddings = 2048;
            init_std = 0.02;
            dropout = 0.1;
            attention_dropout = 0.0;
            activation_dropout = 0.0;
            do_layer_norm_before = true;
            prefix = "</s>";
            torch_dtype = "float16";
            use_cache = true;
        } else {
            throw std::runtime_error("Unsupported model size");
        }
        names_config.init();
        cache_limit = token_limit;
    }
};

#endif // CONFIG_OPT_HPP