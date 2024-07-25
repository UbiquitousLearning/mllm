
#ifndef CONFIG_DEEPSEEK_HPP
#define CONFIG_DEEPSEEK_HPP
#include "Types.hpp"
#include "models/transformer/configuration_transformer.hpp"

using namespace mllm;

class DeepseekNameConfig : public TransformerNameConfig {
public:
    /**
     * @brief Deepseek following the hugging face naming method
     *
     * @param type RoPEType
     */
    void init() {
        blk_name = "model.layers.";
        _attn_base_name = "self_attn.";
        _ffn_base_name = "mlp.";
        _q_proj_name = "q_proj";
        _kv_a_proj_with_mqa_name = "kv_a_proj_with_mqa";
        _kv_a_layernorm_name = "kv_a_layernorm";
        _kv_b_proj_name = "kv_b_proj";
        _o_proj_name = "o_proj";
        _gate_proj_name = "gate_proj";
        _up_proj_name = "up_proj";
        _down_proj_name = "down_proj";
        _attn_norm_name = "input_layernorm";
        _ffn_norm_name = "post_attention_layernorm";
        token_embd_name = "model.embed_tokens";
        post_norm_name = "model.norm";
        lm_head_name = "model.embed_tokens";
    }

    std::string _kv_a_proj_with_mqa_name;
    std::string _kv_a_layernorm_name;
    std::string _kv_b_proj_name;

    std::string blk_name;
    std::string token_embd_name;
    std::string post_norm_name;
    std::string lm_head_name;
    std::string _gate_proj_name;
};

struct DeepseekConfig {
    explicit DeepseekConfig(int token_limit):
        cache_limit(token_limit) {
        names_config.init();
    };

    int vocab_size = 32000;
    int max_position_embeddings = 8192;
    int num_hidden_layers = 12;
    int hidden_size = 768;
    int intermediate_size = 2048;
    int num_heads = 16;
    int qk_rope_head_dim=24;//qk_rope_head_dim
    int qk_nope_head_dim=48; //qk_nope_head_dim = qk_rope_head_dim*2
    int v_head_dim=48;  //v_head_dim=  qk_nope_head_dim*2
    int kv_lora_rank = 192; //kv_lora_rank = 2568* qk_nope_head_dim;

    // int vocab_size = 152064;
    // int max_position_embeddings = 8192;
    // int num_hidden_layers = 12;
    // int hidden_size = 1024;
    // int intermediate_size = 4864;
    // int num_heads = 16;
    // int qk_rope_head_dim=32;//qk_rope_head_dim
    // int qk_nope_head_dim=64; //qk_nope_head_dim = qk_rope_head_dim*2
    // int v_head_dim=64;  //v_head_dim=  qk_nope_head_dim*2
    // int kv_lora_rank = 256; //kv_lora_rank = 2568* qk_nope_head_dim;


    float rms_norm_eps = 1e-6;
    int cache_limit;
    bool do_mask=true;


    DeepseekNameConfig names_config;
};

#endif //! CONFIG_DEEPSEEK_HPP
