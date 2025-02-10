
#ifndef CONFIG_MINICPM_HPP
#define CONFIG_MINICPM_HPP
#include "Types.hpp"
#include "models/transformer/configuration_transformer.hpp"
#include <vector>

using namespace mllm;

// the model naming method is from minicpm3 hf repo
// model.embed_tokens.weight
// model.norm.weight
// model.layers.0.input_layernorm.weight
// model.layers.0.self_attn.q_b_proj.weight
// model.layers.0.self_attn.q_a_proj.weight
// model.layers.0.self_attn.kv_b_proj.weight
// model.layers.0.self_attn.kv_a_proj_with_mqa.weight
// model.layers.0.self_attn.q_a_layernorm.weight
// model.layers.0.self_attn.kv_a_layernorm.weight
// model.layers.0.self_attn.o_proj.weight
// model.layers.0.post_attention_layernorm.weight
// model.layers.0.mlp.gate_proj.weight
// model.layers.0.mlp.up_proj.weight
// model.layers.0.mlp.down_proj.weight

class MiniCPM3NameConfig : public TransformerNameConfig {
public:
    /**
     * @brief MiniCPM3 following the hugging face naming method
     *
     * @param type RoPEType
     */
    void init() {
        blk_name = "model.layers.";
        _attn_base_name = "self_attn.";
        _ffn_base_name = "mlp.";
        _q_b_proj_name = "q_b_proj";
        _q_a_proj_name = "q_a_proj";
        _q_a_layernorm = "q_a_layernorm";
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
    std::string _q_b_proj_name;
    std::string _q_a_proj_name;
    std::string _q_a_layernorm;

    std::string blk_name;
    std::string token_embd_name;
    std::string post_norm_name;
    std::string lm_head_name;
    std::string _gate_proj_name;
};

struct MiniCPM3Config : public TransformerConfig {
    explicit MiniCPM3Config(int token_limit) :
        cache_limit(token_limit) {
        names_config.init();
    };

    int bos_token_id = 1;
    std::vector<int> eos_token_ids = {2, 73440};
    float initializer_range = 0.1f;
    int hidden_size = 2560;
    int num_hidden_layers = 62;
    int intermediate_size = 6400;
    int max_position_embeddings = 32768;
    int num_attention_heads = 40;
    int num_key_value_heads = 40;
    int qk_nope_head_dim = 64;
    int qk_rope_head_dim = 32;
    int q_lora_rank = 768;
    int kv_lora_rank = 256;
    float rms_norm_eps = 1e-05f;

    int vocab_size = 73448;
    int scale_emb = 12;
    int dim_model_base = 256;
    float scale_depth = 1.4f;

    // rope_scaling
    std::string rope_type = "longrope";
    std::vector<float> rope_long_factor = {1.0591234137867171,
                                           1.1241891283591912,
                                           1.2596935748670968,
                                           1.5380380402321725,
                                           2.093982484148734,
                                           3.1446935121267696,
                                           4.937952647693647,
                                           7.524541999994549,
                                           10.475458000005451,
                                           13.062047352306353,
                                           14.85530648787323,
                                           15.906017515851266,
                                           16.461961959767827,
                                           16.740306425132907,
                                           16.87581087164081,
                                           16.940876586213285};
    std::vector<float> rope_short_factor = {1.0591234137867171,
                                            1.1241891283591912,
                                            1.2596935748670968,
                                            1.5380380402321725,
                                            2.093982484148734,
                                            3.1446935121267696,
                                            4.937952647693647,
                                            7.524541999994549,
                                            10.475458000005451,
                                            13.062047352306353,
                                            14.85530648787323,
                                            15.906017515851266,
                                            16.461961959767827,
                                            16.740306425132907,
                                            16.87581087164081,
                                            16.940876586213285};
    float rope_theta = 10000.f;
    int rope_original_max_position_embeddings = 32768;

    float attention_dropout = 0.f;

    int cache_limit;
    bool do_mask = true;

    MiniCPM3NameConfig names_config;
};

#endif //! CONFIG_MINICPM_HPP
