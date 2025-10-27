#ifndef CONFIG_GEMMA2_HPP
#define CONFIG_GEMMA2_HPP
#include "Types.hpp"
#include "models/transformer/configuration_transformer.hpp"

using namespace mllm;

class Gemma2NameConfig : public TransformerNameConfig {
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
            _pre_feedforward_layernorm = "pre_feedforward_layernorm";
            _post_feedforward_layernorm = "post_feedforward_layernorm";
            token_embd_name = "model.embed_tokens";
            post_norm_name = "model.norm";
            lm_head_name = "model.embed_tokens";
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
    std::string _pre_feedforward_layernorm;
    std::string _post_feedforward_layernorm;
};

struct Gemma2Config : public TransformerConfig {
    explicit Gemma2Config(int token_limit, const string billions = "2B", RoPEType type = RoPEType::HFHUBROPE) :
        cache_limit(token_limit) {
        names_config.init(type);
        if (!(billions == "2B" || billions == "2b")) {
            throw std::runtime_error("Unsupported model size");
        }
        RoPE_type = type;
    };

    int vocab_size = 256000;
    int max_position_embeddings = 8192;
    int num_hidden_layers = 26;
    int num_attention_heads = 8;
    int num_key_value_heads = 4;
    int hidden_size = 2304;
    int sliding_window = 4096;
    int intermediate_size = 9216;
    int head_dim = 256;
    float rms_norm_eps = 1e-6;
    float rope_theta = 10000;

    int cache_limit;
    RoPEType RoPE_type = RoPEType::HFHUBROPE;
    Gemma2NameConfig names_config;
};

#endif //! CONFIG_GEMMA2_HPP
