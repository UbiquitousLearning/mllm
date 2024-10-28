#ifndef CONFIG_BERT_HPP
#define CONFIG_BERT_HPP
#include "Types.hpp"
#include "models/transformer/configuration_transformer.hpp"
#include <cctype>
#include <iterator>

using namespace mllm;

class BertNameConfig : public TransformerNameConfig {
public:
    /**
     * @brief QWen2 following the hugging face naming method
     *
     * @param type RoPEType
     */
    void init() {
        embedding_base_name = "embeddings.";

        blk_name = "model.layers.";
        _attn_base_name = "self.";
        _ffn_base_name = "mlp.";
        _q_proj_name = "self.query";
        _k_proj_name = "self.key";
        _v_proj_name = "self.value";
        _o_proj_name = "output.dense";
        _gate_proj_name = "gate_proj";
        _up_proj_name = "up_proj";
        _down_proj_name = "down_proj";
        _attn_norm_name = "input_layernorm";
        _ffn_norm_name = "post_attention_layernorm";
        token_embd_name = "model.embed_tokens";
        post_norm_name = "model.norm";
        lm_head_name = "lm_head";
    }
    std::string embedding_base_name;

    std::string blk_name;
    std::string token_embd_name;
    std::string post_norm_name;
    std::string lm_head_name;
    std::string _gate_proj_name;
};

struct BertConfig : public TransformerConfig {
    explicit BertConfig() {
        attention_dropout = 0.0;
        bos_token_id = 151643;
        eos_token_id = 151645;
        std::string hidden_act = "gelu";
        hidden_size = 384;
        initializer_range = 0.02;
        intermediate_size = 1536;
        max_position_embeddings = 512;
        max_window_layers = 21;
        model_type = "bert";
        num_attention_heads = 12;
        num_hidden_layers = 12;
        num_key_value_heads = 16;
        rms_norm_eps = 1e-6;
        rope_theta = 1000000.0;
        sliding_window = 32768;
        vocab_size = 30522;
        tie_embedding_words = true;

        names_config.init();
    };

    int type_vocab_size = 2;
    float layer_norm_eps = 1e-12;

    float attention_dropout = 0.0;
    int bos_token_id = 151643;
    int eos_token_id = 151643;
    std::string hidden_act = "silu";
    int hidden_size = 1024;
    float initializer_range = 0.02;
    int intermediate_size = 2816;
    int max_position_embeddings = 32768;
    int max_window_layers = 21;
    std::string model_type = "bert";
    int num_attention_heads = 12;
    int num_hidden_layers = 12;
    int num_key_value_heads = 12;
    double rms_norm_eps = 1e-6;
    float rope_theta = 1000000.0;
    int sliding_window = 32768;
    int vocab_size = 151936;
    bool tie_embedding_words = false;

    BertNameConfig names_config;
};

#endif //! CONFIG_BERT_HPP
