#ifndef CONFIG_BERT_HPP
#define CONFIG_BERT_HPP
#include "Types.hpp"
#include "models/transformer/configuration_transformer.hpp"
#include <cctype>
#include <iterator>

using namespace mllm;

class BertNameConfig : public TransformerNameConfig {
public:
    void init() {
        embedding_base_name = "embeddings.";

        blk_name = "encoder.layer.";
        _attn_base_name = "attention.";
        _q_proj_name = "self.query";
        _k_proj_name = "self.key";
        _v_proj_name = "self.value";
        _o_proj_name = "output.dense";
        _up_proj_name = "intermediate.dense";
        _down_proj_name = "output.dense";
        _attn_norm_name = "output.LayerNorm";
        _ffn_norm_name = "output.LayerNorm";
    }
    std::string embedding_base_name;

    std::string blk_name;
};

struct BertConfig : public TransformerConfig {
    explicit BertConfig() {
        hidden_act = "GELU";
        pooling_type = "mean";
        hidden_size = 384;
        intermediate_size = 1536;
        max_position_embeddings = 512;
        model_type = "bert";
        num_attention_heads = 12;
        num_hidden_layers = 12;
        vocab_size = 30522;
        names_config.init();
    };

    int type_vocab_size = 2;
    float layer_norm_eps = 1e-12;

    std::string hidden_act = "GELU";
    std::string pooling_type = "mean";
    int hidden_size = 1024;
    int intermediate_size = 2816;
    int max_position_embeddings = 32768;
    std::string model_type = "bert";
    int num_attention_heads = 12;
    int num_hidden_layers = 12;


    int vocab_size = 151936;

    BertNameConfig names_config;
};

#endif //! CONFIG_BERT_HPP
