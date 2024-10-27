#ifndef MODELING_BERT_HPP
#define MODELING_BERT_HPP

#include "Backend.hpp"
#include "Layer.hpp"
#include "Module.hpp"
#include "Tensor.hpp"
#include "configuration_bert.hpp"
#include <cmath>
using namespace mllm;

class BertEmbeddings : public Module {
public:
    BertEmbeddings() = default;
    BertEmbeddings(int vocal_size, int hidden_size, int type_size, int max_position_embeddings, float eps, BertNameConfig &config) {
        word_embeddings = Embedding(vocal_size, hidden_size, config.embedding_base_name+"word_embeddings");
        token_type_embeddings = Embedding(type_size, hidden_size, config.embedding_base_name+"token_type_embeddings");
        position_embeddings = Embedding(max_position_embeddings, hidden_size, config.embedding_base_name+"position_embeddings");
        layer_norm = LayerNorm(hidden_size, true, eps, config.embedding_base_name+"LayerNorm");
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto inputs_embeds = word_embeddings(inputs[0]);
//        if (Tensor::tensor_status == TENSOR_STATIC_READY)
//            inputs_embeds.printData<float>();
        auto type_embeds = token_type_embeddings(inputs[1]);
        auto position_embeds = position_embeddings(inputs[2]);
        auto embeddings = inputs_embeds + type_embeds + position_embeds;
        return {layer_norm(embeddings)};
    }

private:
    Layer word_embeddings;
    Layer token_type_embeddings;
    Layer position_embeddings;
    Layer layer_norm;
};

class BertSelfAttention : public Module {
public:
    BertSelfAttention() = default;
    BertSelfAttention(BertConfig &config, const string &base_name) {
        num_attention_heads = config.num_attention_heads;
        attention_head_size = config.hidden_size / num_attention_heads;
        all_head_size = num_attention_heads * attention_head_size;

        query = Linear(config.hidden_size, all_head_size, true, base_name + config.names_config._q_proj_name);
        key = Linear(config.hidden_size, all_head_size, true, base_name + config.names_config._k_proj_name);
        value = Linear(config.hidden_size, all_head_size, true, base_name + config.names_config._v_proj_name);

        softmax = Softmax(DIMENSION, base_name + "softmax");
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        if (Tensor::tensor_status == TENSOR_STATIC_READY) {
            std::cout << "emb type: " << inputs[0].ctype() << std::endl;
            inputs[0].printData<float>();
        }

        auto key_states = key(inputs[0]);
        auto query_states = query(inputs[1]);
        auto value_states = value(inputs[2]);

//        auto key_len = key_states.sequence();

        query_states = query_states.view(-1, num_attention_heads, -1, attention_head_size);
        key_states = key_states.view(-1, num_attention_heads, -1, attention_head_size);
        value_states = value_states.view(-1, num_attention_heads, -1, attention_head_size);

        auto attn_weight =
            Tensor::mm(query_states, key_states.transpose(Chl::SEQUENCE, Chl::DIMENSION)) / std::sqrt(attention_head_size);
        auto attn_score = softmax(attn_weight);
        auto attn_output = Tensor::mm(attn_score, value_states);
        attn_output = attn_output.view(-1,1, -1, num_attention_heads * attention_head_size);
        return {attn_output};
    }

private:
    int num_attention_heads;
    int attention_head_size;
    int all_head_size;

    Layer query;
    Layer key;
    Layer value;

    Layer softmax;
};

class BertAttention : public Module {
public:


    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        return {};
    }

private:

};

class BertModel : public Module {
public:
    BertModel(BertConfig &config){
        embeddings = BertEmbeddings(config.vocab_size, config.hidden_size, config.type_vocab_size, config.max_position_embeddings,
                                    config.layer_norm_eps, config.names_config);
        self_attention = BertSelfAttention(config, "encoder.layer.0.attention.self.");
    }


    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto emb = embeddings(inputs, args)[0];
//        if (Tensor::tensor_status == TENSOR_STATIC_READY) {
//            std::cout << "emb type: " << emb.ctype() << std::endl;
//            emb.printData<float>();
//        }
        auto attn = self_attention({emb, emb, emb});
        return {attn[0]};
    }

private:
    BertEmbeddings embeddings;
    BertSelfAttention self_attention;
};

#endif //! MODELING_BERT_HPP