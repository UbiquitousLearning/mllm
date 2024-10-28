#ifndef MODELING_BERT_HPP
#define MODELING_BERT_HPP

#include "Backend.hpp"
#include "Layer.hpp"
#include "Module.hpp"
#include "Tensor.hpp"
#include "configuration_bert.hpp"
#include "models/transformer/modeling_transformer.hpp"
using namespace mllm;

class BertEmbeddings : public Module {
public:
    BertEmbeddings() = default;
    BertEmbeddings(int vocal_size, int hidden_size, int type_size, int max_position_embeddings, float eps, BertNameConfig &config) {
        word_embeddings = Embedding(vocal_size, hidden_size, config.embedding_base_name + "word_embeddings");
        token_type_embeddings = Embedding(type_size, hidden_size, config.embedding_base_name + "token_type_embeddings");
        position_embeddings = Embedding(max_position_embeddings, hidden_size, config.embedding_base_name + "position_embeddings");
        layer_norm = LayerNorm(hidden_size, true, eps, config.embedding_base_name + "LayerNorm");
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto inputs_embeds = word_embeddings(inputs[0]);
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

class BertModel : public Module {
public:
    BertModel(BertConfig &config) {
        embeddings = BertEmbeddings(config.vocab_size, config.hidden_size, config.type_vocab_size, config.max_position_embeddings, config.layer_norm_eps, config.names_config);

        attention = MultiHeadAttention(config.hidden_size, config.num_attention_heads, config.num_attention_heads, config.hidden_size / config.num_attention_heads, SPLIT_NONE, false, false, RoPEType::NONE, -1, -1, 0, false, true, config.names_config, "encoder.layer.0.attention.");
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto emb = embeddings(inputs, args)[0];
        auto attn = attention({emb, emb, emb});
        return {attn[0]};
    }

private:
    BertEmbeddings embeddings;
    MultiHeadAttention attention;
};

#endif //! MODELING_BERT_HPP