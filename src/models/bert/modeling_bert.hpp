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

class BertLayer : public Module {
public:
    BertLayer() = default;
    BertLayer(const BertConfig &config, const string &base_name) {
        // base_name: encoder.layer.n.
        attention = MultiHeadAttention(config.hidden_size, config.num_attention_heads, config.num_attention_heads, config.hidden_size / config.num_attention_heads, SPLIT_NONE, PostQkv_NONE, false, RoPEType::NONE, -1, -1, 0, false, true, true, config.attn_implementation, config.names_config, base_name + config.names_config._attn_base_name);

        feed_forward = FeedForward(config.hidden_size, config.intermediate_size,
                                   config.hidden_act, true, config.names_config, base_name);

        attn_norm = LayerNorm(config.hidden_size, true, config.layer_norm_eps,
                              base_name + config.names_config._attn_base_name + config.names_config._attn_norm_name);

        ff_norm = LayerNorm(config.hidden_size, true, config.layer_norm_eps,
                            base_name + config.names_config._ffn_norm_name);
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto hidden_states = inputs[0];
        auto attn_out = attention({hidden_states, hidden_states, hidden_states})[0];
        hidden_states = attn_norm({hidden_states + attn_out});
        auto ff_out = feed_forward({hidden_states})[0];
        hidden_states = ff_norm({hidden_states + ff_out});
        return {hidden_states};
    }

private:
    MultiHeadAttention attention;
    FeedForward feed_forward;

    Layer attn_norm, ff_norm;
};

class BertAvgPooler : public Module {
public:
    BertAvgPooler() = default;
    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = inputs[0];
        x = x.mean(SEQUENCE);
        return {x};
    }
};

class BertModel : public Module {
public:
    BertModel(BertConfig &config) {
        embeddings = BertEmbeddings(config.vocab_size, config.hidden_size, config.type_vocab_size, config.max_position_embeddings, config.layer_norm_eps, config.names_config);
        layers = List<BertLayer>(config.num_hidden_layers, config, config.names_config.blk_name);

        if (config.pooling_type == "mean") {
            pooler = BertAvgPooler();
        } else {
            // print not support pooling type and exit
            std::cout << "Not support pooling type: " << config.pooling_type << std::endl;
            exit(0);
        }
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = embeddings(inputs, args)[0];
        for (auto &layer : layers) {
            x = layer({x})[0];
        }
        x = pooler({x})[0];
        return {x};
    }

private:
    BertEmbeddings embeddings;
    std::vector<BertLayer> layers;
    BertAvgPooler pooler;
};

#endif //! MODELING_BERT_HPP