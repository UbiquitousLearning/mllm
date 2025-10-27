#ifndef MODELING_OPT_HPP
#define MODELING_OPT_HPP

#include "Layer.hpp"
#include "Module.hpp"
#include "configuration_opt.hpp"
#include "models/transformer/modeling_transformer.hpp"

using namespace mllm;

class OPTBlock final : public Module {
    MultiHeadAttention attention;
    FeedForward mlp;
    Layer norm1;
    Layer norm2;

public:
    OPTBlock() = default;
    OPTBlock(int hidden_dim, int head_size, int ffn_hidden, int cache_limit,
             string attn_implementation, const optNameConfig &names, const string &base_name) {
        attention = MultiHeadAttention(hidden_dim, head_size, head_size,
                                       hidden_dim / head_size, SPLIT_NONE, PostQkv_NONE, false,
                                       NONE, -1, -1, cache_limit, true, true, true, 
                                       attn_implementation,
                                       names, base_name + names._attn_base_name);
        mlp = FeedForward(hidden_dim, ffn_hidden, "ReLU", true,
                          names, base_name + names._ffn_base_name);
        norm1 = LayerNorm(hidden_dim, true, 1e-05, base_name + names._attn_norm_name);
        norm2 = LayerNorm(hidden_dim, true, 1e-05, base_name + names._ffn_norm_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = norm1(inputs[0]);
        x = attention({x, x, x})[0];
        auto tmp = x + inputs[0];
        x = norm2(tmp);
        x = mlp({x})[0];
        x = x + tmp;
        return {x};
    }
};

class OPTModel final : public Module {
    Layer embedding;
    Layer pos_embedding;
    Layer pos;
    vector<OPTBlock> blocks;
    Layer norm;
    Layer lm_head;
    int offset = 1;
    bool flag = true;

public:
    explicit OPTModel(const OPTConfig &config) :
        OPTModel(config.vocab_size, config.hidden_dim,
                 config.head_size, config.ffn_hidden, config.block_num, config.cache_limit,
                 config.attn_implementation,
                 config.names_config, config.names_config.blk_name) {
    }
    OPTModel(int vocab_size, int hidden_dim, int head_size, int ffn_hidden, int block_num, int cache_limit,
             string attn_implementation,
             const optNameConfig &names, const string &base_name) {
        embedding = Embedding(vocab_size, hidden_dim, names.token_embd_name);
        pos_embedding = Embedding(2050, hidden_dim, names.pos_name);
        pos = Position("pos");
        blocks = List<OPTBlock>(block_num, hidden_dim, head_size, ffn_hidden, cache_limit, attn_implementation, names, base_name);
        norm = LayerNorm(hidden_dim, true, 1e-05, names.post_norm_name);
        lm_head = Linear(hidden_dim, vocab_size, false, names.lm_head_name);
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = embedding(inputs[0]);
        x = x + pos_embedding(pos(inputs[0]));
        for (auto &block : blocks) {
            x = block({x})[0];
        }
        x = norm(x);
        x = lm_head(x);
        return {x};
    }
};

#endif // MODELING_OPT_HPP