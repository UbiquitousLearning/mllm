//
// Created by Rongjie Yi on 24-3-7.
//

#ifndef MODELING_TINYLLAMA_HPP
#define MODELING_TINYLLAMA_HPP

#include "Layer.hpp"
#include "Module.hpp"
#include "configuration_tinyllama.hpp"
#include "models/llama/modeling_llama.hpp"

using namespace mllm;

class TinyLLaMABlock final : public Module {
    MultiHeadAttention attention;
    LLaMAMLP mlp;
    Layer norm1;
    Layer norm2;

public:
    TinyLLaMABlock() = default;
    TinyLLaMABlock(int hidden_dim, int head_size, int kv_head_size, int ffn_hidden, RoPEType RoPE_type, float rope_theta, int max_position_embeddings, int cache_limit, string attn_implementation, const LLaMANameConfig &names, const string &base_name) {
        attention = MultiHeadAttention(hidden_dim, head_size, kv_head_size,
                                       hidden_dim / head_size, SPLIT_NONE, PostQkv_NONE, false,
                                       RoPE_type, rope_theta, max_position_embeddings,
                                       cache_limit, true, false, false,
                                       attn_implementation, names, base_name + names._attn_base_name);
        mlp = LLaMAMLP(hidden_dim, ffn_hidden, names, base_name + names._ffn_base_name);
        norm1 = RMSNorm(hidden_dim, 1e-6, base_name + names._attn_norm_name);
        norm2 = RMSNorm(hidden_dim, 1e-6, base_name + names._ffn_norm_name);
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
    MultiHeadAttention &get_attention() {
        return attention;
    }
};

class TinyLLaMAModel final : public Module {
    Layer embedding;
    vector<TinyLLaMABlock> blocks;
    Layer norm;
    Layer lm_head;

public:
    explicit TinyLLaMAModel(const TinyLLaMAConfig &config) :
        TinyLLaMAModel(config.vocab_size, config.hidden_dim, config.head_size, config.kv_head_size, config.ffn_hidden, config.block_num,
                       config.RoPE_type, config.rope_theta, config.max_position_embeddings, config.cache_limit, config.attn_implementation,
                       config.names_config, config.names_config.blk_name) {
    }
    TinyLLaMAModel(int vocab_size, int hidden_dim, int head_size, int kv_head_size, int ffn_hidden, int block_num,
                   RoPEType RoPE_type, float rope_theta, int max_position_embeddings, int cache_limit, string attn_implementation,
                   const LLaMANameConfig &names, const string &base_name) {
        embedding = Embedding(vocab_size, hidden_dim, names.token_embd_name);
        blocks = List<TinyLLaMABlock>(block_num, hidden_dim, head_size, kv_head_size, ffn_hidden, RoPE_type, rope_theta, max_position_embeddings, cache_limit, attn_implementation, names, base_name);
        norm = RMSNorm(hidden_dim, 1e-6, names.post_norm_name);
        lm_head = Linear(hidden_dim, vocab_size, false, names.lm_head_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = embedding(inputs[0]);
        for (auto &block : blocks) {
            x = block({x})[0];
        }
        x = norm(x);
        x = lm_head(x);
        return {x};
    }
    void clear_kvcache() override {
        for (auto &block : blocks) {
            auto kvcache = block.get_attention().get_cache();
            for (auto &cache : kvcache) { cache->clearCache(); }
            auto ropes = block.get_attention().get_rope();
            for (auto &rope : ropes) { rope->clearCache(); }
        }
    }
};

#endif // MODELING_TINYLLAMA_HPP
