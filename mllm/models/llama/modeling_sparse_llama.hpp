//
// Created by shrelic on 24-4-27.
//

#ifndef MLLM_MODELING_SPARSE_LLAMA_HPP
#define MLLM_MODELING_SPARSE_LLAMA_HPP

#include "Layer.hpp"
#include "Module.hpp"
#include "configuration_llama.hpp"
#include "models/transformer/modeling_transformer.hpp"

using namespace mllm;

class SparseLLaMAMLP final : public Module {
    Layer gate_proj;
    Layer relu;
    Layer up_proj;
    Layer down_proj;

public:
    SparseLLaMAMLP() = default;
    SparseLLaMAMLP(int hidden_dim, int ffn_hidden, const LLaMANameConfig &names, const string &base_name, bool is_down_sparse) {
        gate_proj = Linear(hidden_dim, ffn_hidden, false, base_name + names._gate_proj_name);
        relu = ReLU(base_name + "act");
        up_proj = SparseIdLinear(hidden_dim, ffn_hidden, base_name + names._up_proj_name);
        if (is_down_sparse) {
            down_proj = SparseLinear(ffn_hidden, hidden_dim, base_name + names._down_proj_name);
        } else {
            down_proj = Linear(ffn_hidden, hidden_dim, false, base_name + names._down_proj_name);
        }
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = inputs[0];
        auto id = gate_proj(inputs[0]);
        auto gate = relu(id);
        auto y = up_proj(x, id);
        x = gate * y;
        x = down_proj(x);
        return {x};
    }
};

class SparseLLaMABlock final : public Module {
    MultiHeadAttention attention;
    SparseLLaMAMLP mlp;
    Layer norm1;
    Layer norm2;

public:
    SparseLLaMABlock() = default;
    SparseLLaMABlock(bool is_down_sparse, int hidden_dim, int head_size, int ffn_hidden, RoPEType RoPE_type, float rope_theta, int max_position_embeddings, int cache_limit, string attn_implementation, const LLaMANameConfig &names, const string &base_name) {
        attention = MultiHeadAttention(hidden_dim, head_size, head_size, hidden_dim / head_size,
                                       SPLIT_NONE, PostQkv_NONE, false,
                                       RoPE_type, rope_theta, max_position_embeddings, cache_limit, true, false, false,
                                       attn_implementation,
                                       names, base_name + names._attn_base_name);
        mlp = SparseLLaMAMLP(hidden_dim, ffn_hidden, names, base_name + names._ffn_base_name, is_down_sparse);
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
};

class SparseLLaMAModel final : public Module {
    Layer embedding;
    vector<SparseLLaMABlock> blocks;
    Layer norm;
    Layer lm_head;

public:
    explicit SparseLLaMAModel(const LLaMAConfig &config, bool is_down_sparse = false) :
        SparseLLaMAModel(config.vocab_size, config.hidden_dim, config.head_size,
                         config.ffn_hidden, config.block_num, config.RoPE_type,
                         config.rope_theta, config.max_position_embeddings, config.cache_limit,
                         config.attn_implementation,
                         config.names_config, config.names_config.blk_name, is_down_sparse) {
    }
    SparseLLaMAModel(int vocab_size, int hidden_dim, int head_size, int ffn_hidden, int block_num, RoPEType RoPE_type,
                     float rope_theta, int max_position_embeddings, int cache_limit,
                     string attn_implementation,
                     const LLaMANameConfig &names, const string &base_name, bool is_down_sparse) {
        embedding = Embedding(vocab_size, hidden_dim, names.token_embd_name);
        blocks = List<SparseLLaMABlock>(block_num, is_down_sparse, hidden_dim, head_size, ffn_hidden, RoPE_type, rope_theta, max_position_embeddings, cache_limit, attn_implementation, names, base_name);
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
};

#endif // MLLM_MODELING_SPARSE_LLAMA_HPP
