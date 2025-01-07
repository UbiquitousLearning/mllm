//
// Created by Guo Xiaoqiang on 2024/8/12.
//
#ifndef MODELING_PHI3_HPP
#define MODELING_PHI3_HPP

#include "Layer.hpp"
#include "Module.hpp"
#include "Tensor.hpp"
#include "configuration_phi3.hpp"
#include "models/transformer/modeling_transformer.hpp"

using namespace mllm;

class Phi3MLP final : public Module {
    Layer gate_up_proj;
    Layer silu;
    Layer down_proj;
    int ffn_hidden_;

public:
    Phi3MLP() = default;
    Phi3MLP(int hidden_dim, int ffn_hidden, const Phi3NameConfig &names, const string &base_name) {
        ffn_hidden_ = ffn_hidden;
        gate_up_proj = Linear(hidden_dim, 2 * ffn_hidden, false, base_name + names._gate_up_proj_name);
        silu = SiLU(base_name + "act");
        down_proj = Linear(ffn_hidden, hidden_dim, false, base_name + names._down_proj_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = gate_up_proj(inputs[0]);
        auto splited_y_12 = x.split({ffn_hidden_, ffn_hidden_}, DIMENSION);
        auto y_1 = splited_y_12[0];
        Tensor y_2 = splited_y_12[1];
        x = y_2 * silu(y_1);
        x = down_proj(x);
        return {x};
    }
};

class Phi3Block final : public Module {
    MultiHeadAttention attention;
    Phi3MLP mlp;
    Layer norm1;
    Layer norm2;

public:
    Phi3Block() = default;
    Phi3Block(int hidden_dim, int head_size, int kv_head_size, int ffn_hidden, RoPEType RoPE_type, float rope_theta, int max_position_embeddings, int cache_limit, const Phi3NameConfig &names, const string &base_name) {
        attention = MultiHeadAttention(hidden_dim, head_size, kv_head_size, hidden_dim / head_size, SPLIT_HD, false, false,
                                       RoPE_type, rope_theta, max_position_embeddings, cache_limit, true, false, names, base_name + names._attn_base_name);
        mlp = Phi3MLP(hidden_dim, ffn_hidden, names, base_name + names._ffn_base_name);
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

class Phi3Model final : public Module {
    Layer embedding;
    vector<Phi3Block> blocks;
    Layer norm;
    Layer lm_head;

public:
    explicit Phi3Model(const Phi3Config &config) :
        Phi3Model(config.vocab_size, config.hidden_dim, config.head_size, config.num_key_value_heads, config.ffn_hidden, config.block_num,
                  config.RoPE_type, config.rope_theta, config.max_position_embeddings, config.cache_limit,
                  config.names_config, config.names_config.blk_name) {
    }
    Phi3Model(int vocab_size, int hidden_dim, int head_size, int kv_head_size, int ffn_hidden, int block_num, RoPEType RoPE_type, float rope_theta, int max_position_embeddings, int cache_limit,
              const Phi3NameConfig &names, const string &base_name) {
        embedding = Embedding(vocab_size, hidden_dim, names.token_embd_name);
        blocks = List<Phi3Block>(block_num, hidden_dim, head_size, kv_head_size, ffn_hidden, RoPE_type, rope_theta, max_position_embeddings, cache_limit, names, base_name);
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

#endif // MODELING_PHI3_HPP