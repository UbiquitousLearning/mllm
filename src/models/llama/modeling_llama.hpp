//
// Created by Rongjie Yi on 2024/2/4 0004.
//

#ifndef MODELING_LLAMA_HPP
#define MODELING_LLAMA_HPP

#include "Layer.hpp"
#include "Module.hpp"
#include "configuration_llama.hpp"
#include "models/transformer/modeling_transformer.hpp"

using namespace mllm;

class LLaMAMLP final : public Module {
    Layer gate_proj;
    Layer silu;
    Layer up_proj;
    Layer down_proj;

public:
    LLaMAMLP() = default;
    LLaMAMLP(int hidden_dim, int ffn_hidden, const LLaMANameConfig &names, const string &base_name) {
        gate_proj = Linear(hidden_dim, ffn_hidden, false, base_name + names._gate_proj_name);
        silu = SiLU(base_name + "act");
        up_proj = Linear(hidden_dim, ffn_hidden, false, base_name + names._up_proj_name);
        down_proj = Linear(ffn_hidden, hidden_dim, false, base_name + names._down_proj_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = gate_proj(inputs[0]);
        x = silu(x);
        auto y = up_proj(inputs[0]);
        x = x * y;
        x = down_proj(x);
        return {x};
    }
};

class LLaMABlock final : public Module {
    MultiHeadAttention attention;
    LLaMAMLP mlp;
    Layer norm1;
    Layer norm2;

public:
    LLaMABlock() = default;
    LLaMABlock(int hidden_dim, int head_size, int kv_head_size, int ffn_hidden, RoPEType RoPE_type, float rope_theta, int max_position_embeddings, int cache_limit,
               const LLaMANameConfig &names,
               const LLaMAConfig& config,
               const string &base_name) {
        RoPEConfig rope_config;
        if(!config.rope_scaling.empty()){
            rope_config["rope_theta"] = rope_theta;
            rope_config["max_position_embeddings"] = max_position_embeddings;
            rope_config["rope_scaling"] = config.rope_scaling;
        }

        attention = MultiHeadAttention(hidden_dim, head_size, kv_head_size, hidden_dim / head_size, SPLIT_NONE, false, false,
                                       RoPE_type, rope_theta, max_position_embeddings, cache_limit, true, false, names, base_name + names._attn_base_name, rope_config);
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

class LLaMAModel final : public Module {
    Layer embedding;
    vector<LLaMABlock> blocks;
    Layer norm;
    Parameter lm_head;

public:
    explicit LLaMAModel(const LLaMAConfig &config) :
        LLaMAModel(config.vocab_size, config.hidden_dim, config.head_size, config.num_key_value_heads, config.ffn_hidden, config.block_num,
                   config.RoPE_type, config.rope_theta, config.max_position_embeddings, config.cache_limit,
                   config.names_config, config, config.names_config.blk_name) {
    }
    LLaMAModel(int vocab_size, int hidden_dim, int head_size, int kv_head_size, int ffn_hidden, int block_num, RoPEType RoPE_type, float rope_theta, int max_position_embeddings, int cache_limit,
               const LLaMANameConfig &names,
               const LLaMAConfig& config,
               const string &base_name) {
//        printf("vocab_size: %d, hidden_dim: %d, head_size: %d, kv_head_size: %d, ffn_hidden: %d, block_num: %d, RoPE_type: %d, rope_theta: %f, max_position_embeddings: %d, cache_limit: %d\n",
//               vocab_size, hidden_dim, head_size, kv_head_size, ffn_hidden, block_num, RoPE_type, rope_theta, max_position_embeddings, cache_limit);
        embedding = Embedding(vocab_size, hidden_dim, names.token_embd_name);
        blocks = List<LLaMABlock>(block_num, hidden_dim, head_size, kv_head_size, ffn_hidden, RoPE_type, rope_theta, max_position_embeddings, cache_limit, names, config, base_name);
        norm = RMSNorm(hidden_dim, 1e-6, names.post_norm_name);
        // TODO: tie_word_embeddings
        // this is a workaround
        // we just simply use the token embedding as the lm_head
        // but now we are not really tying the word embeddings
        auto lm_head_name = names.lm_head_name;
        if (config.tie_word_embeddings)
            lm_head_name = names.token_embd_name;
        lm_head = Parameter(1, vocab_size, 1, hidden_dim, lm_head_name + ".weight");
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = embedding(inputs[0]);

//        if (Tensor::tensor_status == TENSOR_STATIC_READY) {
//            x.printDataTorchLike<float>();
//            cout << endl;
//        }
        for (auto &block : blocks) {
            x = block({x})[0];
        }
        x = norm(x);
        x = Tensor::mm(x, lm_head().transpose(Chl::SEQUENCE, Chl::DIMENSION));
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

#endif // MODELING_LLAMA_HPP