/**
 * @file modeling_qwen3.hpp
 * @author hyh
 * @brief
 * https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py
 * @version 0.1
 * @date 2025-05-11
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef MODELING_QWEN3_HPP
#define MODELING_QWEN3_HPP

#include "Backend.hpp"
#include "Layer.hpp"
#include "Module.hpp"
#include "Tensor.hpp"
#include "configuration_qwen3.hpp"
#include <cmath>
using namespace mllm;

class QWen3MLP final : public Module {
public:
    QWen3MLP() = default;
    QWen3MLP(int hidden_size, int intermediate_size, const QWen3NameConfig &names,
             const std::string &base_name) {
        gate_proj =
            Linear(hidden_size, intermediate_size, false, base_name + names._gate_proj_name);
        silu = SiLU(base_name + "act");
        up_proj = Linear(hidden_size, intermediate_size, false, base_name + names._up_proj_name);
        down_proj =
            Linear(intermediate_size, hidden_size, false, base_name + names._down_proj_name);
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = gate_proj(inputs[0]);
        x = silu(x);
        auto y = up_proj(inputs[0]);
        x = x * y;
        x = down_proj(x);
        return {x};
    }

private:
    Layer gate_proj;
    Layer up_proj;
    Layer down_proj;

    Layer silu;
};

class QWen3Attention final : public Module {
public:
    QWen3Attention() = default;
    QWen3Attention(const QWen3Config &config, const QWen3NameConfig &names, const string &base_name) {
        hidden_size = config.hidden_size;
        num_heads = config.num_attention_heads;
        head_dim = config.head_dim; // 这里config中有head_dim,不等于相除的结果
        num_key_value_heads = config.num_key_value_heads;
        num_key_value_groups = num_heads / num_key_value_heads;
        rms_norm_eps = config.rms_norm_eps;
        // init layers
        q_proj = Linear(hidden_size, num_heads * head_dim, config.attention_bias, base_name + names._q_proj_name);
        k_proj = Linear(hidden_size, num_key_value_heads * head_dim, config.attention_bias,
                        base_name + names._k_proj_name);
        v_proj = Linear(hidden_size, num_key_value_heads * head_dim, config.attention_bias,
                        base_name + names._v_proj_name);
        o_proj = Linear(num_heads * head_dim, hidden_size, false, base_name + names._o_proj_name);

        // 增加了RMSNorm
        q_norm = RMSNorm(head_dim, rms_norm_eps, base_name + "q_norm");
        k_norm = RMSNorm(head_dim, rms_norm_eps, base_name + "k_norm");
        // 滑动窗口禁用

        q_rope = RoPE(config.RoPE_type, config.rope_theta, config.max_position_embeddings,
                      base_name + "q_rope");
        k_rope = RoPE(config.RoPE_type, config.rope_theta, config.max_position_embeddings,
                      base_name + "k_rope");
        k_cache = KVCache(num_key_value_heads, head_dim, num_key_value_groups, config.cache_limit, base_name + "k_cache");
        v_cache = KVCache(num_key_value_heads, head_dim, num_key_value_groups, config.cache_limit, base_name + "v_cache");
        softmax = Softmax(DIMENSION, true, base_name + "softmax");
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto query_states = q_proj(inputs[0]);
        auto key_states = k_proj(inputs[1]);
        auto value_states = v_proj(inputs[2]);

        // [batch, heads, sequence, dims]
        query_states = query_states.view(-1, num_heads, -1, head_dim);
        key_states = key_states.view(-1, num_key_value_heads, -1, head_dim);
        value_states = value_states.view(-1, num_key_value_heads, -1, head_dim);

        // 加正则化
        query_states = q_norm(query_states);
        key_states = k_norm(key_states);

        // embedding
        query_states = q_rope(query_states);
        key_states = k_rope(key_states);

        // kv cache
        key_states = k_cache(key_states);
        value_states = v_cache(value_states);

        // attention weight
        auto atten_weight =
            Tensor::mm(query_states, key_states.transpose(Chl::SEQUENCE, Chl::DIMENSION))
            / std::sqrt(head_dim);
        atten_weight = softmax(atten_weight, k_cache.getCacheSeqLen());

        // attention output
        auto atten_output = Tensor::mm(atten_weight, value_states);
        atten_output = atten_output.view(-1, 1, -1, head_dim * num_heads);
        atten_output = o_proj(atten_output);
        return {atten_output};
    }

    vector<KVCache *> get_cache() {
        return {&k_cache, &v_cache};
    }
    vector<RoPE *> get_rope() {
        return {&q_rope, &k_rope};
    }

private:
    int hidden_size;
    int num_heads;
    int head_dim;
    int num_key_value_heads;
    int num_key_value_groups;
    double rms_norm_eps;
    Layer q_proj;
    Layer k_proj;
    Layer v_proj;
    Layer o_proj;
    Layer q_norm;
    Layer k_norm;
    RoPE q_rope;
    RoPE k_rope;
    KVCache k_cache;
    KVCache v_cache;
    // Causalmask mask;
    Softmax softmax;
};

class QWen3Decoder final : public Module {
public:
    QWen3Decoder() = default;
    QWen3Decoder(const QWen3Config &config, const QWen3NameConfig &names, const string &base_name) {
        self_atten = QWen3Attention(config, names, base_name + names._attn_base_name);
        mlp = QWen3MLP(config.hidden_size, config.intermediate_size, names,
                       base_name + names._ffn_base_name);
        input_layernorm =
            RMSNorm(config.hidden_size, config.rms_norm_eps, base_name + names._attn_norm_name);
        post_attention_layernorm =
            RMSNorm(config.hidden_size, config.rms_norm_eps, base_name + names._ffn_norm_name);
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = input_layernorm(inputs[0]);
        x = self_atten({x, x, x})[0];
        auto tmp = x + inputs[0];
        x = post_attention_layernorm(tmp);
        x = mlp({x})[0];
        x = x + tmp;
        return {x};
    }

    QWen3Attention &get_attention() {
        return self_atten;
    }

private:
    QWen3Attention self_atten;
    QWen3MLP mlp;
    Layer input_layernorm;
    Layer post_attention_layernorm;
};

class QWen3Model final : public Module {
public:
    QWen3Model() = default;
    QWen3Model(const QWen3Config &config, const QWen3NameConfig &names, const string &base_name) {
        blocks = List<QWen3Decoder>(config.num_hidden_layers, config, names, base_name);
        norm = RMSNorm(config.hidden_size, config.rms_norm_eps, names.post_norm_name);
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = inputs[0];
        for (auto &block : blocks) { x = block({x})[0]; }
        x = norm(x);
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

private:
    std::vector<QWen3Decoder> blocks;
    Layer norm;
};

class QWen3ForCausalLM final : public Module {
public:
    QWen3ForCausalLM(QWen3Config &config) {
        auto names = config.names_config;
        hidden_size = config.hidden_size;
        tie_embedding_words = config.tie_embedding_words;
        embedding = Embedding(config.vocab_size, config.hidden_size, names.token_embd_name);
        model = QWen3Model(config, names, names.blk_name);

        // Qwen-0.5 use tied embedding
        // Others use nn.Linear()
        if (tie_embedding_words) {
            lm_head = Parameter(1, config.vocab_size, 1, config.hidden_size,
                                names.token_embd_name + ".weight");
        } else {
            lm_head_layer =
                Linear(config.hidden_size, config.vocab_size, false, names.lm_head_name);
        }
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = embedding(inputs[0]);

        // go through model
        auto outputs = model({x})[0];
        if (tie_embedding_words) {
            outputs = Tensor::mm(outputs, lm_head().transpose(Chl::SEQUENCE, Chl::DIMENSION));
        } else {
            outputs = lm_head_layer(outputs);
        }
        return {outputs};
    }
    void clear_kvcache() override {
        model.clear_kvcache();
    }

private:
    int hidden_size;
    bool tie_embedding_words;
    Layer embedding;
    Parameter lm_head;
    Layer lm_head_layer;
    QWen3Model model;
};

#endif