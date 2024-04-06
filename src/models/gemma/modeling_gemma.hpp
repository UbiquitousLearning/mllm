/**
 * @file modeling_gemma.hpp
 * @author Chenghua Wang (chenghua.wang@gmail.com)
 * @brief The defination of gemma model
 * @version 0.1
 * @date 2024-04-03
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef MODELING_GEMMA_HPP
#define MODELING_GEMMA_HPP

#include "Backend.hpp"
#include "Layer.hpp"
#include "Module.hpp"
#include "configuration_gemma.hpp"
#include "models/transformer/modeling_transformer.hpp"
#include <cmath>

using namespace mllm;

class GemmaMLP final : public Module {
public:
    GemmaMLP() = default;
    GemmaMLP(int hidden_size, int intermediate_size, const GemmaNameConfig &names, const std::string &base_name) {
        gate_proj = Linear(hidden_size, intermediate_size, false, base_name + names._gate_proj_name);
        gelu = GELU(base_name + "act");
        up_proj = Linear(hidden_size, intermediate_size, false, base_name + names._up_proj_name);
        down_proj = Linear(intermediate_size, hidden_size, false, base_name + names._down_proj_name);
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = inputs[0];
        auto gate = gate_proj(x);
        gate = gelu(x);
        auto up = up_proj(x);
        auto fuse = gate * up;
        auto outputs = down_proj(fuse);
        return {outputs};
    }

private:
    Layer gate_proj;
    Layer up_proj;
    Layer down_proj;

    // FIXME: Check the default method is gelu with tanh or not.
    Layer gelu; ///< F.gelu(gate, approximate="tanh")
};

///< gemma-2B use MQA while 7B use MHA
class GemmaAttention final : public Module {
public:
    GemmaAttention() = default;
    GemmaAttention(int hidden_size, int num_heads, int num_kv_heads, int head_dim, int cache_limit, const GemmaNameConfig &names, const string &base_name) :
        hidden_size(hidden_size), num_heads(num_heads), num_kv_heads(num_kv_heads), head_dim(head_dim) {
        assert(num_heads % num_kv_heads == 0 && "When using MQA/GQA. num_heads mod num_kv_heads should 0");

        // for MQA and GQA
        num_queries_per_kv = num_heads / num_kv_heads;
        q_size = num_heads * head_dim;
        kv_size = num_kv_heads * head_dim;

        // scaling
        scaling = 1.f / std::sqrt(head_dim);

        // init layers
        qkv_proj = Linear(hidden_size, (num_heads + 2 * num_kv_heads) * head_dim, false, base_name + names._qkv_proj_name);
        qkv_split = Split({q_size, kv_size, kv_size}, Chl::DIMENSION, base_name + names._qkv_proj_name + ".split");
        o_proj = Linear(num_heads * head_dim, hidden_size, false, base_name + names._o_proj_name);
        // FIXME RoPEType::HFHUBROPE
        q_rope = RoPE(RoPEType::HFHUBROPE, base_name + "q_rope");
        k_rope = RoPE(RoPEType::HFHUBROPE, base_name + "k_rope");
        if (cache_limit > 0) {
            k_cache = KVCache(num_heads / num_kv_heads, cache_limit, base_name + "k_cache");
            v_cache = KVCache(num_heads / num_kv_heads, cache_limit, base_name + "v_cache");
        }
        mask = Causalmask(base_name + "mask");
        softmax = Softmax(DIMENSION, base_name + "softmax");
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto hidden_sates = inputs[0];
        auto hidden_sates_shape = hidden_sates.shape();
        auto batch_size = hidden_sates_shape[0];
        auto input_size = hidden_sates_shape[1];

        auto qkv = qkv_proj(hidden_sates);
        auto qkv_sp = qkv_split(qkv);
        auto xq = qkv_sp[0];
        auto xk = qkv_sp[1];
        auto xv = qkv_sp[2];

        xq = xq.view(batch_size, -1, num_heads, head_dim);
        xk = xk.view(batch_size, -1, num_kv_heads, head_dim);
        xv = xv.view(batch_size, -1, num_kv_heads, head_dim);

        // position embedding
        xq = q_rope(xq);
        xk = k_rope(xk);

        // kv cache
        auto key = k_cache(xk);
        auto value = v_cache(xv);

        // repeat
        if (num_kv_heads != num_heads) {
            std::vector<Tensor> _key_repeat, _value_repeat;
            for (int i = 0; i < num_queries_per_kv; ++i) _key_repeat.push_back(key);
            for (int i = 0; i < num_queries_per_kv; ++i) _value_repeat.push_back(value);
            key = Tensor::cat(_key_repeat, /*dims*/ Chl::SEQUENCE);
            value = Tensor::cat(_value_repeat, /*dims*/ Chl::SEQUENCE);
        }

        // [batch, head, seq, dim]
        auto q = xq.transpose(Chl::HEAD, Chl::SEQUENCE);
        auto k = key.transpose(Chl::HEAD, Chl::SEQUENCE);
        auto v = value.transpose(Chl::HEAD, Chl::SEQUENCE);

        // [batch_size, n_local_heads, input_len, max_seq_len]
        auto scores = Tensor::mm(q, k.transpose(Chl::SEQUENCE, Chl::DIMENSION)) * scaling;
        scores = mask(scores);
        scores = softmax(scores);

        // output
        auto output = Tensor::mm(scores, v);
        // FIXME fix vie size.
        output = output.transpose(Chl::HEAD, Chl::SEQUENCE).view(batch_size, input_size, -1, -1);
        output = o_proj(output);
        return {output};
    }

private:
    int hidden_size;
    int num_heads;
    int num_kv_heads;
    int head_dim;

    int num_queries_per_kv;
    int q_size;
    int kv_size;
    float scaling;

    // layers
    Layer qkv_proj;
    Layer o_proj;
    Split qkv_split;
    Layer q_rope;
    Layer k_rope;
    Layer k_cache;
    Layer v_cache;
    Layer mask;
    Layer softmax;
};

class GemmaDecoder final : public Module {
public:
    GemmaDecoder() = default;
    GemmaDecoder(const GemmaConfig &config, const GemmaNameConfig &names, const string &base_name) {
        self_atten = GemmaAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.cache_limit,
            names,
            base_name);
        mlp = GemmaMLP(config.hidden_size, config.intermediate_size, names, base_name);
        input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, base_name + names._attn_norm_name);
        post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, base_name + names._ffn_norm_name);
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        // self attention
        auto residual = inputs[0];
        auto hidden_sates = input_layernorm(inputs[0]);
        hidden_sates = self_atten({hidden_sates})[0];
        hidden_sates = hidden_sates + residual;

        // mlp
        residual = hidden_sates;
        hidden_sates = post_attention_layernorm(hidden_sates);
        hidden_sates = mlp({hidden_sates})[0];
        hidden_sates = residual + hidden_sates;

        return {hidden_sates};
    }

private:
    GemmaAttention self_atten;
    GemmaMLP mlp;
    Layer input_layernorm;
    Layer post_attention_layernorm;
};

class GemmaModle final : public Module {
public:
    GemmaModle() = default;
    GemmaModle(const GemmaConfig &config, const GemmaNameConfig &names, const string &base_name) {
        for (int i = 0; i < config.num_hidden_layers; ++i) layers.push_back(GemmaDecoder(config, names, base_name));
        norm = RMSNorm(config.hidden_size, config.rms_norm_eps, base_name + names.post_norm_name);
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = inputs[0];
        for (auto &layer : layers) {
            x = layer({x})[0];
        }
        x = norm(x);
        return {x};
    }

private:
    std::vector<GemmaDecoder> layers;
    Layer norm;
};

class GemmaForCausalLM final : public Module {
public:
    GemmaForCausalLM(GemmaConfig &config) {
        auto names = config.names_config;
        embedding = Embedding(config.vocab_size, config.hidden_size, names.token_embd_name);
        model = GemmaModle(config, names, names.blk_name);
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = embedding(inputs[0]);
        auto outputs = model({x});
        return outputs;
    }

private:
    Layer embedding;
    GemmaModle model;
};

#endif //! MODELING_GEMMA_HPP