/**
 * @file modeling_gemma.hpp
 * @author Chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief The defination of gemma model
 * https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma/modeling_gemma.py
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
#include "Tensor.hpp"
#include "configuration_gemma.hpp"
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
        auto x = gate_proj(inputs[0]);
        x = gelu(x);
        auto y = up_proj(inputs[0]);
        x = x * y;
        x = down_proj(x);
        return {x};
    }

private:
    Layer gate_proj;
    Layer up_proj;
    Layer down_proj;

    Layer gelu; ///< F.gelu(gate, approximate="tanh")
};

///< gemma-2B use MQA while 7B use MHA
class GemmaAttention final : public Module {
public:
    GemmaAttention() = default;
    GemmaAttention(const GemmaConfig &config, const GemmaNameConfig &names, const string &base_name) {
        hidden_size = config.hidden_size;
        head_dim = config.head_dim;
        num_heads = config.num_attention_heads;
        num_key_value_heads = config.num_key_value_heads;
        num_key_value_groups = num_heads / num_key_value_heads;

        // init layers
        q_proj = Linear(hidden_size, num_heads * head_dim, false, base_name + names._q_proj_name);
        k_proj = Linear(hidden_size, num_key_value_heads * head_dim, false, base_name + names._k_proj_name);
        v_proj = Linear(hidden_size, num_key_value_heads * head_dim, false, base_name + names._v_proj_name);
        o_proj = Linear(num_heads * head_dim, hidden_size, false, base_name + names._o_proj_name);
        q_rope = RoPE(config.RoPE_type, base_name + "q_rope");
        k_rope = RoPE(config.RoPE_type, base_name + "k_rope");
        k_cache = KVCache(num_heads / num_key_value_heads, config.cache_limit, base_name + "k_cache");
        v_cache = KVCache(num_heads / num_key_value_heads, config.cache_limit, base_name + "v_cache");
        mask = Causalmask(base_name + "mask");
        softmax = Softmax(DIMENSION, base_name + "softmax");
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto query_states = q_proj(inputs[0]);
        auto key_states = k_proj(inputs[1]);
        auto value_states = v_proj(inputs[2]);

        // [batch, heads, sequence, dims]
        query_states = query_states.view(-1, num_heads, -1, head_dim);
        key_states = key_states.view(-1, num_key_value_heads, -1, head_dim);
        value_states = value_states.view(-1, num_key_value_heads, -1, head_dim);

        // embedding
        query_states = q_rope(query_states);
        key_states = k_rope(key_states);

        // kv cache
        key_states = k_cache(key_states);
        value_states = v_cache(value_states);

        // attention weight
        auto atten_weight = Tensor::mm(query_states, key_states.transpose(Chl::SEQUENCE, Chl::DIMENSION)) / std::sqrt(head_dim);
        atten_weight = mask(atten_weight);
        atten_weight = softmax(atten_weight);

        // attention output
        auto atten_output = Tensor::mm(atten_weight, value_states);
        atten_output = atten_output.view(-1, 1, -1, head_dim * num_heads);
        atten_output = o_proj(atten_output);
        return {atten_output};
    }

private:
    int hidden_size;
    int num_heads;
    int head_dim;
    int num_key_value_heads;
    int num_key_value_groups;
    Layer q_proj;
    Layer k_proj;
    Layer v_proj;
    Layer o_proj;
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
        self_atten = GemmaAttention(config, names, base_name + names._attn_base_name);
        mlp = GemmaMLP(config.hidden_size, config.intermediate_size, names, base_name + names._ffn_base_name);
        input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, true, base_name + names._attn_norm_name);
        post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, true, base_name + names._ffn_norm_name);
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

private:
    GemmaAttention self_atten;
    GemmaMLP mlp;
    Layer input_layernorm;
    Layer post_attention_layernorm;
};

class GemmaModel final : public Module {
public:
    GemmaModel() = default;
    GemmaModel(const GemmaConfig &config, const GemmaNameConfig &names, const string &base_name) {
        blocks = List<GemmaDecoder>(config.num_hidden_layers, config, names, base_name);
        norm = RMSNorm(config.hidden_size, config.rms_norm_eps, true, names.post_norm_name);
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = inputs[0];
        for (auto &block : blocks) {
            x = block({x})[0];
        }
        x = norm(x);
        return {x};
    }

private:
    std::vector<GemmaDecoder> blocks;
    Layer norm;
};

class GemmaForCausalLM final : public Module {
public:
    GemmaForCausalLM(GemmaConfig &config) {
        auto names = config.names_config;
        hidden_size = config.hidden_size;
        embedding = Embedding(config.vocab_size, config.hidden_size, names.token_embd_name);
        model = GemmaModel(config, names, names.blk_name);

        // gemma's lm_head and tok_embedding is tied together.
        // They share same parameters. Use a Transpose to do the lm_head instead.
        lm_head = Parameter(1, config.vocab_size, 1, config.hidden_size, names.lm_head_name + ".weight");
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = embedding(inputs[0]);

        // do nomalize
        x = x * std::sqrt(hidden_size);

        // go through model
        auto outputs = model({x})[0];
        outputs = Tensor::mm(outputs, lm_head().transpose(Chl::SEQUENCE, Chl::DIMENSION));
        return {outputs};
    }

private:
    int hidden_size;
    Layer embedding;
    Parameter lm_head;
    GemmaModel model;
};

#endif //! MODELING_GEMMA_HPP