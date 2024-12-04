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
class GemmaDecoder final : public Module {
public:
    GemmaDecoder() = default;
    GemmaDecoder(const GemmaConfig &config, const GemmaNameConfig &names, const string &base_name) {
        self_atten = MultiHeadAttention(config.hidden_size, config.num_attention_heads, config.num_key_value_heads,
                                        config.hidden_size / config.num_attention_heads, SPLIT_NONE, false, false,
                                        config.RoPE_type, config.rope_theta, config.max_position_embeddings, config.cache_limit, true, false, names, base_name + names._attn_base_name);
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

    MultiHeadAttention &get_attention() {
        return self_atten;
    }

private:
    MultiHeadAttention self_atten;
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

    void clear_kvcache() override {
        for (auto &block : blocks) {
            auto kvcache = block.get_attention().get_cache();
            for (auto &cache : kvcache) { cache->clearCache(); }
            auto ropes = block.get_attention().get_rope();
            for (auto &rope : ropes) { rope->clearCache(); }
        }
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
    void clear_kvcache() override {
        model.clear_kvcache();
    }

private:
    int hidden_size;
    Layer embedding;
    Parameter lm_head;
    GemmaModel model;
};

#endif //! MODELING_GEMMA_HPP