/**
 * @file modeling_mistral.hpp
 * @author Chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief Mistral 7B instruction 0.2V in Huggingface
 * @version 0.1
 * @date 2024-05-25
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef MODELING_MISTRAL_HPP
#define MODELING_MISTRAL_HPP

#include "Backend.hpp"
#include "Layer.hpp"
#include "Module.hpp"
#include "Tensor.hpp"
#include "configuration_mistral.hpp"
#include "models/transformer/modeling_transformer.hpp"
#include <cmath>
using namespace mllm;

class MistralMLP final : public Module {
public:
    MistralMLP() = default;
    MistralMLP(int hidden_size, int intermediate_size, const MistralNameConfig &names, const std::string &base_name) {
        gate_proj = Linear(hidden_size, intermediate_size, false, base_name + names._gate_proj_name);
        silu = SiLU(base_name + "act");
        up_proj = Linear(hidden_size, intermediate_size, false, base_name + names._up_proj_name);
        down_proj = Linear(intermediate_size, hidden_size, false, base_name + names._down_proj_name);
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

class MistralDecoder final : public Module {
public:
    MistralDecoder() = default;
    MistralDecoder(const MistralConfig &config, const MistralNameConfig &names, const string &base_name) {
        self_atten = MultiHeadAttention(config.hidden_size, config.num_attention_heads, config.num_key_value_heads,
                                        config.hidden_size / config.num_attention_heads, SPLIT_NONE, false, false,
                                        config.RoPE_type, config.rope_theta, config.max_position_embeddings, config.cache_limit,
                                        true, false, names, base_name + names._attn_base_name);
        mlp = MistralMLP(config.hidden_size, config.intermediate_size, names, base_name + names._ffn_base_name);
        input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, base_name + names._attn_norm_name);
        post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, base_name + names._ffn_norm_name);
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
    MultiHeadAttention self_atten;
    MistralMLP mlp;
    Layer input_layernorm;
    Layer post_attention_layernorm;
};

class MistralModel final : public Module {
public:
    MistralModel() = default;
    MistralModel(const MistralConfig &config, const MistralNameConfig &names, const string &base_name) {
        blocks = List<MistralDecoder>(config.num_hidden_layers, config, names, base_name);
        norm = RMSNorm(config.hidden_size, config.rms_norm_eps, names.post_norm_name);
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
    std::vector<MistralDecoder> blocks;
    Layer norm;
};

class MistralForCausalLM final : public Module {
public:
    MistralForCausalLM(MistralConfig &config) {
        auto names = config.names_config;
        hidden_size = config.hidden_size;
        embedding = Embedding(config.vocab_size, config.hidden_size, names.token_embd_name);
        model = MistralModel(config, names, names.blk_name);
        lm_head = Linear(hidden_size, config.vocab_size, false, names.lm_head_name);
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = embedding(inputs[0]);

        // go through model
        auto outputs = model({x})[0];
        if (outputs.sequence() > 1) {
            outputs = outputs.clip({}, {}, {-1}, {});
        }
        outputs = lm_head(outputs);
        return {outputs};
    }

private:
    int hidden_size;
    bool tie_embedding_words;
    Layer embedding;
    Layer lm_head;
    MistralModel model;
};

#endif //! MODELING_MISTRAL_HPP