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

#include "Layer.hpp"
#include "Module.hpp"
#include "configuration_gemma.hpp"
#include "models/transformer/modeling_transformer.hpp"

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

class GemmaAttention final : public Module {
public:
    GemmaAttention() = default;
    GemmaAttention(int hidden_size, int num_heads, int num_kv_heads, int head_dims, int cache_limit, const GemmaNameConfig &names, const string &base_name) {
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        return {attention({inputs[0], inputs[0], inputs[0]})[0]};
    }

private:
    MultiHeadAttention attention;
};

class GemmaDecoder final : public Module {
public:
    GemmaDecoder() = default;
    GemmaDecoder(const GemmaNameConfig &names, const string &base_name) {
        // TODO
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
private:
};

#endif //! MODELING_GEMMA_HPP