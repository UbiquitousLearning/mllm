#ifndef MODELING_MINICPM_HPP
#define MODELING_MINICPM_HPP

#include "Layer.hpp"
#include "Module.hpp"
#include "configuration_minicpm.hpp"
#include "models/transformer/modeling_transformer.hpp"
#include <any>
#include <cmath>

using namespace mllm;

class MiniCPMMLP final : public Module {
public:
    MiniCPMMLP() = default;
    MiniCPMMLP(int hidden_size, int intermediate_size, const MiniCPMNameConfig &names, const std::string &base_name) {
        gate_proj = Linear(hidden_size, intermediate_size, false, base_name + names._gate_proj_name);
        silu = SiLU(base_name + "act");
        up_proj = Linear(hidden_size, intermediate_size, false, base_name + names._up_proj_name);
        down_proj = Linear(intermediate_size, hidden_size, false, base_name + names._down_proj_name);
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = gate_proj(inputs[0]);
        x = silu(x);
        auto y = up_proj(inputs[0]); // ERROR
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

class MiniCPMDecoder final : public Module {
public:
    MiniCPMDecoder() = default;
    MiniCPMDecoder(const MiniCPMConfig &config, const MiniCPMNameConfig &names, const string &base_name) {
        self_atten = MultiHeadAttention(config.hidden_size, config.num_attention_heads, config.num_key_value_heads,
                                        config.hidden_size / config.num_attention_heads, SPLIT_NONE, false, false,
                                        config.RoPE_type, config.rope_theta, config.max_position_embeddings, config.cache_limit,
                                        true, false, names, base_name + names._attn_base_name);
        mlp = MiniCPMMLP(config.hidden_size, config.intermediate_size, names, base_name + names._ffn_base_name);
        input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, base_name + names._attn_norm_name);
        post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, base_name + names._ffn_norm_name);
        scale_depth = config.scale_depth;
        num_hidden_layers = config.num_hidden_layers;
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto hidden_states = input_layernorm(inputs[0]);
        hidden_states = self_atten({hidden_states, hidden_states, hidden_states})[0];
        auto tmp = hidden_states * (scale_depth / std::sqrt(num_hidden_layers)) + inputs[0];
        hidden_states = post_attention_layernorm(tmp);
        hidden_states = mlp({hidden_states})[0];
        hidden_states = hidden_states * (scale_depth / std::sqrt(num_hidden_layers)) + tmp;
        return {hidden_states};
    }

    MultiHeadAttention &get_attention() {
        return self_atten;
    }

private:
    MultiHeadAttention self_atten;
    MiniCPMMLP mlp;
    Layer input_layernorm;
    Layer post_attention_layernorm;
    float scale_depth;
    int num_hidden_layers;
};

class MiniCPMModel final : public Module {
public:
    MiniCPMModel() = default;
    MiniCPMModel(const MiniCPMConfig &config, const MiniCPMNameConfig &names, const string &base_name) {
        blocks = List<MiniCPMDecoder>(config.num_hidden_layers, config, names, base_name);
        norm = RMSNorm(config.hidden_size, config.rms_norm_eps, names.post_norm_name);
    }
    // receive embeds
    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto hidden_states = inputs[0];
        for (auto &block : blocks) {
            hidden_states = block({hidden_states})[0];
        }
        hidden_states = norm(hidden_states);
        return {hidden_states};
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
    std::vector<MiniCPMDecoder> blocks;
    Layer norm;
};

class MiniCPMForCausalLM final : public Module {
public:
    MiniCPMForCausalLM(MiniCPMConfig &config) {
        auto names = config.names_config;
        scale_emb = config.scale_emb;
        dim_model_base = config.dim_model_base;
        hidden_size = config.hidden_size;
        embedding = Embedding(config.vocab_size, config.hidden_size, names.token_embd_name);
        model = MiniCPMModel(config, names, names.blk_name);
        lm_head = Parameter(1, config.vocab_size, 1, config.hidden_size, names.token_embd_name + ".weight");
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = embedding(inputs[0]) * scale_emb;
        auto outputs = model({x})[0];
        outputs = outputs / (hidden_size / dim_model_base);
        outputs = Tensor::mm(outputs, lm_head().transpose(Chl::SEQUENCE, Chl::DIMENSION));
        return {outputs};
    }
    void clear_kvcache() override {
        model.clear_kvcache();
    }

private:
    int hidden_size;
    float dim_model_base;
    bool tie_embedding_words;
    float scale_emb;
    Layer embedding;
    Parameter lm_head;
    MiniCPMModel model;
};

#endif // MODELING_MINICPM_HPP