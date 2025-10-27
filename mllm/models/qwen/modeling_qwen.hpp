/**
 * @file modeling_qwen.hpp
 * @author Chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py
 * @version 0.1
 * @date 2024-04-29
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef MODELING_QWEN_HPP
#define MODELING_QWEN_HPP

#include "Backend.hpp"
#include "DataType.hpp"
#include "Layer.hpp"
#include "Module.hpp"
#include "Tensor.hpp"
#include "Types.hpp"
#include "configuration_qwen.hpp"
#include "models/transformer/modeling_transformer.hpp"
#include <cmath>
using namespace mllm;

// Copied from GemmaMLP with Gemma->Qwen and using silu
class QWenMLP final : public Module {
public:
    QWenMLP() = default;
    QWenMLP(int hidden_size, int intermediate_size, const QWenNameConfig &names,
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

class QWenDecoder final : public Module {
public:
    QWenDecoder() = default;
    QWenDecoder(const QWenConfig &config, const QWenNameConfig &names, const string &base_name) {
        self_atten = MultiHeadAttention(config.hidden_size, config.num_attention_heads,
                                        config.num_key_value_heads, config.hidden_size / config.num_attention_heads,
                                        SPLIT_NONE, PostQkv_NONE, false,
                                        config.RoPE_type, config.rope_theta,
                                        config.max_position_embeddings,
                                        config.cache_limit,
                                        true, true, false,
                                        config.attn_implementation, names,
                                        base_name + names._attn_base_name);
        mlp = QWenMLP(config.hidden_size, config.intermediate_size, names,
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

    MultiHeadAttention &get_attention() {
        return self_atten;
    }

private:
    MultiHeadAttention self_atten;
    QWenMLP mlp;
    Layer input_layernorm;
    Layer post_attention_layernorm;
};

// Copied from GemmaModel with Gemma->Qwen and set RmsNorm(without add_unit_offset)
class QWenModel final : public Module {
public:
    QWenModel() = default;
    QWenModel(const QWenConfig &config, const QWenNameConfig &names, const string &base_name) {
        blocks = List<QWenDecoder>(config.num_hidden_layers, config, names, base_name);
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

    void clear_kvcache() override {
        for (auto &block : blocks) {
            auto kvcache = block.get_attention().get_cache();
            for (auto &cache : kvcache) { cache->clearCache(); }
            auto ropes = block.get_attention().get_rope();
            for (auto &rope : ropes) { rope->clearCache(); }
        }
    }

private:
    std::vector<QWenDecoder> blocks;
    Layer norm;
};

class QWenForCausalLM final : public Module {
public:
    CHAINABLE_MODULE_METHODS(QWenForCausalLM)
    QWenForCausalLM(QWenConfig &config) {
        dtype = config.dtype;
        auto names = config.names_config;
        hidden_size = config.hidden_size;
        tie_embedding_words = config.tie_embedding_words;
        embedding = Embedding(config.vocab_size, config.hidden_size, names.token_embd_name);
        model = QWenModel(config, names, names.blk_name);

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
        auto x = embedding(inputs[0]).to(dtype);
        // go through model
        auto outputs = model({x})[0];
        if (outputs.sequence() > 1) {
            outputs = outputs.clip({}, {}, {-1}, {});
        }
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
    QWenModel model;
    DataType dtype;
};

#endif //! MODELING_QWEN_HPP