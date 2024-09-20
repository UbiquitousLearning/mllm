#ifndef MODELING_QWEN_HPP
#define MODELING_QWEN_HPP

#include "Backend.hpp"
#include "Layer.hpp"
#include "Module.hpp"
#include "Tensor.hpp"
#include "Types.hpp"
#include "configuration_qwen.hpp"
#include <cmath>
#include <type_traits>
using namespace mllm;

// NPU QKV part
class QwenDecoderNPUPart1 final : public Module {
    int hidden_size;
    int num_heads;
    int head_dim;
    int num_key_value_heads;
    int num_key_value_groups;
    Layer q_proj;
    Layer k_proj;
    Layer v_proj;
    Layer o_proj;

public:
    QwenDecoderNPUPart1() = default;
    QwenDecoderNPUPart1(const QWenConfig &config, const QWenNameConfig &names, const string &base_name) {
        hidden_size = config.hidden_size;
        num_heads = config.num_attention_heads;
        head_dim = config.hidden_size / num_heads;
        num_key_value_heads = config.num_key_value_heads;
        num_key_value_groups = num_heads / num_key_value_heads;

        q_proj = Linear(hidden_size, num_heads * head_dim, true, base_name + names._attn_base_name + names._q_proj_name);
        k_proj = Linear(hidden_size, num_key_value_heads * head_dim, true, base_name + names._attn_base_name + names._k_proj_name);
        v_proj = Linear(hidden_size, num_key_value_heads * head_dim, true, base_name + names._attn_base_name + names._v_proj_name);
        o_proj = Linear(num_heads * head_dim, hidden_size, false, base_name + names._attn_base_name + names._o_proj_name);
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto query_states = q_proj(inputs[0]);
        auto key_states = k_proj(inputs[0]);
        auto value_states = v_proj(inputs[0]);

        // [batch, heads, sequence, dims]
        // TODO: qnn tensorFunc
        // query_states = query_states.view(-1, num_heads, -1, head_dim);
        // key_states = key_states.view(-1, num_key_value_heads, -1, head_dim);
        // value_states = value_states.view(-1, num_key_value_heads, -1, head_dim);

        // TODO: dequantize q,k,v, transpose v using qnn layer
        // query_states = query_states.toFloat(); ??

        // value_states = value_states.transpose(SEQUENCE, DIMENSION);
        return {query_states, key_states, value_states};
    }
};

// CPU QKV MM part
class QwenQKVmm final : public Module {
    Layer softmax;
    Layer q_rope;
    Layer k_rope;
    Layer k_cache;
    Layer v_cache;

    int head_size_{};
    int attn_hidden_dim_{};

public:
    QwenQKVmm() = default;
    QwenQKVmm(const QWenConfig &config, const QWenNameConfig &names, const string &base_name) {
        attn_hidden_dim_ = config.hidden_size;
        head_size_ = config.num_attention_heads * config.hidden_size / config.num_attention_heads;

        q_rope = RoPE(config.RoPE_type, config.rope_theta, config.max_position_embeddings, base_name + "q_rope");
        k_rope = RoPE(config.RoPE_type, config.rope_theta, config.max_position_embeddings, base_name + "k_rope");

        if (config.cache_limit > 0) {
            k_cache = KVCache(config.num_attention_heads / config.num_key_value_heads, config.cache_limit, base_name + names._attn_base_name + "k_cache");
            v_cache = KVCache(config.num_attention_heads / config.num_key_value_heads, config.cache_limit, base_name + names._attn_base_name + "v_cache");
        }

        softmax = Softmax(DIMENSION, true, base_name + "softmax");
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto q = inputs[0];
        auto k = inputs[1];
        auto v = inputs[2];

        q = q_rope(q);
        k = k_rope(k);

        if (k_cache.ready() && v_cache.ready()) {
            k = k_cache(k);
            v = v_cache(v);
        }

        k = k.transpose(SEQUENCE, DIMENSION);
        auto qk = Tensor::mm(q, k);
        qk = qk / std::sqrt(attn_hidden_dim_);
        qk = softmax(qk);
        auto o = Tensor::mm(qk, v);

        return {o};
    }
};

// QNN mlp part
class QwenDecoderNPUPart2 final : public Module {
    Layer gate_proj;
    Layer up_proj;
    Layer down_proj;

    Layer silu;

public:
    QwenDecoderNPUPart2() = default;
    QwenDecoderNPUPart2(const QWenConfig &config, const QWenNameConfig &names, const string &base_name) {
        int hidden_size = config.hidden_size;
        int intermediate_size = config.intermediate_size;

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
};

class QwenNPU_CPUDecoder final : public Module {
    Layer input_layernorm;
    QwenDecoderNPUPart1 part1;
    QwenQKVmm qkv_mm;
    QwenDecoderNPUPart2 part2;

public:
    QwenNPU_CPUDecoder() = default;
    QwenNPU_CPUDecoder(const QWenConfig &config, const QWenNameConfig &names, const string &base_name) {
        input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, base_name + names._attn_norm_name);

        part1 = QwenDecoderNPUPart1(config, names, base_name);
        part1.to(MLLM_QNN);

        qkv_mm = QwenQKVmm(config, names, base_name);
        qkv_mm.to(MLLM_CPU);

        part2 = QwenDecoderNPUPart2(config, names, base_name);
        part2.to(MLLM_QNN);
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = input_layernorm(inputs[0]);
        // TODO: quantize x to int8
        if (x.device() != MLLM_QNN) {
            x = Tensor::toQNN({x})[0];
        }

        auto q_k_v = part1({x}); // q,k,v
        auto o_x = qkv_mm(q_k_v);

        auto tmp = o_x[0] + inputs[0];

        o_x = Tensor::toQNN(o_x);
        x = part2(o_x)[0];
        x = x + tmp;

        return {x};
    }
};

// Copied from GemmaModel with Gemma->Qwen and set RmsNorm(without add_unit_offset)
class QWenModel final : public Module {
public:
    QWenModel() = default;
    QWenModel(const QWenConfig &config, const QWenNameConfig &names, const string &base_name) {
        // TODO: only one block, change it to config.num_hidden_layers
        blocks = List<QwenNPU_CPUDecoder>(1, config, names, base_name);
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
    std::vector<QwenNPU_CPUDecoder> blocks;
    Layer norm;
};

class QWenForCausalLM final : public Module {
public:
    QWenForCausalLM(QWenConfig &config) {
        auto names = config.names_config;
        hidden_size = config.hidden_size;
        tie_embedding_words = config.tie_embedding_words;
        embedding = Embedding(config.vocab_size, config.hidden_size, names.token_embd_name);
        model = QWenModel(config, names, names.blk_name);

        // Qwen-0.5 use tied embedding
        // Others use nn.Linear()
        if (tie_embedding_words) {
            lm_head = Parameter(1, config.vocab_size, 1, config.hidden_size, names.token_embd_name + ".weight");
        } else {
            lm_head_layer = Linear(config.hidden_size, config.vocab_size, false, names.lm_head_name);
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

    virtual void generate(
        Tensor &input_ids, const LlmTextGeneratorOpts &opt, const std::function<bool(unsigned int)> &call_back = [](unsigned int) -> bool { return true; }) override{
        auto chatPostProcessing = [](unsigned token_idx, Tensor &tokens_tensor, const vector<Tensor *> &clean_tensors) {
            tokens_tensor.reshape(1, 1, 1, 1);
            tokens_tensor.alloc();
            tokens_tensor.setDataAt<float>(0, 0, 0, 0, token_idx);

            for (auto tensor : clean_tensors) {
                tensor->reshape(0, 0, 0, 0);
                tensor->alloc();
            }
        };

        if (!opt.do_sample) {
            // fail to greedy search
            if (!text_generator_ || text_generator_->type() != LLmTextGeneratorType::kGreedySearch)
                text_generator_ = std::make_shared<LlmTextGenerator>(LLmTextGeneratorType::kGreedySearch, opt);
        } else if (opt.do_sample && !opt.top_k && opt.top_p != 0.f) {
            // fail to top p sampling
            if (!text_generator_ || text_generator_->type() != LLmTextGeneratorType::kToppSampling)
                text_generator_ = std::make_shared<LlmTextGenerator>(LLmTextGeneratorType::kToppSampling, opt);
        } else if (opt.do_sample && opt.top_k) {
            // fail to top k sampling
            if (!text_generator_ || text_generator_->type() != LLmTextGeneratorType::kTopkSampling)
                text_generator_ = std::make_shared<LlmTextGenerator>(LLmTextGeneratorType::kTopkSampling, opt);
        }

        for (int step = 0; step < opt.max_new_tokens; ++step) {
            auto _out = (*this)({input_ids});
            auto out_token = text_generator_->generate(_out[0]);
            if (!call_back(out_token)) break;
            chatPostProcessing(out_token, input_ids, {});
            std::cout << "========AFTER PREFILL=========" << std::endl;
            return;
        }
    }

private:
    int hidden_size;
    bool tie_embedding_words;
    Layer embedding;
    Parameter lm_head;
    Layer lm_head_layer;
    QWenModel model;
};

#endif //! MODELING_QWEN_HPP