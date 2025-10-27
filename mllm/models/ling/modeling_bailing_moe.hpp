#pragma once
#include "DataType.hpp"
#include "Layer.hpp"
#include "Module.hpp"
#include "Tensor.hpp"
#include "Trace.hpp"
#include "Types.hpp"
#include "configuration_bailing_moe.hpp"
#include "models/transformer/modeling_transformer.hpp"
#include <any>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

using namespace mllm;

class BailingMoeMLP final : public Module {
public:
    BailingMoeMLP() = default;
    BailingMoeMLP(int hidden_size, int intermediate_size, const BailingMoeNameConfig &names, const std::string &base_name) {
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

class BailingMoeGate final : public Module {
public:
    BailingMoeGate() = default;
    BailingMoeGate(const BailingMoeConfig &config, const BailingMoeNameConfig &names, const std::string &base_name) {
        gate = Linear(config.hidden_size, config.num_experts, false, base_name + "gate");
        softmax = Softmax(DIMENSION, false, base_name + "softmax");
        num_experts_per_tok = config.num_experts_per_tok;
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto scores = softmax(gate(inputs[0]));
        auto experts_w_i = Tensor::topk(scores, num_experts_per_tok, DIMENSION);
        auto topk_weight = experts_w_i[0];                      //  1, batch*seq, 1, k
        auto topk_idx = experts_w_i[1];                         //  1, batch*seq, 1, k
        topk_idx = topk_idx.view(-1, 1, 1, -1);                 // 1, 1, 1, k* batch*seq
        topk_weight = topk_weight / topk_weight.sum(DIMENSION); //  1, batch*seq, 1, k
        return {scores, topk_weight, topk_idx};
    }

private:
    Layer gate;
    Softmax softmax;
    int num_experts_per_tok{};
};

class BailingMoeSparseMoeBlock final : public Module {
public:
    BailingMoeSparseMoeBlock() = default;
    BailingMoeSparseMoeBlock(const BailingMoeConfig &config, const BailingMoeNameConfig &names, const string &base_name) {
        experts = List<BailingMoeMLP>(config.num_experts, config.hidden_size, config.moe_intermediate_size, names, base_name + "experts.");
        gate = BailingMoeGate(config, names, base_name);
        num_experts_per_tok = config.num_experts_per_tok;
        num_shared_experts = config.num_shared_experts;
        if (num_shared_experts > 0) {
            shared_experts = BailingMoeMLP(config.hidden_size,
                                           config.moe_intermediate_size * config.num_shared_experts,
                                           names, base_name + "shared_experts.");
        }
    }
    // receive embeds
    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto hidden_states = inputs[0];
        auto identity = hidden_states;
        if (hidden_states.batch() > 1) {
            hidden_states = hidden_states.view(1, -1, ANYDIM, -1); // 1, batch*seq, 1, hidden
        }
        auto gates_t = gate({hidden_states});                            //  1, batch*seq, 1, num_experts
        auto scores = gates_t[0];                                        // 1, batch*seq, 1, num_experts
        auto topk_weight = gates_t[1];                                   // 1, batch*seq,
        auto topk_idx = gates_t[2];                                      // 1, batch*seq, 1, k
        hidden_states = moe_infer(hidden_states, topk_weight, topk_idx); // 1, batch*seq, 1, hidden
        if (num_shared_experts) {
            hidden_states = hidden_states + shared_experts({identity})[0]; // add shared experts
        }
        if (hidden_states.batch() > 1) {
            // expert_cache.view(ANYDIM, seq, -1, -1);//TODO
        }
        return {hidden_states};
    }
    Tensor moe_infer(Tensor hidden_states,
                     Tensor &topk_weight,
                     Tensor &topk_idx) {
        auto dtype = topk_idx.dtype();
        auto device = topk_idx.device();
        topk_idx = topk_idx.fp32().cpu();
        auto idxs = topk_idx.argsort();               // 1, 1, 1, k* batch*seq
        auto tokens_per_expert = topk_idx.bincount(); // (1, 1, 1, 0) 1, 1, 1, k
        idxs = idxs.to(device).to(dtype);
        auto token_idxs = idxs / num_experts_per_tok; // 1, 1, 1, k* batch*seq
        int start_idx = 0;
        int end_idx = start_idx;
        auto expert_cache = Tensor::zero_like(hidden_states); // 1, batch*seq, 1, hidden
        for (int i = 0; i < experts.size(); ++i) {
            if (tokens_per_expert.dimension() != 0 && i >= tokens_per_expert.dimension())
                break;
            int this_token_num = tokens_per_expert.dimension() == 0 ?
                                     0 :
                                     tokens_per_expert.d<float>(0, 0, 0, i);
            if (tokens_per_expert.dimension() != 0 && this_token_num == 0)
                continue;
            end_idx = start_idx + this_token_num;
            //
            auto exp_token_idx = token_idxs.clip({}, {}, {}, {start_idx, end_idx});             //(1, 1, 1, 0) 1, 1, 1, e-s
            auto exp_idx = idxs.clip({}, {}, {}, {start_idx, end_idx});                         //(1, 1, 1, 0) 1, 1, 1, e-s
            auto expert_tokens = hidden_states.clip(exp_token_idx, SEQUENCE);                   //(1, 0, 1, hidden) 1, e-s, 1, hidden
            auto expert_out = experts[i]({expert_tokens})[0];                                   //(1, 0, 1, hidden) 1, e-s, 1,
            if (topk_weight.dimension() != 1) { topk_weight = topk_weight.view(-1, -1, 1, 1); } // 1, k* batch*seq, 1, 1
            auto expert_weights_clip = topk_weight.clip(exp_idx, SEQUENCE);                     //(1, 0, 1, 1) 1, e-s, 1, 1
            expert_out = expert_out * expert_weights_clip;                                      //(1, 0, 1, hidden) 1, e-s, 1, hidden
            expert_cache.scatter_add(expert_out, exp_token_idx);                                // 1, batch*seq, 1, hidden
            //
            start_idx = end_idx;
        }
        return expert_cache; // 1, batch*seq, 1, hidden
    }

private:
    BailingMoeMLP shared_experts;
    std::vector<BailingMoeMLP> experts;
    BailingMoeGate gate;
    int num_shared_experts{};
    int num_experts_per_tok{};
};

class BailingMoeDecoder final : public Module {
public:
    BailingMoeDecoder() = default;
    BailingMoeDecoder(const BailingMoeConfig &config, const BailingMoeNameConfig &names, const string &base_name) {
        self_atten = MultiHeadAttention(config.hidden_size, config.num_attention_heads,
                                        config.num_key_value_heads,
                                        config.hidden_size / config.num_attention_heads,
                                        SPLIT_HD, PostQkv_NONE, false,
                                        config.RoPE_type, config.rope_theta,
                                        config.max_position_embeddings,
                                        config.cache_limit, config.use_cache, config.use_qkv_bias, config.use_bias,
                                        config.attn_implementation, names, base_name + names._attn_base_name);
        moe = BailingMoeSparseMoeBlock(config, names, base_name + names._ffn_base_name);
        input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, base_name + names._attn_norm_name);
        post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, base_name + names._ffn_norm_name);
        num_hidden_layers = config.num_hidden_layers;
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto hidden_states = input_layernorm(inputs[0]);
        hidden_states = self_atten({hidden_states, hidden_states, hidden_states})[0];
        auto tmp = hidden_states + inputs[0];
        hidden_states = post_attention_layernorm(tmp);
        hidden_states = moe({hidden_states})[0];
        hidden_states = hidden_states + tmp;
        return {hidden_states};
    }

    MultiHeadAttention &get_attention() {
        return self_atten;
    }

private:
    MultiHeadAttention self_atten;
    BailingMoeSparseMoeBlock moe;
    Layer input_layernorm;
    Layer post_attention_layernorm;
    int num_hidden_layers;
};

class BailingMoeModel final : public Module {
public:
    BailingMoeModel() = default;
    BailingMoeModel(const BailingMoeConfig &config, const BailingMoeNameConfig &names, const string &base_name) {
        blocks = List<BailingMoeDecoder>(config.num_hidden_layers, config, names, base_name);
        norm = RMSNorm(config.hidden_size, config.rms_norm_eps, names.post_norm_name);
    }
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
    std::vector<BailingMoeDecoder> blocks;
    Layer norm;
};

class BailingMoeForCausalLM final : public Module {
public:
    CHAINABLE_MODULE_METHODS(BailingMoeForCausalLM)
    BailingMoeForCausalLM(BailingMoeConfig &config) {
        dtype = config.dtype;
        auto names = config.names_config;
        hidden_size = config.hidden_size;
        embedding = Embedding(config.vocab_size, config.hidden_size, names.token_embd_name);
        model = BailingMoeModel(config, names, names.blk_name);
        lm_head = Linear(config.hidden_size, config.vocab_size, false, names.lm_head_name);
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = embedding(inputs[0]).to(dtype);
        auto outputs = model({x})[0];
        if (outputs.sequence() > 1) {
            outputs = outputs.clip({}, {}, {-1}, {});
        }
        outputs = lm_head(outputs);
        return {outputs};
    }
    void clear_kvcache() override {
        model.clear_kvcache();
    }

private:
    int hidden_size;
    bool tie_embedding_words;
    Layer embedding;
    Layer lm_head;
    BailingMoeModel model;
    DataType dtype;
};
