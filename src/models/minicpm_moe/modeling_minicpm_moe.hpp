#ifndef MODELING_MINICPMMOE_HPP
#define MODELING_MINICPMMOE_HPP

#include "Layer.hpp"
#include "Module.hpp"
#include "Tensor.hpp"
#include "Types.hpp"
#include "configuration_minicpm_moe.hpp"
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

class MiniCPMMoE final : public Module {
public:
    MiniCPMMoE() = default;
    MiniCPMMoE(const MiniCPMConfig &config, const MiniCPMNameConfig &names, const string &base_name) {
        experts = List<MiniCPMMLP>(config.num_experts, config.hidden_size, config.intermediate_size, names, base_name + "experts.");
        gate = Linear(config.hidden_size, config.num_experts, false, base_name + "gate");
        softmax = Softmax(DIMENSION, false, base_name + "softmax");
        num_experts_per_tok = config.num_experts_per_tok;
    }
    // receive embeds
    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto hidden_states = inputs[0];
        if (hidden_states.batch() > 1) {
            hidden_states = hidden_states.view(1, -1, ANYDIM, -1); // 1, batch*seq, 1, hidden
        }
        auto scores = gate(hidden_states); //  1, batch*seq, 1, num_experts
        scores = softmax(scores);
        auto experts_w_i = Tensor::topk(scores, num_experts_per_tok, DIMENSION);
        auto expert_weights = experts_w_i[0].get();                      //  1, batch*seq, 1, k
        auto expert_indices = experts_w_i[1].get();                      //  1, batch*seq, 1, k
        expert_indices = expert_indices.view(-1, 1, 1, -1);              // 1, 1, 1, k* batch*seq
        expert_weights = expert_weights / expert_weights.sum(DIMENSION); //  1, batch*seq, 1, k
        expert_weights = expert_weights.view(-1, -1, 1, 1);              // 1, k* batch*seq, 1, 1
        // moe_infer
        auto idxs = expert_indices.argsort();               // 1, 1, 1, k* batch*seq
        auto tokens_per_expert = expert_indices.bincount(); // (1, 1, 1, 0) 1, 1, 1, k
        auto token_idxs = idxs / num_experts_per_tok;       // 1, 1, 1, k* batch*seq
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
            auto exp_token_idx = token_idxs.clip({}, {}, {}, {start_idx, end_idx}); //(1, 1, 1, 0) 1, 1, 1, e-s
            auto exp_idx = idxs.clip({}, {}, {}, {start_idx, end_idx});             //(1, 1, 1, 0) 1, 1, 1, e-s
            auto expert_tokens = hidden_states.clip(exp_token_idx, SEQUENCE);       //(1, 0, 1, hidden) 1, e-s, 1, hidden
            auto expert_out = experts[i]({expert_tokens})[0];                       //(1, 0, 1, hidden) 1, e-s, 1,
            auto expert_weights_clip = expert_weights.clip(exp_idx, SEQUENCE);      //(1, 0, 1, 1) 1, e-s, 1, 1
            expert_out = expert_out * expert_weights_clip;                          //(1, 0, 1, hidden) 1, e-s, 1, hidden
            expert_cache.scatter_reduce(expert_out, exp_token_idx);                 // 1, batch*seq, 1, hidden
            //
            start_idx = end_idx;
        }
        if (hidden_states.batch() > 1) {
            // expert_cache.view(ANYDIM, seq, -1, -1);//TODO
        }
        return {expert_cache};
    }

private:
    std::vector<MiniCPMMLP> experts;
    Layer gate;
    Softmax softmax;

    int num_experts_per_tok{};
};

class MiniCPMDecoder final : public Module {
public:
    MiniCPMDecoder() = default;
    MiniCPMDecoder(const MiniCPMConfig &config, const MiniCPMNameConfig &names, const string &base_name) {
        self_atten = MultiHeadAttention(config.hidden_size, config.num_attention_heads, config.num_key_value_heads,
                                        config.hidden_size / config.num_attention_heads, SPLIT_NONE, false, false,
                                        config.RoPE_type, config.rope_theta, config.max_position_embeddings, config.cache_limit,
                                        true, false, names, base_name + names._attn_base_name);
        moe = MiniCPMMoE(config, names, base_name + names._ffn_base_name);
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
        hidden_states = moe({hidden_states})[0];
        hidden_states = hidden_states * (scale_depth / std::sqrt(num_hidden_layers)) + tmp;
        return {hidden_states};
    }

    MultiHeadAttention &get_attention() {
        return self_atten;
    }

private:
    MultiHeadAttention self_atten;
    MiniCPMMoE moe;
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
        KVCache_TYPE = 32;
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

#endif // MODELING_MINICPMMOE_HPP