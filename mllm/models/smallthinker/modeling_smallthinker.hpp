#ifndef MODELING_SMOLTHINKER_HPP
#define MODELING_SMOLTHINKER_HPP

#include "Layer.hpp"
#include "Module.hpp"
#include "Tensor.hpp"
#include "Types.hpp"
#include "configuration_smallthinker.hpp"
#include "models/transformer/modeling_transformer.hpp"
#include <any>
using namespace mllm;

class SmallThinkerMLP final : public Module {
public:
    SmallThinkerMLP() = default;
    SmallThinkerMLP(int hidden_size, int intermediate_size, const SmallThinkerNameConfig &names, const std::string &base_name) {
        gate_proj = Linear(hidden_size, intermediate_size, false, base_name + names._gate_proj_name);
        relu = ReLU(base_name + "relu");
        up_proj = Linear(hidden_size, intermediate_size, false, base_name + names._up_proj_name);
        down_proj = Linear(intermediate_size, hidden_size, false, base_name + names._down_proj_name);
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = gate_proj(inputs[0]);
        x = relu(x);
        auto y = up_proj(inputs[0]);
        x = x * y;
        x = down_proj(x);
        return {x};
    }

private:
    Layer gate_proj;
    Layer up_proj;
    Layer down_proj;
    Layer relu;
};

class SmallThinkerMoeBlock final : public Module {
public:
    SmallThinkerMoeBlock() = default;
    SmallThinkerMoeBlock(const SmallThinkerConfig &config, const SmallThinkerNameConfig &names, const string &base_name) {
        experts = List<SmallThinkerMLP>(config.num_experts, config.hidden_size, config.intermediate_size, names, base_name + "experts.");
        // primary_router = Linear(config.hidden_size, config.num_experts, false, base_name + "primary_router");
        sigmoid = Sigmoid(base_name + "sigmoid");
        num_experts_per_tok = config.num_experts_per_tok;
    }
    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto hidden_states = inputs[0];
        if (hidden_states.batch() > 1) hidden_states = hidden_states.view(1, -1, ANYDIM, -1); // 1, batch*seq, 1, hidden
        auto router_logits = inputs[1];
        auto expert_indices = inputs[2];
        auto expert_weights = sigmoid(router_logits);
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
            expert_cache.scatter_add(expert_out, exp_token_idx);                    // 1, batch*seq, 1, hidden
            //
            start_idx = end_idx;
        }
        if (hidden_states.batch() > 1) {
            // expert_cache.view(ANYDIM, seq, -1, -1);//TODO
        }
        return {expert_cache};
    }

private:
    std::vector<SmallThinkerMLP> experts;
    // Layer primary_router;
    Layer sigmoid;
    int num_experts_per_tok{};
};

class SmallThinkerDecoder final : public Module {
public:
    SmallThinkerDecoder() = default;
    SmallThinkerDecoder(const SmallThinkerConfig &config, const SmallThinkerNameConfig &names, const string &base_name) {
        self_atten = MultiHeadAttention(config.hidden_size, config.num_attention_heads,
                                        config.num_key_value_heads,
                                        config.hidden_size / config.num_attention_heads,
                                        SPLIT_NONE, PostQkv_NONE, false,
                                        config.RoPE_type, config.rope_theta, config.max_position_embeddings, config.cache_limit,
                                        true, false, false,
                                        config.attn_implementation, names, base_name + names._attn_base_name);
        block_sparse_moe = SmallThinkerMoeBlock(config, names, base_name + names._ffn_base_name);
        input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, base_name + names._attn_norm_name);
        post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, base_name + names._ffn_norm_name);
        num_hidden_layers = config.num_hidden_layers;
        primary_router = Linear(config.hidden_size, config.num_experts, false, base_name + names._ffn_base_name + "primary_router");
        num_experts_per_tok = config.num_experts_per_tok;
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto router_input = inputs[0];
        if (router_input.batch() > 1) router_input = router_input.view(1, -1, 1, -1); // 1, batch*seq, 1, hidden
        auto router_logits = primary_router(router_input);                            //  1, batch*seq, 1, num_experts
        auto experts_w_i = Tensor::topk(router_logits, num_experts_per_tok, DIMENSION);
        router_logits = experts_w_i[0];                     //  1, batch*seq, 1, k
        auto expert_indices = experts_w_i[1];               //  1, batch*seq, 1, k
        expert_indices = expert_indices.view(-1, 1, 1, -1); // 1, 1, 1, k* batch*seq
        auto hidden_states = input_layernorm(router_input);
        hidden_states = self_atten({hidden_states, hidden_states, hidden_states})[0];
        auto residual = hidden_states + inputs[0];
        hidden_states = post_attention_layernorm(residual);
        hidden_states = block_sparse_moe({hidden_states, router_logits, expert_indices})[0];
        hidden_states = hidden_states + residual;
        return {hidden_states};
    }

    MultiHeadAttention &get_attention() {
        return self_atten;
    }

private:
    MultiHeadAttention self_atten;
    SmallThinkerMoeBlock block_sparse_moe;
    Layer input_layernorm;
    Layer post_attention_layernorm;
    Layer primary_router;
    int num_hidden_layers;
    int num_experts_per_tok{};
};

class SmallThinkerModel final : public Module {
public:
    SmallThinkerModel() = default;
    SmallThinkerModel(const SmallThinkerConfig &config, const SmallThinkerNameConfig &names, const string &base_name) {
        blocks = List<SmallThinkerDecoder>(config.num_hidden_layers, config, names, base_name);
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
    std::vector<SmallThinkerDecoder> blocks;
    Layer norm;
};

class SmallThinkerForCausalLM final : public Module {
public:
    CHAINABLE_MODULE_METHODS(SmallThinkerForCausalLM)
    SmallThinkerForCausalLM(SmallThinkerConfig &config) {
        auto names = config.names_config;
        hidden_size = config.hidden_size;
        embedding = Embedding(config.vocab_size, config.hidden_size, names.token_embd_name);
        model = SmallThinkerModel(config, names, names.blk_name);
        tie_embedding_words = config.tie_embedding_words;
        if (tie_embedding_words) {
            lm_head = Parameter(1, config.vocab_size, 1, config.hidden_size, names.token_embd_name + ".weight");
        } else {
            lm_head_layer = Linear(config.hidden_size, config.vocab_size, false, names.lm_head_name);
        }
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = embedding(inputs[0]);
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
    SmallThinkerModel model;
};

#endif // MODELING_SMOLTHINKER_HPP