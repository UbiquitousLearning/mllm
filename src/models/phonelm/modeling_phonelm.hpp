
#ifndef MODELING_PHONELM_HPP
#define MODELING_PHONELM_HPP

#include "Backend.hpp"
#include "Layer.hpp"
#include "Module.hpp"
#include "Tensor.hpp"
#include "configuration_phonelm.hpp"
#include <cmath>
using namespace mllm;

class PhoneLMMLP final : public Module {
public:
    PhoneLMMLP() = default;
    PhoneLMMLP(int hidden_size, int intermediate_size, const string &act_fn_type, const PhoneLMNameConfig &names,
               const std::string &base_name) {
        gate_proj =
            Linear(hidden_size, intermediate_size, false, base_name + names._gate_proj_name);
        // act = ReLU(base_name + "act");
        act = ACT_FN[act_fn_type](base_name + "act");
        up_proj = Linear(hidden_size, intermediate_size, false, base_name + names._up_proj_name);
        down_proj =
            Linear(intermediate_size, hidden_size, false, base_name + names._down_proj_name);
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = gate_proj(inputs[0]);
        x = act(x);
        auto y = up_proj(inputs[0]);
        x = x * y;
        x = down_proj(x);
        return {x};
    }

private:
    Layer gate_proj;
    Layer up_proj;
    Layer down_proj;

    Layer act;
};

class PhoneLMAttention final : public Module {
public:
    PhoneLMAttention() = default;
    PhoneLMAttention(const PhoneLMConfig &config, const PhoneLMNameConfig &names, const string &base_name) {
        hidden_size = config.hidden_size;
        num_heads = config.num_attention_heads;
        head_dim = config.hidden_size / num_heads;
        num_key_value_heads = config.num_key_value_heads;
        num_key_value_groups = num_heads / num_key_value_heads;

        // init layers
        q_proj = Linear(hidden_size, num_heads * head_dim, false, base_name + names._q_proj_name);
        k_proj = Linear(hidden_size, num_key_value_heads * head_dim, false,
                        base_name + names._k_proj_name);
        v_proj = Linear(hidden_size, num_key_value_heads * head_dim, false,
                        base_name + names._v_proj_name);
        o_proj = Linear(num_heads * head_dim, hidden_size, false, base_name + names._o_proj_name);
        q_rope = IRoPE(config.RoPE_type, config.rope_theta, config.max_position_embeddings,
                       base_name + "q_rope");
        k_rope = IRoPE(config.RoPE_type, config.rope_theta, config.max_position_embeddings,
                       base_name + "k_rope");
        k_cache = KVCache(num_key_value_groups, config.cache_limit, base_name + "k_cache");
        v_cache = KVCache(num_key_value_groups, config.cache_limit, base_name + "v_cache");
        softmax = Softmax(DIMENSION, true, base_name + "softmax");
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        Tensor q, k, v;
        q = q_proj(inputs[0]);
        k = k_proj(inputs[1]);
        v = v_proj(inputs[2]);
        q = q.view(-1, num_heads, -1, head_dim);
        k = k.view(-1, num_key_value_heads, -1, head_dim);
        v = v.view(-1, num_key_value_heads, -1, head_dim);

        if (q_rope.ready() && k_rope.ready()) {
            q = q_rope(q);
            k = k_rope(k);
        }
        if (k_cache.ready() && v_cache.ready()) {
            k = k_cache(k);
            v = v_cache(v);
        }
        k = k.transpose(SEQUENCE, DIMENSION);
        auto qk = Tensor::mm(q, k);
        qk = qk / std::sqrt(head_dim);
        if (k_cache.ready() && v_cache.ready()) {
            qk = softmax(qk, k_cache.getCacheSeqLen());
        } else {
            qk = softmax(qk);
        }
        auto o = Tensor::mm(qk, v);
        o = o.view(-1, 1, -1, head_dim * num_heads);
        o = o_proj(o);
        return {o};
    }

    vector<KVCache *> get_cache() {
        return {&k_cache, &v_cache};
    }
    vector<IRoPE *> get_rope() {
        return {&q_rope, &k_rope};
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
    IRoPE q_rope;
    IRoPE k_rope;
    KVCache k_cache;
    KVCache v_cache;
    Softmax softmax;
};

class PhoneLMDecoder final : public Module {
public:
    PhoneLMDecoder() = default;
    PhoneLMDecoder(const PhoneLMConfig &config, const PhoneLMNameConfig &names, const string &base_name) {
        self_atten = PhoneLMAttention(config, names, base_name + names._attn_base_name);
        mlp = PhoneLMMLP(config.hidden_size, config.intermediate_size, config.hidden_act, names, base_name + names._ffn_base_name);
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

    PhoneLMAttention &get_attention() {
        return self_atten;
    }

private:
    PhoneLMAttention self_atten;
    PhoneLMMLP mlp;
    Layer input_layernorm;
    Layer post_attention_layernorm;
};

class PhoneLMModel final : public Module {
public:
    PhoneLMModel() = default;
    PhoneLMModel(const PhoneLMConfig &config, const PhoneLMNameConfig &names, const string &base_name) {
        blocks = List<PhoneLMDecoder>(config.num_hidden_layers, config, names, base_name);
        norm = RMSNorm(config.hidden_size, config.rms_norm_eps, names.post_norm_name);
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = inputs[0];
        for (auto &block : blocks) { x = block({x})[0]; }
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
    std::vector<PhoneLMDecoder> blocks;
    Layer norm;
};

class PhoneLMForCausalLM final : public Module {
public:
    PhoneLMForCausalLM(PhoneLMConfig &config) {
        auto names = config.names_config;
        hidden_size = config.hidden_size;
        tie_embedding_words = config.tie_embedding_words;
        embedding = Embedding(config.vocab_size, config.hidden_size, names.token_embd_name);
        model = PhoneLMModel(config, names, names.blk_name);
        if (tie_embedding_words) {
            lm_head = Parameter(1, config.vocab_size, 1, config.hidden_size,
                                names.token_embd_name + ".weight");
        } else {
            lm_head_layer =
                Linear(config.hidden_size, config.vocab_size, false, names.lm_head_name);
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
    void clear_kvcache() override {
        model.clear_kvcache();
    }

private:
    int hidden_size;
    bool tie_embedding_words;
    Layer embedding;
    Parameter lm_head;
    Layer lm_head_layer;
    PhoneLMModel model;
};

#endif //! MODELING_PHONELM_HPP