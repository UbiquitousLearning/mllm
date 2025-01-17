#ifndef MODELING_GEMMA2_HPP
#define MODELING_GEMMA2_HPP

#include "Backend.hpp"
#include "Layer.hpp"
#include "Module.hpp"
#include "Tensor.hpp"
#include "configuration_gemma2.hpp"
#include <cmath>
using namespace mllm;

class Gemma2Attention final : public Module {
public:
    Gemma2Attention() {}
    Gemma2Attention(const Gemma2Config &config, const Gemma2NameConfig &names, const string &base_name) {
        hidden_size = config.hidden_size;
        num_heads = config.num_attention_heads;
        // in gemma2, the head_dim is fixed to 2048 / num_heads rather than hidden_size(2304) / num_heads
        head_dim = 2048 / num_heads;
        num_key_value_heads = config.num_key_value_heads;
        num_key_value_groups = num_heads / num_key_value_heads;

        // init layers
        q_proj = Linear(hidden_size, head_dim * num_heads, false, base_name + names._q_proj_name);
        k_proj = Linear(hidden_size, head_dim * num_key_value_heads, false,
                        base_name + names._k_proj_name);
        v_proj = Linear(hidden_size, head_dim * num_key_value_heads, false,
                        base_name + names._v_proj_name);
        o_proj = Linear(head_dim * num_heads, hidden_size, false, base_name + names._o_proj_name);
        q_rope = RoPE(config.RoPE_type, config.rope_theta, config.max_position_embeddings,
                      base_name + "q_rope");
        k_rope = RoPE(config.RoPE_type, config.rope_theta, config.max_position_embeddings,
                      base_name + "k_rope");
        k_cache = KVCache(num_key_value_groups, config.cache_limit, base_name + "k_cache");
        v_cache = KVCache(num_key_value_groups, config.cache_limit, base_name + "v_cache");

        softmax = Softmax(DIMENSION, true, base_name + "softmax");
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto query_states = q_proj(inputs[0]);
        auto key_states = k_proj(inputs[1]);
        auto value_states = v_proj(inputs[2]);

        // [batch, heads, sequence, dims]
        query_states = query_states.view(-1, num_heads, -1, head_dim);
        key_states = key_states.view(-1, num_key_value_heads, -1, head_dim);
        value_states = value_states.view(-1, num_key_value_heads, -1, head_dim);

        // embedding
        query_states = q_rope(query_states);
        key_states = k_rope(key_states);

        // kv cache
        key_states = k_cache(key_states);
        value_states = v_cache(value_states);

        // attention weight
        auto atten_weight =
            Tensor::mm(query_states, key_states.transpose(Chl::SEQUENCE, Chl::DIMENSION))
            / std::sqrt(head_dim);

        atten_weight = softmax(atten_weight, k_cache.getCacheSeqLen());

        // attention output
        auto atten_output = Tensor::mm(atten_weight, value_states);
        atten_output = atten_output.view(-1, 1, -1, head_dim * num_heads);
        atten_output = o_proj(atten_output);
        return {atten_output};
    }

    vector<KVCache *> get_cache() {
        return {&k_cache, &v_cache};
    }
    vector<RoPE *> get_rope() {
        return {&q_rope, &k_rope};
    }

private:
    int hidden_size;
    int num_heads;
    int head_dim;
    int num_key_value_heads;
    int num_key_value_groups;
    int layer_num = 0;
    Layer q_proj;
    Layer k_proj;
    Layer v_proj;
    Layer o_proj;
    RoPE q_rope;
    RoPE k_rope;
    KVCache k_cache;
    KVCache v_cache;
    Softmax softmax;
};

class Gemma2MLP final : public Module {
public:
    Gemma2MLP() = default;
    Gemma2MLP(int hidden_size, int intermediate_size, const Gemma2NameConfig &names, const std::string &base_name) {
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

class Gemma2Decoder final : public Module {
public:
    Gemma2Decoder() = default;
    Gemma2Decoder(const Gemma2Config &config, const Gemma2NameConfig &names, const string &base_name) {
        self_attn = Gemma2Attention(config, names, base_name + names._attn_base_name);
        mlp = Gemma2MLP(config.hidden_size, config.intermediate_size, names, base_name + names._ffn_base_name);
        input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, true, base_name + names._attn_norm_name);
        post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, true, base_name + names._ffn_norm_name);
        pre_feedforward_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, true, base_name + names._pre_feedforward_layernorm);
        post_feedforward_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, true, base_name + names._post_feedforward_layernorm);
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = input_layernorm(inputs[0]);
        x = self_attn({x, x, x})[0];
        x = post_attention_layernorm(x);
        auto tmp = x + inputs[0];
        x = pre_feedforward_layernorm(tmp);
        x = mlp({x})[0];
        x = post_feedforward_layernorm(x);
        x = x + tmp;
        return {x};
    }

    Gemma2Attention &get_attention() {
        return self_attn;
    }

private:
    // MultiHeadAttention self_attn;
    Gemma2Attention self_attn;
    Gemma2MLP mlp;
    Layer input_layernorm;
    Layer post_attention_layernorm;
    Layer pre_feedforward_layernorm;
    Layer post_feedforward_layernorm;
};

class Gemma2Model final : public Module {
public:
    Gemma2Model() = default;
    Gemma2Model(const Gemma2Config &config, const Gemma2NameConfig &names, const string &base_name) {
        blocks = List<Gemma2Decoder>(config.num_hidden_layers, config, names, base_name);
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
    std::vector<Gemma2Decoder> blocks;
    Layer norm;
};

class Gemma2ForCausalLM final : public Module {
public:
    Gemma2ForCausalLM(Gemma2Config &config) {
        auto names = config.names_config;
        hidden_size = config.hidden_size;
        embedding = Embedding(config.vocab_size, config.hidden_size, names.token_embd_name);
        model = Gemma2Model(config, names, names.blk_name);

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
    Gemma2Model model;
};

#endif //! MODELING_GEMMA2_HPP
