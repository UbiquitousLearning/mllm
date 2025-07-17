//
// Created by Lu Yiwen on 2025/6/3.
//
#ifndef MODELING_PHI4_HPP
#define MODELING_PHI4_HPP

#include <any>
#include "Layer.hpp"
#include "Module.hpp"
#include "Tensor.hpp"
#include "configuration_phi4.hpp"
// #include "models/transformer/modeling_transformer.hpp"

using namespace mllm;

class Phi4Attention final : public Module {
public:
    Phi4Attention() = default;
    Phi4Attention(const Phi4Config &config, const Phi4NameConfig &names, const string &base_name) {
        hidden_size = config.hidden_dim;
        num_heads = config.num_attention_heads;
        head_dim = config.hidden_dim / num_heads;
        num_key_value_heads = config.num_key_value_heads;
        num_key_value_groups = num_heads / num_key_value_heads;

        int head_dim = config.hidden_dim / config.num_attention_heads; // 128
        int rotary_dim = head_dim * config.partial_rotary_factor;      // 96
        
        qkv_proj = Linear(
            hidden_size,
            num_heads * head_dim + num_key_value_heads * head_dim * 2,
            config.attention_bias,
            base_name + names._qkv_proj_name);
        o_proj = Linear(num_heads * head_dim, hidden_size, false, base_name + names._o_proj_name);
        
        q_rope = NTKRoPE(
            config.RoPE_type,                            
            config.rope_theta,                           
            config.max_position_embeddings,              
            config.rope_original_max_position_embeddings,
            config.rope_long_factor,                      
            config.rope_short_factor,                     
            base_name + "q_ntkrope",                     
            config.partial_rotary_factor);
        k_rope = NTKRoPE(
            config.RoPE_type,
            config.rope_theta,
            config.max_position_embeddings,
            config.rope_original_max_position_embeddings,
            config.rope_long_factor,
            config.rope_short_factor,
            base_name + "k_ntkrope",
            config.partial_rotary_factor);
        k_cache = KVCache(num_key_value_heads, head_dim, num_key_value_groups, config.cache_limit, base_name + "k_cache");
        v_cache = KVCache(num_key_value_heads, head_dim, num_key_value_groups, config.cache_limit, base_name + "v_cache");
        //  mask = SlidingWindowMask(config.sliding_window, base_name + "mask");
        //  mask = Causalmask(base_name + "mask");
        softmax = Softmax(DIMENSION, true, base_name + "softmax");
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        int head_dim = hidden_size / num_heads;
        int Q_dim = num_heads * head_dim;            // 3072 
        int KV_dim = num_key_value_heads * head_dim; // 1024 
        int total_proj_dim = Q_dim + 2 * KV_dim;     // 5120 

        auto qkv = qkv_proj(inputs[0]);
        auto qkv_sp = qkv.split({Q_dim, KV_dim, KV_dim}, Chl::DIMENSION);
        auto query_raw = qkv_sp[0];
        auto key_raw = qkv_sp[1];
        auto value_raw = qkv_sp[2];

        auto query_states = query_raw.view(-1, num_heads, -1, head_dim);
        auto key_states = key_raw.view(-1, num_key_value_heads, -1, head_dim);
        auto value_states = value_raw.view(-1, num_key_value_heads, -1, head_dim);

        //  embedding
        query_states = q_rope(query_states);
        key_states = k_rope(key_states);

        //  kv cache
        key_states = k_cache(key_states);
        value_states = v_cache(value_states);

        // attention weight
        auto atten_weight =
            Tensor::mm(query_states, key_states.transpose(Chl::SEQUENCE, Chl::DIMENSION))
            / std::sqrt(head_dim);
        // atten_weight = mask(atten_weight, k_cache.getCacheSeqLen());
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
    vector<NTKRoPE *> get_rope() {
        return {&q_rope, &k_rope};
    }

private:
    int hidden_size;
    int num_heads;
    int head_dim;
    int num_key_value_heads;
    int num_key_value_groups;
    Layer qkv_proj;
    Layer o_proj;
    NTKRoPE q_rope;
    NTKRoPE k_rope;
    // RoPE q_rope;
    // RoPE k_rope;
    KVCache k_cache;
    KVCache v_cache;
    // Causalmask mask;
    Softmax softmax;
};

class Phi4MLP final : public Module {
    Layer gate_up_proj;
    Layer silu;
    Layer down_proj;
    int ffn_hidden_;

public:
    Phi4MLP() = default;
    Phi4MLP(int hidden_dim, int ffn_hidden, const Phi4NameConfig &names, const string &base_name) {
        ffn_hidden_ = ffn_hidden;
        gate_up_proj = Linear(hidden_dim, 2 * ffn_hidden, false, base_name + names._gate_up_proj_name);
        silu = SiLU(base_name + "act");
        down_proj = Linear(ffn_hidden, hidden_dim, false, base_name + names._down_proj_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = gate_up_proj(inputs[0]);
        auto splited_y_12 = x.split({ffn_hidden_, ffn_hidden_}, DIMENSION);
        auto y_1 = splited_y_12[0];
        Tensor y_2 = splited_y_12[1];
        x = y_2 * silu(y_1);
        x = down_proj(x);
        return {x};
    }
};

class Phi4Block final : public Module {
    Phi4Attention attention;
    Phi4MLP mlp;
    Layer norm1;
    Layer norm2;

public:
    Phi4Block() = default;
    Phi4Block(const Phi4Config &config,
              const Phi4NameConfig &names,
              const string &base_name) {
        attention = Phi4Attention(config, names, base_name + names._attn_base_name);
        mlp = Phi4MLP(config.hidden_dim, config.ffn_hidden, names, base_name + names._ffn_base_name);
        norm1 = RMSNorm(config.hidden_dim, 1e-5, base_name + names._attn_norm_name);
        norm2 = RMSNorm(config.hidden_dim, 1e-5, base_name + names._ffn_norm_name);

    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = norm1(inputs[0]);

        x = attention({x, x, x})[0];
        auto tmp = x + inputs[0];
        x = norm2(tmp);
        x = mlp({x})[0];
        x = x + tmp;
        return {x};
    }

    Phi4Attention &get_attention() {
        return attention;
    }
};

class Phi4Model final : public Module {
    Layer embedding;
    vector<Phi4Block> blocks;
    Layer norm;
    Layer lm_head;
    Parameter lm_head_weight; // 形状 (1, vocab_size, 1, hidden_dim)，与 embedding.weight 绑定

public:
    explicit Phi4Model(const Phi4Config &config) {
        embedding = Embedding(config.vocab_size, config.hidden_dim, config.names_config.token_embd_name);
        norm = RMSNorm(config.hidden_dim, 1e-6, config.names_config.post_norm_name);
        // lm_head = Linear(config.hidden_dim, config.vocab_size, false, config.names_config.lm_head_name);
        lm_head_weight = Parameter{1, config.vocab_size, 1, config.hidden_dim, config.names_config.token_embd_name + ".weight"};
        const auto &names = config.names_config;
        const std::string base_name = names.blk_name;
        blocks = List<Phi4Block>(config.block_num, config, names, base_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = embedding(inputs[0]);
        for (auto &block : blocks) {
            x = block({x})[0];
        }
        x = norm(x);

        x = Tensor::mm(x, lm_head_weight().transpose(Chl::SEQUENCE, Chl::DIMENSION));
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
};

#endif // MODELING_PHI4_HPP