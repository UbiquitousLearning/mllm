/**
 * @file modeling_openelm.hpp
 * @author Chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-09-25
 *
 * @copyright Copyright (c) 2024
 *
 */
#pragma once

#include "Layer.hpp"
#include "Module.hpp"
#include "Types.hpp"
#include "configuration_openelm.hpp"
#include <algorithm>
#include <cassert>

using namespace mllm;

template <typename T>
T makeDivisible(T v, int divisor = 8, T min_value = T()) {
    if (min_value == T()) {
        min_value = static_cast<T>(divisor);
    }
    T new_v = std::max(min_value, static_cast<T>((static_cast<int>(v + divisor / 2) / divisor) * divisor));
    if (new_v < 0.9 * v) {
        new_v += divisor;
    }
    return new_v;
}

class OpenELMMultiHeadCausalAttention final : public Module {
    int layer_idx_;
    int head_dim_;
    int q_heads_;
    int k_heads_;
    int v_heads_;

    Layer qkv_proj;
    RoPE q_rope;
    RoPE k_rope;
    Layer q_norm;
    Layer k_norm;
    Layer out_proj;

    KVCache k_cache;
    KVCache v_cache;

    Softmax softmax;

    int iter = 0;

public:
    OpenELMMultiHeadCausalAttention() = default;
    OpenELMMultiHeadCausalAttention(const OpenELMConfig &cfg, int layer_idx, const std::string &base_name) {
        layer_idx_ = layer_idx;
        head_dim_ = cfg.head_dim;
        q_heads_ = cfg.num_query_heads[layer_idx];
        k_heads_ = cfg.num_kv_heads[layer_idx];
        v_heads_ = cfg.num_kv_heads[layer_idx];

        qkv_proj = Linear(cfg.model_dim, (q_heads_ + k_heads_ + v_heads_) * head_dim_, false, base_name + "qkv_proj");
        q_rope = RoPE(cfg.RoPE_type, cfg.rope_freq_constant, cfg.rope_max_length, base_name + "q_rope");
        k_rope = RoPE(cfg.RoPE_type, cfg.rope_freq_constant, cfg.rope_max_length, base_name + "k_rope");

        q_norm = RMSNorm(cfg.head_dim, 1e-6, base_name + "q_norm");
        k_norm = RMSNorm(cfg.head_dim, 1e-6, base_name + "k_norm");

        out_proj = Linear(q_heads_ * head_dim_, cfg.model_dim, false, base_name + "out_proj");

        k_cache = KVCache(q_heads_ / k_heads_, cfg.cache_limit, base_name + "k_cache");
        v_cache = KVCache(q_heads_ / v_heads_, cfg.cache_limit, base_name + "v_cache");

        softmax = Softmax(DIMENSION, true, base_name + "softmax");
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto qkv = qkv_proj(inputs[0]);

        auto qkv_sp = qkv.split({q_heads_ * head_dim_, k_heads_ * head_dim_, v_heads_ * head_dim_}, DIMENSION);
        Tensor q, k, v;
        q = qkv_sp[0];
        k = qkv_sp[1];
        v = qkv_sp[2];

        q = q.view(-1, q_heads_, -1, head_dim_);
        k = k.view(-1, k_heads_, -1, head_dim_);
        v = v.view(-1, v_heads_, -1, head_dim_);

        q = q_norm(q);
        k = k_norm(k);

        q = q_rope(q);
        k = k_rope(k);

        k = k_cache(k);
        v = v_cache(v);

        k = k.transpose(SEQUENCE, DIMENSION);
        auto qk = Tensor::mm(q, k);

        qk = qk / std::sqrt(head_dim_);

        qk = softmax(qk, k_cache.getCacheSeqLen());
        auto o = Tensor::mm(qk, v);
        o = o.view(-1, 1, -1, q_heads_ * head_dim_);
        o = out_proj(o);

        return {o};
    }
    vector<KVCache *> get_cache() {
        return {&k_cache, &v_cache};
    }
    vector<RoPE *> get_rope() {
        return {&q_rope, &k_rope};
    }
};

class OpenELMFeedForwardNetwork final : public Module {
    float ffn_multiplier_;
    int intermediate_dim_;

    Layer proj_1;
    Layer proj_2;
    Layer act; // swish

public:
    OpenELMFeedForwardNetwork() = default;
    OpenELMFeedForwardNetwork(const OpenELMConfig &cfg, int layer_idx, const std::string &base_name) {
        ffn_multiplier_ = cfg.ffn_multipliers[layer_idx];
        intermediate_dim_ = int(makeDivisible(ffn_multiplier_ * cfg.model_dim, cfg.ffn_dim_divisor));

        // ffn_with_glu
        proj_1 = Linear(cfg.model_dim, 2 * intermediate_dim_, false, base_name + "proj_1");
        proj_2 = Linear(intermediate_dim_, cfg.model_dim, false, base_name + "proj_2");

        // swish act
        act = SiLU(base_name + "silu");
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = inputs[0];
        auto y_12 = proj_1(x);

        // FIXME: may be buggy in split
        auto splited_y_12 = y_12.split({intermediate_dim_, intermediate_dim_}, DIMENSION);
        auto y_1 = splited_y_12[0];
        auto y_2 = splited_y_12[1];
        auto y = act(y_1) * y_2;
        return {proj_2(y)};
    }
};

class OpenELMDecoderLayer final : public Module {
    OpenELMMultiHeadCausalAttention attn;
    OpenELMFeedForwardNetwork ffn;
    Layer ffn_norm;
    Layer attn_norm;

public:
    OpenELMDecoderLayer() = default;
    OpenELMDecoderLayer(const OpenELMConfig &cfg, int layer_idx, const std::string &base_name) {
        attn = OpenELMMultiHeadCausalAttention(cfg, layer_idx, base_name + "attn.");
        ffn = OpenELMFeedForwardNetwork(cfg, layer_idx, base_name + "ffn.");
        ffn_norm = RMSNorm(cfg.model_dim, 1e-6, base_name + "ffn_norm");
        attn_norm = RMSNorm(cfg.model_dim, 1e-6, base_name + "attn_norm");
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = attn_norm(inputs[0]);
        x = attn({x, x, x})[0];
        auto tmp = x + inputs[0];
        x = ffn_norm(tmp);
        x = ffn({x})[0];
        x = x + tmp;
        return {x};
    }
    OpenELMMultiHeadCausalAttention &get_attention() {
        return attn;
    }
};

class OpenElMModel final : public Module {
    Layer token_embeddings;
    Layer norm;
    std::vector<OpenELMDecoderLayer> decode_layers;
    Parameter lm_head;

public:
    OpenElMModel() = default;
    OpenElMModel(const OpenELMConfig &cfg) {
        Layer::use_layername_2_tensorname = false; // for OpenELM only.
        token_embeddings = Embedding(cfg.vocab_size, cfg.model_dim, "transformer.token_embeddings");
        norm = RMSNorm(cfg.model_dim, 1e-6, "transformer.norm");

        // decode layers
        listIdx = 0;
        for (int i = 0; i < cfg.num_transformer_layers; ++i) {
            decode_layers.push_back(OpenELMDecoderLayer(cfg, i, cfg.names_config.blk_name + std::to_string(i) + "."));
            listIdx++;
        }
        listIdx = 0;

        assert(decode_layers.size() == 28);

        // tied embeddings
        lm_head = Parameter(1, cfg.vocab_size, 1, cfg.model_dim,
                            "transformer.token_embeddings.weight");
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto inputs_embeds = token_embeddings(inputs[0]);
        auto hidden_states = inputs_embeds;

        // decoders
        for (auto &it : decode_layers) {
            hidden_states = it({hidden_states})[0];
        }

        hidden_states = norm(hidden_states);

        // tied embeddings
        auto logits = Tensor::mm(hidden_states, lm_head().transpose(Chl::SEQUENCE, Chl::DIMENSION));

        return {logits};
    }
    void clear_kvcache() override {
        for (auto &block : decode_layers) {
            auto kvcache = block.get_attention().get_cache();
            for (auto &cache : kvcache) { cache->clearCache(); }
            auto ropes = block.get_attention().get_rope();
            for (auto &rope : ropes) { rope->clearCache(); }
        }
    }
};