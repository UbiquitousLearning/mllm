/**
 * @file modeling_dclm.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2024-09-26
 * @ref https://github.com/mlfoundations/open_lm/blob/main/open_lm/model.py
 *
 * @copyright Copyright (c) 2024
 *
 */
#pragma once

#include "Layer.hpp"
#include "Module.hpp"
#include "Types.hpp"
#include "configuration_dclm.hpp"

using namespace mllm;

class DCLMFFN final : public Module {
    Layer w12;
    Layer w3;
    Layer silu;
    int hidden_dim_;

public:
    DCLMFFN() = default;
    DCLMFFN(int in_dim, int hidden_dim, int out_dim, const std::string &base_name) {
        hidden_dim_ = hidden_dim;
        w12 = Linear(in_dim, 2 * hidden_dim, false, base_name + "w12");
        w3 = Linear(hidden_dim, out_dim, false, base_name + "w3");
        silu = SiLU(base_name + "silu");
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = inputs[0];
        x = w12(x);
        auto x_sp = x.split({hidden_dim_, hidden_dim_}, DIMENSION);
        Tensor gate;
        gate = x_sp[0];
        x = x_sp[1];
        x = silu(gate) * x;
        return {w3(x)};
    }
};

class DCLMAttention final : public Module {
    Layer in_proj;
    Layer out_proj;
    Layer q_norm;
    Layer k_norm;
    RoPE q_rope;
    RoPE k_rope;
    KVCache k_cache;
    KVCache v_cache;
    Softmax softmax;

    int attn_hidden_dim_;
    int head_dim_;
    int n_heads_;

public:
    DCLMAttention() = default;
    DCLMAttention(const DCLMConfig &cfg, const std::string &base_name) {
        int head_dim = cfg.dim / cfg.n_heads;
        attn_hidden_dim_ = cfg.n_heads * head_dim;
        head_dim_ = head_dim;
        n_heads_ = cfg.n_heads;
        in_proj = Linear(cfg.dim, 3 * cfg.n_heads * head_dim, false, base_name + "in_proj");
        out_proj = Linear(cfg.n_heads * head_dim, cfg.dim, false, base_name + "out_proj");
        q_norm = LayerNorm(cfg.n_heads * head_dim, false, cfg.norm_eps, base_name + "q_norm");
        k_norm = LayerNorm(cfg.n_heads * head_dim, false, cfg.norm_eps, base_name + "k_norm");
        q_rope = RoPE(cfg.RoPE_type, 10000, cfg.seq_len, base_name + "q_rope");
        k_rope = RoPE(cfg.RoPE_type, 10000, cfg.seq_len, base_name + "k_rope");
        k_cache = KVCache(cfg.n_heads, head_dim, 1, cfg.cache_limit, base_name + "k_cache");
        v_cache = KVCache(cfg.n_heads, head_dim, 1, cfg.cache_limit, base_name + "v_cache");
        softmax = Softmax(DIMENSION, true, base_name + "softmax");
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto qkv = in_proj(inputs[0]);
        auto qkv_sp = qkv.split({attn_hidden_dim_, attn_hidden_dim_, attn_hidden_dim_}, DIMENSION);

        Tensor q, k, v;
        q = qkv_sp[0];
        k = qkv_sp[1];
        v = qkv_sp[2];

        q = q_norm(q);
        k = k_norm(k);
        q = q.view(-1, n_heads_, -1, head_dim_);
        k = k.view(-1, n_heads_, -1, head_dim_);
        v = v.view(-1, n_heads_, -1, head_dim_);

        q = q_rope(q);
        k = k_rope(k);

        k = k_cache(k);
        v = v_cache(v);

        k = k.transpose(SEQUENCE, DIMENSION);
        auto qk = Tensor::mm(q, k);
        qk = qk / std::sqrt(head_dim_);

        qk = softmax(qk, k_cache.getCacheSeqLen());

        auto o = Tensor::mm(qk, v);
        o = o.view(-1, 1, -1, n_heads_ * head_dim_);
        o = out_proj(o);
        return {o};
    }
};

class DCLMDecoder final : public Module {
    int n_heads_;
    int dim_;
    int head_dim_;
    int hidden_dim_;
    DCLMFFN feed_forward;
    Layer attention_norm;
    Layer ffn_norm;

    DCLMAttention attention;

public:
    DCLMDecoder() = default;
    DCLMDecoder(const DCLMConfig &cfg, const std::string &base_name) {
        n_heads_ = cfg.n_heads;
        dim_ = cfg.dim;
        head_dim_ = cfg.dim / cfg.n_heads;

        attention = DCLMAttention(cfg, base_name + "attention.");
        // swiglu_torch
        hidden_dim_ = 256 * ((int(2 * 4 * cfg.dim / 3) + 256 - 1) / 256);
        feed_forward = DCLMFFN(cfg.dim, hidden_dim_, cfg.dim, base_name + "feed_forward.");
        // lp_layer_norm
        // we do not use low precision here， it's basiclly a normal layernorm without bias
        attention_norm = LayerNorm(cfg.dim, false, cfg.norm_eps, base_name + "attention_norm");
        ffn_norm = LayerNorm(cfg.dim, false, cfg.norm_eps, base_name + "ffn_norm");
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = inputs[0];
        auto h = attention({attention_norm(x)})[0];
        h = h + x;
        auto ffn_out = feed_forward({ffn_norm(h)})[0];
        auto out = h + ffn_out;
        return {out};
    }
};

class DCLM final : public Module {
    std::string base_name_ = "model.";

    Layer tok_embeddings;
    std::vector<DCLMDecoder> layers;
    Layer norm;
    Parameter lm_head;

public:
    DCLM() = default;
    DCLM(const DCLMConfig &cfg) {
        tok_embeddings = Embedding(cfg.vocab_size, cfg.dim, base_name_ + "tok_embeddings");
        layers = List<DCLMDecoder>(cfg.n_layers, cfg, base_name_ + "layers.");
        norm = LayerNorm(cfg.dim, false, cfg.norm_eps, base_name_ + "norm");
        lm_head = Parameter(1, cfg.vocab_size, 1, cfg.dim,
                            base_name_ + "output.weight");
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = inputs[0];
        x = tok_embeddings(x);

        for (auto &layer : layers) {
            x = layer({x})[0];
        }

        x = norm(x);
        auto out = Tensor::mm(x, lm_head().transpose(Chl::SEQUENCE, Chl::DIMENSION));
        return {out};
    }
};
