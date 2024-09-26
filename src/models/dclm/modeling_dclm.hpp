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
#include "models/transformer/modeling_transformer.hpp"

using namespace mllm;

// the swiglu_torch in DCLM has no bias.
// set bias = false
class SwiGLUTorch final : public Module {
    Layer w12_;
    Layer w3_;
    Layer silu_;
    int hidden_dim_;

public:
    SwiGLUTorch() = default;
    SwiGLUTorch(int in_dim, int hidden_dim, int out_dim, const std::string &base_name) {
        hidden_dim_ = hidden_dim;
        w12_ = Linear(in_dim, 2 * hidden_dim, false, base_name + "w12");
        w3_ = Linear(hidden_dim, out_dim, false, base_name + "w3");
        silu_ = SiLU(base_name + "silu");
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = inputs[0];
        x = w12_(x);

        auto x_sp = Tensor::split(x, {hidden_dim_, hidden_dim_}, DIMENSION);
        Tensor gate;

        gate = x_sp[0];
        x = x_sp[1];

        x = silu_(gate) * x;

        return {w3_(x)};
    }
};

class DCLMAttention final : public Module {
    Layer in_proj_;
    Layer out_proj_;
    Layer pos_embed_;
    Layer q_norm_;
    Layer k_norm_;
    Layer q_rope_;
    Layer k_rope_;
    KVCache k_cache_;
    KVCache v_cache_;
    Layer softmax_;

    int attn_hidden_dim_;
    int head_dim_;
    int n_heads_;

public:
    DCLMAttention() = default;
    DCLMAttention(const DCLMConfig &cfg, int layer_id, const std::string &base_name) {
        int head_dim = cfg.dim / cfg.n_heads;
        attn_hidden_dim_ = cfg.n_heads * head_dim;
        head_dim_ = head_dim;
        n_heads_ = cfg.n_heads;
        in_proj_ = Linear(cfg.dim, 3 * cfg.n_heads * head_dim, false, base_name + "in_proj");
        out_proj_ = Linear(cfg.n_heads * head_dim, cfg.dim, false, base_name + "out_proj");
        q_norm_ = LayerNorm(cfg.n_heads * head_dim, false, cfg.norm_eps, base_name + "q_norm");
        k_norm_ = LayerNorm(cfg.n_heads * head_dim, false, cfg.norm_eps, base_name + "k_norm");
        q_rope_ = RoPE(cfg.RoPE_type, 10000, cfg.seq_len, base_name + "q_rope");
        k_rope_ = RoPE(cfg.RoPE_type, 10000, cfg.seq_len, base_name + "k_rope");
        k_cache_ = KVCache(1, cfg.cache_limit, base_name + "k_cache");
        v_cache_ = KVCache(1, cfg.cache_limit, base_name + "v_cache");
        softmax_ = Softmax(DIMENSION, true, base_name + "softmax");
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto qkv = in_proj_(inputs[0]);
        auto qkv_sp = Tensor::split(qkv, {attn_hidden_dim_, attn_hidden_dim_, attn_hidden_dim_}, DIMENSION, -1);

        // [B, H=1, S, D=(n_heads * head_dim)]
        Tensor q, k, v;
        q = qkv_sp[0];
        k = qkv_sp[1];
        v = qkv_sp[2];

        // layer norm: cfg.n_heads * head_dim
        // self.q_norm = (
        //     args.norm_type(
        //         args.n_heads * self.head_dim,
        //         eps=args.norm_eps,
        //     )
        //     if self.apply_qk_norm
        //     else nn.Identity()
        // )
        // self.k_norm = (
        //     args.norm_type(
        //         args.n_heads * self.head_dim,
        //         eps=args.norm_eps,
        //     )
        //     if self.apply_qk_norm
        //     else nn.Identity()
        // )
        q = q_norm_(q);
        k = k_norm_(k);

        // view to
        auto batchsize = q.batch();
        auto q_len = q.sequence();
        q = q.view(batchsize, n_heads_, q_len, head_dim_);
        k = k.view(batchsize, n_heads_, q_len, head_dim_);
        k = v.view(batchsize, n_heads_, q_len, head_dim_);

        q = q_rope_(q);
        k = k_rope_(k);

        k = k_cache_(k);
        v = v_cache_(v);

        k = k.transpose(SEQUENCE, DIMENSION);
        auto qk = Tensor::mm(q, k);
        qk = qk / std::sqrt(attn_hidden_dim_);

        qk = softmax_(qk);

        auto o = Tensor::mm(qk, v);
        o = o.view(-1, 1, -1, attn_hidden_dim_ * head_dim_);
        o = out_proj_(o);
        return {o};
    }
};

// swiglu_torch.
// there is no need to reset params.
class Block final : public Module {
    int n_heads_;
    int dim_;
    int head_dim_;
    int hidden_dim_;
    int layer_id_;
    SwiGLUTorch feed_forward_;

    // lp_layer_norm
    // we do not use low precision here， it's basiclly a normal layernorm without bias
    Layer attention_norm_;
    Layer ffn_norm_;

    DCLMAttention attention_;

public:
    Block() = default;
    Block(const DCLMConfig &cfg, int layer_id, const std::string &base_name) {
        layer_id_ = layer_id;
        n_heads_ = cfg.n_heads;
        dim_ = cfg.dim;
        head_dim_ = cfg.dim / cfg.n_heads;

        // swiglu_torch
        hidden_dim_ = 256 * ((int(2 * 4 * cfg.dim / 3) + 256 - 1) / 256);
        feed_forward_ = SwiGLUTorch(cfg.dim, hidden_dim_, cfg.dim, base_name + "feed_forward.");

        // lp_layer_norm
        // we do not use low precision here， it's basiclly a normal layernorm without bias
        attention_norm_ = LayerNorm(cfg.dim, false, cfg.norm_eps, base_name + "attention_norm");
        ffn_norm_ = LayerNorm(cfg.dim, false, cfg.norm_eps, base_name + "ffn_norm");

        attention_ = DCLMAttention(cfg, layer_id, base_name + "attention.");
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = inputs[0];
        auto h = attention_({attention_norm_(x)})[0];
        h = h + x;
        auto ffn_out = feed_forward_({ffn_norm_(h)})[0];
        auto out = h + ffn_out;
        return {out};
    }
};

class DCLMTransformer final : public Module {
    std::string base_name_ = "model.";

    Layer tok_embeddings_;
    std::vector<Block> layers_;
    Layer norm_;
    Parameter lm_head_;

public:
    DCLMTransformer() = default;
    DCLMTransformer(const DCLMConfig &cfg) {
        tok_embeddings_ = Embedding(cfg.vocab_size, cfg.dim, base_name_ + "tok_embeddings");

        listIdx = 0;
        for (int i = 0; i < cfg.n_layers; ++i) {
            layers_.push_back(Block(cfg, i, base_name_ + "layers." + std::to_string(i) + "."));
            listIdx++;
        }
        listIdx = 0;

        norm_ = LayerNorm(cfg.dim, false, cfg.norm_eps, base_name_ + "norm");
        lm_head_ = Parameter(1, cfg.vocab_size, 1, cfg.dim,
                             base_name_ + "output.weight");
    }

    std::vector<Tensor> Forward(std::vector<Tensor> inputs, std::vector<std::any> args) override {
        auto x = inputs[0];
        x = tok_embeddings_(x);

        for (auto &layer : layers_) {
            x = layer({x})[0];
        }

        x = norm_(x);
        auto out = Tensor::mm(x, lm_head_().transpose(Chl::SEQUENCE, Chl::DIMENSION));
        return {out};
    }
};
