//
// Created by Rongjie Yi on 24-2-29.
//

#ifndef MODELING_TRANSFORMER_HPP
#define MODELING_TRANSFORMER_HPP
#ifdef USE_QNN

#include "Tensor.hpp"
#include "Types.hpp"
#include "configuration_opt_qnn.hpp"

using namespace mllm;

class OPTEncoderBlockPart1 final : public Module {
    Layer q_proj;
    Layer k_proj;
    Layer v_proj;
    RoPE q_rope;
    RoPE k_rope;
    Layer norm1;
    int head_size_{};
    int kv_head_size_{};
    int attn_hidden_dim_{};

public:
    OPTEncoderBlockPart1() = default;
    OPTEncoderBlockPart1(int hidden_dim, int head_size, int kv_head_size, int attn_hidden_dim,
                         RoPEType RoPE_type, bool bias,
                         const OPTNameConfig &names, const string &base_name) {
        attn_hidden_dim_ = attn_hidden_dim;
        head_size_ = head_size;
        kv_head_size_ = kv_head_size;
        norm1 = LayerNorm(hidden_dim, true, 1e-6, base_name + names._attn_norm_name, MLLM_QNN);

        q_proj = Linear(hidden_dim, head_size * attn_hidden_dim, bias, base_name + names._attn_base_name + names._q_proj_name);
        k_proj = Linear(hidden_dim, kv_head_size * attn_hidden_dim, bias, base_name + names._attn_base_name + names._k_proj_name);
        v_proj = Linear(hidden_dim, kv_head_size * attn_hidden_dim, bias, base_name + names._attn_base_name + names._v_proj_name);

        if (RoPE_type > 0) {
            q_rope = RoPE(RoPE_type, base_name + names._attn_base_name + "q_rope");
            k_rope = RoPE(RoPE_type, base_name + names._attn_base_name + "k_rope");
        }
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        Tensor x = inputs[0];
        auto i = norm1(x);
        auto q = q_proj(i);
        auto k = k_proj(i);
        auto v = v_proj(i);
        q = q.view(-1, head_size_, -1, attn_hidden_dim_);
        k = k.view(-1, kv_head_size_, -1, attn_hidden_dim_);
        v = v.view(-1, kv_head_size_, -1, attn_hidden_dim_);
        if (q_rope.ready() && k_rope.ready()) {
            q = q_rope(q);
            k = k_rope(k);
        }
        return {q, k, v, x};
    }
};

class OPTQKVmm final : public Module {
    Layer mask;
    Layer softmax;
    Layer k_cache;
    Layer v_cache;

    int head_size_{};
    int attn_hidden_dim_{};

public:
    OPTQKVmm() = default;
    OPTQKVmm(int head_size, int kv_head_size, int attn_hidden_dim, bool do_mask, int cache_limit,
             const OPTNameConfig &names, const string &base_name) {
        attn_hidden_dim_ = attn_hidden_dim;
        head_size_ = head_size;
        if (cache_limit > 0) {
            k_cache = KVCache(head_size / kv_head_size, cache_limit, false, base_name + names._attn_base_name + "k_cache");
            v_cache = KVCache(head_size / kv_head_size, cache_limit, false, base_name + names._attn_base_name + "v_cache");
        }

        if (do_mask) {
            mask = Causalmask(base_name + "mask");
        }
        softmax = Softmax(DIMENSION, base_name + "softmax");
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        Tensor q = inputs[0];
        Tensor k = inputs[1];
        Tensor v = inputs[2];

        if (k_cache.ready() && v_cache.ready()) {
            k = k_cache(k);
            v = v_cache(v);
        }

        k = k.transpose(SEQUENCE, DIMENSION);
        auto qk = Tensor::mm(q, k);
        qk = qk / std::sqrt(attn_hidden_dim_);
        if (mask.ready()) {
            qk = mask(qk);
        }
        qk = softmax(qk);
        auto o = Tensor::mm(qk, v);
        // o = o.view(-1, 1, -1, attn_hidden_dim_ * head_size_);
        return {o, inputs[3]};
    }
};

class OPTEncoderBlockPart2 final : public Module {
    Layer o_proj;
    Layer up_proj;
    Layer act;
    Layer down_proj;
    Layer norm2;
    int hidden_dim_{};

public:
    OPTEncoderBlockPart2() = default;
    OPTEncoderBlockPart2(int hidden_dim, int head_size, int ffn_hidden, const string &act_fn_type, bool bias, const OPTNameConfig &names, const string &base_name) {
        hidden_dim_ = hidden_dim;
        o_proj = Linear(hidden_dim, hidden_dim, bias, base_name + names._attn_base_name + names._o_proj_name);
        norm2 = LayerNorm(hidden_dim, true, 1e-6, base_name + names._ffn_norm_name, MLLM_QNN);
        up_proj = Linear(hidden_dim, ffn_hidden, bias, base_name + names._ffn_base_name + names._up_proj_name);
        act = ACT_FN[act_fn_type](base_name + names._ffn_base_name + "act");
        down_proj = Linear(ffn_hidden, hidden_dim, bias, base_name + names._ffn_base_name + names._down_proj_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto o = inputs[0].view(-1, 1, -1, hidden_dim_);
        o = o_proj(o);
        o = o + inputs[1];
        auto x = norm2(o);
        x = up_proj(x);
        x = act(x);
        x = down_proj(x);
        x = x + o;
        return {x};
    }
};

class QNNEncoderBlock final : public Module {
    OPTEncoderBlockPart1 part1;
    OPTQKVmm qkv_mm;
    OPTEncoderBlockPart2 part2;
    Layer norm1;
    Layer norm2;

public:
    QNNEncoderBlock() = default;
    QNNEncoderBlock(int hidden_dim, int head_size, int ffn_hidden, int cache_limit, const OPTNameConfig &names, const string &base_name) {
        part1 = OPTEncoderBlockPart1(hidden_dim, head_size, head_size, hidden_dim / head_size,
                                     HFHUBROPE, true, names, base_name);
        part1.to(MLLM_QNN);

        qkv_mm = OPTQKVmm(head_size, head_size, hidden_dim / head_size, true, cache_limit, names, base_name);

        part2 = OPTEncoderBlockPart2(hidden_dim, head_size, ffn_hidden, "ReLU", true,
                                     names, base_name);
        part2.to(MLLM_QNN);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = inputs[0];
        if (x.device() != MLLM_QNN) {
            x = Tensor::toQNN({x})[0];
        }
        auto q_k_v_x = part1({x});
        q_k_v_x = Tensor::toCPU(q_k_v_x);
        auto o_x = qkv_mm(q_k_v_x);
        o_x = Tensor::toQNN(o_x);
        auto out = part2(o_x)[0];
        // out = Tensor::toCPU({out})[0];
        return {out};
    }
};
#endif

#endif // MODELING_TRANSFORMER_HPP
