//
// Created by Rongjie Yi on 24-2-29.
//

#ifndef MODELING_TRANSFORMER_HPP
#define MODELING_TRANSFORMER_HPP

#include "Layer.hpp"
#include "configuration_transformer.hpp"
#include <vector>

using namespace mllm;


enum AttnQKVSplitType {
    SPLIT_NONE = 0,
    SPLIT_HD = Chl::HD,
    SPLIT_D_HD = Chl::D_HD,
};


class MultiHeadAttention final : public Module {
    Layer qkv_proj;
    Split qkv_split;
    Layer q_proj;
    Layer k_proj;
    Layer v_proj;
    Layer q_rope;
    Layer k_rope;
    Layer q_norm;
    Layer k_norm;
    KVCache k_cache;
    KVCache v_cache;
    Softmax softmax;
    Layer o_proj;
    Parameter bias_k;
    Parameter bias_v;
    int head_size_{};
    int kv_head_size_{};
    int attn_hidden_dim_{};

public:
    MultiHeadAttention() = default;
    MultiHeadAttention(int hidden_dim, int head_size,int kv_head_size, int attn_hidden_dim,
                       AttnQKVSplitType do_qkv_proj, bool post_qkv_norm, bool bias_kv_cat,
                       RoPEType RoPE_type, float rope_theta, int max_position_embeddings, 
                       int cache_limit, bool do_mask, bool bias,
                       const TransformerNameConfig &names, const string &base_name) {
        attn_hidden_dim_ = attn_hidden_dim;
        head_size_ = head_size;
        kv_head_size_ = kv_head_size;
        if (do_qkv_proj > 0) {
            qkv_proj = Linear(hidden_dim, head_size * attn_hidden_dim * 3, bias, base_name + names._qkv_proj_name);
            qkv_split = Split(3, (Chl)do_qkv_proj, head_size, base_name + names._qkv_proj_name + ".split");
        } else {
            q_proj = Linear(hidden_dim, head_size * attn_hidden_dim, bias, base_name + names._q_proj_name);
            k_proj = Linear(hidden_dim, kv_head_size * attn_hidden_dim, bias, base_name + names._k_proj_name);
            v_proj = Linear(hidden_dim, kv_head_size * attn_hidden_dim, bias, base_name + names._v_proj_name);
        }
        if (post_qkv_norm) {
            q_norm = LayerNorm(attn_hidden_dim, true, 1e-6, base_name + names._q_norm_name);
            k_norm = LayerNorm(attn_hidden_dim, true, 1e-6, base_name + names._k_norm_name);
        }
        if (RoPE_type > 0) {
            q_rope = RoPE(RoPE_type, rope_theta, max_position_embeddings, base_name + "q_rope");
            k_rope = RoPE(RoPE_type, rope_theta, max_position_embeddings, base_name + "k_rope");
        }
        if (cache_limit > 0) {
            k_cache = KVCache(head_size/kv_head_size, cache_limit, base_name + "k_cache");
            v_cache = KVCache(head_size/kv_head_size, cache_limit, base_name + "v_cache");
        }
        softmax = Softmax(DIMENSION, do_mask, base_name + "softmax");
        o_proj = Linear(head_size * attn_hidden_dim, hidden_dim, bias, base_name + names._o_proj_name);
        if (bias_kv_cat) {
            bias_k = Parameter(1, 1, head_size, attn_hidden_dim, base_name + "bias_k");
            bias_v = Parameter(1, 1, head_size, attn_hidden_dim, base_name + "bias_v");
        }
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override  {
        Tensor q, k, v;
        if (qkv_proj.ready()) {
            auto qkv = qkv_proj(inputs[0]);
            auto qkv_sp = qkv_split(qkv);
            q = qkv_sp[0];
            k = qkv_sp[1];
            v = qkv_sp[2];
        } else {
            q = q_proj(inputs[0]);
            k = k_proj(inputs[1]);
            v = v_proj(inputs[2]);
            q = q.view(-1, head_size_, -1, attn_hidden_dim_);
            k = k.view(-1, kv_head_size_, -1, attn_hidden_dim_);
            v = v.view(-1, kv_head_size_, -1, attn_hidden_dim_);
        }
        if (q_norm.ready() && k_norm.ready()) {
            q = q_norm(q);
            k = k_norm(k);
        }
        if (bias_k.ready() && bias_v.ready()) {
            k = Tensor::cat({k, bias_k()}, SEQUENCE);
            v = Tensor::cat({v, bias_v()}, SEQUENCE);
        }
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
        qk = qk / std::sqrt(attn_hidden_dim_);
        if (k_cache.ready() && v_cache.ready()) {
            qk = softmax(qk, k_cache.getCacheSeqLen());          
        }else{
            qk = softmax(qk);
        }
        auto o = Tensor::mm(qk, v);
        o = o.view(-1, 1, -1, attn_hidden_dim_ * head_size_);
        o = o_proj(o);
        return {o};
    }
    vector<KVCache*> get_cache() {
        return {&k_cache,&v_cache};
    }
};

class FeedForward final : public Module {
    Layer up_proj;
    Layer act;
    Layer down_proj;

public:
    FeedForward() = default;
    FeedForward(int hidden_dim, int ffn_hidden, const string &act_fn_type, bool bias, const TransformerNameConfig &names, const string &base_name) {
        up_proj = Linear(hidden_dim, ffn_hidden, bias, base_name + names._up_proj_name);
        act = ACT_FN[act_fn_type](base_name + "act");
        down_proj = Linear(ffn_hidden, hidden_dim, bias, base_name + names._down_proj_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = up_proj(inputs[0]);
        x = act(x);
        x = down_proj(x);
        return {x};
    }
};

#endif // MODELING_TRANSFORMER_HPP
