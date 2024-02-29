//
// Created by Rongjie Yi on 24-2-29.
//

#ifndef MODELING_TRANSFORMER_HPP
#define MODELING_TRANSFORMER_HPP

#include "configuration_transformer.hpp"

using namespace mllm;

class MultiHeadAttention final : public Module {
    Layer q_proj;
    Layer k_proj;
    Layer v_proj;
    Layer o_proj;
    Layer q_rope;
    Layer k_rope;
    Layer k_cache;
    Layer v_cache;
    Layer mask;
    Layer softmax;
    int head_size_{};
    int attn_hidden_dim_{};

public:
    MultiHeadAttention() = default;
    MultiHeadAttention(int hidden_dim, int head_size, int attn_hidden_dim,
                       RoPEType RoPE_type, int cache_limit, bool do_mask, bool bias,
                       const TransformerNameConfig &names, const string &base_name) {
        head_size_ = head_size;
        attn_hidden_dim_ = attn_hidden_dim;
        q_proj = Linear(hidden_dim, head_size * attn_hidden_dim, bias, base_name + names._q_proj_name);
        k_proj = Linear(hidden_dim, head_size * attn_hidden_dim, bias, base_name + names._k_proj_name);
        v_proj = Linear(hidden_dim, head_size * attn_hidden_dim, bias, base_name + names._v_proj_name);
        o_proj = Linear(head_size * attn_hidden_dim, hidden_dim, bias, base_name + names._o_proj_name);
        if (RoPE_type > 0) {
            q_rope = RoPE(RoPE_type, base_name + "q_rope");
            k_rope = RoPE(RoPE_type, base_name + "k_rope");
        }
        if (cache_limit > 0) {
            k_cache = KVCache(cache_limit, base_name + "k_cache");
            v_cache = KVCache(cache_limit, base_name + "v_cache");
        }
        if (do_mask) {
            mask = Causalmask(base_name + "mask");
        }
        softmax = Softmax(DIMENSION, base_name + "softmax");
    }
    vector<Tensor> Forward(vector<Tensor> inputs) override {
        auto q = q_proj(inputs[0]);
        auto k = k_proj(inputs[1]);
        auto v = v_proj(inputs[2]);
        q = q.view(-1, head_size_, -1, attn_hidden_dim_);
        k = k.view(-1, head_size_, -1, attn_hidden_dim_);
        v = v.view(-1, head_size_, -1, attn_hidden_dim_);
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
        if (mask.ready()) {
            qk = mask(qk);
        }
        qk = softmax(qk);
        auto o = Tensor::mm(qk, v);
        o = o.view(-1, 1, -1, attn_hidden_dim_ * head_size_);
        o = o_proj(o);
        return {o};
    }
};

#endif // MODELING_TRANSFORMER_HPP
