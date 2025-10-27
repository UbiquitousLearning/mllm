//
// Created by Rongjie Yi on 24-2-29.
//

#ifndef MODELING_TRANSFORMER_HPP
#define MODELING_TRANSFORMER_HPP

#include "DataType.hpp"
#include "Layer.hpp"
#include "Types.hpp"
#include "configuration_transformer.hpp"
#include <vector>

using namespace mllm;

struct MultiHeadAttentionConfig {
    int hidden_dim;
    int num_heads;
    int num_key_value_heads;
    int head_dim;
    AttnQKVSplitType do_qkv_proj = SPLIT_NONE; // Options: SPLIT_NONE, SPLIT_HD, SPLIT_D_HD
    AttnPostQkvNormType post_qkv_norm = PostQkv_NONE;
    bool bias_kv_cat = false;            // Only used when do_qkv_proj > 0
    RoPEType RoPE_type = RoPEType::NONE; // Options: NONE, ALIBI, ROPE, PERSIMMONROPE
    float rope_theta;
    int max_position_embeddings;
    float partial_rotary_factor = 1.0f; // Used for PERSIMMONROPE
    int cache_limit;
    bool is_causal;
    bool qkv_bias;
    bool o_bias;
    string attn_implementation = "flash_attention_2"; // Options: "flash_attention_2", "eager"
};

class MultiHeadAttention final : public Module {
    Layer qkv_proj;
    Layer q_proj;
    Layer k_proj;
    Layer v_proj;
    RoPE q_rope;
    RoPE k_rope;
    Layer q_norm;
    Layer k_norm;
    KVCache k_cache;
    KVCache v_cache;
    Softmax softmax;
    Layer o_proj;
    Parameter bias_k;
    Parameter bias_v;
    int num_heads_{};
    int num_key_value_heads_{};
    int head_dim_{};
    Chl split_chl_{};
    bool causal_mask = true;
    string attn_implementation_ = "flash_attention_2"; // Options: "flash_attention_2", "eager"
    bool head_first_attn = false;                      // 是否是head-first的注意力排布实现

public:
    MultiHeadAttention() = default;
    MultiHeadAttention(MultiHeadAttentionConfig config,
                       const TransformerNameConfig &names, const string &base_name) :
        MultiHeadAttention(config.hidden_dim, config.num_heads,
                           config.num_key_value_heads, config.head_dim,
                           config.do_qkv_proj, config.post_qkv_norm, config.bias_kv_cat,
                           config.RoPE_type, config.rope_theta, config.partial_rotary_factor,
                           config.max_position_embeddings,
                           config.cache_limit, config.is_causal,
                           config.qkv_bias, config.o_bias,
                           config.attn_implementation, names, base_name) {
    }
    MultiHeadAttention(int hidden_dim, int num_heads, int num_key_value_heads, int head_dim,
                       AttnQKVSplitType do_qkv_proj, AttnPostQkvNormType post_qkv_norm, bool bias_kv_cat,
                       RoPEType RoPE_type, float rope_theta, int max_position_embeddings,
                       int cache_limit, bool is_causal, bool qkv_bias, bool o_bias,
                       string attn_implementation,
                       const TransformerNameConfig &names, const string &base_name) :
        MultiHeadAttention(hidden_dim, num_heads, num_key_value_heads, head_dim,
                           do_qkv_proj, post_qkv_norm, bias_kv_cat,
                           RoPE_type, rope_theta, 1.0f, max_position_embeddings,
                           cache_limit, is_causal, qkv_bias, o_bias,
                           attn_implementation, names, base_name) {
    }
    MultiHeadAttention(int hidden_dim, int num_heads, int num_key_value_heads, int head_dim,
                       AttnQKVSplitType do_qkv_proj, AttnPostQkvNormType post_qkv_norm, bool bias_kv_cat,
                       RoPEType RoPE_type, float rope_theta, float partial_rotary_factor, int max_position_embeddings,
                       int cache_limit, bool is_causal, bool qkv_bias, bool o_bias,
                       string attn_implementation,
                       const TransformerNameConfig &names, const string &base_name) {
        head_dim_ = head_dim;
        num_heads_ = num_heads;
        num_key_value_heads_ = num_key_value_heads;
        causal_mask = is_causal;
        attn_implementation_ = attn_implementation;
        if (do_qkv_proj > 0) {
            split_chl_ = (Chl)do_qkv_proj;
            if (do_qkv_proj == SPLIT_HD) {
                qkv_proj = Linear(hidden_dim, (num_heads_ + num_key_value_heads_ + num_key_value_heads_) * head_dim, qkv_bias, base_name + names._qkv_proj_name);
            } else {
                qkv_proj = Linear(hidden_dim, num_heads * head_dim * 3, qkv_bias, base_name + names._qkv_proj_name);
            }
        } else {
            q_proj = Linear(hidden_dim, num_heads * head_dim, qkv_bias, base_name + names._q_proj_name);
            k_proj = Linear(hidden_dim, num_key_value_heads * head_dim, qkv_bias, base_name + names._k_proj_name);
            v_proj = Linear(hidden_dim, num_key_value_heads * head_dim, qkv_bias, base_name + names._v_proj_name);
        }
        if (post_qkv_norm == PostQkv_LayerNorm) {
            q_norm = LayerNorm(head_dim, true, 1e-6, base_name + names._q_norm_name);
            k_norm = LayerNorm(head_dim, true, 1e-6, base_name + names._k_norm_name);
        } else if (post_qkv_norm == PostQkv_RMSNorm) {
            q_norm = RMSNorm(head_dim, 1e-6, base_name + names._q_norm_name);
            k_norm = RMSNorm(head_dim, 1e-6, base_name + names._k_norm_name);
        }
        if (RoPE_type > 0) {
            q_rope = RoPE(RoPE_type, rope_theta, partial_rotary_factor, max_position_embeddings, base_name + "q_rope");
            k_rope = RoPE(RoPE_type, rope_theta, partial_rotary_factor, max_position_embeddings, base_name + "k_rope");
        }
        if (cache_limit > 0) {
            k_cache = KVCache(num_key_value_heads, head_dim,
                              num_heads / num_key_value_heads, cache_limit,
                              attn_implementation_, base_name + "k_cache");
            v_cache = KVCache(num_key_value_heads, head_dim,
                              num_heads / num_key_value_heads, cache_limit,
                              attn_implementation_, base_name + "v_cache");
        }
        softmax = Softmax(DIMENSION, is_causal, base_name + "softmax");
        o_proj = Linear(num_heads * head_dim, hidden_dim, o_bias, base_name + names._o_proj_name);
        if (bias_kv_cat) {
            bias_k = Parameter(1, 1, num_heads, head_dim, base_name + "bias_k");
            bias_v = Parameter(1, 1, num_heads, head_dim, base_name + "bias_v");
        }
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        Tensor q, k, v;
        if (qkv_proj.ready()) {
            auto qkv = qkv_proj(inputs[0]);
            if (split_chl_ == HD) {
                auto qkv_sp = qkv.split({head_dim_ * num_heads_,
                                         head_dim_ * num_key_value_heads_,
                                         head_dim_ * num_key_value_heads_},
                                        DIMENSION);
                q = qkv_sp[0];
                k = qkv_sp[1];
                v = qkv_sp[2];
                q = q.view(-1, num_heads_, -1, head_dim_);
                k = k.view(-1, num_key_value_heads_, -1, head_dim_);
                v = v.view(-1, num_key_value_heads_, -1, head_dim_);
            } else {
                auto qkv_sp = qkv.split({head_dim_, head_dim_, head_dim_}, split_chl_, num_heads_);
                q = qkv_sp[0];
                k = qkv_sp[1];
                v = qkv_sp[2];
            }
        } else {
            q = q_proj(inputs[0]);
            k = k_proj(inputs[1]);
            v = v_proj(inputs[2]);
            q = q.view(-1, num_heads_, -1, head_dim_);
            k = k.view(-1, num_key_value_heads_, -1, head_dim_);
            v = v.view(-1, num_key_value_heads_, -1, head_dim_);
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
        if (attn_implementation_ == "eager") {
            q = q.transpose(HEAD, SEQUENCE);
            k = k.transpose(HEAD, SEQUENCE);
            v = v.transpose(HEAD, SEQUENCE);
        }
        if (k_cache.ready() && v_cache.ready()) {
            k = k_cache(k);
            v = v_cache(v);
        }
        Tensor o;
        if (attn_implementation_ == "flash_attention_2") {
            o = Tensor::flash_attention2_forward(q, k, v, causal_mask);
        } else if (attn_implementation_ == "sage_attention") {
            o = Tensor::sage_attention_forward(q, k, v, causal_mask);
        } else if (attn_implementation_ == "eager") { // eager implementation
            q = q / std::sqrt(head_dim_);
            k = k.transpose(SEQUENCE, DIMENSION);
            auto qk = Tensor::mm(q, k);
            if (k_cache.ready() && v_cache.ready() && k_cache.getCacheSeqLen() != qk.sequence() && qk.sequence() > 1) {
                qk = softmax(qk, k_cache.getCacheSeqLen());
            } else {
                qk = softmax(qk);
            }
            o = Tensor::mm(qk, v);
            o = o.transpose(HEAD, SEQUENCE);
        } else if (attn_implementation_ == "eager_notrans") { // eager no transpose mplementation
            q = q / std::sqrt(head_dim_);
            k = k.transpose(SEQUENCE, DIMENSION);
            auto qk = Tensor::mm(q, k);
            if (k_cache.ready() && v_cache.ready() && k_cache.getCacheSeqLen() != qk.sequence() && qk.sequence() > 1) {
                qk = softmax(qk, k_cache.getCacheSeqLen());
            } else {
                qk = softmax(qk);
            }
            o = Tensor::mm(qk, v);
        }
        o = o.view(-1, 1, -1, head_dim_ * num_heads_);
        o = o_proj(o);
        return {o};
    }
    vector<KVCache *> get_cache() {
        return {&k_cache, &v_cache};
    }
    vector<RoPE *> get_rope() {
        return {&q_rope, &k_rope};
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
