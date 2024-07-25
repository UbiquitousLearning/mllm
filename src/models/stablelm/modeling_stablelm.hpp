#ifndef MODELING_STABLELM_HPP
#define MODELING_STABLELM_HPP

#include "Layer.hpp"
#include "Module.hpp"
#include "configuration_stablelm.hpp"
#include "models/transformer/modeling_transformer.hpp"
#include <chrono>

using namespace mllm;

class StableLMMultiHeadAttention final : public Module {
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
    Causalmask mask;
    Softmax softmax;
    Layer o_proj;
    Parameter bias_k;
    Parameter bias_v;
    int head_size_{};
    int kv_head_size_{};
    int attn_hidden_dim_{};

public:
    StableLMMultiHeadAttention() = default;
    StableLMMultiHeadAttention(int hidden_dim, int head_size, int kv_head_size, int attn_hidden_dim,
                                  AttnQKVSplitType do_qkv_proj, bool post_qkv_norm, bool bias_kv_cat,
                                  RoPEType RoPE_type, int cache_limit, bool do_mask, bool bias,
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
            q_rope = RoPE(RoPE_type, 10000, 0.25, 4096, base_name + "q_rope");
            k_rope = RoPE(RoPE_type, 10000, 0.25, 4096, base_name + "k_rope");
        }
        if (cache_limit > 0) {
            k_cache = KVCache(head_size / kv_head_size, cache_limit, base_name + "k_cache");
            v_cache = KVCache(head_size / kv_head_size, cache_limit, base_name + "v_cache");
        }
        if (do_mask) {
            mask = Causalmask(base_name + "mask");
        }
        softmax = Softmax(DIMENSION, base_name + "softmax");
        o_proj = Linear(head_size * attn_hidden_dim, hidden_dim, false, base_name + names._o_proj_name);
        if (bias_kv_cat) {
            bias_k = Parameter(1, 1, head_size, attn_hidden_dim, base_name + "bias_k");
            bias_v = Parameter(1, 1, head_size, attn_hidden_dim, base_name + "bias_v");
        }
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
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
        if (mask.ready()) {
            qk = mask(qk, k_cache.getCacheSeqLen());
        }
        qk = softmax(qk, k_cache.getCacheSeqLen());
        auto o = Tensor::mm(qk, v);
        o = o.view(-1, 1, -1, attn_hidden_dim_ * head_size_);
        o = o_proj(o);
        return {o};
    }
};

class StableLMMLP final : public Module {
    Layer gate_proj;
    Layer silu;
    Layer up_proj;
    Layer down_proj;

public:
    StableLMMLP() = default;
    StableLMMLP(int hidden_dim, int ffn_hidden, const stablelmNameConfig &names, const string &base_name) {
        gate_proj = Linear(hidden_dim, ffn_hidden, false, base_name + names._gate_proj_name);
        silu = SiLU(base_name + "act");
        up_proj = Linear(hidden_dim, ffn_hidden, false, base_name + names._up_proj_name);
        down_proj = Linear(ffn_hidden, hidden_dim, false, base_name + names._down_proj_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = gate_proj(inputs[0]);
        x = silu(x);
        auto y = up_proj(inputs[0]);
        x = x * y;
        x = down_proj(x);
        return {x};
    }
};

class StableLMBlock final : public Module {
    StableLMMultiHeadAttention attention;
    StableLMMLP mlp;
    Layer norm1;
    Layer norm2;

public:
    StableLMBlock() = default;
    StableLMBlock(int hidden_dim, int head_size, int ffn_hidden, RoPEType RoPE_type, int cache_limit, const stablelmNameConfig &names, const string &base_name) {
        attention = StableLMMultiHeadAttention(hidden_dim, head_size, head_size, hidden_dim / head_size, SPLIT_NONE, false, false,
                                                  RoPE_type, cache_limit, true, true, names, base_name + names._attn_base_name);
        mlp = StableLMMLP(hidden_dim, ffn_hidden, names, base_name + names._ffn_base_name);
        norm1 = LayerNorm(hidden_dim, true, 1e-5, base_name + names._attn_norm_name);
        norm2 = LayerNorm(hidden_dim, true, 1e-5, base_name + names._ffn_norm_name);
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
};

class StableLMModel final : public Module {
    Layer embedding;
    vector<StableLMBlock> blocks;
    Layer norm;
    Layer lm_head;

public:
    explicit StableLMModel(const StableLMConfig &config) :
        StableLMModel(config.vocab_size, config.hidden_dim, config.head_size, config.ffn_hidden, config.block_num, config.RoPE_type, config.cache_limit,
                      config.names_config, config.names_config.blk_name) {
    }
    StableLMModel(int vocab_size, int hidden_dim, int head_size, int ffn_hidden, int block_num, RoPEType RoPE_type, int cache_limit,
                  const stablelmNameConfig &names, const string &base_name) {
        embedding = Embedding(vocab_size, hidden_dim, names.token_embd_name);
        blocks = List<StableLMBlock>(block_num, hidden_dim, head_size, ffn_hidden, RoPE_type, cache_limit, names, base_name);
        norm = LayerNorm(hidden_dim, true, 1e-5, names.post_norm_name);
        lm_head = Linear(hidden_dim, vocab_size, false, names.lm_head_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = embedding(inputs[0]);

        for (auto &block : blocks) {
            x = block({x})[0];
        }

        x = norm(x);
        x = lm_head(x);
        return {x};
    }
};

#endif // MODELING_STABLELM_HPP