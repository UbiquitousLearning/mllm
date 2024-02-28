//
// Created by Rongjie Yi on 2024/2/14 0004.
//

#ifndef MODELING_FUYU_HPP
#define MODELING_FUYU_HPP

#include "Layer.hpp"
#include "Module.hpp"
#include "configuration_fuyu.hpp"

using namespace mllm;

class PersimmonAttention final : public Module {
    Layer qkv_proj;
    Split qkv_split;
    Layer q_norm;
    Layer k_norm;
    Layer q_rope;
    Layer k_rope;
    Layer k_cache;
    Layer v_cache;
    Layer mask;
    Layer softmax;
    Layer o_proj;
    int hidden_dim_{};
    int head_size_{};
    int attn_hidden_dim_{};

public:
    PersimmonAttention() = default;
    PersimmonAttention(int hidden_dim, int head_size, int attn_hidden_dim, int cache_limit, const FuyuNameConfig &names, const string &base_name) {
        attn_hidden_dim_ = attn_hidden_dim;
        head_size_ = head_size;
        hidden_dim_ = hidden_dim;
        qkv_proj = Linear(hidden_dim, head_size * attn_hidden_dim * 3, true, base_name + names._qkv_proj_name);
        qkv_split = Split(3, Chl::D_HD, head_size, base_name + names._qkv_proj_name + ".split");
        q_norm = LayerNorm(attn_hidden_dim, true, 1e-6, base_name + names._q_norm_name);
        k_norm = LayerNorm(attn_hidden_dim, true, 1e-6, base_name + names._k_norm_name);
        q_rope = RoPE(RoPEType::PERSIMMONROPE, base_name + "q_rope");
        k_rope = RoPE(RoPEType::PERSIMMONROPE, base_name + "k_rope");
        k_cache = KVCache(cache_limit, base_name + "k_cache");
        v_cache = KVCache(cache_limit, base_name + "v_cache");
        mask = Causalmask(base_name + "mask");
        softmax = Softmax(DIMENSION, base_name + "softmax");
        o_proj = Linear(head_size * attn_hidden_dim, hidden_dim, true, base_name + names._o_proj_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs) override {
        auto qkv = qkv_proj(inputs[0]);
        auto qkv_sp = qkv_split(qkv);
        auto q = qkv_sp[0];
        auto k = qkv_sp[1];
        auto v = qkv_sp[2];
        q = q_norm(q);
        k = k_norm(k);
        q = q_rope(q);
        k = k_rope(k);
        k = k_cache(k);
        v = v_cache(v);
        k = k.transpose(SEQUENCE, DIMENSION);
        auto qk = Tensor::mm(q, k);
        qk = qk / std::sqrt(attn_hidden_dim_);
        qk = mask(qk);
        qk = softmax(qk);
        auto o = Tensor::mm(qk, v);
        o = o.view(-1, 1, -1, attn_hidden_dim_ * head_size_);
        o = o_proj(o);
        return {o};
    }
};

class PersimmonMLP final : public Module {
    Layer up_proj;
    Layer relu2;
    Layer down_proj;

public:
    PersimmonMLP() = default;
    PersimmonMLP(int hidden_dim, int mlp_hidden, const FuyuNameConfig &names, const string &base_name) {
        up_proj = Linear(hidden_dim, mlp_hidden, true, base_name + names._up_proj_name);
        relu2 = ReLUSquaredActivation(base_name + "act");
        down_proj = Linear(mlp_hidden, hidden_dim, true, base_name + names._down_proj_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs) override {
        auto x = up_proj(inputs[0]);
        x = relu2(x);
        x = down_proj(x);
        return {x};
    }
};

class PersimmonBlock final : public Module {
    PersimmonAttention attention;
    PersimmonMLP mlp;
    Layer norm1;
    Layer norm2;

public:
    PersimmonBlock() = default;
    PersimmonBlock(int hidden_dim, int head_size, int mlp_hidden, int cache_limit, const FuyuNameConfig &names, const string &base_name) {
        attention = PersimmonAttention(hidden_dim, head_size, hidden_dim / head_size, cache_limit, names, base_name + names._attn_base_name);
        mlp = PersimmonMLP(hidden_dim, mlp_hidden, names, base_name + names._ffn_base_name);
        norm1 = LayerNorm(hidden_dim, true, 1e-6, base_name + names._attn_norm_name);
        norm2 = LayerNorm(hidden_dim, true, 1e-6, base_name + names._ffn_norm_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs) override {
        auto x = norm1(inputs[0]);
        x = attention({x, x, x})[0];
        auto tmp = x + inputs[0];
        x = norm2(tmp);
        x = mlp({x})[0];
        x = x + tmp;
        return {x};
    }
};

class Persimmon final : public Module {
    vector<PersimmonBlock> blocks;
    Layer norm;
    Layer lm_head;

public:
    Persimmon() = default;
    Persimmon(int hidden_dim, int head_size, int mlp_hidden, int cache_limit, int block_num, int vocab_size, const FuyuNameConfig &names) {
        blocks = List<PersimmonBlock>(block_num, hidden_dim, head_size, mlp_hidden, cache_limit, names, names.blk_name);
        norm = LayerNorm(hidden_dim, true, 1e-6, names.post_norm_name);
        lm_head = Linear(hidden_dim, vocab_size, false, names.lm_head_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs) override {
        auto x = inputs[0];
        for (auto &block : blocks) {
            x = block({x})[0];
        }
        x = norm(x);
        x = lm_head(x);
        return {x};
    }
};

class FuyuGather final : public Layer {
public:
    FuyuGather() = default;
    explicit FuyuGather(std::string name) {
        init(std::move(name), OpType::GATHER);
    }
    Tensor &operator()(Tensor &input_ids, Tensor &image_patches, Tensor &image_patches_indices) {
        return _3I1O_OP(input_ids, image_patches, image_patches_indices);
    }
};

class FuyuModel final : public Module {
    Layer embed_tokens;
    Layer vision_embed_tokens;
    FuyuGather fuyu_gather;
    Persimmon persimmon;

public:
    explicit FuyuModel(const FuyuConfig &config) :
        FuyuModel(config.vocab_size, config.hidden_dim, config.head_size, config.mlp_hidden, config.block_num,
                  config.cache_limit, config.patch_size, config.chl_size,
                  config.name_config) {
    }
    FuyuModel(int vocab_size, int hidden_dim, int head_size, int mlp_hidden, int block_num,
              int cache_limit, int patch_size, int chl_size,
              const FuyuNameConfig &names) {
        embed_tokens = Embedding(vocab_size, hidden_dim, names.token_embd_name);
        vision_embed_tokens = Linear(patch_size * patch_size * chl_size, hidden_dim, true, names.vision_embed_tokens_name);
        fuyu_gather = FuyuGather("gather");
        persimmon = Persimmon(hidden_dim, head_size, mlp_hidden, cache_limit, block_num, vocab_size, names);
    }
    vector<Tensor> Forward(vector<Tensor> inputs) override {
        auto input_ids = embed_tokens(inputs[0]);
        if (inputs[1].batch() > 0) {
            auto image_patches = vision_embed_tokens(inputs[1]);
            input_ids = fuyu_gather(input_ids, image_patches, inputs[2]);
        }
        return persimmon({input_ids});
    }
};

#endif // MODELING_FUYU_HPP