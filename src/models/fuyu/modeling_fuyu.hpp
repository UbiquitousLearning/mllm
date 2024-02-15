//
// Created by Rongjie Yi on 2024/2/14 0004.
//

#ifndef MODELING_FUYU_HPP
#define MODELING_FUYU_HPP


#include "Layer.hpp"
#include "Module.hpp"
#include "configuration_fuyu.hpp"

using namespace mllm;

class PersimmonAttention final: public Module, public FuyuConfig {
    Linear qkv_proj = Linear(hidden_dim, head_size*attn_hidden_dim*3, true, qkv_proj_name);
    Split qkv_split = Split(3, Chl::D_HD, head_size, qkv_proj_name + ".split");
    LayerNorm q_norm = LayerNorm(attn_hidden_dim, true,1e-6, q_norm_name);
    LayerNorm k_norm = LayerNorm(attn_hidden_dim, true,1e-6, k_norm_name);
    RoPE q_rope = RoPE( RoPEType::PERSIMMONROPE, attn_base_name+"q_rope");
    RoPE k_rope = RoPE( RoPEType::PERSIMMONROPE, attn_base_name+"k_rope");
    KVCache k_cache = KVCache(cache_limit, attn_base_name+"k_cache");
    KVCache v_cache = KVCache(cache_limit, attn_base_name+"v_cache");
    Matmul qk_mm = Matmul(false, true, attn_base_name+"qk_mm");
    Matmul qkv_mm = Matmul(false, false, attn_base_name+"qkv_mm");
    Causalmask mask = Causalmask(attn_base_name+"mask");
    Softmax softmax = Softmax(DIMENSION, attn_base_name+"softmax");
    Linear o_proj = Linear(head_size*attn_hidden_dim, hidden_dim, true, o_proj_name);

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
        auto qk = qk_mm(q, k);
        qk = qk / std::sqrt(attn_hidden_dim);
        qk = mask(qk);
        qk = softmax(qk);
        auto o = qkv_mm(qk, v);
        o = o.view(-1, 1, -1, attn_hidden_dim * head_size);
        o = o_proj(o);
        return {o};
    }
};

class PersimmonMLP final: public Module, public FuyuConfig {
    ReLUSquaredActivation relu2 = ReLUSquaredActivation( ffn_base_name+"act");
    Linear up_proj = Linear(hidden_dim, mlp_hidden, true, up_proj_name);
    Linear down_proj = Linear(mlp_hidden, hidden_dim, true, down_proj_name);

    vector<Tensor> Forward(vector<Tensor> inputs) override {
        auto x = up_proj(inputs[0]);
        x =relu2(x);
        x = down_proj(x);
        return {x};
    }
};

class PersimmonBlock final: public Module, public FuyuConfig {
    PersimmonAttention attention = PersimmonAttention();
    PersimmonMLP mlp = PersimmonMLP();
    LayerNorm norm1 = LayerNorm(hidden_dim, true, 1e-6, attn_norm_name);
    LayerNorm norm2 = LayerNorm(hidden_dim,true,  1e-6, ffn_norm_name);

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

class Persimmon final: public Module, public FuyuConfig {
    // Embedding embedding = Embedding(vocab_size, hidden_dim, token_embd_name);
    vector<PersimmonBlock> blocks = List<PersimmonBlock>(block_num);
    LayerNorm norm = LayerNorm(hidden_dim, true, 1e-6, post_norm_name);
    Linear lm_head = Linear(hidden_dim, vocab_size, false, lm_head_name);

    vector<Tensor> Forward(vector<Tensor> inputs) override {
        // auto x = embedding(inputs[0]);
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
    explicit FuyuGather(std::string name) {
        init(std::move(name), OpType::GATHER);
    }
    Tensor& operator()(Tensor &input_ids, Tensor &image_patches, Tensor &image_patches_indices) {
        return _3I1O_OP(input_ids, image_patches, image_patches_indices);
    }
};

class FuyuModel final: public Module, public FuyuConfig {
    Embedding embed_tokens = Embedding(vocab_size, hidden_dim, token_embd_name);
    Linear vision_embed_tokens = Linear(patch_size * patch_size * chl_size, hidden_dim, true, vision_embed_tokens_name);
    FuyuGather fuyu_gather = FuyuGather("gather");
    Persimmon persimmon = Persimmon();

    vector<Tensor> Forward(vector<Tensor> inputs) override {
        auto input_ids = embed_tokens(inputs[0]);
        if(inputs[1].batch()>0) {
            auto image_patches = vision_embed_tokens(inputs[1]);
            input_ids = fuyu_gather(input_ids, image_patches, inputs[2]);
        }
        return persimmon({input_ids});
    }
};


#endif //MODELING_FUYU_HPP