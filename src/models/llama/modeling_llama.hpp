//
// Created by Rongjie Yi on 2024/2/4 0004.
//

#ifndef MODELING_LLAMA_HPP
#define MODELING_LLAMA_HPP


#include "Layer.hpp"
#include "Module.hpp"
#include "configuration_llama.hpp"

using namespace mllm;

class LLaMAAttention final: public Module, public LLaMAConfig {
    Linear q_proj = Linear(hidden_dim, head_size*attn_hidden_dim, false, q_proj_name);
    Linear k_proj = Linear(hidden_dim, head_size*attn_hidden_dim, false,k_proj_name);
    Linear v_proj = Linear(hidden_dim, head_size*attn_hidden_dim,false,v_proj_name);
    Linear o_proj = Linear(head_size*attn_hidden_dim, hidden_dim, false, o_proj_name);
    RoPE q_rope = RoPE( RoPE_type, attn_base_name+"q_rope");
    RoPE k_rope = RoPE( RoPE_type, attn_base_name+"k_rope");
    KVCache k_cache = KVCache(cache_limit, attn_base_name+"k_cache");
    KVCache v_cache = KVCache(cache_limit, attn_base_name+"v_cache");
    Matmul qk_mm = Matmul(false, true, attn_base_name+"qk_mm");
    Matmul qkv_mm = Matmul(false, false, attn_base_name+"qkv_mm");
    Causalmask mask = Causalmask(attn_base_name+"mask");
    Softmax softmax = Softmax(DIMENSION, attn_base_name+"softmax");

    vector<Tensor> Forward(vector<Tensor> inputs) override {
        auto q = q_proj(inputs[0]);
        auto k = k_proj(inputs[1]);
        auto v = v_proj(inputs[2]);
        q = q.view(-1, head_size, -1, attn_hidden_dim);
        k = k.view(-1, head_size, -1, attn_hidden_dim);
        v = v.view(-1, head_size, -1, attn_hidden_dim);
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

class LLaMAMLP final: public Module, public LLaMAConfig {
    Linear gate_proj = Linear(hidden_dim, mlp_hidden, false, gate_proj_name);
    SiLU silu = SiLU( ffn_base_name+"act");
    Linear up_proj = Linear(hidden_dim, mlp_hidden, false, up_proj_name);
    Linear down_proj = Linear(mlp_hidden, hidden_dim, false, down_proj_name);

    vector<Tensor> Forward(vector<Tensor> inputs) override {
        auto x = gate_proj(inputs[0]);
        x = silu(x);
        auto y = up_proj(inputs[0]);
        x = x * y;
        x = down_proj(x);
        return {x};
    }
};

class LLaMABlock final: public Module, public LLaMAConfig {
    LLaMAAttention attention = LLaMAAttention();
    LLaMAMLP mlp = LLaMAMLP();
    RMSNorm norm1 = RMSNorm(hidden_dim, 1e-6, attn_norm_name);
    RMSNorm norm2 = RMSNorm(hidden_dim, 1e-6, ffn_norm_name);

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

class LLaMAModel final: public Module, public LLaMAConfig {
    Embedding embedding = Embedding(vocab_size, hidden_dim, token_embd_name);
    vector<LLaMABlock> blocks = List<LLaMABlock>(block_num);
    RMSNorm norm = RMSNorm(hidden_dim, 1e-6, post_norm_name);
    Linear mlp_head = Linear(hidden_dim, vocab_size, false, lm_head_name);

    vector<Tensor> Forward(vector<Tensor> inputs) override {
        auto x = embedding(inputs[0]);
        for (auto &block : blocks) {
            x = block({x})[0];
        }
        x = norm(x);
        x = mlp_head(x);
        return {x};
    }
};


#endif //MODELING_LLAMA_HPP