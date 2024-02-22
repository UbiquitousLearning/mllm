//
// Created by Rongjie Yi on 24-2-16.
//

#ifndef MODELING_VIT_HPP
#define MODELING_VIT_HPP
#include "Layer.hpp"
#include "Module.hpp"
#include "configuration_vit.hpp"

using namespace mllm;

class ViTAttention final: public Module, public ViTConfig {
    Linear q_proj = Linear(hidden_dim, head_size*attn_hidden_dim, true, q_proj_name);
    Linear k_proj = Linear(hidden_dim, head_size*attn_hidden_dim, true,k_proj_name);
    Linear v_proj = Linear(hidden_dim, head_size*attn_hidden_dim,true,v_proj_name);
    Linear o_proj = Linear(head_size*attn_hidden_dim, hidden_dim, true, o_proj_name);
    Matmul qk_mm = Matmul(false, true, attn_base_name+"qk_mm");
    Matmul qkv_mm = Matmul(false, false, attn_base_name+"qkv_mm");
    Softmax softmax = Softmax(DIMENSION, attn_base_name+"softmax");

    vector<Tensor> Forward(vector<Tensor> inputs) override {
        auto q = q_proj(inputs[0]);
        auto k = k_proj(inputs[1]);
        auto v = v_proj(inputs[2]);
        q = q.view(-1, head_size, -1, attn_hidden_dim);
        k = k.view(-1, head_size, -1, attn_hidden_dim);
        v = v.view(-1, head_size, -1, attn_hidden_dim);
        auto qk = qk_mm(q, k);
        qk = qk / std::sqrt(attn_hidden_dim);
        qk = softmax(qk);
        auto o = qkv_mm(qk, v);
        o = o.view(-1, 1, -1, attn_hidden_dim * head_size);
        o = o_proj(o);
        return {o};
    }
};

class ViTMLP final: public Module, public ViTConfig {
    Linear up_proj = Linear(hidden_dim, mlp_hidden, true, up_proj_name);
    Layer act = ACT_FN[act_fn_type](ffn_base_name+"act");
    Linear down_proj = Linear(mlp_hidden, hidden_dim, true, down_proj_name);

    vector<Tensor> Forward(vector<Tensor> inputs) override {
        auto x = up_proj(inputs[0]);
        x =act(x);
        x = down_proj(x);
        return {x};
    }
};
class ViTBlock final: public Module, public ViTConfig {
    ViTAttention attention = ViTAttention();
    ViTMLP mlp = ViTMLP();
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

class ViTEmbedding final: public Module, public ViTConfig {
    Convolution2D patch_embedding = Convolution2D(3, hidden_dim, {patch, patch}, {patch, patch}, VALID, true, patch_embedding_name);
    Parameter cls_token = Parameter(1, 1, 1, hidden_dim, cls_token_name);
    Parameter position_embeddings = Parameter(1, int(img_hw/patch) * int(img_hw/patch) + 1, 1, hidden_dim,position_embeddings_name);

    vector<Tensor> Forward(vector<Tensor> inputs) override {
        auto embd = patch_embedding(inputs[0]);
        embd = embd.transpose(SEQUENCE, DIMENSION);
        embd = embd.flatten(HEAD, SEQUENCE);
        embd = Tensor::cat({cls_token(), embd}, SEQUENCE);
        embd = position_embeddings() + embd;
        return {embd};
    }
};

class ViTModel final: public Module, public ViTConfig {
    ViTEmbedding embedding = ViTEmbedding();
    vector<ViTBlock> blocks = List<ViTBlock>(block_num);
    LayerNorm norm = LayerNorm(hidden_dim, true, 1e-6, post_norm_name);
    Linear lm_head = Linear(hidden_dim, class_size, false, lm_head_name);

    vector<Tensor> Forward(vector<Tensor> inputs) override {
        auto x = embedding(inputs)[0];
        for (auto &block : blocks) {
            x = block({x})[0];
        }
        x = x.clip( {}, {}, {0}, {});
        x = norm(x);
        x = lm_head(x);
        return {x};
    }
};

#endif //MODELING_VIT_HPP
