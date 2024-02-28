//
// Created by Rongjie Yi on 24-2-16.
//

#ifndef MODELING_VIT_HPP
#define MODELING_VIT_HPP
#include "Layer.hpp"
#include "Module.hpp"
#include "configuration_vit.hpp"

using namespace mllm;

class ViTAttention final : public Module {
    Layer q_proj;
    Layer k_proj;
    Layer v_proj;
    Layer o_proj;
    Layer softmax;
    int head_size_{};
    int attn_hidden_dim_{};

public:
    ViTAttention() = default;
    ViTAttention(int hidden_dim, int head_size, int attn_hidden_dim, const ViTNameConfig &names, const string &base_name) {
        head_size_ = head_size;
        attn_hidden_dim_ = attn_hidden_dim;
        q_proj = Linear(hidden_dim, head_size * attn_hidden_dim, true, base_name + names._q_proj_name);
        k_proj = Linear(hidden_dim, head_size * attn_hidden_dim, true, base_name + names._k_proj_name);
        v_proj = Linear(hidden_dim, head_size * attn_hidden_dim, true, base_name + names._v_proj_name);
        o_proj = Linear(head_size * attn_hidden_dim, hidden_dim, true, base_name + names._o_proj_name);
        softmax = Softmax(DIMENSION, base_name + "softmax");
    }
    vector<Tensor> Forward(vector<Tensor> inputs) override {
        auto q = q_proj(inputs[0]);
        auto k = k_proj(inputs[1]);
        auto v = v_proj(inputs[2]);
        q = q.view(-1, head_size_, -1, attn_hidden_dim_);
        k = k.view(-1, head_size_, -1, attn_hidden_dim_);
        v = v.view(-1, head_size_, -1, attn_hidden_dim_);
        k = k.transpose(SEQUENCE, DIMENSION);
        auto qk = Tensor::mm(q, k);
        qk = qk / std::sqrt(attn_hidden_dim_);
        qk = softmax(qk);
        auto o = Tensor::mm(qk, v);
        o = o.view(-1, 1, -1, attn_hidden_dim_ * head_size_);
        o = o_proj(o);
        return {o};
    }
};

class ViTMLP final : public Module {
    Layer up_proj;
    Layer act;

public:
    ViTMLP() = default;
    ViTMLP(int hidden_dim, int mlp_hidden, const string &act_fn_type, const ViTNameConfig &names, const string &base_name) {
        up_proj = Linear(hidden_dim, mlp_hidden, true, base_name + names._up_proj_name);
        act = ACT_FN[act_fn_type](base_name + names._ffn_base_name + "act");
    }
    vector<Tensor> Forward(vector<Tensor> inputs) override {
        auto x = up_proj(inputs[0]);
        x = act(x);
        return {x};
    }
};

class ViTBlock final : public Module {
    ViTAttention attention;
    ViTMLP mlp;
    Layer down_proj;
    Layer norm1;
    Layer norm2;

public:
    ViTBlock() = default;
    ViTBlock(int hidden_dim, int head_size, int mlp_hidden, const string &act_fn_type, const ViTNameConfig &names, const string &base_name) {
        attention = ViTAttention(hidden_dim, head_size, hidden_dim / head_size, names, base_name + names._attn_base_name);
        mlp = ViTMLP(hidden_dim, mlp_hidden, act_fn_type, names, base_name + names._ffn_base_name);
        down_proj = Linear(mlp_hidden, hidden_dim, true, base_name + names._down_proj_name);
        norm1 = LayerNorm(hidden_dim, true, 1e-6, base_name + names._attn_norm_name);
        norm2 = LayerNorm(hidden_dim, true, 1e-6, base_name + names._ffn_norm_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs) override {
        auto x = norm1(inputs[0]);
        x = attention({x, x, x})[0];
        auto tmp = x + inputs[0];
        x = norm2(tmp);
        x = mlp({x})[0];
        x = down_proj(x);
        x = x + tmp;
        return {x};
    }
};

class ViTEmbedding final : public Module {
    Layer patch_embedding;
    Parameter cls_token;
    Parameter position_embeddings;

public:
    ViTEmbedding() = default;
    ViTEmbedding(int hidden_dim, int patch, int img_hw, const ViTNameConfig &names, const string &base_name) {
        patch_embedding = Convolution2D(3, hidden_dim, {patch, patch}, {patch, patch}, VALID, true, base_name + names._patch_embedding_name);
        cls_token = Parameter(1, 1, 1, hidden_dim, base_name + names._cls_token_name);
        position_embeddings = Parameter(1, int(img_hw / patch) * int(img_hw / patch) + 1, 1, hidden_dim, base_name + names._position_embeddings_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs) override {
        auto embd = patch_embedding(inputs[0]);
        embd = embd.transpose(SEQUENCE, DIMENSION);
        embd = embd.flatten(HEAD, SEQUENCE);
        embd = Tensor::cat({cls_token(), embd}, SEQUENCE);
        embd = position_embeddings() + embd;
        return {embd};
    }
};

class ViTModel final : public Module {
    ViTEmbedding embedding;
    vector<ViTBlock> blocks;
    Layer norm;
    Layer lm_head;

public:
    explicit ViTModel(const ViTConfig &config) :
        ViTModel(config.hidden_dim, config.head_size, config.mlp_hidden, config.act_fn_type, config.patch, config.img_hw, config.block_num, config.class_size,
                 config.names_config, config.names_config.vison_model_name) {
    }
    ViTModel(int hidden_dim, int head_size, int mlp_hidden, const string &act_fn_type, int patch, int img_hw, int block_num, int class_size,
             const ViTNameConfig &names, const string &base_name) {
        embedding = ViTEmbedding(hidden_dim, patch, img_hw, names, base_name + names._embd_name);
        blocks = List<ViTBlock>(block_num, hidden_dim, head_size, mlp_hidden, act_fn_type, names, base_name + names._layer_name);
        norm = LayerNorm(hidden_dim, true, 1e-6, base_name + names._post_norm_name);
        lm_head = Linear(hidden_dim, class_size, false, names.lm_head_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs) override {
        auto x = embedding(inputs)[0];
        for (auto &block : blocks) {
            x = block({x})[0];
        }
        x = x.clip({}, {}, {0}, {});
        x = norm(x);
        x = lm_head(x);
        return {x};
    }
};

#endif // MODELING_VIT_HPP
