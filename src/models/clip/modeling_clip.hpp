//
// Created by Rongjie Yi on 24-2-19.
//

#ifndef MODELING_CLIP_HPP
#define MODELING_CLIP_HPP

#include "models/vit/modeling_vit.hpp"
#include "configuration_clip.hpp"

class ClipVisionEmbedding final : public Module, public ClipConfig {
    Convolution2D patch_embedding = Convolution2D(3, hidden_dim, {patch, patch}, {patch, patch}, VALID, false, patch_embedding_name);
    Parameter cls_token = Parameter(1, 1, 1, hidden_dim, cls_token_name);
    Parameter position_ids = Parameter(1, std::ceil(img_hw / patch) * std::ceil(img_hw / patch) + 1, 1, 1, position_ids_name);
    Embedding position_embedding = Embedding(std::ceil(img_hw / patch) * std::ceil(img_hw / patch) + 1, hidden_dim, position_embeddings_name);
    LayerNorm pre_layrnorm = LayerNorm(hidden_dim, true, 1e-6, vision_pre_layrnorm_name);

    vector<Tensor> Forward(vector<Tensor> inputs) override {
        auto embd = patch_embedding(inputs[0]);
        embd = embd.transpose(SEQUENCE, DIMENSION);
        embd = embd.flatten(HEAD, SEQUENCE);
        embd = Tensor::cat({cls_token(), embd}, SEQUENCE);
        embd = position_embedding(position_ids()) + embd;
        embd = pre_layrnorm(embd);
        return {embd};
    }
};

class CLipVisionModel final : public Module, public ClipConfig {
    ClipVisionEmbedding embedding = ClipVisionEmbedding();
    vector<ViTBlock> blocks = List<ViTBlock>(block_num);
    LayerNorm norm = LayerNorm(hidden_dim, true, 1e-6, post_norm_name);

    vector<Tensor> Forward(vector<Tensor> inputs) override {
        auto x = embedding(inputs)[0];
        for (auto &block : blocks) {
            x = block({x})[0];
        }
        x = x.clip({}, {}, {0}, {});
        x = norm(x);
        return {x};
    }
};

class ClipTextAttention final : public Module, public ClipConfig {
    Linear q_proj = Linear(text_hidden_dim, text_head_size *text_attn_hidden_dim, true, text_q_proj_name);
    Linear k_proj = Linear(text_hidden_dim, text_head_size *text_attn_hidden_dim, true, text_k_proj_name);
    Linear v_proj = Linear(text_hidden_dim, text_head_size *text_attn_hidden_dim, true, text_v_proj_name);
    Linear o_proj = Linear(text_head_size * text_attn_hidden_dim, text_hidden_dim, true, text_o_proj_name);
    Causalmask mask = Causalmask(text_attn_base_name + "mask");
    Softmax softmax = Softmax(DIMENSION, text_attn_base_name + "softmax");

    vector<Tensor> Forward(vector<Tensor> inputs) override {
        auto q = q_proj(inputs[0]);
        auto k = k_proj(inputs[1]);
        auto v = v_proj(inputs[2]);
        q = q.view(-1, text_head_size, -1, text_attn_hidden_dim);
        k = k.view(-1, text_head_size, -1, text_attn_hidden_dim);
        v = v.view(-1, text_head_size, -1, text_attn_hidden_dim);
        k = k.transpose(SEQUENCE, DIMENSION);
        auto qk = Tensor::mm(q, k);
        qk = qk / std::sqrt(text_attn_hidden_dim);
        qk = mask(qk);
        qk = softmax(qk);
        auto o = Tensor::mm(qk, v);
        o = o.view(-1, 1, -1, text_attn_hidden_dim * text_head_size);
        o = o_proj(o);
        return {o};
    }
};
class ClipTextMLP final : public Module, public ClipConfig {
    Linear up_proj = Linear(text_hidden_dim, text_mlp_hidden, true, text_up_proj_name);
    Layer act = ACT_FN[act_fn_type](text_ffn_base_name + "act");
    Linear down_proj = Linear(text_mlp_hidden, text_hidden_dim, true, text_down_proj_name);

    vector<Tensor> Forward(vector<Tensor> inputs) override {
        auto x = up_proj(inputs[0]);
        x = act(x);
        x = down_proj(x);
        return {x};
    }
};
class ClipTextBlock final : public Module, public ClipConfig {
    ClipTextAttention attention = ClipTextAttention();
    ClipTextMLP mlp = ClipTextMLP();
    LayerNorm norm1 = LayerNorm(text_hidden_dim, true, 1e-6, text_attn_norm_name);
    LayerNorm norm2 = LayerNorm(text_hidden_dim, true, 1e-6, text_ffn_norm_name);

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
class ClipTextEmbedding final : public Module, public ClipConfig {
    Embedding token_embedding = Embedding(text_vocab_size, text_hidden_dim, text_token_embedding_name);
    Parameter position_ids = Parameter(1, max_position_embeddings, 1, 1, text_position_ids_name);
    Embedding position_embedding = Embedding(max_position_embeddings, text_hidden_dim, text_position_embeddings_name);

    vector<Tensor> Forward(vector<Tensor> inputs) override {
        auto embd = token_embedding(inputs[0]);
        auto pos_embd = position_ids().clip({}, {}, {0, embd.sequence()}, {});
        auto p_embd = position_embedding(pos_embd);
        auto out_embd = p_embd + embd;
        return {out_embd};
    }
};

class CLipTextModel final : public Module, public ClipConfig {
    ClipTextEmbedding embedding = ClipTextEmbedding();
    vector<ClipTextBlock> blocks = List<ClipTextBlock>(text_block_num);
    LayerNorm norm = LayerNorm(text_hidden_dim, true, 1e-6, text_post_norm_name);

    vector<Tensor> Forward(vector<Tensor> inputs) override {
        auto x = embedding(inputs)[0];
        for (auto &block : blocks) {
            x = block({x})[0];
        }
        x = norm(x);
        x = x.clip({}, {}, {-1}, {});
        return {x};
    }
};

class CLipModel final : public Module, public ClipConfig {
    CLipTextModel text_model = CLipTextModel();
    Linear text_projection = Linear(text_hidden_dim, text_hidden_dim, false, "text_projection");
    CLipVisionModel vision_model = CLipVisionModel();
    Linear visual_projection = Linear(hidden_dim, text_hidden_dim, false, "visual_projection");
    Matmul out_mm = Matmul(false, true, "out_mm");

    vector<Tensor> Forward(vector<Tensor> inputs) override {
        auto text = text_model({inputs[0]})[0];
        text = text_projection(text);
        text = text/text.norm(2);
        auto vision = vision_model({inputs[1]})[0];
        vision = visual_projection(vision);
        vision = vision/vision.norm(2);
        auto out = out_mm(text, vision)*100;
        return {out};
    }
};

#endif // MODELING_CLIP_HPP
