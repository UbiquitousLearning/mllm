//
// Created by Rongjie Yi on 24-2-19.
//

#ifndef MODELING_CLIP_HPP
#define MODELING_CLIP_HPP

#include "models/vit/modeling_vit.hpp"
#include "configuration_clip.hpp"
#include "models/transformer/modeling_transformer.hpp"

class ClipVisionEmbedding final : public Module {
    Layer patch_embedding;
    Parameter cls_token;
    Parameter position_ids;
    Layer position_embedding;

public:
    ClipVisionEmbedding() = default;
    ClipVisionEmbedding(int hidden_dim, int patch, int img_hw, const ViTNameConfig &names, const string &base_name) {
        patch_embedding = Convolution2D(3, hidden_dim, {patch, patch}, {patch, patch}, VALID, false, base_name + names._patch_embedding_name);
        cls_token = Parameter(1, 1, 1, hidden_dim, base_name + names._cls_token_name);
        position_ids = Parameter(1, std::ceil(img_hw / patch) * std::ceil(img_hw / patch) + 1, 1, 1, base_name + names._position_ids_name);
        position_embedding = Embedding(std::ceil(img_hw / patch) * std::ceil(img_hw / patch) + 1, hidden_dim, base_name + names._position_embeddings_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto embd = patch_embedding(inputs[0]);
        embd = embd.transpose({{SEQUENCE, DIMENSION}, {HEAD, SEQUENCE}}); // BSHD->BDHS->BDSH
        embd = embd.flatten(HEAD, SEQUENCE);
        embd = Tensor::cat({cls_token(), embd}, SEQUENCE);
        embd = position_embedding(position_ids()) + embd;
        return {embd};
    }
};

class CLipVisionModel final : public Module {
    ClipVisionEmbedding embedding;
    Layer pre_layrnorm;
    vector<ViTBlock> blocks;
    Layer norm;

public:
    CLipVisionModel() = default;
    CLipVisionModel(int hidden_dim, int head_size, int ffn_hidden, const string &act_fn_type, int patch, int img_hw, int block_num,
                    string attn_implementation,
                    const ViTNameConfig &names, const string &base_name) {
        embedding = ClipVisionEmbedding(hidden_dim, patch, img_hw, names, base_name + names._embd_name);
        pre_layrnorm = LayerNorm(hidden_dim, true, 1e-6, base_name + names._vision_pre_layrnorm_name);
        blocks = List<ViTBlock>(block_num, hidden_dim, head_size, ffn_hidden, act_fn_type, attn_implementation, names, base_name + names._layer_name);
        norm = LayerNorm(hidden_dim, true, 1e-6, base_name + names._post_norm_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = embedding(inputs)[0];
        x = pre_layrnorm(x);
        for (auto &block : blocks) {
            x = block({x})[0];
        }
        x = x.clip({}, {}, {0}, {});
        x = norm(x);
        return {x};
    }
};

class ClipTextMLP final : public Module {
    Layer up_proj;
    Layer act;

public:
    ClipTextMLP() = default;
    ClipTextMLP(int hidden_dim, int ffn_hidden, const string &act_fn_type, const ClipTextNameConfig &names, const string &base_name) {
        up_proj = Linear(hidden_dim, ffn_hidden, true, base_name + names._up_proj_name);
        act = ACT_FN[act_fn_type](base_name + names._ffn_base_name + "act");
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = up_proj(inputs[0]);
        x = act(x);
        return {x};
    }
};

class ClipTextBlock final : public Module {
    MultiHeadAttention attention;
    ClipTextMLP mlp;
    Layer down_proj;
    Layer norm1;
    Layer norm2;

public:
    ClipTextBlock() = default;
    ClipTextBlock(int hidden_dim, int head_size, int ffn_hidden, const string &act_fn_type,
                  string attn_implementation, const ClipTextNameConfig &names, const string &base_name) {
        attention = MultiHeadAttention(hidden_dim, head_size, head_size,
                                       hidden_dim / head_size, SPLIT_NONE, PostQkv_NONE, false,
                                       RoPEType::NONE, -1, -1, 0, true, true, true, attn_implementation,
                                       names, base_name + names._attn_base_name);
        mlp = ClipTextMLP(hidden_dim, ffn_hidden, act_fn_type, names, base_name + names._ffn_base_name);
        down_proj = Linear(ffn_hidden, hidden_dim, true, base_name + names._down_proj_name);
        norm1 = LayerNorm(hidden_dim, true, 1e-6, base_name + names._attn_norm_name);
        norm2 = LayerNorm(hidden_dim, true, 1e-6, base_name + names._ffn_norm_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
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

class ClipTextEmbedding final : public Module {
    Layer token_embedding;
    Parameter position_ids;
    Layer position_embedding;

public:
    ClipTextEmbedding() = default;
    ClipTextEmbedding(int vocab_size, int hidden_dim, int max_position_embeddings, const ClipTextNameConfig &names, const string &base_name) {
        token_embedding = Embedding(vocab_size, hidden_dim, base_name + names._token_embedding_name);
        position_ids = Parameter(1, max_position_embeddings, 1, 1, base_name + names._position_ids_name);
        position_embedding = Embedding(max_position_embeddings, hidden_dim, base_name + names._position_embeddings_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto embd = token_embedding(inputs[0]);
        auto pos_embd = position_ids().clip({}, {}, {0, embd.sequence()}, {});
        auto p_embd = position_embedding(pos_embd);
        auto out_embd = p_embd + embd;
        return {out_embd};
    }
};

class CLipTextModel final : public Module {
    ClipTextEmbedding embedding;
    vector<ClipTextBlock> blocks;
    Layer norm;

public:
    CLipTextModel() = default;
    CLipTextModel(int hidden_dim, int head_size, int ffn_hidden, const string &act_fn_type,
                  int max_position_embeddings, int vocab_size, int block_num,
                  string attn_implementation,
                  const ClipTextNameConfig &names, const string &base_name) {
        embedding = ClipTextEmbedding(vocab_size, hidden_dim, max_position_embeddings, names, base_name + names._embd_name);
        blocks = List<ClipTextBlock>(block_num, hidden_dim, head_size, ffn_hidden, act_fn_type,
                                     attn_implementation, names, base_name + names._layer_name);
        norm = LayerNorm(hidden_dim, true, 1e-6, base_name + names._post_norm_name);
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = embedding(inputs)[0];
        for (auto &block : blocks) {
            x = block({x})[0];
        }
        x = norm(x);
        x = x.clip({}, {}, {-1}, {});
        return {x};
    }
};

class CLipModel final : public Module {
    CLipTextModel text_model;
    Layer text_projection;
    CLipVisionModel vision_model;
    Layer visual_projection;

public:
    explicit CLipModel(const ClipConfig &config) :
        CLipModel(config.text_hidden_dim, config.text_head_size, config.text_ffn_hidden,
                  config.hidden_dim, config.head_size, config.ffn_hidden,
                  config.act_fn_type, config.max_position_embeddings, config.text_vocab_size, config.text_block_num,
                  config.patch, config.img_hw, config.block_num,
                  config.attn_implementation,
                  config.text_names_config, "text_model",
                  config.names_config, "vision_model"){};
    CLipModel(int text_hidden_dim, int text_head_size, int text_ffn_hidden,
              int vision_hidden_dim, int vision_head_size, int vision_ffn_hidden,
              const string &act_fn_type, int max_position_embeddings, int vocab_size, int text_block_num,
              int patch, int img_hw, int vision_block_num,
              string attn_implementation,
              const ClipTextNameConfig &text_names, const string &text_base_name,
              const ViTNameConfig &vit_names, const string &vision_base_name) {
        text_model = CLipTextModel(text_hidden_dim, text_head_size, text_ffn_hidden,
                                   act_fn_type, max_position_embeddings,
                                   vocab_size, text_block_num,
                                   attn_implementation,
                                   text_names, text_base_name);
        text_projection = Linear(text_hidden_dim, text_hidden_dim, false, "text_projection");
        vision_model = CLipVisionModel(vision_hidden_dim, vision_head_size, vision_ffn_hidden, act_fn_type, patch, img_hw, vision_block_num,
                                       attn_implementation, vit_names, vision_base_name);
        visual_projection = Linear(vision_hidden_dim, text_hidden_dim, false, "visual_projection");
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto text = text_model({inputs[0]})[0];
        text = text_projection(text);
        text = text / text.norm(2);
        auto vision = vision_model({inputs[1]})[0];
        vision = visual_projection(vision);
        vision = vision / vision.norm(2);
        vision = vision.transpose(SEQUENCE, DIMENSION);
        auto out = Tensor::mm(text, vision) * 100;
        return {out};
    }
};

#endif // MODELING_CLIP_HPP
