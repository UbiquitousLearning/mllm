//
// Created by ey on 24-2-29.
//

#ifndef MODELING_IMAGEBIND_HPP
#define MODELING_IMAGEBIND_HPP

#include "Layer.hpp"
#include "Module.hpp"
#include "configuration_imagebind.hpp"
#include "models/transformer/modeling_transformer.hpp"

using namespace mllm;

class EncoderBlock final : public Module {
    MultiHeadAttention attention;
    FeedForward ffn;
    Layer norm1;
    Layer norm2;

public:
    EncoderBlock() = default;
    EncoderBlock(int hidden_dim, int head_size, int ffn_hidden, const string &model_type, const ImagebindNameConfig &names, const string &base_name) {
        bool do_mask = false;
        bool bias_kv_cat = false;
        if (model_type == "text") {
            do_mask = true;
        } else if (model_type == "audio") {
            bias_kv_cat = true;
        }
        attention = MultiHeadAttention(hidden_dim, head_size, hidden_dim / head_size,
                                       true, false, bias_kv_cat,
                                       RoPEType::NONE, 0, do_mask, true,
                                       names, base_name + names._attn_base_name);
        ffn = FeedForward(hidden_dim, ffn_hidden, "GELU", true,
                          names, base_name + names._ffn_base_name);
        norm1 = LayerNorm(hidden_dim, true, 1e-6, base_name + names._attn_norm_name);
        norm2 = LayerNorm(hidden_dim, true, 1e-6, base_name + names._ffn_norm_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = norm1(inputs[0]);
        x = attention({x, x, x})[0];
        auto tmp = x + inputs[0];
        x = norm2(tmp);
        x = ffn({x})[0];
        x = x + tmp;
        return {x};
    }
};

class ImagebindVisionEmbedding final : public Module {
    Layer patch_embedding;
    Parameter cls_token;
    Parameter pos_embed;

public:
    ImagebindVisionEmbedding() = default;
    ImagebindVisionEmbedding(int hidden_dim, int patch, int patch_time, int img_hw, const ImagebindNameConfig &names, const string &base_name) {
        patch_embedding = Convolution3D(3, hidden_dim, {patch_time, patch, patch}, {patch_time, patch, patch}, VALID, false, base_name + names._patch_embedding_name);
        cls_token = Parameter(1, 1, 1, hidden_dim, base_name + names._cls_token_name);
        int pos_embd_seq = std::ceil(img_hw / patch) * std::ceil(img_hw / patch) * std::ceil(3 / patch_time) + 1;
        pos_embed = Parameter(1, pos_embd_seq, 1, hidden_dim, base_name + names._vision_pos_embed_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto embd = patch_embedding(inputs[0]);
        // embd = embd.transpose(THW, CHANNLE);
        embd = embd.transpose({{CHANNLE, TIME}, {CHANNLE, WIDTH}, {CHANNLE, HEIGHT}});// BCTHW->-->BTHWC
        embd = embd.flatten(TIME, WIDTH);
        embd = Tensor::cat({cls_token(), embd}, SEQUENCE);
        embd = pos_embed() + embd;
        return {embd};
    }
};

class ImagebindVisionModel final : public Module {
    ImagebindVisionEmbedding embedding;
    Layer pre_transformer_layer;
    vector<EncoderBlock> blocks;
    Layer norm;
    Layer head;

public:
    ImagebindVisionModel() = default;
    ImagebindVisionModel(int hidden_dim, int head_size, int ffn_hidden, int head_hidden_dim,
                         int patch, int patch_time, int img_hw, int block_num,
                         const ImagebindNameConfig &names) {
        embedding = ImagebindVisionEmbedding(hidden_dim, patch, patch_time, img_hw, names, names._vision_embd_name);
        pre_transformer_layer = LayerNorm(hidden_dim, true, 1e-6, names.vision_pre_transformer_layer_name);
        blocks = List<EncoderBlock>(block_num, hidden_dim, head_size, ffn_hidden, "vision", names, names._vision_blocks_name);
        norm = LayerNorm(hidden_dim, true, 1e-6, names.vision_post_norm_name);
        head = Linear(hidden_dim, head_hidden_dim, false, names.vision_head_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = embedding(inputs)[0];
        x = pre_transformer_layer(x);
        for (auto &block : blocks) {
            x = block({x})[0];
        }
        x = norm(x);
        x = x.clip({}, {}, {0}, {});
        x = head(x);
        x = x / x.norm(2);
        return {x};
    }
};

class ImagebindTextEmbedding final : public Module {
    Layer token_embedding;
    Parameter pos_embd;

public:
    ImagebindTextEmbedding() = default;
    ImagebindTextEmbedding(int vocab_size, int hidden_dim, int max_position_embeddings, const ImagebindNameConfig &names, const string &base_name) {
        token_embedding = Embedding(vocab_size, hidden_dim, base_name + names._token_embedding_name);
        pos_embd = Parameter(1, max_position_embeddings, 1, hidden_dim, base_name + names._pos_embed_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto embd = token_embedding(inputs[0]);
        embd = embd + pos_embd();
        return {embd};
    }
};

class ImagebindTextModel final : public Module {
    ImagebindTextEmbedding embedding;
    vector<EncoderBlock> blocks;
    Layer norm;
    Layer head;

public:
    ImagebindTextModel() = default;
    ImagebindTextModel(int hidden_dim, int head_size, int ffn_hidden, int head_hidden_dim,
                       int vocab_size, int max_position_embeddings, int block_num,
                       const ImagebindNameConfig &names) {
        embedding = ImagebindTextEmbedding(vocab_size, hidden_dim, max_position_embeddings, names, names._text_embd_name);
        blocks = List<EncoderBlock>(block_num, hidden_dim, head_size, ffn_hidden, "text", names, names._text_blocks_name);
        norm = LayerNorm(hidden_dim, true, 1e-6, names.text_post_norm_name);
        head = Linear(hidden_dim, head_hidden_dim, false, names.text_head_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        int in_len_ = std::any_cast<int>(args[0]);
        auto x = embedding(inputs)[0];
        for (auto &block : blocks) {
            x = block({x})[0];
        }
        x = x.clip({}, {}, {in_len_}, {});
        x = norm(x);
        x = head(x);
        x = x / x.norm(2);
        x = x * 100;
        return {x};
    }
};

class ImagebindModel final : public Module {
    ImagebindTextModel text_model;
    ImagebindVisionModel vision_model;
    Layer vision_text_softmax;

public:
    explicit ImagebindModel(const ImagebindConfig &config):
        ImagebindModel(config.vision_hidden_dim, config.vision_head_size, config.vision_ffn_hidden, config.patch, config.patch_time, config.img_hw, config.vision_block_num,
                       config.text_hidden_dim, config.text_head_size, config.text_ffn_hidden, config.vocab_size, config.max_position_embeddings, config.text_block_num,
                       config.head_hidden_dim,
                       config.names_config) {};
    ImagebindModel(int vision_hidden_dim, int vision_head_size, int vision_ffn_hidden, int patch, int patch_time, int img_hw, int vision_block_num,
                   int text_hidden_dim, int text_head_size, int text_ffn_hidden, int vocab_size, int max_position_embeddings, int text_block_num,
                   int head_hidden_dim,
                   const ImagebindNameConfig &names) {
        vision_model = ImagebindVisionModel(vision_hidden_dim, vision_head_size, vision_ffn_hidden, head_hidden_dim,
                                            patch, patch_time, img_hw, vision_block_num, names);
        text_model = ImagebindTextModel(text_hidden_dim, text_head_size, text_ffn_hidden, head_hidden_dim,
                                        vocab_size, max_position_embeddings, text_block_num, names);
        vision_text_softmax = Softmax( DIMENSION, "final.vision@text.softmax");
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        int in_len_ = std::any_cast<int>(args[0]);
        auto text = text_model({inputs[0]}, in_len_)[0];
        auto vision = vision_model({inputs[1]})[0];

        text = text.transpose(BATCH, SEQUENCE);
        vision = vision.transpose(BATCH, SEQUENCE);

        auto out = Tensor::mm(vision, text);
        out = vision_text_softmax( out);
        return {out};
    }
};

#endif // MODELING_IMAGEBIND_HPP
