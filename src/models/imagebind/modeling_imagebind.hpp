//
// Created by Rongjie Yi on 24-2-29.
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
        attention = MultiHeadAttention(hidden_dim, head_size, head_size, hidden_dim / head_size,
                                       SPLIT_HD, false, bias_kv_cat,
                                       RoPEType::NONE, -1, -1, 0, do_mask, true,
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
        embd = embd.transpose({{WIDTH, HEIGHT}, {WIDTH, TIME}, {WIDTH, CHANNLE}});
        embd = embd.flatten(CHANNLE, HEIGHT);
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
    ImagebindVisionModel(const ImagebindConfig &config) :
        ImagebindVisionModel(config.vision_hidden_dim, config.vision_head_size, config.vision_ffn_hidden, config.head_hidden_dim,
                             config.patch, config.patch_time, config.img_hw, config.vision_block_num,
                             config.names_config){};
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
    ImagebindTextModel(const ImagebindConfig &config) :
        ImagebindTextModel(config.text_hidden_dim, config.text_head_size, config.text_ffn_hidden, config.head_hidden_dim,
                           config.vocab_size, config.max_position_embeddings, config.text_block_num,
                           config.names_config){};
    ImagebindTextModel(int hidden_dim, int head_size, int ffn_hidden, int head_hidden_dim,
                       int vocab_size, int max_position_embeddings, int block_num,
                       const ImagebindNameConfig &names) {
        embedding = ImagebindTextEmbedding(vocab_size, hidden_dim, max_position_embeddings, names, names._text_embd_name);
        blocks = List<EncoderBlock>(block_num, hidden_dim, head_size, ffn_hidden, "text", names, names._text_blocks_name);
        norm = LayerNorm(hidden_dim, true, 1e-6, names.text_post_norm_name);
        head = Linear(hidden_dim, head_hidden_dim, false, names.text_head_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        vector<int> in_len_ = std::any_cast<vector<int>>(args[0]);
        auto x = embedding(inputs)[0];
        for (auto &block : blocks) {
            x = block({x})[0];
        }
        x = x.clip(BATCH, {}, {}, in_len_, {});
        x = norm(x);
        x = head(x);
        x = x / x.norm(2);
        x = x * 100;
        return {x};
    }
};

class ImagebindAudioEmbedding final : public Module {
    Layer patch_embedding;
    Layer norm_layer;
    Parameter cls_token;
    Parameter position_embeddings;

public:
    ImagebindAudioEmbedding() = default;
    ImagebindAudioEmbedding(int hidden_dim, int patch, int stride, int img_h, int img_w, const ImagebindNameConfig &names, const string &base_name) {
        patch_embedding = Convolution2D(1, hidden_dim, {patch, patch}, {stride, stride}, VALID, false, base_name + names._rgbt_stem_name);
        norm_layer = LayerNorm(hidden_dim, true, 1e-6, base_name + names._norm_layer_name);
        cls_token = Parameter(1, 1, 1, hidden_dim, base_name + names._cls_token_name);
        int seq_len = (int)((img_h - patch) / stride + 1) * (int)((img_w - patch) / stride + 1) + 1;
        position_embeddings = Parameter(1, seq_len, 1, hidden_dim, base_name + names._helper_pos_embed_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto embd = patch_embedding(inputs[0]);
        embd = embd.transpose({{SEQUENCE, DIMENSION}, {HEAD, SEQUENCE}});
        embd = embd.flatten(HEAD, SEQUENCE);
        embd = norm_layer(embd);
        embd = Tensor::cat({cls_token(), embd}, SEQUENCE);
        embd = position_embeddings() + embd;
        return {embd};
    }
};

class ImagebindAudioModel final : public Module {
    ImagebindAudioEmbedding embedding;
    vector<EncoderBlock> blocks;
    Layer norm;
    Layer head;

public:
    ImagebindAudioModel() = default;
    ImagebindAudioModel(const ImagebindConfig &config) :
        ImagebindAudioModel(config.audio_hidden_dim, config.audio_head_size,
                            config.audio_ffn_hidden, config.head_hidden_dim,
                            config.audio_kernal, config.audio_stride, config.audio_h, config.audio_w, config.audio_block_num,
                            config.names_config){};
    ImagebindAudioModel(int hidden_dim, int head_size, int ffn_hidden, int head_hidden_dim,
                        int patch, int stride, int img_h, int img_w, int block_num,
                        const ImagebindNameConfig &names) {
        embedding = ImagebindAudioEmbedding(hidden_dim, patch, stride, img_h, img_w, names, names._audio_embd_name);
        blocks = List<EncoderBlock>(block_num, hidden_dim, head_size, ffn_hidden, "audio", names, names._audio_blocks_name);
        norm = LayerNorm(hidden_dim, true, 1e-6, names.audio_post_norm_name);
        head = Linear(hidden_dim, head_hidden_dim, false, names.audio_head_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = embedding(inputs)[0];
        for (auto &block : blocks) {
            x = block({x})[0];
        }
        x = norm(x);
        x = x.clip({}, {}, {0}, {});
        x = head(x);
        x = x / x.norm(2);
        x = x * 20;
        x = x.view(ANYDIM, -1, 3, -1);
        x = x.mean(SEQUENCE);
        return {x};
    }
};

class ImagebindModel final : public Module {
    ImagebindTextModel text_model;
    ImagebindVisionModel vision_model;
    ImagebindAudioModel audio_model;
    Layer softmax;
    Layer softmax2;

public:
    explicit ImagebindModel(const ImagebindConfig &config) :
        ImagebindModel(config.vision_hidden_dim, config.vision_head_size, config.vision_ffn_hidden, config.patch, config.patch_time, config.img_hw, config.vision_block_num,
                       config.text_hidden_dim, config.text_head_size, config.text_ffn_hidden, config.vocab_size, config.max_position_embeddings, config.text_block_num,
                       config.audio_hidden_dim, config.audio_head_size, config.audio_ffn_hidden, config.audio_kernal, config.audio_stride, config.audio_h, config.audio_w, config.audio_block_num,
                       config.head_hidden_dim,
                       config.names_config){};
    ImagebindModel(int vision_hidden_dim, int vision_head_size, int vision_ffn_hidden, int patch, int patch_time, int img_hw, int vision_block_num,
                   int text_hidden_dim, int text_head_size, int text_ffn_hidden, int vocab_size, int max_position_embeddings, int text_block_num,
                   int audio_hidden_dim, int audio_head_size, int audio_ffn_hidden, int audio_kernal, int audio_stride, int audio_h, int audio_w, int audio_block_num,
                   int head_hidden_dim,
                   const ImagebindNameConfig &names) {
        text_model = ImagebindTextModel(text_hidden_dim, text_head_size, text_ffn_hidden, head_hidden_dim,
                                        vocab_size, max_position_embeddings, text_block_num, names);
        vision_model = ImagebindVisionModel(vision_hidden_dim, vision_head_size, vision_ffn_hidden, head_hidden_dim,
                                            patch, patch_time, img_hw, vision_block_num, names);
        audio_model = ImagebindAudioModel(audio_hidden_dim, audio_head_size, audio_ffn_hidden, head_hidden_dim,
                                          audio_kernal, audio_stride, audio_h, audio_w, audio_block_num, names);
        softmax = Softmax(DIMENSION, "final.softmax1");
        softmax2 = Softmax(DIMENSION, "final.softmax2");
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        vector<int> in_len_ = std::any_cast<vector<int>>(args[0]);
        auto text = text_model({inputs[0]}, in_len_)[0];
        auto vision = vision_model({inputs[1]})[0];
        auto audio = audio_model({inputs[2]})[0];

        text = text.transpose(BATCH, SEQUENCE);
        vision = vision.transpose(BATCH, SEQUENCE);
        audio = audio.transpose(BATCH, SEQUENCE);

        text = text.transpose(SEQUENCE, DIMENSION);
        auto out = Tensor::mm(vision, text);
        out = softmax(out);

        audio = audio.transpose(SEQUENCE, DIMENSION);
        auto out_2 = Tensor::mm(vision, audio);
        out_2 = softmax2(out_2);
        return {out, out_2};
    }
};

#endif // MODELING_IMAGEBIND_HPP
