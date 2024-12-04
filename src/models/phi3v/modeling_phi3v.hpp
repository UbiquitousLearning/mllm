//
// Created by Guo Xiaoqiang on 2024/8/12.
//
#ifndef MODELING_PHI3V_HPP
#define MODELING_PHI3V_HPP

#include "Layer.hpp"
#include "Module.hpp"
#include "Tensor.hpp"
#include "Types.hpp"
#include "configuration_phi3v.hpp"
#include "models/phi3/modeling_phi3.hpp"
#include "models/vit/modeling_vit.hpp"
#include <cassert>
#include <string>

using namespace mllm;

class Phi3VisionEmbedding final : public Module {
    Layer patch_embedding;
    Parameter cls_token;
    Layer position_embedding;
    int range_len_{};

public:
    Phi3VisionEmbedding() = default;
    Phi3VisionEmbedding(int hidden_dim, int patch, int img_hw, const Phi3VNameConfig &names, const string &base_name) {
        patch_embedding = Convolution2D(3, hidden_dim, {patch, patch}, {patch, patch}, VALID, false, base_name + names._patch_embedding_name);
        cls_token = Parameter(1, 1, 1, hidden_dim, base_name + names._cls_token_name);
        range_len_ = std::ceil(img_hw / patch) * std::ceil(img_hw / patch) + 1;
        position_embedding = Embedding(range_len_, hidden_dim, base_name + names._position_embeddings_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto embd = patch_embedding(inputs[0]);
        embd = embd.transpose({{SEQUENCE, DIMENSION}, {HEAD, SEQUENCE}}); // BSHD->BDHS->BDSH
        embd = embd.flatten(HEAD, SEQUENCE);
        embd = Tensor::cat({cls_token(), embd}, SEQUENCE);
        auto position_ids = Tensor::range(0, range_len_);
        embd = position_embedding(position_ids) + embd;
        return {embd};
    }
};

class Phi3VisionModel final : public Module {
    Phi3VisionEmbedding embedding;
    Layer pre_layrnorm;
    vector<ViTBlock> blocks;
    int clip_len_{};

public:
    Phi3VisionModel() = default;
    Phi3VisionModel(int hidden_dim, int head_size, int ffn_hidden, const string &act_fn_type, int patch, int img_hw, int block_num,
                    const Phi3VNameConfig &names, const string &base_name) {
        embedding = Phi3VisionEmbedding(hidden_dim, patch, img_hw, names, base_name + names._embd_name);
        pre_layrnorm = LayerNorm(hidden_dim, true, 1e-5, base_name + names._vision_pre_layrnorm_name);
        clip_len_ = std::ceil(img_hw / patch) * std::ceil(img_hw / patch) + 1;
        blocks = List<ViTBlock>(block_num, hidden_dim, head_size, ffn_hidden, act_fn_type, names, base_name + names._layer_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = embedding(inputs)[0];
        x = pre_layrnorm(x);
        for (auto &block : blocks) {
            x = block({x})[0];
        }
        x = x.clip({}, {}, {1, clip_len_}, {});
        return {x};
    }
};

class Phi3Embedding final : public Module {
    Phi3VisionModel img_processor;
    Layer embed_tokens;
    Parameter glb_GN;
    Parameter sub_GN;

    int image_dim_out;
    Layer img_projector_linear1;
    Layer img_projector_relu;
    Layer img_projector_linear2;
    string project_cls;

public:
    Phi3Embedding() = default;
    explicit Phi3Embedding(int vocab_size, int hidden_dim, int head_size, int ffn, int vision_hidden_dim, string &projection_cls, const Phi3VNameConfig &nameconfig, const string &base_name, const string &embd_name) {
        embed_tokens = Embedding(vocab_size, hidden_dim, embd_name);
        img_processor = Phi3VisionModel(vision_hidden_dim, 16, vision_hidden_dim * 4, "QuickGELU", 14, 336, 23, nameconfig, nameconfig.vison_model_name);
        glb_GN = Parameter(1, 1, 1, vision_hidden_dim * 4, nameconfig._vision_model_prefix + nameconfig._glb_GN);
        sub_GN = Parameter(1, 1, 1, vision_hidden_dim * 4, nameconfig._vision_model_prefix + nameconfig._sub_GN);
        project_cls = projection_cls;
        if (project_cls == "Linear") {
            img_projector_linear1 = Linear(1024 * 4, hidden_dim, true, nameconfig._vision_model_prefix + nameconfig._projection + ".0");
        } else if (project_cls == "MLP") {
            img_projector_linear1 = Linear(1024 * 4, hidden_dim, true, nameconfig._vision_model_prefix + nameconfig._projection + ".0");
            img_projector_relu = GELU(nameconfig._vision_model_prefix + nameconfig._projection + ".1");
            img_projector_linear2 = Linear(hidden_dim, hidden_dim, true, nameconfig._vision_model_prefix + nameconfig._projection + ".2");
        } else {
            throw std::runtime_error("Unsupported projection_cls");
        }
    }
    Tensor add_image_newline(Tensor image_features_hd) {
        auto newline_embeddings = sub_GN().expand(-1, -1, image_features_hd.sequence(), -1);
        image_features_hd = Tensor::cat({image_features_hd, newline_embeddings}, HEAD);
        image_features_hd = image_features_hd.flatten(HEAD, SEQUENCE);
        return image_features_hd;
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        bool have_img = inputs[1].batch() > 0;
        auto text_features = embed_tokens({inputs[0]});
        if (have_img) {
            auto image_features = img_processor({inputs[1]})[0];
            auto global_image_features = image_features.clip({0}, {}, {}, {});
            auto global_image_features_hd = Tensor::phi3v_hd_merge(global_image_features, 1, 1);
            auto global_image_features_hd_newline = add_image_newline(global_image_features_hd);

            auto img_nums = inputs[2].sequence();
            assert(img_nums == 1 && "phi3v only supports one images now!!");
            Tensor all_image_embeddings;
            for (int i = 0; i < inputs[2].sequence(); i++) {
                auto img_h = int(inputs[2].d<float>(0, 0, 0, 0));
                auto img_w = int(inputs[2].d<float>(0, 0, 0, 1));
                auto h_crop = img_h / 336;
                auto w_crop = img_w / 336;
                auto num_crops = h_crop * w_crop;
                auto sub_image_features = image_features.clip({1, num_crops + 1}, {}, {}, {});
                auto sub_image_features_hd = Tensor::phi3v_hd_merge(sub_image_features, h_crop, w_crop);
                auto sub_image_features_hd_newline = add_image_newline(sub_image_features_hd);
                all_image_embeddings = Tensor::cat({sub_image_features_hd_newline, glb_GN(), global_image_features_hd_newline}, SEQUENCE);
            }
            //  img projection
            image_features = img_projector_linear1(all_image_embeddings);
            if (project_cls == "MLP") {
                image_features = img_projector_relu(image_features);
                image_features = img_projector_linear2(image_features);
            }
            for (int i = 0; i < inputs[2].sequence(); i++) {
                auto where_idx = inputs[0].where(-1 * (i + 1), SEQUENCE);
                text_features = text_features.index_put(image_features, where_idx, false);
            }
        }
        return {text_features};
    }
};

class Phi3VModel final : public Module {
    Phi3Embedding vision_embed_tokens;
    vector<Phi3Block> blocks;
    Layer norm;
    Layer lm_head;

public:
    explicit Phi3VModel(const Phi3VConfig &config) :
        Phi3VModel(config.vocab_size, config.hidden_dim, config.head_size, config.num_key_value_heads, config.ffn_hidden, config.block_num,
                   config.RoPE_type, config.rope_theta, config.max_position_embeddings, config.cache_limit, config.vision_hidden_dim, config.projection_cls, config.name_config,
                   config.names_config, config.names_config.blk_name) {
    }
    Phi3VModel(int vocab_size, int hidden_dim, int head_size, int kv_head_size, int ffn_hidden, int block_num, RoPEType RoPE_type, float rope_theta, int max_position_embeddings, int cache_limit, int vision_hidden_dim, string projection_cls, const Phi3VNameConfig &visionconfig,
               const Phi3NameConfig &names, const string &base_name) {
        norm = RMSNorm(hidden_dim, 1e-6, names.post_norm_name);
        vision_embed_tokens = Phi3Embedding(vocab_size, hidden_dim, head_size, ffn_hidden, vision_hidden_dim, projection_cls, visionconfig, base_name, names.token_embd_name);
        blocks = List<Phi3Block>(block_num, hidden_dim, head_size, kv_head_size, ffn_hidden, RoPE_type, rope_theta, max_position_embeddings, cache_limit, names, base_name);
        norm = RMSNorm(hidden_dim, 1e-6, names.post_norm_name);
        lm_head = Linear(hidden_dim, vocab_size, false, names.lm_head_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = vision_embed_tokens(inputs)[0];
        for (auto &block : blocks) {
            x = block({x})[0];
        }
        x = norm(x);
        x = lm_head(x);
        return {x};
    }

    void clear_kvcache() override {
        for (auto &block : blocks) {
            auto kvcahce = block.get_attention().get_cache();
            for (auto &cache : kvcahce) {
                cache->clearCache();
            }
        }
    }
};
#endif // MODELING_PHI3_HPP