//
// Created by Rongjie Yi on 2024/2/4 0004.
//

#ifndef CONFIG_VIT_HPP
#define CONFIG_VIT_HPP

#include <utility>
#include "models/transformer/configuration_transformer.hpp"

using namespace mllm;

class ViTNameConfig : public TransformerNameConfig {
public:
    string vison_model_name;
    string _layer_name;
    string _post_norm_name;
    string lm_head_name;
    string _embd_name;
    string _patch_embedding_name;
    string _cls_token_name;
    string _position_ids_name;
    string _position_embeddings_name;
    string _vision_pre_layrnorm_name;

    void init(const string &name_type = "vit") {
        if (name_type == "vit") {
            vison_model_name = "vit";
            _patch_embedding_name = "patch_embeddings.projection";
            _cls_token_name = "cls_token";
            _position_ids_name = "position_ids";
            _position_embeddings_name = "position_embeddings";
            _layer_name = ".encoder.layer.";
            _attn_base_name = "attention.";
            _ffn_base_name = "intermediate.";
            _q_proj_name = "attention.query";
            _k_proj_name = "attention.key";
            _v_proj_name = "attention.value";
            _o_proj_name = "output.dense";
            _up_proj_name = "dense";
            _down_proj_name = "output.dense";
            _attn_norm_name = "layernorm_before";
            _ffn_norm_name = "layernorm_after";
            _post_norm_name = ".layernorm";
            lm_head_name = "classifier";
        } else if (name_type == "clip") {
            vison_model_name = "vision_model";
            _patch_embedding_name = "patch_embedding";
            _cls_token_name = "class_embedding";
            _position_ids_name = "position_ids";
            _position_embeddings_name = "position_embedding";
            _layer_name = ".encoder.layers.";
            _attn_base_name = "self_attn.";
            _ffn_base_name = "mlp.";
            _q_proj_name = "q_proj";
            _k_proj_name = "k_proj";
            _v_proj_name = "v_proj";
            _o_proj_name = "out_proj";
            _up_proj_name = "fc1";
            _down_proj_name = _ffn_base_name + "fc2";
            _attn_norm_name = "layer_norm1";
            _ffn_norm_name = "layer_norm2";
            _post_norm_name = ".post_layernorm";
            _vision_pre_layrnorm_name = ".pre_layrnorm";
        } else if (name_type == "phi3v") {
            vison_model_name = "vision_model";
            _patch_embedding_name = "patch_embedding";
            _cls_token_name = "class_embedding";
            // _position_ids_name = "";
            _position_embeddings_name = "position_embedding";
            _layer_name = ".encoder.layers.";
            _attn_base_name = "self_attn.";
            _ffn_base_name = "mlp.";
            _q_proj_name = "q_proj";
            _k_proj_name = "k_proj";
            _v_proj_name = "v_proj";
            _o_proj_name = "out_proj";
            _up_proj_name = "fc1";
            _down_proj_name = _ffn_base_name + "fc2";
            _attn_norm_name = "layer_norm1";
            _ffn_norm_name = "layer_norm2";
            _post_norm_name = ".post_layernorm";
            _vision_pre_layrnorm_name = ".pre_layrnorm";
        }
        _embd_name = ".embeddings.";
    }
};

class ViTConfig : public TransformerConfig {
public:
    ViTNameConfig names_config;
    int class_size;
    int patch;
    int img_hw;
    int hidden_dim;
    int head_size;
    int ffn_hidden;
    int block_num;
    string act_fn_type;

    explicit ViTConfig(const string &model_type = "base", int patch_ = 16, int hw = 224, int classes = 1000, string act_fn_type_ = "GELU", const string &name_type = "vit") {
        names_config.init(name_type);
        class_size = classes;
        patch = patch_;
        img_hw = hw;
        if (model_type == "base") {
            hidden_dim = 768;
            head_size = 12;
            ffn_hidden = 3072;
            block_num = 12;
        } else {
            throw std::runtime_error("Unsupported model size");
        }
        act_fn_type = std::move(act_fn_type_);
    }
};

#endif // CONFIG_ViT_HPP
