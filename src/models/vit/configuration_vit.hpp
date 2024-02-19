//
// Created by Rongjie Yi on 2024/2/4 0004.
//

#ifndef CONFIG_VIT_HPP
#define CONFIG_VIT_HPP

#include <utility>

#include "Module.hpp"

using namespace mllm;

class ViTConfig {
public:
    static int class_size;
    static int patch;
    static int img_hw;
    static int hidden_dim;
    static int head_size;
    int attn_hidden_dim = hidden_dim / head_size;
    static int mlp_hidden;
    static int block_num;

    static std::string act_fn_type;

    static string vison_model_name;
    static std::string _q_proj_name;
    static std::string _k_proj_name;
    static std::string _v_proj_name;
    static std::string _o_proj_name;
    static std::string _up_proj_name;
    static std::string _down_proj_name;
    static std::string _attn_base_name;
    static std::string _ffn_base_name;
    static std::string _attn_norm_name;
    static std::string _ffn_norm_name;
    static std::string _layer_name;
    std::string embd_name = vison_model_name +".embeddings.";
    std::string patch_embedding_name = embd_name +_patch_embedding_name;
    std::string cls_token_name = embd_name +_cls_token_name;
    std::string position_ids_name = embd_name + _position_ids_name;
    std::string position_embeddings_name = embd_name + _position_embeddings_name;
    std::string encoder_host_name = vison_model_name +".encoder.";
    std::string list_base_name = encoder_host_name + _layer_name + std::to_string(Module::listIdx) + ".";
    std::string attn_base_name = list_base_name  + _attn_base_name;
    std::string ffn_base_name = list_base_name + _ffn_base_name;
    std::string q_proj_name = attn_base_name + _q_proj_name;
    std::string k_proj_name = attn_base_name + _k_proj_name;
    std::string v_proj_name = attn_base_name + _v_proj_name;
    std::string o_proj_name = attn_base_name + _o_proj_name;
    std::string up_proj_name = ffn_base_name + _up_proj_name;
    std::string down_proj_name = list_base_name + _down_proj_name;
    std::string attn_norm_name = list_base_name + _attn_norm_name;
    std::string ffn_norm_name = list_base_name + _ffn_norm_name;
    std::string  post_norm_name = vison_model_name + _post_norm_name;
    static std::string _post_norm_name;
    static std::string lm_head_name;

    static std::string _patch_embedding_name;
    static std::string _cls_token_name;
    static std::string _position_ids_name;
    static std::string _position_embeddings_name;

    static void init(const string& model_type = "base", int patch_ = 16, int hw = 224,  int classes = 1000, string act_fn_type_ = "GELU", string name_type = "vit") {
        class_size = classes;
        patch = patch_;
        img_hw = hw;
        if (model_type == "base") {
            hidden_dim = 768;
            head_size = 12;
            mlp_hidden = 3072;
            block_num = 12;
        } else {
            throw std::runtime_error("Unsupported model size");
        }
        act_fn_type = std::move(act_fn_type_);
        if(name_type == "vit") {
            vison_model_name = "vit";
            _patch_embedding_name = "patch_embeddings.projection";
            _cls_token_name = "cls_token";
            _position_ids_name = "position_ids";
            _position_embeddings_name = "position_embeddings";
            _layer_name = "layer.";
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
        } else if(name_type == "vision_model") {
            vison_model_name = "vision_model";
            _patch_embedding_name = "patch_embeddings";
            _cls_token_name = "class_embedding";
            _position_ids_name = "position_ids";
            _position_embeddings_name = "position_embeddings";
            _layer_name = "layers.";
            _attn_base_name = "self_attn.";
            _ffn_base_name = "mlp.";
            _q_proj_name = "q_proj";
            _k_proj_name = "k_proj";
            _v_proj_name = "v_proj";
            _o_proj_name = "out_proj";
            _up_proj_name = "fc1";
            _down_proj_name = "fc2";
            _attn_norm_name = "layer_norm1";
            _ffn_norm_name = "layer_norm2";
            _post_norm_name = ".post_layernorm";
        }
    }
};
int ViTConfig::class_size;
int ViTConfig::patch;
int ViTConfig::img_hw;
int ViTConfig::hidden_dim;
int ViTConfig::head_size;
int ViTConfig::mlp_hidden;
int ViTConfig::block_num;
std::string ViTConfig::act_fn_type;
std::string ViTConfig::vison_model_name;
std::string ViTConfig::_layer_name;
std::string ViTConfig::_attn_base_name;
std::string ViTConfig::_ffn_base_name;
std::string ViTConfig::_q_proj_name;
std::string ViTConfig::_k_proj_name;
std::string ViTConfig::_v_proj_name;
std::string ViTConfig::_o_proj_name;
std::string ViTConfig::_up_proj_name;
std::string ViTConfig::_down_proj_name;
std::string ViTConfig::_attn_norm_name;
std::string ViTConfig::_ffn_norm_name;
std::string ViTConfig::_post_norm_name;
std::string ViTConfig::lm_head_name;
std::string ViTConfig::_patch_embedding_name;
std::string ViTConfig::_cls_token_name;
std::string ViTConfig::_position_ids_name;
std::string ViTConfig::_position_embeddings_name;

#endif // CONFIG_ViT_HPP
