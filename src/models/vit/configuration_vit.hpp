//
// Created by Rongjie Yi on 2024/2/4 0004.
//

#ifndef CONFIG_VIT_HPP
#define CONFIG_VIT_HPP

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

    static std::string host_name;
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
    std::string base_name = host_name + "layer." + std::to_string(Module::listIdx) + ".";
    std::string attn_base_name = base_name  + _attn_base_name;
    std::string ffn_base_name = base_name + _ffn_base_name;
    std::string q_proj_name = attn_base_name + _q_proj_name;
    std::string k_proj_name = attn_base_name + _k_proj_name;
    std::string v_proj_name = attn_base_name + _v_proj_name;
    std::string o_proj_name = attn_base_name + _o_proj_name;
    std::string up_proj_name = ffn_base_name + _up_proj_name;
    std::string down_proj_name = base_name + _down_proj_name;
    std::string attn_norm_name = base_name + _attn_norm_name;
    std::string ffn_norm_name = base_name + _ffn_norm_name;
    static std::string token_embd_name;
    static std::string post_norm_name;
    static std::string lm_head_name;

    static std::string patch_embedding_name;
    static std::string cls_token_name;
    static std::string position_embeddings_name;

    static void init(const string& model_type = "base", int patch_ = 16, int hw = 224,  int classes = 1000) {
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
        host_name = "vit.encoder.";
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
        token_embd_name = "embed_tokens";
        post_norm_name = "vit.layernorm";
        lm_head_name = "classifier";
        patch_embedding_name = "vit.embeddings.patch_embeddings.projection";
        cls_token_name = "vit.embeddings.cls_token";
        position_embeddings_name = "vit.embeddings.position_embeddings";
    }
};
int ViTConfig::class_size;
int ViTConfig::patch;
int ViTConfig::img_hw;
int ViTConfig::hidden_dim;
int ViTConfig::head_size;
int ViTConfig::mlp_hidden;
int ViTConfig::block_num;
std::string ViTConfig::host_name;
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
std::string ViTConfig::token_embd_name;
std::string ViTConfig::post_norm_name;
std::string ViTConfig::lm_head_name;
std::string ViTConfig::patch_embedding_name;
std::string ViTConfig::cls_token_name;
std::string ViTConfig::position_embeddings_name;

#endif // CONFIG_ViT_HPP
