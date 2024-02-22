//
// Created by Rongjie Yi on 2024/2/4 0004.
//

#ifndef CONFIG_CLIP_HPP
#define CONFIG_CLIP_HPP

#include "Module.hpp"
#include "models/vit/configuration_vit.hpp"

using namespace mllm;

class ClipConfig: public ViTConfig{

public:
    static int text_vocab_size;
    static int max_position_embeddings;

    static int text_hidden_dim;
    static int text_head_size;
    int text_attn_hidden_dim = text_hidden_dim / text_head_size;
    static int text_mlp_hidden;
    static int text_block_num;



    std::string text_token_embedding_name = "text_model.embeddings.token_embedding";
    std::string text_position_ids_name = "text_model.embeddings.position_ids";
    std::string text_position_embeddings_name = "text_model.embeddings.position_embedding";
    string text_attn_base_name = "text_model.encoder.layers." + std::to_string(Module::listIdx) + ".self_attn.";
    std::string text_q_proj_name = text_attn_base_name + "q_proj";
    std::string text_k_proj_name = text_attn_base_name + "k_proj";
    std::string text_v_proj_name = text_attn_base_name + "v_proj";
    std::string text_o_proj_name = text_attn_base_name + "out_proj";
    string text_ffn_base_name = "text_model.encoder.layers." + std::to_string(Module::listIdx) + ".mlp.";
    std::string text_up_proj_name   = text_ffn_base_name +"fc1";
    std::string text_down_proj_name = text_ffn_base_name +"fc2";
    std::string text_attn_norm_name = "text_model.encoder.layers." + std::to_string(Module::listIdx) + ".layer_norm1";
    std::string text_ffn_norm_name  = "text_model.encoder.layers." + std::to_string(Module::listIdx) + ".layer_norm2";
    std::string text_post_norm_name = "text_model.final_layer_norm";

    string vision_pre_layrnorm_name = "vision_model.pre_layrnorm";


    static void init(const string& model_type = "base", int patch_ = 32, int hw = 224, int text_vocab_size_ = 49408) {
        ViTConfig::init(model_type, patch_, hw, 1000,  "QuickGELU", "vision_model");
        text_vocab_size = text_vocab_size_;
        if (model_type == "base") {
            max_position_embeddings = 77;
            text_hidden_dim = 512;
            text_head_size = 8;
            text_mlp_hidden = 2048;
            text_block_num = 12;
        }
    }
};

int ClipConfig::text_vocab_size;
int ClipConfig::max_position_embeddings;

int ClipConfig::text_hidden_dim;
int ClipConfig::text_head_size;
int ClipConfig::text_mlp_hidden;
int ClipConfig::text_block_num;

#endif // CONFIG_ViT_HPP
