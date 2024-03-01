//
// Created by Rongjie Yi on 2024/2/4 0004.
//

#ifndef CONFIG_CLIP_HPP
#define CONFIG_CLIP_HPP

#include "Module.hpp"
#include "models/vit/configuration_vit.hpp"

using namespace mllm;

class ClipTextNameConfig: public TransformerNameConfig{
public:
    string _layer_name;
    string _post_norm_name;
    string lm_head_name;
    string _embd_name;
    string _patch_embedding_name;
    string _cls_token_name;
    string _position_ids_name;
    string _position_embeddings_name;
    string _token_embedding_name;
    void init() {
        _token_embedding_name = "token_embedding";
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
        _down_proj_name = "mlp.fc2";
        _attn_norm_name = "layer_norm1";
        _ffn_norm_name = "layer_norm2";
        _post_norm_name = ".final_layer_norm";
        _embd_name = ".embeddings.";
    }
};

class ClipConfig : public ViTConfig {
public:
    ClipTextNameConfig text_names_config;

    int text_vocab_size{};
    int max_position_embeddings{};
    int text_hidden_dim{};
    int text_head_size{};
    int text_ffn_hidden{};
    int text_block_num{};

    explicit ClipConfig(const string &model_type = "base", int patch_ = 32, int hw = 224, int text_vocab_size_ = 49408) {
        names_config.init("clip");
        text_names_config.init();
        patch = patch_;
        img_hw = hw;
        act_fn_type = "GELU";
        text_vocab_size = text_vocab_size_;
        if (model_type == "base") {
            hidden_dim = 768;
            head_size = 12;
            ffn_hidden = 3072;
            block_num = 12;
            max_position_embeddings = 77;
            text_hidden_dim = 512;
            text_head_size = 8;
            text_ffn_hidden = 2048;
            text_block_num = 12;
        } else {
            throw std::runtime_error("Unsupported model size");
        }
    }
};

#endif // CONFIG_ViT_HPP
