//
// Created by ey on 24-2-29.
//

#ifndef CONFIGURATION_IMAGEBIND_HPP
#define CONFIGURATION_IMAGEBIND_HPP

#include "models/vit/configuration_vit.hpp"

using namespace mllm;
class ImagebindNameConfig : public TransformerNameConfig {
public:
    string _vision_embd_name;
    string _patch_embedding_name;
    string _cls_token_name;
    string _vision_pos_embed_name;
    string vision_pre_transformer_layer_name;
    string _vision_blocks_name;
    string vision_post_norm_name;
    string vision_head_name;

    string _text_embd_name;
    string _token_embedding_name;
    string _pos_embed_name;
    string _text_blocks_name;
    string text_post_norm_name;
    string text_head_name;


    void init() {
        _qkv_proj_name = "in_proj";
        _bias_k_name = "bias_k";
        _bias_v_name = "bias_v";
        _o_proj_name = "out_proj";
        _up_proj_name = "fc1";
        _down_proj_name = "fc2";
        _attn_norm_name = "norm_1";
        _ffn_norm_name = "norm_2";

        _vision_embd_name = "modality_preprocessors.vision.";
        _patch_embedding_name = "rgbt_stem.proj.1";
        _cls_token_name = "class_embedding";
        _vision_pos_embed_name = "pos_embedding_helper.pos_embed";

        vision_pre_transformer_layer_name = "modality_trunks.vision.pre_transformer_layer.0";
        vision_post_norm_name = "modality_heads.vision.0";
        vision_head_name ="modality_heads.vision.2";
        _vision_blocks_name = "modality_trunks.vision.blocks.";

        _text_embd_name = "modality_preprocessors.text.";
        _token_embedding_name = "token_embedding";
        _pos_embed_name = "pos_embed";

        text_post_norm_name = "modality_heads.text.0";
        text_head_name ="modality_heads.text.1";
        _text_blocks_name = "modality_trunks.text.blocks.";

    }

};
class ImagebindConfig {
public:
};

#endif // CONFIGURATION_IMAGEBIND_HPP
