//
// Created by Rongjie Yi on 24-2-29.
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
    string _norm_layer_name;
    string _rgbt_stem_name;
    string vision_pre_transformer_layer_name;
    string _vision_blocks_name;
    string vision_post_norm_name;
    string vision_head_name;

    string _text_embd_name;
    string _audio_embd_name;
    string _token_embedding_name;
    string _pos_embed_name;
    string _helper_pos_embed_name;
    string _text_blocks_name;
    string text_post_norm_name;
    string text_head_name;
    string _audio_blocks_name;
    string audio_post_norm_name;
    string audio_head_name;

    void init() {
        _qkv_proj_name = "in_proj";
        _bias_k_name = "bias_k";
        _bias_v_name = "bias_v";
        _o_proj_name = "out_proj";
        _up_proj_name = "fc1";
        _down_proj_name = "fc2";
        _attn_norm_name = "norm_1";
        _ffn_norm_name = "norm_2";
        _attn_base_name = "attn.";
        _ffn_base_name = "mlp.";

        _vision_embd_name = "modality_preprocessors.vision.";
        _patch_embedding_name = "rgbt_stem.proj.1";
        _norm_layer_name = "rgbt_stem.norm_layer";
        _rgbt_stem_name = "rgbt_stem.proj";
        _cls_token_name = "cls_token";
        _vision_pos_embed_name = "pos_embedding_helper.pos_embed";

        vision_pre_transformer_layer_name = "modality_trunks.vision.pre_transformer_layer.0";
        vision_post_norm_name = "modality_heads.vision.0";
        vision_head_name = "modality_heads.vision.2";
        _vision_blocks_name = "modality_trunks.vision.blocks.";

        _text_embd_name = "modality_preprocessors.text.";
        _token_embedding_name = "token_embedding";
        _pos_embed_name = "pos_embed";
        _helper_pos_embed_name = "pos_embedding_helper.pos_embed";

        text_post_norm_name = "modality_heads.text.proj.0";
        text_head_name = "modality_heads.text.proj.1";
        _text_blocks_name = "modality_trunks.text.blocks.";

        _audio_embd_name = "modality_preprocessors.audio.";

        audio_post_norm_name = "modality_heads.audio.0";
        audio_head_name = "modality_heads.audio.2";
        _audio_blocks_name = "modality_trunks.audio.blocks.";
    }
};
class ImagebindConfig : public TransformerConfig {
public:
    ImagebindNameConfig names_config;

    int vision_hidden_dim = 1280;
    int vision_head_size = 16;
    int vision_ffn_hidden = 5120;
    int patch = 14;
    int patch_time = 2;
    int img_hw = 224;
    int vision_block_num = 32;

    int text_hidden_dim = 1024;
    int text_head_size = 16;
    int text_ffn_hidden = 4096;
    int vocab_size = 49408;
    int max_position_embeddings = 77;
    int text_block_num = 24;

    int audio_hidden_dim = 768;
    int audio_head_size = 12;
    int audio_ffn_hidden = 3072;
    int audio_kernal = 16;
    int audio_stride = 10;
    int audio_h = 128;
    int audio_w = 204;
    int audio_block_num = 12;

    int head_hidden_dim = 1024;

    explicit ImagebindConfig(const string &model_type = "huge") {
        if (model_type != "huge") {
            MLLM_LOG_ERROR_STREAM << "model type not supported" << std::endl;
        }
        names_config.init();
    }
};

#endif // CONFIGURATION_IMAGEBIND_HPP
