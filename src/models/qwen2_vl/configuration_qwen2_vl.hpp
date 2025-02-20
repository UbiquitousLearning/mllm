//
// Created by Rongjie Yi on 25-2-9.
//
#ifndef CONFIG_PHI3V_HPP
#define CONFIG_PHI3V_HPP
#include "models/qwen/configuration_qwen.hpp"
#include "models/vit/configuration_vit.hpp"
// #include <vector>

using namespace mllm;

class Qwen2VLNameConfig : public ViTNameConfig {
    public:
        // string token_embd_name = "model.embed_tokens";
        string patch_embed_name = ".patch_embed"; //
        string _merger_name = ".merger"; //
        string _ln_q_name = ".ln_q"; //
        string _m_mlp_0_name = ".mlp.0"; //
        string _m_mlp_2_name = ".mlp.2"; //
        void init_qwen2vl() {
            vison_model_name = "visual"; //
            _patch_embedding_name = ".proj"; //
            _layer_name = ".blocks."; //
            _attn_base_name = "attn."; //
            _ffn_base_name = "mlp."; //
            _qkv_proj_name = "qkv"; //
            _o_proj_name = "proj"; //
            _up_proj_name = "fc1"; //
            _down_proj_name = "fc2"; //
            _attn_norm_name = "norm1"; //
            _ffn_norm_name = "norm2"; //
        }
};

class Qwen2VLConfig : public QWenConfig {
public:
    int vision_embed_dim;
    int spatial_merge_size= 2;
    string projection_cls;
    
    int bos_token_id = 151643;
    int eos_token_id = 151645;
    int vision_start_token_id = 151652;
    int vision_end_token_id = 151653;
    int vision_token_id = 151654;
    int image_token_id = 151655;
    int video_token_id = 151656;
    vector<int> mrope_section = {16, 24, 24};

    Qwen2VLNameConfig vision_names_config;
    Qwen2VLConfig(int token_limit, string billions = "1.5b", RoPEType type = HFHUBROPE, int vocab = 32064, string project_cls = "MLP") :
        QWenConfig(token_limit, billions, type) {
        // names_config.init(type);
        projection_cls = project_cls;
        hidden_size = 1536;
        vision_embed_dim = 1280;
        vision_names_config.init_qwen2vl();
    }
};

#endif // CONFIG_PHI3V_HPP
