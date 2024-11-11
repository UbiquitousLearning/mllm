//
// Created by Guo Xiaoqiang on 2024/8/12 .
//
#ifndef CONFIG_PHI3V_HPP
#define CONFIG_PHI3V_HPP
#include "models/phi3/configuration_phi3.hpp"
#include "models/vit/configuration_vit.hpp"

using namespace mllm;

class Phi3VNameConfig : public ViTNameConfig {
public:
    // string vision_name = "model.vision_embed_tokens.";
    string _GN = "glb.GN";
    string _vision_model_prefix = "model.vision_embed_tokens.";
    string _vision_model = "img_processor.vision_model";
    string _projection = "img_projection";
    void init_phi3V() {
        init("phi3v");
        vison_model_name = _vision_model_prefix + _vision_model;
    }
};

class Phi3VConfig : public Phi3Config {
public:
    string embed_layer;
    int img_dim;
    string projection_cls;
    Phi3VNameConfig vision_model_config;
    Phi3NameConfig text_model_config;

    Phi3VConfig(int token_limit, string billions = "3.8B", RoPEType type = HFHUBROPE, int vocab = 32064, string embed_modal = "default", string project_cls = "Linear",int imgdim = 1024) :
    Phi3Config(token_limit, billions, type, vocab) {
    // names_config.init(type);
        embed_layer = embed_modal;
        projection_cls = project_cls;
        img_dim = imgdim;
        vision_model_config.init_phi3V();
    }
};

#endif // CONFIG_PHI3V_HPP
