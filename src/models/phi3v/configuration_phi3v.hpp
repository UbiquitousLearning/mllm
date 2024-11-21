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
    string _glb_GN = "glb_GN";
    string _sub_GN = "sub_GN";
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
    int vision_hidden_dim;
    string projection_cls;
    Phi3VNameConfig name_config;
    Phi3VConfig(int token_limit, string billions = "3.8B", RoPEType type = HFHUBROPE, int vocab = 32064, string project_cls = "MLP") :
        Phi3Config(token_limit, billions, type, vocab) {
        // names_config.init(type);
        projection_cls = project_cls;
        vision_hidden_dim = 1024;
        name_config.init_phi3V();
    }
};

#endif // CONFIG_PHI3V_HPP
