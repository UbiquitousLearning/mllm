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
    static void init(const string& model_type = "base", int patch_ = 32, int hw = 224,  int classes = 1000) {
        ViTConfig::init(model_type, patch_, hw, classes,  "QuickGELU", "vision_model");
    }
};

#endif // CONFIG_ViT_HPP
