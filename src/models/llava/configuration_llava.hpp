//
// Created by Rongjie Yi on 24-3-7.
//

#ifndef CONFIGURATION_LLAVA_HPP
#define CONFIGURATION_LLAVA_HPP
#include <utility>

#include "models/llama/configuration_llama.hpp"
#include "models/vit/configuration_vit.hpp"

using namespace mllm;

class LLaVAConfig: public LLaMAConfig {
public:
    ViTNameConfig vit_names_config;
    int patch{};
    int img_hw{};
    int vision_hidden_dim{};
    int vision_head_size{};
    int vision_ffn_hidden{};
    int vision_block_num{};

    explicit LLaVAConfig(int token_limit, string billions = "7B", int vocab = 32064): LLaMAConfig(token_limit, std::move(billions), HFHUBROPE, vocab) {
        names_config.init(HFHUBROPE);
        names_config.blk_name = "language_model.model.layers.";
        names_config.token_embd_name = "language_model.model.embed_tokens";
        names_config.post_norm_name = "language_model.model.norm";
        names_config.lm_head_name = "language_model.lm_head";
        vit_names_config.init("clip");
        vit_names_config._cls_token_name = "class_embedding";
        vit_names_config.vison_model_name = "vision_tower.vision_model";
        vision_hidden_dim = 1024;
        vision_head_size = 16;
        vision_ffn_hidden = 4096;
        vision_block_num = 23;
        patch = 14;
        img_hw= 336;
    }
};

#endif // CONFIGURATION_LLAVA_HPP
