/**
 * @file configuration_dclm.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2024-09-26
 *
 * @copyright Copyright (c) 2024
 *
 */
#pragma once

#include "Types.hpp"
#include "models/transformer/configuration_transformer.hpp"

using namespace mllm;

class DCLMNameConfig : public TransformerNameConfig {
public:
    /**
     * @brief DCLM following the hugging face naming method
     *
     * @param type RoPEType
     */
    void init(RoPEType type = RoPEType::HFHUBROPE) {
        // the dclm's params name is quite different than others.
        // pls set name of layers manually.
    }
};

struct DCLMConfig : public TransformerConfig {
    explicit DCLMConfig(int token_limit, const string billions = "1B", RoPEType type = RoPEType::HFHUBROPE, int vocab = 50432) :
        cache_limit(token_limit) {
        names_config.init(type);
        if (!(billions == "1B" || billions == "1b")) {
            throw std::runtime_error("Unsupported model size");
        }
        RoPE_type = type;
        vocab_size = vocab;
    };

    int dim = 2048;
    float moe_capacity_factor = 1.25;
    bool moe_expert_model_parallelism = false;
    float moe_freq = 0.f;
    float moe_loss_weight = 0.1f;
    int moe_top_k = 2;
    bool moe_weight_parallelism = false;
    int n_heads = 16;
    int n_layers = 24;
    float norm_eps = 1e-06;
    int seq_len = 2048;
    bool weight_tying = false;
    int vocab_size = 50432;
    int cache_limit;
    RoPEType RoPE_type = RoPEType::HFHUBROPE;
    DCLMNameConfig names_config;
};