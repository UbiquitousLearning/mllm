//
// Created by Rongjie Yi on 24-3-7.
//

#ifndef CONFIGURATION_TINYLLAMA_HPP
#define CONFIGURATION_TINYLLAMA_HPP
#include "models/llama/configuration_llama.hpp"

using namespace mllm;

class TinyLLaMAConfig : public TransformerConfig {
public:
    int vocab_size{};
    int hidden_dim{};
    int head_size{};
    int kv_head_size{};
    int ffn_hidden{};
    int block_num{};
    RoPEType RoPE_type;
    int cache_limit{};
    LLaMANameConfig names_config;
    float rope_theta;
    int max_position_embeddings;

    explicit TinyLLaMAConfig(int token_limit, string billions = "1.5B", RoPEType type = HFHUBROPE, int vocab = 32000) {
        names_config.init(type);
        vocab_size = vocab;
        if (billions == "1.5B" || billions == "1.5b") {
            hidden_dim = 2048;
            head_size = 32;
            kv_head_size = 4;
            ffn_hidden = 5632;
            block_num = 22;
            max_position_embeddings = 16384;
            rope_theta = 10000;
        } else if (billions == "1.1B" || billions == "1.1b") {
            hidden_dim = 2048;
            head_size = 32;
            kv_head_size = 4;
            ffn_hidden = 5632;
            block_num = 22;
            max_position_embeddings = 16384;
            rope_theta = 10000;
        } else {
            throw std::runtime_error("Unsupported model size");
        }
        RoPE_type = type;
        cache_limit = token_limit;
    }
};

#endif // CONFIGURATION_TINYLLAMA_HPP
