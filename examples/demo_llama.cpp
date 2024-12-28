//
// Created by Rongjie Yi on 2024/1/26 0026.
//

#include <iostream>
#include "cmdline.h"
#include "models/llama/modeling_llama.hpp"
#include "models/llama/tokenization_llama.hpp"
#include "processor/PostProcess.hpp"

using namespace mllm;

LLaMAConfig config(400, "7B", HFHUBROPE);

void llama2_7b(LLaMAConfig& cfg) {
    // do nothing
}

void llama3_2_1b(LLaMAConfig& cfg) {
    cfg.vocab_size = 128256;
    cfg.hidden_dim = 2048;
    cfg.head_size = 32;
    cfg.num_key_value_heads = 8;
    cfg.ffn_hidden = 8192;
    cfg.block_num = 16;
    cfg.max_position_embeddings = 131072;
    cfg.rope_theta = 500000.0;
    cfg.tie_word_embeddings = true;

    cfg.rope_scaling = {
        {"factor", 32.0f},
        {"high_freq_factor", 4.0f},
        {"low_freq_factor", 1.0f},
        {"original_max_position_embeddings", 8192},
        {"rope_type", std::string("llama3")}
    };
}

map<string, void (*)(LLaMAConfig& config)> CONFIG_MAP = {
    {"llama-2-7b", llama2_7b},
    {"llama-3-2-1b", llama3_2_1b}
};

int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/llama2_vocab.mllm");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/llama-2-7b-chat-q4_0_4_4.mllm");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 400);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.add<string>("model_type", 'M', "specify mllm model type", false, "llama-2-7b");
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string model_path = cmdParser.get<string>("model");
    int tokens_limit = cmdParser.get<int>("limits");
    string MODEL_TYPE = cmdParser.get<string>("model_type");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    auto tokenizer = LLaMATokenizer(vocab_path);

    CONFIG_MAP[MODEL_TYPE](config);
    config.cache_limit = tokens_limit;
    auto model = LLaMAModel(config);
    MultiFileParamLoader loader({model_path});
    model.load(loader);
//    model.load(model_path);

    vector<string> in_strs = {
        "My name is",
        "Hello, who are you?",
        "What can you do?",
        "Please introduce Beijing University of Posts and Telecommunications."};

    for (int i = 0; i < in_strs.size(); ++i) {
//        auto in_str = tokenizer.apply_chat_template(in_strs[i]);
        auto in_str = in_strs[i];
        auto input_tensor = tokenizer.tokenize(in_str);
//        auto input_tensor = LLaMATokenizer::tokens2Input({128000, 5159, 836, 374});
        input_tensor.printDataTorchLike<float>();
        std::cout << "\n-----\n" << std::endl;
        std::cout << "[Q] " << in_strs[i] << std::endl;
        std::cout << "[A] " << std::flush;
        for (int step = 0; step < 100; step++) {
            auto result = model({input_tensor});
            auto [out_string, out_token] = tokenizer.detokenize(result[0]);
            auto [not_end, output_string] = tokenizer.postprocess(out_string);
            if (!not_end) { break; }
            std::cout << output_string << std::flush;
            chatPostProcessing(out_token, input_tensor, {});
        }
        printf("\n");
        model.clear_kvcache();
        model.profiling();
    }

    return 0;
}