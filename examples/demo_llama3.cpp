//
// Created by xwk on 25-1-10.
//

#include <iostream>
#include "cmdline.h"
#include "models/llama/modeling_llama3.hpp"
#include "models/llama/tokenization_llama.hpp"
#include "processor/PostProcess.hpp"

#include "tokenizers/Tiktoken/tiktoken.hpp"

using namespace mllm;

Llama3Config config(400, HFHUBROPE);

string llama3_2_1b(Llama3Config& cfg) {
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
    return "llama3";
}

string llama3_2_3b(Llama3Config& cfg) {
    cfg.vocab_size = 128256;
    cfg.hidden_dim = 3072;
    cfg.head_size = 24;
    cfg.num_key_value_heads = 8;
    cfg.ffn_hidden = 8192;
    cfg.block_num = 28;
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
    return "llama3";
}

map<string, string (*)(Llama3Config& config)> CONFIG_MAP = {
    {"llama-3-2-1b", llama3_2_1b},
    {"llama-3-2-3b", llama3_2_3b}
};

int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/llama3_tokenizer.model");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/Llama-3.2-1B-Instruct.mllm");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 400);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.add<string>("model_type", 'M', "specify mllm model type", false, "llama-3-2-1b");
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string model_path = cmdParser.get<string>("model");
    int tokens_limit = cmdParser.get<int>("limits");
    string MODEL_TYPE = cmdParser.get<string>("model_type");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    auto tokenizer_type = CONFIG_MAP.at(MODEL_TYPE)(config);

    auto tokenizer = TokenizerFactory::createTokenizer(vocab_path, tokenizer_type);

    config.cache_limit = tokens_limit;
    auto model = Llama3Model(config);
    MultiFileParamLoader loader({model_path});
    model.load(loader);
    //    model.load(model_path);

    vector<string> in_strs = {
        "Hello, who are you?",
        "What can you do?",
        "Please introduce Beijing University of Posts and Telecommunications."};

    for (int i = 0; i < in_strs.size(); ++i) {
        auto in_str = tokenizer->apply_chat_template(in_strs[i]);
        //        auto in_str = in_strs[i];
        auto input_tensor = tokenizer->tokenize(in_str);
        //        auto input_tensor = Llama3Tokenizer::tokens2Input({128000, 5159, 836, 374});
        input_tensor.printDataTorchLike<float>();
        std::cout << "\n-----\n" << std::endl;
        std::cout << "[Q] " << in_strs[i] << std::endl;
        std::cout << "[A] " << std::flush;
        for (int step = 0; step < 100; step++) {
            auto result = model({input_tensor});
            auto [out_string, out_token] = tokenizer->detokenize(result[0]);
            auto [not_end, output_string] = tokenizer->postprocess(out_string);
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