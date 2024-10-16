//
// Created by shrelic on 24-4-27.
//

#include <iostream>
#include "cmdline.h"
#include "models/llama/tokenization_llama.hpp"
#include "processor/PostProcess.hpp"
#include "models/llama/modeling_sparse_llama.hpp"

using namespace mllm;

int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/llama2_vocab.mllm");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/ReLULlama_sparse_q4_k.mllm");
    // cmdParser.add<string>("predictor", 'p', "specify mllm model predictor path", false, "../models/ReLULlama_predictor.mllm");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 600);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string model_path = cmdParser.get<string>("model");
    int tokens_limit = cmdParser.get<int>("limits");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");
    // string predictor_path = cmdParser.get<string>("predictor");

    auto tokenizer = LLaMATokenizer(vocab_path);

    LLaMAConfig config(tokens_limit, "7B", HFHUBROPE);
    auto is_down_sparse = true;
    auto model = SparseLLaMAModel(config, is_down_sparse);
    // MultiFileParamLoader param_loader({model_path, predictor_path, "../ReLULlama_q4_k.mllm"});
    MultiFileParamLoader param_loader({model_path, "../models/ReLULlama_q4_k.mllm"});
    model.load(param_loader);

    vector<string> in_strs = {
        " Hello, who are you?",
        " What can you do?",
        "Please introduce Beijing University of Posts and Telecommunications."};

    for (int i = 0; i < in_strs.size(); ++i) {
        auto in_str = in_strs[i];
        auto input_tensor = tokenizer.tokenize(in_str, i);
        std::cout << "[Q] " << in_str << std::endl;
        std::cout << "[A] " << std::flush;
        for (int step = 0; step < 100; step++) {
            auto result = model({input_tensor});
            auto outputs = tokenizer.detokenize(result[0]);
            auto out_string = outputs.first;
            auto out_token = outputs.second;
            if (out_token == 2) {
                break;
            }
            std::cout << out_string << std::flush;
            chatPostProcessing(out_token, input_tensor, {});
        }
        printf("\n");
    }

    return 0;
}