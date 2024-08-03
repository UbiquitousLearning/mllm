//
// Created by Rongjie Yi on 2024/1/26 0026.
//

#include <iostream>
#include "cmdline.h"
#include "models/llama/modeling_llama.hpp"
#include "models/llama/tokenization_llama.hpp"
#include "processor/PostProcess.hpp"

using namespace mllm;

int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/llama_vocab.mllm");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/llama-2-7b-chat-q4_0_4_4.mllm");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 400);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string model_path = cmdParser.get<string>("model");
    int tokens_limit = cmdParser.get<int>("limits");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    auto tokenizer = LLaMATokenizer(vocab_path);

    LLaMAConfig config(tokens_limit, "7B", LLAMAROPE);
    auto model = LLaMAModel(config);
    model.load(model_path);

    vector<string> in_strs = {
        " Hello, who are you?",
        " What can you do?",
        "Please introduce Beijing University of Posts and Telecommunications."
        };

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
        model.clear_kvcache();
        model.profiling();
    }

    return 0;
}