//
// Created by Rongjie Yi on 2024/03/07 0026.
//

#include <iostream>
#include "cmdline.h"
#include "models/tinyllama/modeling_tinyllama.hpp"
#include "models/llama/tokenization_llama.hpp"
#include "processor/PostProcess.hpp"

using namespace mllm;

int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/tinyllama_vocab.mllm");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/tinyllama-1.1b-chat-q4_k.mllm");
    cmdParser.add<int>("limits", 'l',  "max KV cache size", false, 600);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string model_path = cmdParser.get<string>("model");
    int tokens_limit = cmdParser.get<int>("limits");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    auto tokenizer = LLaMATokenizer(vocab_path);

    TinyLLaMAConfig config(tokens_limit, "1.5B", HFHUBROPE);
    auto model = TinyLLaMAModel(config);
    model.load(model_path);

    string system_prompt_start = " You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided.<|USER|>";
    string system_prompt_end = "<|ASSISTANT|>";

    vector<string> in_strs = {
        "Hello, who are you?",
        "Please introduce Beijing University of Posts and Telecommunications."
    };

    for (int i = 0; i < in_strs.size(); ++i) {
        auto in_str_origin = in_strs[i];
        auto in_str = system_prompt_start + in_str_origin + system_prompt_end;
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