//
// Created by Rongjie Yi on 2024/1/26 0026.
//

#include <iostream>
#include "cmdline.h"
#include "models/llama/modeling_elastic_llama.hpp"
#include "models/llama/tokenization_llama.hpp"
#include "processor/PostProcess.hpp"

using namespace mllm;

int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/llama2_vocab.mllm");
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
    auto model = ElasticLLaMAModel(config);
    model.load(model_path);

    vector<string> in_strs = {
        " Hello, who are you?",
        " What can you do?",
        "Please introduce Beijing University of Posts and Telecommunications."};

    for (int i = 0; i < in_strs.size(); ++i) {
        auto in_str = in_strs[i];
        auto input_tensor = tokenizer.tokenize(in_str);
        std::cout << "[Q] " << in_str << std::endl;
        std::cout << "[A] " << std::flush;
        for (int step = 0; step < 100; step++) {
            float ratio = 1.0; // 0.25; //0.5;
            vector<vector<int>> activate_dims = {
                {(int)(32 * ratio), (int)(11008 * ratio)}, // 0
                {(int)(32 * ratio), (int)(11008 * ratio)}, // 1
                {(int)(32 * ratio), (int)(11008 * ratio)}, // 2
                {(int)(32 * ratio), (int)(11008 * ratio)}, // 3
                {(int)(32 * ratio), (int)(11008 * ratio)}, // 4
                {(int)(32 * ratio), (int)(11008 * ratio)}, // 5
                {(int)(32 * ratio), (int)(11008 * ratio)}, // 6
                {(int)(32 * ratio), (int)(11008 * ratio)}, // 7
                {(int)(32 * ratio), (int)(11008 * ratio)}, // 8
                {(int)(32 * ratio), (int)(11008 * ratio)}, // 9
                {(int)(32 * ratio), (int)(11008 * ratio)}, // 10
                {(int)(32 * ratio), (int)(11008 * ratio)}, // 11
                {(int)(32 * ratio), (int)(11008 * ratio)}, // 12
                {(int)(32 * ratio), (int)(11008 * ratio)}, // 13
                {(int)(32 * ratio), (int)(11008 * ratio)}, // 14
                {(int)(32 * ratio), (int)(11008 * ratio)}, // 15
                {(int)(32 * ratio), (int)(11008 * ratio)}, // 16
                {(int)(32 * ratio), (int)(11008 * ratio)}, // 17
                {(int)(32 * ratio), (int)(11008 * ratio)}, // 18
                {(int)(32 * ratio), (int)(11008 * ratio)}, // 19
                {(int)(32 * ratio), (int)(11008 * ratio)}, // 20
                {(int)(32 * ratio), (int)(11008 * ratio)}, // 21
                {(int)(32 * ratio), (int)(11008 * ratio)}, // 22
                {(int)(32 * ratio), (int)(11008 * ratio)}, // 23
                {(int)(32 * ratio), (int)(11008 * ratio)}, // 24
                {(int)(32 * ratio), (int)(11008 * ratio)}, // 25
                {(int)(32 * ratio), (int)(11008 * ratio)}, // 26
                {(int)(32 * ratio), (int)(11008 * ratio)}, // 27
                {(int)(32 * ratio), (int)(11008 * ratio)}, // 28
                {(int)(32 * ratio), (int)(11008 * ratio)}, // 29
                {(int)(32 * ratio), (int)(11008 * ratio)}, // 30
                {(int)(32 * ratio), (int)(11008 * ratio)}  // 31
            };
            auto result = model({input_tensor}, activate_dims);
            auto [out_string, out_token] = tokenizer.detokenize(result[0]);
            auto [not_end, output_string] = tokenizer.postprocess(out_string);
            if (!not_end) { break; }
            std::cout << output_string << std::flush;
            chatPostProcessing(out_token, input_tensor, {});
        }
        printf("\n");
        model.clear_kvcache();
    }

    return 0;
}