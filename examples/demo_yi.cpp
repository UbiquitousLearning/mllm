/**
 * @file demo_yi.cpp
 * @author Chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-07-02
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "cmdline.h"
#include "models/llama/configuration_llama.hpp"
#include "models/llama/modeling_llama.hpp"
#include "models/llama/tokenization_llama.hpp"
#include "processor/PostProcess.hpp"

using namespace mllm;

int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/yi_vocab.mllm");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/yi-1.5-6b-chat-q4_k.mllm");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 400);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string model_path = cmdParser.get<string>("model");
    int tokens_limit = cmdParser.get<int>("limits");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    auto tokenizer = LLaMATokenizer(vocab_path, false);
    LLaMAConfig config(tokens_limit, "6B", RoPEType::HFHUBROPE, 64000);
    auto model = LLaMAModel(config);
    model.load(model_path);

    vector<string> in_strs = {
        "请介绍北京邮电大学，推荐同学们报考。",
    };

    auto processOutput = [&](std::string &text) -> std::pair<bool, std::string> {
        text = std::regex_replace(text, std::regex("▁"), " ");
        if (text == "<|endoftext|>" || text == "<|im_end|>") return {false, ""};
        return {true, text};
    };

    for (int i = 0; i < in_strs.size(); ++i) {
        auto in_str = in_strs[i];
        std::cout << "[Q] " << in_str << std::endl;
        auto input_tensor = tokenizer.tokenize(in_str, i);
        std::cout << "[A] " << std::flush;
        for (int step = 0; step < 1000; step++) {
            auto result = model({input_tensor});
            auto outputs = tokenizer.detokenize(result[0]);
            auto out_string = outputs.first;
            auto out_token = outputs.second;
            auto [isOk, print_string] = processOutput(out_string);
            if (isOk) {
                std::cout << print_string << std::flush;
            } else {
                break;
            }
            chatPostProcessing(out_token, input_tensor, {});
        }
        printf("\n");
    }
}
