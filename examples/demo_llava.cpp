//
// Created by Rongjie Yi on 24-3-7.
//

#ifndef DEMO_LLAVA_HPP
#define DEMO_LLAVA_HPP
#include <iostream>
#include "cmdline.h"
#include "models/llava/modeling_llava.hpp"
#include "models/llava/processing_llava.hpp"
#include "processor/PostProcess.hpp"

using namespace mllm;

int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/llava_vocab.mllm");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/llava-1.5-7b-q4_k.mllm");
    cmdParser.add<string>("merges", 'f', "specify mllm tokenizer merges.txt path", false, "../vocab/llava_merges.txt");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 700);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);

    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string model_path = cmdParser.get<string>("model");
    string merges_path = cmdParser.get<string>("merges");
    int tokens_limit = cmdParser.get<int>("limits");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    auto processor = LLaVAProcessor(vocab_path, merges_path);

    LLaVAConfig config(tokens_limit, "7B", 32064);
    auto model = LLaVAModel(config);
    model.load(model_path);

    vector<string> in_imgs = {
        "../assets/australia.jpg"};
    vector<string> in_strs = {
        "<image>\nUSER: What's the content of the image?\nASSISTANT:"};

    for (int i = 0; i < in_strs.size(); ++i) {
        auto in_str = in_strs[i];
        auto input_tensors = processor.process(in_str, in_imgs[i], 336);
        std::cout << in_str << std::flush;
        for (int step = 0; step < 100; step++) {
            auto result = model({input_tensors[0], input_tensors[1]});
            auto outputs = processor.detokenize(result[0]);
            auto out_string = outputs.first;
            auto out_token = outputs.second;
            auto [isOk, print_string] = processor.postprocess(out_string);
            if (isOk) {
                std::cout << print_string << std::flush;
            } else {
                break;
            }
            chatPostProcessing(out_token, input_tensors[0], {&input_tensors[1]});
        }
        printf("\n");
    }
}
#endif // DEMO_LLAVA_HPP
