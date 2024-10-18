//
// Created by Rongjie Yi on 2024/1/26 0026.
//

#include <iostream>
#include "cmdline.h"
#include "models/fuyu/modeling_fuyu.hpp"
#include "models/fuyu/processing_fuyu.hpp"
#include "processor/PostProcess.hpp"

using namespace mllm;

int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/fuyu_vocab.mllm");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/fuyu-8b-q4_k.mllm");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 500);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string model_path = cmdParser.get<string>("model");
    int tokens_limit = cmdParser.get<int>("limits");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    auto processor = FuyuProcessor(vocab_path);

    FuyuConfig config(tokens_limit, "8B");
    auto model = FuyuModel(config);
    model.load(model_path);

    std::vector<vector<string>> in_imgs = {
        {"../assets/bus.png"},
        {"../assets/two_cats.jpg"}};
    vector<string> in_strs = {
        "Generate a coco-style caption.\n",
        "What's this?\n"};

    for (int inId = 0; inId < in_strs.size(); ++inId) {
        auto in_str = in_strs[inId];
        auto in_img = in_imgs[inId];
        auto input_tensors = processor.process(in_str, in_img);
        std::cout << "[Q] [";
        if (!in_img.empty()) {
            std::cout << in_img[0];
        }
        std::cout << "]" << in_str << std::endl;
        std::cout << "[A] " << std::flush;
        for (int step = 0; step < 20; step++) {
            auto result = model({input_tensors[0], input_tensors[1], input_tensors[2]});
            auto outputs = processor.detokenize(result[0]);
            auto out_string = outputs.first;
            auto out_token = outputs.second;
            auto [end, string] = processor.postprocess(out_string);
            if (!end) { break; }
            std::cout << string << std::flush;
            chatPostProcessing(out_token, input_tensors[0], {&input_tensors[1], &input_tensors[2]});
        }
        printf("\n");
    }

    return 0;
}