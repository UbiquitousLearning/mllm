#include <iostream>
#include "cmdline.h"
#include "models/clip/modeling_clip.hpp"
#include "models/clip/processing_clip.hpp"

using namespace mllm;

int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/clip_vocab.mllm");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/clip-vit-base-patch32-q4_k.mllm");
    cmdParser.add<string>("merges", 'f', "specify mllm tokenizer merges.txt path", false, "../vocab/clip_merges.txt");
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string model_path = cmdParser.get<string>("model");
    string merges_path = cmdParser.get<string>("merges");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    auto processor = ClipProcessor(vocab_path, merges_path);

    ClipConfig config("base", 32, 224, 49408);
    auto model = CLipModel(config);
    model.load(model_path);

    auto input_tensors = processor.process({"a photo of a cat", "a photo of a dog"}, "../assets/cat.jpg", 224);
    auto result = model({input_tensors[0], input_tensors[1]});
    auto token_idx = processor.postProcess(result[0]);
    for (auto prob : token_idx) {
        std::cout << prob << "  ";
    }
    std::cout << std::endl;
}