#include <iostream>
#include <utility>
#include "cmdline.h"
#include "models/vit/modeling_vit.hpp"
#include "models/vit/labels_vit.hpp"
#include "models/vit/processing_vit.hpp"

using namespace mllm;

int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/vit-base-patch16-224-q4_k.mllm");
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string model_path = cmdParser.get<string>("model");
    Layer::cpu_thread = cmdParser.get<int>("thread");

    auto processor = ViTProcessor();

    ViTConfig config("base", 16, 224, imagenet_id2label.size());
    auto model = ViTModel(config);
    model.load(model_path);


    auto input_tensor = processor.process("../assets/cat.jpg", 224);
    auto result = model({input_tensor});
    auto token_idx = processor.postProcess(result[0]);
    std::cout << imagenet_id2label[token_idx] << std::endl;
}