#include <iostream>
#include "cmdline.h"
#include "models/phi3v/configuration_phi3v.hpp"
#include "models/phi3v/modeling_phi3v.hpp"
#include "models/phi3v/processing_phi3v.hpp"
#include "processor/PostProcess.hpp"

using namespace mllm;
int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/phi3v_vocab.mllm");
    // cmdParser.add<string>("merges", 'f', "specify mllm tokenizer merges.txt path", false, "../vocab/phi3v_merges.txt");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/phi-3-vision-instruct-fp32.mllm");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 400);
    cmdParser.add<int>("thread", 't', "num of threads", false, 80);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    // string merge_path = cmdParser.get<string>("merges");
    string model_path = cmdParser.get<string>("model");
    int tokens_limit = cmdParser.get<int>("limits");
    int thread_num = cmdParser.get<int>("thread");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    ParamLoader param_loader(model_path);
    auto processor = Phi3VProcessor(vocab_path);
    Phi3VConfig config(tokens_limit, "3.8B", HFHUBROPE, 32064, "vision", "MLP", 1024);
    auto model_config = Phi3VConfig(config);
    auto model = Phi3VModel(model_config);
    model.load(model_path);

    vector<string> in_imgs = {
        "../assets/australia.jpg"};

    vector<string> in_strs = {
        "<|image_1|>\nWhat's the content of the image?",
    };

    for (int i = 0; i < in_strs.size(); ++i) {
        auto in_str = in_strs[i];
        in_str = processor.tokenizer->apply_chat_template(in_str);

        // 图片版，有问题，参数层的问题疑似解决了
        auto input_tensor = processor.process(in_str, in_imgs[i]);
        // 目前的仅文字版的结果，能正常推理
        // auto input_tensor = processor.process(in_str, "");

        std::cout << "[Q] " << in_strs[i] << std::endl;
        std::cout << "[A] " << std::flush;
        for (int step = 0; step < 1; step++) {
            auto result = model(input_tensor);
            auto outputs = processor.detokenize(result[0]);
            auto out_string = outputs.first;
            auto out_token = outputs.second;
            if (out_token == processor.end_id && step != 0) {
                break;
            }
            std::cout << out_string << std::flush;
            chatPostProcessing(out_token, input_tensor[0], {});
        }
        printf("\n");
    }

    return 0;
}