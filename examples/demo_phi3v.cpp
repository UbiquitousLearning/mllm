#include <iostream>
#include "cmdline.h"
#include "models/phi3v/configuration_phi3v.hpp"
#include "models/phi3v/modeling_phi3v.hpp"
#include "models/phi3v/processing_phi3v.hpp"
#include "processor/PostProcess.hpp"

using namespace mllm;
int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/phi3_vocab.mllm");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/phi-3v.mllm");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 400);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string model_path = cmdParser.get<string>("model");
    int tokens_limit = cmdParser.get<int>("limits");
    int thread_num = cmdParser.get<int>("thread");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");
    ParamLoader param_loader(model_path);

    string merge_path = "test"; // TODO check this
    auto processor = Phi3VProcessor(vocab_path,merge_path);
    Phi3VConfig config(tokens_limit, "3.8B", HFHUBROPE, 32064, "vision", "Linear", 1024);
    auto model_config = Phi3VConfig(config);
    auto model = Phi3VModel(model_config);
    model.load(model_path);

    string system_prompt_start = "<|user|>\n";
    string system_prompt_end = " <|end|>\n<|assistant|>";
    string img_prompt = "<|image|>\n";
    vector<string> in_imgs = {
        "../assets/australia.jpg"};
   
    vector<string> in_strs = {
        "What's the content of the image?",
        // "What can you do?",
        // "Please introduce Beijing University of Posts and Telecommunications."
        };

    for (int i = 0; i < in_strs.size(); ++i) {
        auto in_str_origin = in_strs[i];
        auto in_str = system_prompt_start + img_prompt + in_str_origin + system_prompt_end;
        auto input_tensor = processor.process(in_str, in_imgs[i], 336);

        std::cout << "[Q] " << in_str << std::endl;
        std::cout << "[A] " << std::flush;
        for (int step = 0; step < 100; step++) {
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