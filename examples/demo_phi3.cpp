#include <iostream>
#include "cmdline.h"
#include "models/phi3/modeling_phi3.hpp"
#include "models/phi3/tokenization_phi3.hpp"
#include "processor/PostProcess.hpp"

using namespace mllm;

int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/phi3_vocab.mllm");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/phi-3-mini-instruct-q4_k.mllm");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 400);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string model_path = cmdParser.get<string>("model");
    int tokens_limit = cmdParser.get<int>("limits");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    auto tokenizer = Phi3Tokenizer(vocab_path);

    Phi3Config config(tokens_limit, "3.8B", HFHUBROPE);
    auto model = Phi3Model(config);
    model.load(model_path);

    vector<string> in_strs = {
        "who are you?",
        "What can you do?",
        "Please introduce Beijing University of Posts and Telecommunications."};

    for (int i = 0; i < in_strs.size(); ++i) {
        auto in_str_origin = in_strs[i];
        auto in_str = tokenizer.apply_chat_template(in_str_origin);
        auto input_tensor = tokenizer.tokenize(in_str);
        std::cout << "[Q] " << in_str_origin << std::endl;
        std::cout << "[A] " << std::flush;
        for (int step = 0; step < 100; step++) {
            auto result = model({input_tensor});
            auto [out_string, out_token] = tokenizer.detokenize(result[0]);
            auto [not_end, output_string] = tokenizer.postprocess(out_string);
            if (!not_end) { break; }
            std::cout << output_string << std::flush;
            chatPostProcessing(out_token, input_tensor, {});
        }
        printf("\n");
        model.clear_kvcache();
        model.profiling();
    }

    return 0;
}