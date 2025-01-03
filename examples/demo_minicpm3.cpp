//
// Created by Rongjie Yi on 2024/6/26 0026.
//

#include "Types.hpp"
#include "cmdline.h"
#include "models/minicpm/tokenization_minicpm.hpp"
#include "models/minicpm3/modeling_minicpm3.hpp"
#include "processor/PostProcess.hpp"

using namespace mllm;

int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/minicpm_vocab.mllm");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/minicpm3-4b-fp32.mllm");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 400);
    cmdParser.add<int>("thread", 't', "num of threads", false, 1);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string model_path = cmdParser.get<string>("model");
    int tokens_limit = cmdParser.get<int>("limits");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    auto tokenizer = MiniCPMTokenizer(vocab_path, "../vocab/minicpm_merges.txt");

    MiniCPM3Config config(tokens_limit);
    auto model = MiniCPM3ForCausalLM(config);
    model.load(model_path);

    vector<string> in_strs = {
        " Hello, who are you?",
        " What can you do?",
        "Please introduce Beijing University of Posts and Telecommunications."};

    for (int i = 0; i < in_strs.size(); ++i) {
        auto in_str = in_strs[i];
        auto input_tensor = tokenizer.tokenize(in_str);
        input_tensor.saveData<float>();
        std::cout << "[Q] " << in_str << std::flush;
        for (int step = 0; step < 1; step++) {
            auto result = model({input_tensor});
            auto [out_string, out_token] = tokenizer.detokenize(result[0]);
            auto [not_end, output_string] = tokenizer.postprocess(out_string);
            if (!not_end) { break; }
            std::cout << output_string << std::flush;
            chatPostProcessing(out_token, input_tensor, {});
        }
        printf("\n");
    }

    return 0;
}