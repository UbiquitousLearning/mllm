/**
 * @file demo_dclm.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2024-09-26
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "cmdline.h"
#include "models/dclm/configuration_dclm.hpp"
#include "models/dclm/modeling_dclm.hpp"
#include "models/dclm/tokenization_dclm.hpp"
#include "processor/PostProcess.hpp"

using namespace mllm;

int main(int argc, char **argv) {
    std::iostream::sync_with_stdio(false);

    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/dclm_vocab.mllm");
    cmdParser.add<string>("merge", 'e', "specify mllm merge file path", false, "../vocab/dclm_merges.txt");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/dclm-1b-fp32.mllm");
    cmdParser.add<string>("billion", 'b', "[1B]", false, "1B");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 400);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string model_path = cmdParser.get<string>("model");
    string merge_path = cmdParser.get<string>("merge");
    string model_billion = cmdParser.get<string>("billion");
    int tokens_limit = cmdParser.get<int>("limits");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    auto tokenizer = DCLMTokenizer(vocab_path, merge_path);
    DCLMConfig config(tokens_limit, model_billion, RoPEType::HFHUBROPE);
    auto model = DCLM(config);
    model.load(model_path);

    vector<string> in_strs = {
        "Machine learning is",
    };

    for (int i = 0; i < in_strs.size(); ++i) {
        auto in_str = in_strs[i];
        std::cout << in_str << std::flush;
        auto input_tensor = tokenizer.tokenize(in_str);
        for (int step = 0; step < 100; step++) {
            auto result = model({input_tensor});
            auto [out_string, out_token] = tokenizer.detokenize(result[0]);
            auto [not_end, output_string] = tokenizer.postprocess(out_string);
            std::cout << output_string << std::flush;
            chatPostProcessing(out_token, input_tensor, {});
        }
        printf("\n");
    }
}