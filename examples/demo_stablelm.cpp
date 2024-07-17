// ./demo_stablelm -m ../vocab/stablelm_merges.txt -v ../vocab/stablelm_vocab.mllm ../models/stablelm-2-1.6b.mllm

// ./demo_stablelm -m ../vocab/stablelm_merges.txt -v ../vocab/stablelm_vocab.mllm ../models/stablelm-2-1.6b-q4_k.mllm

#include <iostream>
#include "cmdline.h"
#include "models/stablelm/modeling_stablelm.hpp"
#include "models/stablelm/tokenization_stablelm.hpp"
#include "processor/PostProcess.hpp"

using namespace mllm;

int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/stablelm_vocab.mllm");
    cmdParser.add<string>("merge", 'm', "specify mllm merge path", false, "../vocab/stablelm_merges.txt");
    cmdParser.add<string>("model", 'o', "specify mllm model path", false, "../models/stablelm-2-1.6b.mllm");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 400);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string merge_path = cmdParser.get<string>("merge");
    string model_path = cmdParser.get<string>("model");
    int tokens_limit = cmdParser.get<int>("limits");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    auto tokenizer = stablelmTokenizer(vocab_path, merge_path);

    stablelmConfig config(tokens_limit, "1.6B", HFHUBROPE);
    auto model = stablelmModel(config);
    model.load(model_path);

    vector<string> in_strs = {
        " Hello, who are you?",
        " What can you do?",
        "Please introduce Beijing University of Posts and Telecommunications."};

    for (int i = 0; i < in_strs.size(); ++i) {
        auto in_str = in_strs[i];
        std::cout << "[Q] " << in_str << std::endl;
        auto input_tensor = tokenizer.tokenize(in_str, i);
        std::cout << "[A] " << std::flush;
        for (int step = 0; step < 100; step++) {
            auto result = model({input_tensor});
            auto outputs = tokenizer.detokenize(result[0]);
            auto out_string = outputs.first;
            auto out_token = outputs.second;
            if (out_token == 100278) {
                break;
            }

            size_t pos = 0;
            while ((pos = out_string.find("Ċ", pos)) != std::string::npos) {
                out_string.replace(pos, 2, " ");
            }

            pos = 0;

            while ((pos = out_string.find("Ġ", pos)) != std::string::npos) {
                out_string.replace(pos, 2, " ");
            }

            std::cout << out_string << std::flush;
            chatPostProcessing(out_token, input_tensor, {});
        }
        printf("\n");
    }

    return 0;
}