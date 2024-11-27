#include <iostream>
#include "cmdline.h"
#include "models/stablelm/modeling_stablelm.hpp"
#include "models/stablelm/tokenization_stablelm.hpp"
#include "processor/PostProcess.hpp"

using namespace mllm;

int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/stablelm_vocab.mllm");
    cmdParser.add<string>("merge", 'e', "specify mllm merge path", false, "../vocab/stablelm_merges.txt");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/stablelm-2-1.6b-chat-q4_k.mllm");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 600);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string merge_path = cmdParser.get<string>("merge");
    string model_path = cmdParser.get<string>("model");
    int tokens_limit = cmdParser.get<int>("limits");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    auto tokenizer = StableLMTokenizer(vocab_path, merge_path);

    StableLMConfig config(tokens_limit, "1.6B", HFHUBROPE);
    auto model = StableLMModel(config);
    model.load(model_path);

    vector<string> in_strs = {
        "Hello, who are you?",
        "What can you do?",
        "Please introduce Beijing University of Posts and Telecommunications."};

    for (int i = 0; i < in_strs.size(); ++i) {
        const auto &in_str_origin = in_strs[i];
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
    }

    return 0;
}