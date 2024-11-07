#include <iostream>
#include <vector>
#include "Types.hpp"
#include "cmdline.h"
#include "models/phonelm/modeling_phonelm.hpp"
#include "models/smollm/tokenization_smollm.hpp"

using namespace mllm;

int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/phonelm_vocab.mllm");
    cmdParser.add<string>("merge", 'e', "specify mllm merge file path", false, "../vocab/phonelm_merges.txt");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/phonelm-1.5b-instruct-q4_0_4_4.mllm");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 400);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string merge_path = cmdParser.get<string>("merge");
    string vocab_path = cmdParser.get<string>("vocab");
    string model_path = cmdParser.get<string>("model");
    int tokens_limit = cmdParser.get<int>("limits");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    auto tokenizer = SmolLMTokenizer(vocab_path, merge_path);

    string system_prompt_start;
    string system_prompt_end;

    PhoneLMConfig config(tokens_limit, "1.5B");
    auto model = PhoneLMForCausalLM(config);
    model.load(model_path);

    vector<string> in_strs = {
        "Hello, who are you?",
        "What can you do?",
        "Please introduce Beijing University of Posts and Telecommunications.",
    };

    for (int i = 0; i < in_strs.size(); ++i) {
        auto input_str = tokenizer.apply_chat_template(in_strs[i]);
        auto input_tensor = tokenizer.tokenize(input_str);
        std::cout << "[Q] " << in_strs[i] << std::endl;
        std::cout << "[A] " << std::flush;
        LlmTextGeneratorOpts opt{
            .max_new_tokens = 100,
            .do_sample = false,
        };
        model.generate(input_tensor, opt, [&](unsigned int out_token) -> bool {
            auto out_string = tokenizer.detokenize({out_token});
            auto [not_end, output_string] = tokenizer.postprocess(out_string);
            if (!not_end) { return false; }
            std::cout << output_string << std::flush;
            return true;
        });
        model.clear_kvcache();
        std::cout << "\n";
    }
    return 0;
}