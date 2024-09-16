#include "cmdline.h"
#include "models/minicpm/configuration_minicpm.hpp"
#include "models/minicpm/modeling_minicpm.hpp"
#include "models/minicpm/tokenization_minicpm.hpp"
#include "processor/PostProcess.hpp"

using namespace mllm;

int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/minicpm_vocab.mllm");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/minicpm-2b-dpo-q4_k.mllm");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 400);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string model_path = cmdParser.get<string>("model");
    int tokens_limit = cmdParser.get<int>("limits");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    auto tokenizer = MiniCPMTokenizer(vocab_path, "../vocab/minicpm_merges.txt");
    MiniCPMConfig config(tokens_limit, "2B");
    auto model = MiniCPMForCausalLM(config);
    model.load(model_path);

    vector<string> in_strs = {
        "Hello, who are you?",
        "山东省最高的山是哪座山, 它比黄山高还是矮？差距多少？",
        "Please introduce Beijing University of Posts and Telecommunications.",
    };

    string system_prompt_start = tokenizer.token_user_o;
    string system_prompt_end = tokenizer.token_user_c;

    auto processOutput = [&](unsigned int id, std::string &text) -> std::pair<bool, std::string> {
        text = std::regex_replace(text, std::regex("▁"), " ");
        if (text == "<0x0A>") return {true, "\n"};
        if (text == "</s>") return {false, ""};
        if (id == 2) return {false, ""};
        return {true, text};
    };

    for (int i = 0; i < in_strs.size(); ++i) {
        auto in_str_origin = in_strs[i];
        auto in_str = system_prompt_start + in_str_origin + system_prompt_end;
        auto input_tensor = tokenizer.tokenize(in_str);
        std::cout << "[Q] " << in_str << std::endl;
        std::cout << "[A] " << std::flush;
        for (int step = 0; step < 100; step++) {
            auto result = model({input_tensor});
            auto outputs = tokenizer.detokenize(result[0]);
            auto out_string = outputs.first;
            auto out_token = outputs.second;
            auto [isOk, print_string] = processOutput(out_token, out_string);
            if (isOk) {
                std::cout << print_string << std::flush;
            } else {
                break;
            }
            chatPostProcessing(out_token, input_tensor, {});
        }
        printf("\n");
        model.clear_kvcache();
    }
}