/**
 * @file demo_openelm.cpp
 * @author chenghua.wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2024-09-25
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "cmdline.h"
#include "models/openelm/configuration_openelm.hpp"
#include "models/openelm/modeling_openelm.hpp"
#include "models/llama/tokenization_llama.hpp"

using namespace mllm;

int main(int argc, char **argv) {
    std::iostream::sync_with_stdio(false);

    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/llama2_hf_vocab.mllm");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/openelm-1.1b-Instruct-q4_k.mllm");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 400);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string model_path = cmdParser.get<string>("model");
    int tokens_limit = cmdParser.get<int>("limits");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    auto tokenizer = LLaMATokenizer(vocab_path);
    OpenELMConfig config(tokens_limit, "1.1B", RoPEType::HFHUBROPE);
    auto model = OpenElMModel(config);
    model.load(model_path);

    vector<string> in_strs = {
        "Hello, who are you?",
        "What can you do?",
        "Please introduce Beijing University of Posts and Telecommunications.",
    };

    auto processOutput = [&](std::string &text, unsigned int &out_token) -> std::pair<bool, std::string> {
        size_t pos = 0;
        std::string from = "▁";
        std::string to = " ";
        while ((pos = text.find(from, pos)) != std::string::npos) {
            text.replace(pos, from.length(), to);
            pos += to.length();
        }
        if (out_token == 2) return {false, ""};
        return {true, text};
    };

    auto addSystemPrompt = [](const std::string &text) -> std::string {
        std::string ret;
        std::string pre =
            "<s>[INST] ";
        ret = pre + text;
        std::string end = " [/INST]";
        ret = ret + end;
        return ret;
    };

    for (int i = 0; i < in_strs.size(); ++i) {
        auto input_str = addSystemPrompt(in_strs[i]);
        auto input_tensor = tokenizer.tokenize(input_str);
        std::cout << "[Q] " << in_strs[i] << std::endl;
        std::cout << "[A] " << std::flush;

        LlmTextGeneratorOpts opt{
            .max_new_tokens = 100,
            .do_sample = true,
            .temperature = 0.3F,
            .top_k = 50,
            .top_p = 0.F,
        };
        model.generate(input_tensor, opt, [&](unsigned int out_token) -> bool {
            auto out_string = tokenizer.detokenize({out_token});
            auto [isOk, print_string] = processOutput(out_string, out_token);
            if (isOk) {
                std::cout << print_string << std::flush;
            } else {
                return false;
            }
            return true;
        });
        std::cout << "\n";
    }
}