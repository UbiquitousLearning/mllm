#include "cmdline.h"
#include "models/qwen/configuration_qwen.hpp"
#include "models/qwen/modeling_qwen_npu.hpp"
#include "models/qwen/tokenization_qwen.hpp"

using namespace mllm;

int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/qwen_vocab.mllm");
    cmdParser.add<string>("merge", 'e', "specify mllm merge file path", false, "../vocab/qwen_merges.txt");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/qwen-1.5-1.8b-q8_0.mllm");
    cmdParser.add<string>("billion", 'b', "[0.5B | 1.8B]", false, "1.8B");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 400);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string merge_path = cmdParser.get<string>("merge");
    string model_path = cmdParser.get<string>("model");
    string model_billion = cmdParser.get<string>("billion");
    int tokens_limit = cmdParser.get<int>("limits");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    auto tokenizer = QWenTokenizer(vocab_path, merge_path);
    QWenConfig config(tokens_limit, model_billion, RoPEType::HFHUBROPE);
    auto model = QWenForCausalLM(config);
    model.load(model_path);

    vector<string> in_strs = {
        " Give me a short introduction to large language model.",
    };

    auto processOutput = [&](std::string &text) -> std::pair<bool, std::string> {
        if (text == "<|im_start|>" || text == "<|im_end|>" || text == "<unk>") return {true, ""};
        if (text == "<|endoftext|>") return {false, ""};
        return {true, text};
    };

    auto addSystemPrompt = [](const std::string &text) -> std::string {
        std::string ret;
        std::string pre = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n";
        ret = pre + text;
        std::string end = "<|im_end|>\n<|im_start|>assistant\n";
        ret = ret + end;
        return ret;
    };

    for (int i = 0; i < in_strs.size(); ++i) {
        auto input_str = addSystemPrompt(in_strs[i]);
        auto input_tensor = tokenizer.tokenize(input_str, i);
        std::cout << "[Q] " << in_strs[i] << std::endl;
        std::cout << "[A] " << std::flush;

        LlmTextGeneratorOpts opt{
            .max_new_tokens = 100,
            .do_sample = true,
            .temperature = 0.3f,
            .top_k = 50,
            .top_p = 0.f,
        };
        model.generate(input_tensor, opt, [&](unsigned int out_token) -> bool {
            auto out_string = tokenizer.detokenize({out_token});
            auto [isOk, print_string] = processOutput(out_string);
            if (isOk) {
                std::cout << print_string << std::flush;
            } else {
                return false;
            }
            return true;
        });
        std::cout << "FINISH\n";
    }
}