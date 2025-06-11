/**
 * @file demo_qwen_sd.cpp
 * @author Zhiyang Chen (zhiyangchen@stu.pku.edu.cn)
 * @brief
 * @date 2025-3-11
 *
 *
 */
#include "cmdline.h"
#include "models/qwen/configuration_qwen.hpp"
#include "models/qwen/modeling_qwen_sd.hpp"
#include "models/qwen/tokenization_qwen.hpp"

using namespace mllm;

int main(int argc, char **argv) {
    std::iostream::sync_with_stdio(false);

    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/qwen_vocab.mllm");
    cmdParser.add<string>("merge", 'e', "specify mllm merge file path", false, "../vocab/qwen_merges.txt");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/qwen-1.5-1.8b-q8_0.mllm");
    cmdParser.add<string>("billion", 'b', "[0.5B | 1.8B | 1.5B |]", false, "1.8B");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 1000);
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
        "Summarize: Hillary Clinton\u2019s security detail arrived at a suburban Des Moines, Iowa fruit processing company on Tuesday with an added vehicle \u2013 a second Scooby. After her signature oversize black Chevy conversion van dropped her off at Capitol Fruit Company in Norwalk, Iowa, a visually identical GMC van drove up to the building with a nearly identical Secret Service escort vehicle. Both armored vehicles have raised roofs, deep-tinted windows and New York license plates. But while the original van \u2013 the one nicknamed 'Scooby' after the Scooby-Doo cartoon show \u2013 sports a mustard-yellow New York tag, the second has blue and white plates of a different design. Scroll down for video. WHY BUY ONE WHEN YOU CAN HAVE TWO AT TWICE THE PRICE? The first picture of both of Hillary Clinton's Scooby mobiles. One is a GMC and the other is a Chevrolet, but they are mechanically identical. CONVOY: Scooby-one and Scooby-two took up positions in Hillary's motorcade on a freeway near Des Moines",
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
            .max_new_tokens = 50,
            .do_sample = false, // TODO 实现投机解码的核采样
        };
        model.generate(input_tensor, opt, [&](unsigned int out_token) -> bool {
            auto out_string = tokenizer.detokenize({out_token});
            auto [not_end, output_string] = tokenizer.postprocess(out_string);
            if (!not_end) { return false; }
            std::cout << output_string << std::flush;
            return true;
        });
        std::cout << "\n";
    }
}
