/**
 * @file demo_qwen.cpp
 * @author Chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2024-05-01
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "cmdline.h"
#include "models/qwen/configuration_qwen.hpp"
#include "models/qwen/modeling_qwen.hpp"
#include "models/qwen/tokenization_qwen.hpp"
#include <string>
#include <vector>

using namespace mllm;

int main(int argc, char **argv) {
    std::iostream::sync_with_stdio(false);

    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/qwen2.5_vocab.mllm");
    cmdParser.add<string>("merge", 'e', "specify mllm merge file path", false, "../vocab/qwen2.5_merges.txt");
#ifdef ARM
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/qwen-2.5-1.5b-instruct-kai_q4_0_lm.mllm");
#else
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/qwen-2.5-1.5b-instruct-q4_0_4_4.mllm");
#endif
    cmdParser.add<string>("billion", 'b', "[0.5B | 1.8B | 1.5B | 3B |]", false, "1.5b-lm");
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
    // config.attn_implementation = "sage_attention"; // 使用Sage Attention实现
    auto model = QWenForCausalLM(config);
    model.load(model_path);

    vector<string> in_strs = {
        "Give me a short introduction to large language model.",
        "介绍一下你自己。",
        "什么是北京市的旧称？",
    };
    vector<string> input_strs;
    for (int i = 0; i < in_strs.size(); ++i) {
        std::cout << "[Q" << i << "] " << in_strs[i] << std::endl;
        auto input_str = tokenizer.apply_chat_template(in_strs[i]);
        input_strs.push_back(input_str);
    }
    auto input_tensor = tokenizer.tokenize(input_strs);

    LlmTextGeneratorOpts opt{
        .max_new_tokens = 200,
        .do_sample = false,
        .temperature = 0.3F,
        .top_k = 50,
        .top_p = 0.F,
    };
    auto output_tokens = model.generate(input_tensor, opt, tokenizer.eos_id_);
    for (int i = 0; i < output_tokens.size(); ++i) {
        auto out_token = output_tokens[i];
        auto out_string = tokenizer.detokenize(out_token);
        std::cout << "[A" << i << "] " << out_string << std::endl;
    }
    model.clear_kvcache();
    model.profiling();
}
