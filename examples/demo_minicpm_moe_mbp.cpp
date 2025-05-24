#include "cmdline.h"
#include "models/minicpm_moe/configuration_minicpm_moe.hpp"
#include "models/minicpm_moe/mbp/modeling_minicpm_moe_mbp.hpp"
// #include "models/minicpm_moe/modeling_minicpm_moe.hpp"
#include "models/minicpm/tokenization_minicpm.hpp"
#include "processor/PostProcess.hpp"

using namespace mllm;

int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/minicpm_vocab.mllm");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/minicpm-moe-8x2b-q4_k.mllm");
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
        "\"Large Language Models (LLMs) are advanced artificial intelligence systems designed to understand and generate human-like text. These models are trained on vast amounts of data, enabling them to perform a wide range of tasks, from answering questions and summarizing text to generating creative content and engaging in conversational dialogue. LLMs like GPT-3 and GPT-4, developed by OpenAI, have set new benchmarks in natural language processing by leveraging deep learning architectures, particularly transformer models, which excel at capturing context and relationships within text. The scalability and versatility of LLMs make them invaluable tools for applications in education, customer service, content creation, and more. However, their deployment also raises ethical considerations, including issues of bias, misinformation, and the potential for misuse. As the field continues to evolve, ongoing research and responsible deployment strategies are essential to harnessing the full potential of these powerful AI systems while mitigating their risks.\"\nGenerate a title based on the above text.", // 203
        // "Hello, who are you?", // 13
        // "山东省最高的山是哪座山, 它比黄山高还是矮？差距多少？",
        // "Please introduce Beijing University of Posts and Telecommunications.",
    };
    minicpmmoe_mbp_init(config.num_hidden_layers, config.num_experts);
    for (int i = 0; i < in_strs.size(); ++i) {
        auto in_str = tokenizer.apply_chat_template(in_strs[i]);
        auto input_tensor = tokenizer.tokenize(in_str);
        std::cout << "[Q] " << in_strs[i] << std::endl;
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
        model.clear_kvcache();
        model.profiling();
        // prinMBPtimes();
    }
}