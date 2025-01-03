#include "cmdline.h"
// #include "models/minicpm_moe/configuration_minicpm_moe_mbm.hpp"
#include "models/minicpm_moe/mbm/settings_minicpm_moe_mbm.hpp"
#include "models/minicpm_moe/mbm/modeling_minicpm_moe_mbm.hpp"
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
    minicpmmoe_mbm_init(config.num_hidden_layers);
    auto model = MiniCPMForCausalLM(config);
    model.load(model_path);

    vector<string> in_strs = {
        "Hello, who are you?",
        // "山东省最高的山是哪座山, 它比黄山高还是矮？差距多少？",
        // "Please introduce Beijing University of Posts and Telecommunications.",
        // "Area, volume, and speed are all examples of what type of units?",
    };

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
    }
}