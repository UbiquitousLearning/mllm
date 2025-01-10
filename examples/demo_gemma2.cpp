#include "cmdline.h"
#include "models/gemma2/configuration_gemma2.hpp"
#include "models/gemma2/modeling_gemma2.hpp"
#include "models/gemma/tokenization_gemma.hpp"
#include "processor/PostProcess.hpp"

using namespace mllm;

int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/gemma2_vocab.mllm");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/gemma-2-2b-q4_k.mllm");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 400);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string model_path = cmdParser.get<string>("model");
    int tokens_limit = cmdParser.get<int>("limits");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    // gemma2 uses the same tokenizer as gemma
    auto tokenizer = GemmaTokenizer(vocab_path);

    Gemma2Config config(tokens_limit, "2B", RoPEType::HFHUBROPE);
    auto model = Gemma2ForCausalLM(config);
    model.load(model_path);

    vector<string> in_strs = {
        "Hello, who are you?",
        "What can you do?",
        "Please introduce Beijing University of Posts and Telecommunications."};

    for (int i = 0; i < in_strs.size(); ++i) {
        auto in_str = in_strs[i];
        auto input_tensor = tokenizer.tokenize(in_str);

        std::cout << "[Q] " << in_str << std::endl;
        std::cout << "[A] " << std::flush;
        for (int step = 0; step < 200; step++) {
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

    return 0;
}
