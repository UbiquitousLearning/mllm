#ifdef USE_QNN
#include "backends/cpu/CPUBackend.hpp"
#include "cmdline.h"
#include "models/phonelm/modeling_phonelm.hpp"
#include "models/phonelm/modeling_phonelm_npu.hpp"
#include "models/smollm/tokenization_smollm.hpp"

using namespace mllm;

int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/phonelm_vocab.mllm");
    cmdParser.add<string>("merge", 'e', "specify mllm merge file path", false, "../vocab/phonelm_merges.txt");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/PhoneLM-1.5B-Instruct-128.mllm");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 400);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string merge_path = cmdParser.get<string>("merge");
    string model_path = cmdParser.get<string>("model");
    int tokens_limit = cmdParser.get<int>("limits");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    auto tokenizer = SmolLMTokenizer(vocab_path, merge_path);
    PhoneLMConfig config(tokens_limit, "1.5B");
    auto model = PhoneLMForCausalLM_NPU(config);
    model.load(model_path);
    auto decoding_model = PhoneLMForCausalLM(config);
    decoding_model.load("../models/phonelm-1.5b-instruct-q4_0_4_4.mllm");

    vector<string> in_strs = {
        "Give me a short introduction to large language model.",
    };

    for (int i = 0; i < in_strs.size(); ++i) {
        auto input_str = tokenizer.apply_chat_template(in_strs[i]);
        auto [real_seq_length, input_tensor] = tokenizer.tokenizeWithPadding(input_str, 64, config.vocab_size);
        std::cout << real_seq_length << endl;
        std::cout << input_str << std::endl;
        std::cout << "[Q] " << in_strs[i] << std::endl;
        std::cout << "[A] " << std::flush;

        LlmTextGeneratorOpts opt{
            .max_new_tokens = 1,
            .do_sample = false,
            .temperature = 0.3f,
            .top_k = 50,
            .top_p = 0.f,
            .is_padding = true,
            .seq_before_padding = real_seq_length,
        };
        model.generate(input_tensor, opt, [&](unsigned int out_token) -> bool {
            auto out_string = tokenizer.detokenize({out_token});
            auto [not_end, output_string] = tokenizer.postprocess(out_string);
            if (!not_end) { return false; }
            std::cout << output_string << std::flush;
            return true;
        });

        static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setSequenceLength(real_seq_length);
        static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->switchDecodeTag();

        LlmTextGeneratorOpts decoding_opt{
            .max_new_tokens = 100,
            .do_sample = false,
            .temperature = 0.3f,
            .top_k = 50,
            .top_p = 0.f,
            .is_padding = false,
        };
        bool isSwitched = false;
        decoding_model.generate(input_tensor, decoding_opt, [&](unsigned int out_token) -> bool {
            // call only once of switchDecodeTag
            if (!isSwitched) {
                static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->switchDecodeTag();
                isSwitched = true;
            }
            auto out_string = tokenizer.detokenize({out_token});
            auto [isOk, print_string] = tokenizer.postprocess(out_string);
            if (isOk) {
                std::cout << print_string << std::flush;
            } else {
                return false;
            }
            return true;
        });
        std::cout << "\n---------------" << std::endl;
    }
}
#endif