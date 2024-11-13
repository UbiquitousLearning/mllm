#include "Module.hpp"
#include "Types.hpp"
#include <memory>
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
    cmdParser.add<int>("chunk", 'c', "chunk size", false, 64);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string merge_path = cmdParser.get<string>("merge");
    string model_path = cmdParser.get<string>("model");
    int tokens_limit = cmdParser.get<int>("limits");
    int chunk_size = cmdParser.get<int>("chunk");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    auto tokenizer = SmolLMTokenizer(vocab_path, merge_path);
    PhoneLMConfig config(tokens_limit, "1.5B");
    auto model = PhoneLMForCausalLM_NPU(config);
    model.load(model_path);
    auto decoding_model = PhoneLMForCausalLM(config);
    decoding_model.load("../models/phonelm-1.5b-instruct-q4_0_4_4.mllm");

    vector<string> in_strs = {
        "Give me a short introduction to large language model.",
        "What is the Beijing University of Posts and Telecommunications.",
        "What is the meaning of life?",
    };

    // turn on the multi-chunk prefilling
    Module::isMultiChunkPrefilling = true;

    for (int i = 0; i < in_strs.size(); ++i) {
        auto input_str = tokenizer.apply_chat_template(in_strs[i]);
        auto [real_seq_length, input_tensor] = tokenizer.tokenizePaddingByChunk(input_str, chunk_size, config.vocab_size);

        const int seq_length_padding = (chunk_size - real_seq_length % chunk_size) + real_seq_length;
        const int chunk_num = seq_length_padding / chunk_size;
        bool isSwitched = false;

        std::cout << "real seq length: " << real_seq_length << " padding to: " << seq_length_padding << " chunk num: " << chunk_num << std::endl;

        std::cout << "[Q] " << in_strs[i] << std::endl;
        std::cout << "[A] " << std::flush;

        // tensor vectors to save the chunked tensors of the QNN prefilling input
        vector<Tensor> chunked_tensors(chunk_num);
        LlmTextGeneratorOpts opt{
            .max_new_tokens = 1,
            .do_sample = false,
            .is_padding = true,
            .seq_before_padding = real_seq_length,
            .chunk_size = chunk_size,
        };

        for (int chunk_id = 0; chunk_id < chunk_num; ++chunk_id) {
            chunked_tensors[chunk_id].setBackend(Backend::global_backends[MLLM_CPU]);
            chunked_tensors[chunk_id].setTtype(INPUT_TENSOR);
            chunked_tensors[chunk_id].reshape(1, 1, chunk_size, 1);
            chunked_tensors[chunk_id].setName("input-chunk-" + to_string(chunk_id));
            chunked_tensors[chunk_id].deepCopyFrom(&input_tensor, false, {0, 0, chunk_id * chunk_size, 0});

            model.generate(chunked_tensors[chunk_id], opt, [&](unsigned int out_token) -> bool {
                if (i != 0 && !isSwitched && chunk_id == 0) {
                    // turn off switching at the first chunk of following inputs
                    static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->toggleSwitching();
                    isSwitched = true;
                }
                auto out_string = tokenizer.detokenize({out_token});
                auto [not_end, output_string] = tokenizer.postprocess(out_string);
                if (!not_end) { return false; }

                if (chunk_id == chunk_num - 1) { // print the output of the last chunk
                    std::cout << output_string << std::flush;
                }
                return true;
            });

            Module::isFirstChunk = false;
        }

        // turn on switching, set sequence length and execution type
        static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setSequenceLength(real_seq_length);
        static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setExecutionType(AUTOREGRESSIVE);
        static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->toggleSwitching();

        LlmTextGeneratorOpts decoding_opt{
            .max_new_tokens = 100,
            .do_sample = false,
            .temperature = 0.3f,
            .top_k = 50,
            .top_p = 0.f,
            .is_padding = false,
        };

        isSwitched = false;
        decoding_model.generate(chunked_tensors.back(), decoding_opt, [&](unsigned int out_token) -> bool {
            if (!isSwitched) {
                // turn off switching
                static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->toggleSwitching();
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
        // turn on switching, set sequence length and execution type
        static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setSequenceLength(0);
        static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setExecutionType(PROMPT);
        static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->toggleSwitching();
    }
}
#endif