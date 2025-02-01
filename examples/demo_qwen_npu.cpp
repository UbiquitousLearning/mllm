#include "backends/cpu/CPUBackend.hpp"
#include "cmdline.h"
#include "models/qwen/configuration_qwen.hpp"
#include "models/qwen/modeling_qwen_npu.hpp"
#include "models/qwen/modeling_qwen.hpp"
#include "models/qwen/tokenization_qwen.hpp"
#include "processor/PostProcess.hpp"

using namespace mllm;

int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/qwen_vocab.mllm");
    cmdParser.add<string>("merge", 'e', "specify mllm merge file path", false, "../vocab/qwen_merges.txt");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/qwen-1.5-1.8b-chat-int8.mllm");
    cmdParser.add<string>("billion", 'b', "[0.5B | 1.8B]", false, "1.8B");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 400);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string merge_path = cmdParser.get<string>("merge");
    string model_path = cmdParser.get<string>("model");
    string model_billion = cmdParser.get<string>("billion");
    int tokens_limit = cmdParser.get<int>("limits");
    const int chunk_size = 128;
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    auto tokenizer = QWenTokenizer(vocab_path, merge_path);
    QWenConfig config(tokens_limit, model_billion, RoPEType::HFHUBROPE);
    auto model = QWenForCausalLM_NPU(config, chunk_size);
    model.load(model_path);
    auto decoding_model = QWenForCausalLM(config);
    decoding_model.load("../models/qwen-1.5-1.8b-chat-q4k.mllm");

    // warmup START
    std::string input_str = " ";
    auto [real_seq_length, input_tensor] = tokenizer.tokenizePaddingByChunk(input_str, chunk_size, config.vocab_size);
    LlmTextGeneratorOpts opt{
        .max_new_tokens = 1,
        .do_sample = false,
        .is_padding = true,
        .seq_before_padding = real_seq_length,
        .chunk_size = chunk_size,
    };
    model.generate(input_tensor, opt, [&](unsigned int out_token) -> bool {
        auto out_string = tokenizer.detokenize({out_token});
        auto [not_end, output_string] = tokenizer.postprocess(out_string);
        if (!not_end) { return false; }
        return true;
    });
    Module::isFirstChunk = false;
    static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setCurSequenceLength(0);
    static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setExecutionType(PROMPT);
    static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->toggleSwitching();
    // turn on the multi-chunk prefilling
    Module::isMultiChunkPrefilling = true;
    // warmup END
    std::cout << "Warmup finished." << std::endl;

    vector<string> in_strs = {
        // " Give me a short introduction to large language model.",
        "\"Large Language Models (LLMs) are advanced artificial intelligence systems designed to understand and generate human-like text. These models are trained on vast amounts of data, enabling them to perform a wide range of tasks, from answering questions and summarizing text to generating creative content and engaging in conversational dialogue. LLMs like GPT-3 and GPT-4, developed by OpenAI, have set new benchmarks in natural language processing by leveraging deep learning architectures, particularly transformer models, which excel at capturing context and relationships within text. The scalability and versatility of LLMs make them invaluable tools for applications in education, customer service, content creation, and more. However, their deployment also raises ethical considerations, including issues of bias, misinformation, and the potential for misuse. As the field continues to evolve, ongoing research and responsible deployment strategies are essential to harnessing the full potential of these powerful AI systems while mitigating their risks.\"\nGenerate a title based on the above text."};

    for (int i = 0; i < in_strs.size(); ++i) {
        auto input_str = tokenizer.apply_chat_template(in_strs[i]);
        auto [real_seq_length, input_tensor] = tokenizer.tokenizePaddingByChunk(input_str, chunk_size, config.vocab_size);
        const int seq_length_padding = (chunk_size - real_seq_length % chunk_size) + real_seq_length;
        const int chunk_num = seq_length_padding / chunk_size;

        std::cout << "[Q] " << in_strs[i] << std::endl;
        std::cout << "[A] " << std::flush;

        // set total seq length for HeadLinear execute, which can not get the real seq length from Opts
        static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setTotalSequenceLength(real_seq_length);
        // set chunk size for the HeadLinear execute, which can not get the chunk size from Opts
        static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setChunkSize(chunk_size);

        LlmTextGeneratorOpts opt{
            .max_new_tokens = 1,
            .do_sample = false,
            .is_padding = true,
            .seq_before_padding = real_seq_length,
            .chunk_size = chunk_size,
        };

        // tensor vectors to save the chunked tensors of the QNN prefilling input
        bool isSwitched = false;
        vector<Tensor> chunked_tensors(chunk_num);
        for (int chunk_id = 0; chunk_id < chunk_num; ++chunk_id) {
            chunked_tensors[chunk_id].setBackend(Backend::global_backends[MLLM_CPU]);
            chunked_tensors[chunk_id].setTtype(INPUT_TENSOR);
            chunked_tensors[chunk_id].reshape(1, 1, chunk_size, 1);
            chunked_tensors[chunk_id].setName("input-chunk-" + to_string(chunk_id));
            chunked_tensors[chunk_id].shallowCopyFrom(&input_tensor, false, {0, 0, chunk_id * chunk_size, 0});

            model.generate(chunked_tensors[chunk_id], opt, [&](unsigned int out_token) -> bool {
                if (!isSwitched && chunk_id == 0 && static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->isStageSwitching()) {
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

        static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setCurSequenceLength(real_seq_length);
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
            // call only once of switchDecodeTag
            if (!isSwitched) {
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

        // turn on switching, set sequence length and execution type
        static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setCurSequenceLength(0);
        static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setExecutionType(PROMPT);
        static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->toggleSwitching();
        std::cout << "\n";
    }
}