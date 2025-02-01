#include "Module.hpp"
#include "Types.hpp"
#include <memory>
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
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/phonelm-1.5b-instruct-int8.mllm");
    cmdParser.add<string>("decoding", 'd', "specify mllm decoding model path", false, "../models/phonelm-1.5b-instruct-q4_0_4_4.mllm");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 400);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.add<int>("chunk", 'c', "chunk size", false, 64);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string merge_path = cmdParser.get<string>("merge");
    string model_path = cmdParser.get<string>("model");
    string decoding_path = cmdParser.get<string>("decoding");
    int tokens_limit = cmdParser.get<int>("limits");
    int chunk_size = cmdParser.get<int>("chunk");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    auto tokenizer = SmolLMTokenizer(vocab_path, merge_path);
    PhoneLMConfig config(tokens_limit, "1.5B");
    auto model = PhoneLMForCausalLM_NPU(config, chunk_size);
    model.load(model_path);
    auto decoding_model = PhoneLMForCausalLM(config);
    decoding_model.load(decoding_path);

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
        "Give me a short introduction to large language model.",
        "What is the Beijing University of Posts and Telecommunications.",
        "What is the meaning of life?",
        "Hello, who are you?",
        "What can you do?",
        "Please introduce Beijing University of Posts and Telecommunications.",
        "\"Large Language Models (LLMs) are advanced artificial intelligence systems designed to understand and generate human-like text. These models are trained on vast amounts of data, enabling them to perform a wide range of tasks, from answering questions and summarizing text to generating creative content and engaging in conversational dialogue. LLMs like GPT-3 and GPT-4, developed by OpenAI, have set new benchmarks in natural language processing by leveraging deep learning architectures, particularly transformer models, which excel at capturing context and relationships within text. The scalability and versatility of LLMs make them invaluable tools for applications in education, customer service, content creation, and more. However, their deployment also raises ethical considerations, including issues of bias, misinformation, and the potential for misuse. As the field continues to evolve, ongoing research and responsible deployment strategies are essential to harnessing the full potential of these powerful AI systems while mitigating their risks.\"\nGenerate a title based on the above text."};

    for (int i = 0; i < in_strs.size(); ++i) {
        auto input_str = tokenizer.apply_chat_template(in_strs[i]);
        auto [real_seq_length, input_tensor] = tokenizer.tokenizePaddingByChunk(input_str, chunk_size, config.vocab_size);
        const int seq_length_padding = (chunk_size - real_seq_length % chunk_size) + real_seq_length;
        const int chunk_num = seq_length_padding / chunk_size;
        bool isSwitched = false;
        // std::cout << "real seq length: " << real_seq_length << " padding to: " << seq_length_padding << " chunk num: " << chunk_num << std::endl;
        std::cout << "[Q] " << in_strs[i] << std::endl;
        std::cout << "[A] " << std::flush;

        // set total seq length for HeadLinear execute, which can not get the real seq length from Opts
        static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setTotalSequenceLength(real_seq_length);
        // set chunk size for the HeadLinear execute, which can not get the chunk size from Opts
        static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setChunkSize(chunk_size);

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
            chunked_tensors[chunk_id].shallowCopyFrom(&input_tensor, false, {0, 0, chunk_id * chunk_size, 0});

            model.generate(chunked_tensors[chunk_id], opt, [&](unsigned int out_token) -> bool {
                // if (i != 0 && !isSwitched && chunk_id == 0) {
                if (!isSwitched && chunk_id == 0) {
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
            if (!isSwitched) { // turn off switching
                static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->toggleSwitching();
                isSwitched = true;
            }
            auto out_string = tokenizer.detokenize({out_token});
            auto [not_end, output_string] = tokenizer.postprocess(out_string);
            if (!not_end) { return false; }
            std::cout << output_string << std::flush;
            return true;
        });

        // turn on switching, set sequence length and execution type
        static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setCurSequenceLength(0);
        static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setExecutionType(PROMPT);
        static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->toggleSwitching();
        std::cout << "\n";
    }
}