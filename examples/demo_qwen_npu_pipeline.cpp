#include "Context.hpp"
#include "Backend.hpp"
#include "QNNBackend.hpp"
#include "Trace.hpp"
#include "Types.hpp"
#include "backends/cpu/CPUBackend.hpp"
#include "cmdline.h"
#include "models/qwen/configuration_qwen.hpp"
#include "models/qwen/modeling_qwen_npu_v2.hpp"
#include "models/qwen/modeling_qwen.hpp"
#include "models/qwen/tokenization_qwen.hpp"
#include "processor/PostProcess.hpp"
#include "Parallel.hpp"

using namespace mllm;

int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/qwen2.5_vocab.mllm");
    // "../vocab/qwen_vocab.mllm"
    cmdParser.add<string>("merge", 'e', "specify mllm merge file path", false, "../vocab/qwen2.5_merges.txt");
    // "../vocab/qwen_merges.txt"
    cmdParser.add<string>("qnn-model", 'm', "specify mllm model path", false, "../models/Qwen2.5-1.5B-Instruct_rotated-noshadow.mllm");
    // "../models/qwen1.5-1.8b-chat-rot-qnn.mllm"
    cmdParser.add<string>("decoding-model", '\0', "specify mllm model path", false, "../models/Qwen2.5-1.5B-Instruct_rotated-Q40.mllm");
    // "../models/qwen1.5-1.8b-chat-rot_q4_0.mllm"
    cmdParser.add<string>("billion", 'b', "[0.5B | 1.8B | 1.5B | [1.5B, 1.8B]-rotated]", false, "1.5B-rotated");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 400);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string merge_path = cmdParser.get<string>("merge");
    string model_path = cmdParser.get<string>("qnn-model");
    string decoding_model_path = cmdParser.get<string>("decoding-model");
    string model_billion = cmdParser.get<string>("billion");
    int tokens_limit = cmdParser.get<int>("limits");
    const int chunk_size = 128;
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    Module::initBackend(MLLM_QNN);

    auto tokenizer = QWenTokenizer(vocab_path, merge_path);
    QWenNPUConfig config(tokens_limit, model_billion, RoPEType::HFHUBROPE);
    auto model = v2::QWenForCausalLM_NPU(config, chunk_size);
    model.load(model_path);
    config.attn_implementation = "eager_notrans";
    auto decoding_model = QWenForCausalLM(config);
    decoding_model.load(decoding_model_path);

    string trace_string = " ";
    auto [_, input_tensor] = tokenizer.tokenizePaddingByChunk(trace_string, chunk_size, config.vocab_size);
    Tracer::trace(&model, {input_tensor});
    std::cout << "Trace and Warmup finished" << std::endl;

    if (!std::filesystem::exists("qnn_context.bin")) {
        static_cast<QNNBackend *>(Backend::global_backends[MLLM_QNN].get())->saveQNNContext();
    }

    vector<string> in_strs = {
        " Give me a short introduction to large language model.",
        // "\"Large Language Models (LLMs) are advanced artificial intelligence systems designed to understand and generate human-like text. These models are trained on vast amounts of data, enabling them to perform a wide range of tasks, from answering questions and summarizing text to generating creative content and engaging in conversational dialogue. LLMs like GPT-3 and GPT-4, developed by OpenAI, have set new benchmarks in natural language processing by leveraging deep learning architectures, particularly transformer models, which excel at capturing context and relationships within text. The scalability and versatility of LLMs make them invaluable tools for applications in education, customer service, content creation, and more. However, their deployment also raises ethical considerations, including issues of bias, misinformation, and the potential for misuse. As the field continues to evolve, ongoing research and responsible deployment strategies are essential to harnessing the full potential of these powerful AI systems while mitigating their risks.\"\nGenerate a title based on the above text."
    };

    for (int i = 0; i < in_strs.size(); ++i) {
        auto input_str = tokenizer.apply_chat_template(in_strs[i]);
        auto [real_seq_length, input_tensor] = tokenizer.tokenizePaddingByChunk(input_str, chunk_size, config.vocab_size);

        // set total seq length for HeadLinear execute, which can not get the real seq length from Opts
        Context::Instance().inference_state().setTotalSequenceLength(real_seq_length);
        // set chunk size for the HeadLinear execute, which can not get the chunk size from Opts
        Context::Instance().inference_state().setChunkSize(chunk_size);

        std::cout << "[Q] " << in_strs[i] << std::endl;
        std::cout << "[A] " << std::flush;
        std::cout << "real_seq_length: " << real_seq_length << std::endl;

        LlmTextGeneratorOpts opt{
            .max_new_tokens = 1,
            .do_sample = false,
            .is_padding = true,
            .seq_before_padding = real_seq_length,
            .chunk_size = chunk_size,
        };

        // tensor vectors to save the chunked tensors of the QNN prefilling input
        bool isSwitched = false;

        ChunkPipeline pipeline(real_seq_length, chunk_size);
        auto prefill_result = pipeline.run(input_tensor, opt, tokenizer, model, isSwitched);

        Context::Instance().inference_state().setQnnGraphFrozen(true);

        Context::Instance().inference_state().setCurSequenceLength(real_seq_length);
        Context::Instance().inference_state().setExecutionType(AUTOREGRESSIVE);
        Context::Instance().inference_state().toggleSwitching();

        LlmTextGeneratorOpts decoding_opt{
            .max_new_tokens = 100,
            .do_sample = false,
            .temperature = 0.3f,
            .top_k = 50,
            .top_p = 0.f,
            .is_padding = false,
        };
        isSwitched = false;

        Tensor decoding_input;
        decoding_input.setBackend(Backend::global_backends[MLLM_CPU].get());
        decoding_input.setTtype(INPUT_TENSOR);
        decoding_input.reshape(1, 1, 1, 1);
        decoding_input.setName("input0");
        decoding_input.alloc();
        decoding_input.setDataAt(0, 0, 0, 0, prefill_result->dataAt<float>(0, 0, 0, 0));
        decoding_model.generate(decoding_input, decoding_opt, [&](unsigned int out_token) -> bool {
            // call only once of switchDecodeTag
            if (!isSwitched) {
                Context::Instance().inference_state().toggleSwitching();
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
        Context::Instance().inference_state().setCurSequenceLength(0);
        Context::Instance().inference_state().setExecutionType(PROMPT);
        Context::Instance().inference_state().toggleSwitching();
        std::cout << "\n";

        decoding_model.profiling("Decoding");
    }
}