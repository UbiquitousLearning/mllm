#include "Context.hpp"
#include "QNNBackend.hpp"
#include "Types.hpp"
#include "backends/cpu/CPUBackend.hpp"
#include "cmdline.h"
#include "models/qwen/configuration_qwen.hpp"
#include "models/qwen/modeling_qwen.hpp"
#include "models/qwen/modeling_qwen_npu_v2.hpp"
#include "models/qwen/tokenization_qwen.hpp"
#include "processor/PostProcess.hpp"

using namespace mllm;

int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/qwen2.5_vocab.mllm");
    cmdParser.add<string>("merge", 'e', "specify mllm merge file path", false, "../vocab/qwen2.5_merges.txt");
    cmdParser.add<string>("qnn-model", 'm', "specify mllm model path", false, "../models/Qwen2.5-1.5B-Instruct_rotated-noshadow.mllm");
    cmdParser.add<string>("decoding-model", '\0', "specify mllm model path", false, "../models/Qwen2.5-1.5B-Instruct_rotated-Q40.mllm");
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
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    Module::initBackend(MLLM_QNN);

    auto tokenizer = QWenTokenizer(vocab_path, merge_path);
    QWenNPUConfig config(tokens_limit, "1.5b-rotated", RoPEType::HFHUBROPE);
    auto model = v2::QWenForCausalLM_NPU(config, 256);
    config.attn_implementation = "eager_notrans";
    model.load(model_path);
    auto decoding_model = QWenForCausalLM(config);
    decoding_model.load(decoding_model_path);

    vector<string> in_strs = {
        // " Give me a short introduction to large language model.",
        "\"Large Language Models (LLMs) are advanced artificial intelligence systems designed to understand and generate human-like text. These models are trained on vast amounts of data, enabling them to perform a wide range of tasks, from answering questions and summarizing text to generating creative content and engaging in conversational dialogue. LLMs like GPT-3 and GPT-4, developed by OpenAI, have set new benchmarks in natural language processing by leveraging deep learning architectures, particularly transformer models, which excel at capturing context and relationships within text. The scalability and versatility of LLMs make them invaluable tools for applications in education, customer service, content creation, and more. However, their deployment also raises ethical considerations, including issues of bias, misinformation, and the potential for misuse. As the field continues to evolve, ongoing research and responsible deployment strategies are essential to harnessing the full potential of these powerful AI systems while mitigating their risks.\"\nGenerate a title based on the above text.",
        // " Hello, Who are you?"
    };

    for (int i = 0; i < in_strs.size(); ++i) {
        auto input_str = tokenizer.apply_chat_template(in_strs[i]);
        auto [real_seq_length, input_tensor] = tokenizer.tokenizeWithPadding(input_str, 256, config.vocab_size);
        // real_seq_length = 256;
        std::cout << "[Q] " << in_strs[i] << std::endl;
        std::cout << "[A] " << std::flush;
        std::cout << "real_seq_length: " << real_seq_length << std::endl;

        // set total seq length for HeadLinear execute, which can not get the real seq length from Opts
        Context::Instance().inference_state().setTotalSequenceLength(real_seq_length);

        LlmTextGeneratorOpts opt{
            .max_new_tokens = 1,
            .do_sample = false,
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

        Context::Instance().inference_state().setCurSequenceLength(real_seq_length);
        Context::Instance().inference_state().setExecutionType(AUTOREGRESSIVE);
        Context::Instance().inference_state().toggleSwitching();

        LlmTextGeneratorOpts decoding_opt{
            .max_new_tokens = 50,
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

        if (!std::filesystem::exists("qnn_context.bin")) {
            // static_cast<QNNBackend *>(Backend::global_backends[MLLM_QNN].get())->saveQNNContext();
            static_cast<QNNBackend *>(Backend::global_backends[MLLM_QNN].get())->saveQNNContext();
        }
    }
}
