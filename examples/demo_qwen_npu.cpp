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
    cmdParser.add<string>("prompt", 'p', "prompt used for the inference smoke test", false, "Hello! Introduce yourself in one sentence.");
    cmdParser.add<int>("max-new-tokens", 'n', "total number of generated tokens", false, 12);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string merge_path = cmdParser.get<string>("merge");
    string model_path = cmdParser.get<string>("qnn-model");
    string decoding_model_path = cmdParser.get<string>("decoding-model");
    string model_billion = cmdParser.get<string>("billion");
    int tokens_limit = cmdParser.get<int>("limits");
    string prompt = cmdParser.get<string>("prompt");
    int max_new_tokens = cmdParser.get<int>("max-new-tokens");
    if (max_new_tokens < 1) {
        std::cerr << "--max-new-tokens must be at least 1" << std::endl;
        return 2;
    }
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
        prompt,
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
        std::string generated_text;
        model.generate(input_tensor, opt, [&](unsigned int out_token) -> bool {
            auto out_string = tokenizer.detokenize({out_token});
            auto [not_end, output_string] = tokenizer.postprocess(out_string);
            if (!not_end) { return false; }
            generated_text += output_string;
            std::cout << output_string << std::flush;
            return true;
        });

        Context::Instance().inference_state().setCurSequenceLength(real_seq_length);
        Context::Instance().inference_state().setExecutionType(AUTOREGRESSIVE);
        Context::Instance().inference_state().toggleSwitching();

        LlmTextGeneratorOpts decoding_opt{
            .max_new_tokens = static_cast<size_t>(max_new_tokens - 1),
            .do_sample = false,
            .temperature = 0.3f,
            .top_k = 50,
            .top_p = 0.f,
            .is_padding = false,
        };
        bool isSwitched = false;
        if (max_new_tokens > 1) {
            decoding_model.generate(input_tensor, decoding_opt, [&](unsigned int out_token) -> bool {
                // call only once of switchDecodeTag
                if (!isSwitched) {
                    Context::Instance().inference_state().toggleSwitching();

                    isSwitched = true;
                }
                auto out_string = tokenizer.detokenize({out_token});
                auto [isOk, print_string] = tokenizer.postprocess(out_string);
                if (isOk) {
                    generated_text += print_string;
                    std::cout << print_string << std::flush;
                } else {
                    return false;
                }
                return true;
            });
        }

        // turn on switching, set sequence length and execution type
        Context::Instance().inference_state().setCurSequenceLength(0);
        Context::Instance().inference_state().setExecutionType(PROMPT);
        Context::Instance().inference_state().toggleSwitching();
        std::cout << "\n";
        std::cout << "[MLLM_RESULT_BEGIN]\n";
        std::cout << "Question: " << in_strs[i] << "\n";
        std::cout << "Answer: " << generated_text << "\n";
        std::cout << "[MLLM_RESULT_END]\n";

        if (!std::filesystem::exists("qnn_context.bin")) {
            // static_cast<QNNBackend *>(Backend::global_backends[MLLM_QNN].get())->saveQNNContext();
            static_cast<QNNBackend *>(Backend::global_backends[MLLM_QNN].get())->saveQNNContext();
        }
    }
}
