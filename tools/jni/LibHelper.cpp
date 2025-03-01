//
// Created by Xiang Li on 2023/12/16.
//

// #ifdef ANDROID_API

#include "LibHelper.hpp"
#include <Types.hpp>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "Generate.hpp"
#include "models/bert/configuration_bert.hpp"
#include "models/bert/modeling_bert.hpp"
#include "models/bert/tokenization_bert.hpp"
#include "models/fuyu/configuration_fuyu.hpp"
#include "models/fuyu/modeling_fuyu.hpp"
#include "models/phonelm/configuration_phonelm.hpp"
#include "models/phonelm/modeling_phonelm.hpp"
#include "models/qwen/configuration_qwen.hpp"
#include "models/qwen/modeling_qwen.hpp"
#include "models/qwen/tokenization_qwen.hpp"
#include "models/smollm/tokenization_smollm.hpp"
#include "tokenizers/Unigram/Unigram.hpp"
#include "models/fuyu/processing_fuyu.hpp"
#include "processor/PostProcess.hpp"
using namespace mllm;

#ifdef USE_QNN
#include "models/qwen/modeling_qwen_npu.hpp"
#include "models/phonelm/modeling_phonelm_npu.hpp"

#endif
inline bool exists_test(const std::string &name) {
    std::ifstream f(name.c_str());
    return f.good();
}

unsigned int LibHelper::postProcessing(shared_ptr<Tensor> result, shared_ptr<Tensor> &out_result) const {
    // switch (model_) {
    // case LLAMA: {
    //     return 0;
    // }
    // case FUYU: {
    //     // return chatPostProcessing(unsigned int token_idx, Tensor &tokens_tensor, const int &clean_tensors);
    // }
    // default: return 0;
    // }
    return 0;
}

bool LibHelper::setUp(const std::string &base_path, std::string weights_path, std::string qnn_weights_path, std::string vocab_path, std::string merge_path, PreDefinedModel model, MLLMBackendType backend_type) {
    FuyuConfig fuyuconfig(tokens_limit, "8B");
    QWenConfig qwconfig(tokens_limit, "1.5B");
    BertConfig bertconfig;
    PhoneLMConfig phone_config(tokens_limit, "1.5B");
    vocab_path = base_path + vocab_path;
    merge_path = base_path + merge_path;
    weights_path = base_path + weights_path;
    qnn_weights_path = base_path + qnn_weights_path;
    model_ = model;
    backend_ = backend_type;

    LOGI("Loading qnn model from %s", qnn_weights_path.c_str());
    LOGI("Loading model from %s", weights_path.c_str());

    switch (model) {
    case QWEN25:
        qwconfig = QWenConfig(tokens_limit, "1.5B");
        tokenizer_ = make_shared<QWenTokenizer>(vocab_path, merge_path);
        module_ = make_shared<QWenForCausalLM>(qwconfig);
        break;
    case QWEN15:
        qwconfig = QWenConfig(tokens_limit, "1.8B");
        tokenizer_ = make_shared<QWenTokenizer>(vocab_path, merge_path);
        module_ = make_shared<QWenForCausalLM>(qwconfig);
#ifdef USE_QNN
        if (backend_type == MLLMBackendType::QNN) {
            int chunk_size = 64;
            prefill_module_ = make_shared<QWenForCausalLM_NPU>(qwconfig, chunk_size);
            prefill_module_->load(qnn_weights_path);

            auto tokenizer = dynamic_pointer_cast<QWenTokenizer>(tokenizer_);
            // warmup START
            std::string input_str = " ";
            auto res = tokenizer->tokenizePaddingByChunk(input_str, chunk_size, 151936);
            auto input_tensor = res.second;
            auto real_seq_length = res.first;
            LlmTextGeneratorOpts opt{
                .max_new_tokens = 1,
                .do_sample = false,
                .is_padding = true,
                .seq_before_padding = real_seq_length,
                .chunk_size = chunk_size,
            };
            prefill_module_->generate(input_tensor, opt, [&](unsigned int out_token) -> bool {
                auto out_string = tokenizer_->detokenize({out_token});
                auto [not_end, output_string] = tokenizer_->postprocess(out_string);
                if (!not_end) { return false; }
                return true;
            });
            Module::isFirstChunk = false;
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setCurSequenceLength(0);
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setExecutionType(PROMPT);
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->toggleSwitching();
            Module::isMultiChunkPrefilling = true;
            // warmup END
            LOGE("QNN Warmup finished.");
        }
#endif
        break;

    case FUYU:
        processor_ = new FuyuProcessor(vocab_path, 224, 224);
        module_ = make_shared<FuyuModel>(fuyuconfig);
        break;
    case Bert:
        tokenizer_ = make_shared<BertTokenizer>(vocab_path, true);
        module_ = make_shared<BertModel>(bertconfig);
        break;

    case PhoneLM:
        tokenizer_ = make_shared<SmolLMTokenizer>(vocab_path, merge_path);
        module_ = make_shared<PhoneLMForCausalLM>(phone_config);
#ifdef USE_QNN
        if (backend_type == MLLMBackendType::QNN) {
            prefill_module_ = make_shared<PhoneLMForCausalLM_NPU>(phone_config);
            prefill_module_->load(qnn_weights_path);

            auto tokenizer = dynamic_pointer_cast<SmolLMTokenizer>(tokenizer_);
            // warmup START
            std::string input_str = " ";
            int chunk_size = 64;
            auto res = tokenizer->tokenizePaddingByChunk(input_str, chunk_size, 49152);
            auto input_tensor = res.second;
            auto real_seq_length = res.first;
            LlmTextGeneratorOpts opt{
                .max_new_tokens = 1,
                .do_sample = false,
                .is_padding = true,
                .seq_before_padding = real_seq_length,
                .chunk_size = chunk_size,
            };
            prefill_module_->generate(input_tensor, opt, [&](unsigned int out_token) -> bool {
                auto out_string = tokenizer_->detokenize({out_token});
                auto [not_end, output_string] = tokenizer_->postprocess(out_string);
                if (!not_end) { return false; }
                return true;
            });
            Module::isFirstChunk = false;
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setCurSequenceLength(0);
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setExecutionType(PROMPT);
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->toggleSwitching();
            Module::isMultiChunkPrefilling = true;
            // warmup END
            LOGE("QNN Warmup finished.");
        }
#endif
        break;
    }
    module_->load(weights_path);
    is_first_run_cond_ = true;

    return true;
}

void LibHelper::setCallback(callback_t callback) {
    this->callback_ = std::move(callback);
}

void LibHelper::run(std::string &input_str, uint8_t *image, unsigned max_step, unsigned int image_length, bool chat_template) {
    std::string output_string_;
    LOGE("Running model %d", model_);
    unsigned max_new_tokens = 500;
    LOGE("Running backend %d", backend_);
    vector<double> profiling_data(3);

    if (model_ == QWEN15 || model_ == QWEN25) {
        auto tokenizer = dynamic_pointer_cast<QWenTokenizer>(tokenizer_);
        if (chat_template) input_str = tokenizer_->apply_chat_template(input_str);
        if (backend_ == MLLMBackendType::QNN) {
            int chunk_size = 64;
            auto res = tokenizer->tokenizePaddingByChunk(input_str, chunk_size, 151936);
            auto input_tensor = res.second;
            max_new_tokens = tokens_limit - input_tensor.sequence();
            auto real_seq_length = res.first;
            const int seq_length_padding = (chunk_size - real_seq_length % chunk_size) + real_seq_length;
            const int chunk_num = seq_length_padding / chunk_size;
            bool isSwitched = false;

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
            std::vector<Tensor> chunked_tensors(chunk_num);
            for (int chunk_id = 0; chunk_id < chunk_num; ++chunk_id) {
                chunked_tensors[chunk_id].setBackend(Backend::global_backends[MLLM_CPU]);
                chunked_tensors[chunk_id].setTtype(INPUT_TENSOR);
                chunked_tensors[chunk_id].reshape(1, 1, chunk_size, 1);
                chunked_tensors[chunk_id].setName("input-chunk-" + to_string(chunk_id));
                chunked_tensors[chunk_id].shallowCopyFrom(&input_tensor, false, {0, 0, chunk_id * chunk_size, 0});

                prefill_module_->generate(chunked_tensors[chunk_id], opt, [&](unsigned int out_token) -> bool {
                    if (!isSwitched && chunk_id == 0 && static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->isStageSwitching()) {
                        // turn off switching at the first chunk of following inputs
                        static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->toggleSwitching();
                        isSwitched = true;
                    }
                    // switch_flag = true;
                    auto out_string = tokenizer_->detokenize({out_token});
                    auto [not_end, output_string] = tokenizer_->postprocess(out_string);
                    if (chunk_id == chunk_num - 1) { // print the output of the last chunk
                        output_string_ += output_string;
                        if (!not_end) {
                            auto profile_res = prefill_module_->profiling("Prefilling");
                            if (profile_res.size() == 3) {
                                profiling_data[0] += profile_res[0];
                                profiling_data[1] = profile_res[1];
                            }
                            callback_(output_string_, !not_end, profiling_data);
                        }
                        callback_(output_string_, !not_end, {});
                    }
                    return true;
                });
                Module::isFirstChunk = false;
            }
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setCurSequenceLength(real_seq_length);
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setExecutionType(AUTOREGRESSIVE);
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->toggleSwitching();

            opt = LlmTextGeneratorOpts{
                .max_new_tokens = max_new_tokens - 1,
                .do_sample = false,
                .temperature = 0.3f,
                .top_k = 50,
                .top_p = 0.f,
                .is_padding = false,
            };
            isSwitched = false;
            module_->generate(chunked_tensors.back(), opt, [&](unsigned int out_token) -> bool {
                if (!isSwitched) {
                    static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->toggleSwitching();
                    isSwitched = true;
                }
                auto out_token_string = tokenizer_->detokenize({out_token});
                auto [not_end, output_string] = tokenizer_->postprocess(out_token_string);
                output_string_ += output_string;
                if (!not_end) {
                    auto profile_res = module_->profiling("Inference");
                    if (profile_res.size() == 3) {
                        profiling_data[0] += profile_res[0];
                        profiling_data[2] = profile_res[2];
                    }
                    callback_(output_string_, !not_end, profiling_data);
                }
                callback_(output_string_, !not_end, {});
                if (!not_end) { return false; }
                return true;
            });
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setCurSequenceLength(0);
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setExecutionType(PROMPT);
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->toggleSwitching();
        } else { // CPU
            auto input_tensor = tokenizer_->tokenize(input_str);
            max_new_tokens = tokens_limit - input_tensor.sequence();
            LlmTextGeneratorOpts opt{
                .max_new_tokens = max_new_tokens,
                .do_sample = false,
            };
            module_->generate(input_tensor, opt, [&](unsigned int out_token) -> bool {
                auto out_token_string = tokenizer_->detokenize({out_token});
                auto [not_end, output_string] = tokenizer_->postprocess(out_token_string);
                output_string_ += output_string;
                if (!not_end) {
                    auto profile_res = module_->profiling("Inference");
                    if (profile_res.size() == 3) {
                        profiling_data = profile_res;
                    }
                    callback_(output_string_, !not_end, profiling_data);
                }
                callback_(output_string_, !not_end, {});
                if (!not_end) { return false; }
                return true;
            });
            module_->clear_kvcache();
        }

    } else if (model_ == FUYU) {
        auto processor = dynamic_cast<FuyuProcessor *>(processor_);
        auto input_tensors = processor->process(input_str, {image}, {image_length});
        for (int step = 0; step < 100; step++) {
            auto result = (*module_)({input_tensors[0], input_tensors[1], input_tensors[2]});
            auto outputs = processor->detokenize(result[0]);
            auto out_string = outputs.first;
            auto out_token = outputs.second;
            auto [end, string] = processor->postprocess(out_string);
            output_string_ += string;
            callback_(output_string_, !end, {});
            if (!end) { break; }
            chatPostProcessing(out_token, input_tensors[0], {&input_tensors[1], &input_tensors[2]});
        }
        module_->clear_kvcache();
    } else if (model_ == Bert) {
        LOGE("Bert model is not supported in this version.");
    } else if (model_ == PhoneLM) {
        // static bool switch_flag = false;
        auto tokenizer = dynamic_pointer_cast<SmolLMTokenizer>(tokenizer_);
        if (chat_template) input_str = tokenizer_->apply_chat_template(input_str);
        if (backend_ == MLLMBackendType::QNN) {
            int chunk_size = 64;
            auto res = tokenizer->tokenizePaddingByChunk(input_str, chunk_size, 49152);
            auto input_tensor = res.second;
            max_new_tokens = tokens_limit - input_tensor.sequence();
            auto real_seq_length = res.first;
            const int seq_length_padding = (chunk_size - real_seq_length % chunk_size) + real_seq_length;
            const int chunk_num = seq_length_padding / chunk_size;
            bool isSwitched = false;

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
            std::vector<Tensor> chunked_tensors(chunk_num);
            for (int chunk_id = 0; chunk_id < chunk_num; ++chunk_id) {
                chunked_tensors[chunk_id].setBackend(Backend::global_backends[MLLM_CPU]);
                chunked_tensors[chunk_id].setTtype(INPUT_TENSOR);
                chunked_tensors[chunk_id].reshape(1, 1, chunk_size, 1);
                chunked_tensors[chunk_id].setName("input-chunk-" + to_string(chunk_id));
                chunked_tensors[chunk_id].shallowCopyFrom(&input_tensor, false, {0, 0, chunk_id * chunk_size, 0});

                prefill_module_->generate(chunked_tensors[chunk_id], opt, [&](unsigned int out_token) -> bool {
                    // if (switch_flag && !isSwitched && chunk_id == 0) {
                    if (!isSwitched && chunk_id == 0) {
                        // turn off switching at the first chunk of following inputs
                        static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->toggleSwitching();
                        isSwitched = true;
                    }
                    // switch_flag = true;
                    auto out_string = tokenizer_->detokenize({out_token});
                    auto [not_end, output_string] = tokenizer_->postprocess(out_string);
                    if (chunk_id == chunk_num - 1) { // print the output of the last chunk
                        output_string_ += output_string;
                        if (!not_end) {
                            auto profile_res = prefill_module_->profiling("Prefilling");
                            if (profile_res.size() == 3) {
                                profiling_data[0] += profile_res[0];
                                profiling_data[1] = profile_res[1];
                            }
                            callback_(output_string_, !not_end, profiling_data);
                        }
                        callback_(output_string_, !not_end, {});
                    }
                    if (!not_end) { return false; }
                    return true;
                });
                Module::isFirstChunk = false;
            }
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setCurSequenceLength(real_seq_length);
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setExecutionType(AUTOREGRESSIVE);
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->toggleSwitching();

            opt = LlmTextGeneratorOpts{
                .max_new_tokens = max_new_tokens - 1,
                .do_sample = false,
                .temperature = 0.3f,
                .top_k = 50,
                .top_p = 0.f,
                .is_padding = false,
            };
            isSwitched = false;
            module_->generate(chunked_tensors.back(), opt, [&](unsigned int out_token) -> bool {
                if (!isSwitched) {
                    static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->toggleSwitching();
                    isSwitched = true;
                }
                auto out_token_string = tokenizer_->detokenize({out_token});
                auto [not_end, output_string] = tokenizer_->postprocess(out_token_string);
                output_string_ += output_string;
                if (!not_end) {
                    auto profile_res = module_->profiling("Inference");
                    if (profile_res.size() == 3) {
                        profiling_data[0] += profile_res[0];
                        profiling_data[2] = profile_res[2];
                    }
                    callback_(output_string_, !not_end, profiling_data);
                }
                callback_(output_string_, !not_end, {});
                if (!not_end) { return false; }
                return true;
            });
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setCurSequenceLength(0);
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setExecutionType(PROMPT);
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->toggleSwitching();
        } else { // CPU
            auto input_tensor = tokenizer_->tokenize(input_str);
            max_new_tokens = tokens_limit - input_tensor.sequence();
            LlmTextGeneratorOpts opt{
                .max_new_tokens = max_new_tokens,
                .do_sample = false,
            };
            module_->generate(input_tensor, opt, [&](unsigned int out_token) -> bool {
                auto out_token_string = tokenizer_->detokenize({out_token});
                auto [not_end, output_string] = tokenizer_->postprocess(out_token_string);
                output_string_ += output_string;
                if (!not_end) {
                    auto profile_res = module_->profiling("Inference");
                    if (profile_res.size() == 3) {
                        profiling_data = profile_res;
                    }
                    callback_(output_string_, !not_end, profiling_data);
                }
                callback_(output_string_, !not_end, {});
                if (!not_end) { return false; }
                return true;
            });
            module_->clear_kvcache();
        }
    }
}
std::vector<float> LibHelper::runForResult(std::string &input_str) {
    LOGE("Running model %d", model_);
    if (model_ == Bert) {
        // auto bert_tokenizer = dynamic_pointer_cast<BertTokenizer>(tokenizer_);
        auto inputs = tokenizer_->tokenizes(input_str);
        auto result = (*module_)(inputs)[0];
        auto output_arr = result.hostPtr<float>();
        return std::vector<float>(output_arr, output_arr + result.count());
    } else {
        return {};
    }
}

LibHelper::~LibHelper() {
    delete processor_;
}
// #endif
