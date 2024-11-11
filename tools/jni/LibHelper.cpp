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

bool LibHelper::setUp(const std::string &base_path, std::string weights_path, std::string vocab_path, std::string merge_path, PreDefinedModel model, MLLMBackendType backend_type) {
    FuyuConfig fuyuconfig(500, "8B");
    QWenConfig qwconfig(tokens_limit, "1.5B");
    BertConfig bertconfig;
    PhoneLMConfig phone_config(tokens_limit, "1.5B");
    vocab_path = base_path + vocab_path;
    merge_path = base_path + merge_path;
    weights_path = base_path + weights_path;
    model_ = model;
    backend_ = backend_type;

    LOGI("Loading model from %s", weights_path.c_str());

    switch (model) {
    case QWEN:
        tokenizer_ = make_shared<QWenTokenizer>(vocab_path, merge_path);
#ifdef USE_QNN
        if (backend_type == MLLMBackendType::QNN) {
            prefill_module_ = make_shared<QWenForCausalLM_NPU>(qwconfig);
            prefill_module_->load(base_path + "model/qwen-1.5-1.8b-chat-int8.mllm");
        }
#endif
        module_ = make_shared<QWenForCausalLM>(qwconfig);
        break;

    case FUYU:
        processor_ = new FuyuProcessor(vocab_path);
        module_ = make_shared<FuyuModel>(fuyuconfig);
        break;
    case Bert:
        tokenizer_ = make_shared<BertTokenizer>(vocab_path, true);
        module_ = make_shared<BertModel>(bertconfig);
        break;

    case PhoneLM:
        tokenizer_ = make_shared<SmolLMTokenizer>(vocab_path, merge_path);
#ifdef USE_QNN
        if (backend_type == MLLMBackendType::QNN) {
            prefill_module_ = make_shared<PhoneLMForCausalLM_NPU>(phone_config);
            prefill_module_->load(base_path + "model/PhoneLM-1.5B-Instruct-128.mllm");
        }
#endif
        module_ = make_shared<PhoneLMForCausalLM>(phone_config);
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

    if (model_ == QWEN) {
        auto tokenizer = dynamic_pointer_cast<QWenTokenizer>(tokenizer_);
        if (chat_template) input_str = tokenizer_->apply_chat_template(input_str);
        auto input_tensor = tokenizer_->tokenize(input_str);
        max_new_tokens = tokens_limit - input_tensor.sequence();
        LlmTextGeneratorOpts opt{
            .max_new_tokens = max_new_tokens,
            .do_sample = false,
        };
        if (backend_ == MLLMBackendType::QNN) {
            auto res = tokenizer->tokenizeWithPadding(input_str, 64, 151936);
            input_tensor = res.second;
            auto real_seq_length = res.first;
            opt.is_padding = true;
            opt.seq_before_padding = real_seq_length;
            opt.max_new_tokens = 1;
            opt.do_sample = false;
            prefill_module_->generate(input_tensor, opt, [&](unsigned int out_token) -> bool {
                auto out_token_string = tokenizer_->detokenize({out_token});
                auto [not_end, output_string] = tokenizer_->postprocess(out_token_string);
                output_string_ += output_string;
                callback_(output_string_, !not_end);
                if (!not_end) { return false; }
                return true;
            });
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setSequenceLength(real_seq_length);
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->switchDecodeTag();
            opt = LlmTextGeneratorOpts{
                .max_new_tokens = max_new_tokens,
                .do_sample = false,
                .temperature = 0.3f,
                .top_k = 50,
                .top_p = 0.f,
                .is_padding = false,
            };
        }
        bool isSwitched = false;
        module_->generate(input_tensor, opt, [&](unsigned int out_token) -> bool {
            if (!isSwitched && backend_ == MLLMBackendType::QNN) {
                static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->switchDecodeTag();
                isSwitched = true;
            }
            auto out_token_string = tokenizer_->detokenize({out_token});
            auto [not_end, output_string] = tokenizer_->postprocess(out_token_string);
            output_string_ += output_string;
            callback_(output_string_, !not_end);
            if (!not_end) { return false; }
            return true;
        });
        if (backend_ == MLLMBackendType::CPU)
            module_->clear_kvcache();

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
            callback_(output_string_, !end);
            if (!end) { break; }
            chatPostProcessing(out_token, input_tensors[0], {&input_tensors[1], &input_tensors[2]});
        }
        module_->clear_kvcache();
    } else if (model_ == Bert) {
        LOGE("Bert model is not supported in this version.");
    } else if (model_ == PhoneLM) {
        auto tokenizer = dynamic_pointer_cast<SmolLMTokenizer>(tokenizer_);
        if (chat_template) input_str = tokenizer_->apply_chat_template(input_str);
        auto input_tensor = tokenizer_->tokenize(input_str);
        max_new_tokens = tokens_limit - input_tensor.sequence();
        LlmTextGeneratorOpts opt{
            .max_new_tokens = max_new_tokens,
            .do_sample = false,
        };
        if (backend_ == MLLMBackendType::QNN) {
            auto res = tokenizer->tokenizeWithPadding(input_str, 64, 49152);
            input_tensor = res.second;
            auto real_seq_length = res.first;
            opt.is_padding = true;
            opt.seq_before_padding = real_seq_length;
            opt.max_new_tokens = 1;
            opt.do_sample = false;
            prefill_module_->generate(input_tensor, opt, [&](unsigned int out_token) -> bool {
                auto out_token_string = tokenizer_->detokenize({out_token});
                auto [not_end, output_string] = tokenizer_->postprocess(out_token_string);
                output_string_ += output_string;
                callback_(output_string_, !not_end);
                if (!not_end) { return false; }
                return true;
            });
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setSequenceLength(real_seq_length);
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->switchDecodeTag();

            opt = LlmTextGeneratorOpts{
                .max_new_tokens = max_new_tokens,
                .do_sample = false,
                .temperature = 0.3f,
                .top_k = 50,
                .top_p = 0.f,
                .is_padding = false,
            };
        }
        bool isSwitched = false;
        module_->generate(input_tensor, opt, [&](unsigned int out_token) -> bool {
            if (!isSwitched && backend_ == MLLMBackendType::QNN) {
                static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->switchDecodeTag();
                isSwitched = true;
            }
            auto out_token_string = tokenizer_->detokenize({out_token});
            auto [not_end, output_string] = tokenizer_->postprocess(out_token_string);
            output_string_ += output_string;
            callback_(output_string_, !not_end);
            if (!not_end) { return false; }
            return true;
        });
        if (backend_ == MLLMBackendType::CPU) module_->clear_kvcache();
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
