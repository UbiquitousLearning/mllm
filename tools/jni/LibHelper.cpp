//
// Created by Xiang Li on 2023/12/16.
//

// #ifdef ANDROID_API

#include "LibHelper.hpp"
#include <Types.hpp>
#include <memory>
#include <utility>
#include "models/fuyu/configuration_fuyu.hpp"
#include "models/fuyu/modeling_fuyu.hpp"
#include "models/qwen/configuration_qwen.hpp"
#include "models/qwen/modeling_qwen.hpp"
#include "models/qwen/tokenization_qwen.hpp"
#include "tokenizers/Unigram/Unigram.hpp"
using namespace mllm;
#include "models/fuyu/processing_fuyu.hpp"

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
}

bool LibHelper::setUp(const std::string &base_path, std::string weights_path, std::string vocab_path, std::string merge_path, PreDefinedModel model, MLLMBackendType backend_type) {
    FuyuConfig fuyuconfig(tokens_limit, "8B");
    QWenConfig qwconfig(tokens_limit, "1.5B");
    switch (model) {
    case LLAMA:
        tokenizer_ = make_shared<QWenTokenizer>(vocab_path, merge_path);
        module_ = make_shared<QWenForCausalLM>(qwconfig);
        break;

    case FUYU:
        processor_ = new FuyuProcessor(vocab_path);
        module_ = make_shared<FuyuModel>(fuyuconfig);
        break;
    }
    module_->load(weights_path);
    is_first_run_cond_ = true;
    return true;
}

void LibHelper::setCallback(callback_t callback) {
    this->callback_ = std::move(callback);
}

void LibHelper::run(std::string &input_str, uint8_t *image, unsigned max_step, unsigned int image_length) {
    if (model_ == LLAMA) {
        auto in_str = tokenizer_->apply_chat_template(input_str);
        auto input_tensor = tokenizer_->tokenize(in_str);
        LlmTextGeneratorOpts opt{
            .max_new_tokens = max_step,
            .do_sample = true,
            .temperature = 0.3F,
            .top_k = 50,
            .top_p = 0.F,
        };
        module_->generate(input_tensor, opt, [&](unsigned int out_token) -> bool {
            auto out_string = tokenizer_->detokenize({out_token});
            auto [not_end, output_string] = tokenizer_->postprocess(out_string);
            callback_(output_string, !not_end);
            if (!not_end) { return false; }
        });
        module_->clear_kvcache();
    } else if (model_ == FUYU) {
        auto processor = dynamic_cast<FuyuProcessor *>(processor_);
        auto input_tensors = processor->process(input_str, {image}, {image_length});
        for (int step = 0; step < max_step; step++) {
            auto result = (*module_)({input_tensors[0], input_tensors[1], input_tensors[2]});
            auto outputs = processor->detokenize(result[0]);
            auto out_string = outputs.first;
            auto out_token = outputs.second;
            auto [end, string] = processor->postprocess(out_string);
            callback_(string, !end);
            if (!end) { break; }
        }
    }
}
std::string LibHelper::runForResult(std::string &input_str) {
    if (model_ == Bert) {
        auto input_tensor = tokenizer_->tokenize(input_str);
        auto result = (*module_)({input_tensor});
        auto outputs = tokenizer_->detokenize(result[0]);
        return outputs.first;
    } else {
        return "";
    }
}

LibHelper::~LibHelper() {
    delete processor_;
}
// #endif
