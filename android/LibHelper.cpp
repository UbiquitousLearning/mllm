//
// Created by 咸的鱼 on 2023/12/16.
//
#include "helper.hpp"
#include "processor/FuyuPreProcess.hpp"

#ifdef ANDROID_API

#include "LibHelper.hpp"
#include <iostream>
#include <Types.hpp>
#include <utility>
#include <valarray>
#include "Net.hpp"
#include "Executor.hpp"
#include "NetParameter.hpp"
#include "express/Express.hpp"
#include "tokenizers/BPE/Bpe.hpp"
#include "modeling_llama.hpp"
#include "modeling_fuyu.hpp"
#include "tokenizers/Unigram/Unigram.hpp"
using namespace mllm;

inline bool exists_test(const std::string &name) {
    std::ifstream f(name.c_str());
    return f.good();
}


unsigned int LibHelper::postProcessing(shared_ptr<Tensor> result, shared_ptr<Tensor> &out_result) const {
    switch (model_) {
    case LLAMA: {
        return postProcessing_llama(result, out_result);
    }
    case FUYU: return postProcessing_Fuyu(result, out_result);
    default: return 0;
    }
}


bool LibHelper::setUp(const std::string &base_path, std::string weights_path, std::string vacab_path, PreDefinedModel model, MLLMBackendType backend_type) {
    c = new Context();
    BackendConfig bn;
    weights_path = base_path + weights_path;
    vacab_path = base_path + vacab_path;
    LOGI("Setup!");
    //check path exists
    if (!exists_test(weights_path) || !exists_test(vacab_path)) {
        return false;
    }

    const auto param_loader = new ParamLoader(std::move(weights_path));
    executor_ = new Executor(param_loader);
    net_ = new Net(bn);
    if (net_ == nullptr || executor_ == nullptr || !param_loader->isAvailible()) {
        return false;
    }
    auto size = param_loader->getParamSize();
    LOGI("param size:%d", size);
    model_ = model;

    switch (model) {
    case LLAMA: {
        int vocab_size = 32000;
        int hidden_dim = 4096;
        int ffn_hidden_dim = 11008;
        int mutil_head_size = 32;
        llama2(c, vocab_size, hidden_dim, ffn_hidden_dim, mutil_head_size);
        net_->convert(c->sub_param_, BackendType::MLLM_CPU);
        tokenizer_ = new BPETokenizer(vacab_path);
        eos_id_ = 2;
        break;
    }
    case FUYU:
        int vocab_size = 262144;
        int hidden_dim = 4096;
        int ffn_hidden_dim = 4096*4;
        int mutil_head_size = 64;
        int patch_size = 30;
        Fuyu(c, vocab_size, patch_size, 3, hidden_dim, ffn_hidden_dim, mutil_head_size);
        tokenizer_ = new UnigramTokenizer(vacab_path);
        pre_processor_ = new FuyuPreProcess(tokenizer_);
        eos_id_ = 71013;
        break;
    default: {
        return false;
    }
    }
    size = tokenizer_->getVocabSize();
    LOGI("tokenizer size:%d", size);
    if (!tokenizer_->isAvailible()) {
        return false;
    }
    return true;
}

void LibHelper::setCallback(callback_t callback) {
    this->callback_ = std::move(callback);
}

void LibHelper::run(const std::string &input_str, unsigned int max_step) const {
    auto tokens_id = vector<token_id_t>();
    tokenizer_->tokenize(input_str, tokens_id, true);
    auto out_string = input_str;
    shared_ptr<Tensor> input = std::make_shared<Tensor>();
    token2Tensor(input, *net_, tokens_id);

    std::cout << input << std::flush;
    for (int step = 0; step < max_step; step++) {
        executor_->execute(net_, {input});
        auto result = executor_->result();
        auto token_idx = postProcessing(result[0], input);
        const auto out_token = tokenizer_->detokenize({token_idx});
        if (out_token == "</s>" || token_idx == eos_id_) {
            break;
        }
        out_string += out_token;
        //TODO: End with EOS
        callback_(out_string, step == max_step - 1);
    }
}

LibHelper::~LibHelper() {
    delete c;
    delete net_;
    delete executor_;
    delete tokenizer_;
}
#endif
