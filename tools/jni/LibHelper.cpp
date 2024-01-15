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
//#include "NetParameter.hpp"
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
    LOGI("MODEL TYPE:%d", model_);

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
    case FUYU: {
        int vocab_size = 262144;
        int hidden_dim = 4096;
        int ffn_hidden_dim = 4096*4;
        int mutil_head_size = 64;
        int patch_size = 30;
        Fuyu(c, vocab_size, patch_size, 3, hidden_dim, ffn_hidden_dim, mutil_head_size);
        net_->convert(c->sub_param_, BackendType::MLLM_CPU);
        tokenizer_ = new UnigramTokenizer(vacab_path);
        pre_processor_ = new FuyuPreProcess(tokenizer_);
        eos_id_ = 71013;
        break;
    }
    default: {
        return false;
    }
    }
    size = tokenizer_->getVocabSize();
    LOGI("tokenizer size:%d", size);
    if (!tokenizer_->isAvailible()) {
        return false;
    }

    executor_->setup(net_);
    // switch (model_) {
    // case LLAMA: {
    //     shared_ptr<Tensor> initT =  mllm::Tokenizer::token2Tensor( net_, {0});
    //     executor_->setup(net_, {initT});
    //     break;
    // }
    // case FUYU: {
    //     shared_ptr<Tensor> initT = mllm::Tokenizer::token2Tensor( net_, {0});
    //     shared_ptr<Tensor> initIMG = std::make_shared<Tensor>();
    //     shared_ptr<Tensor> imgPatchId= std::make_shared<Tensor>();
    //     fullTensor(initIMG, net_, {0, 0, 0, 0},1.0F);
    //     fullTensor(imgPatchId, net_, {0, 0, 0, 0},1.0F);
    //     executor_->setup(net_, {initT, initIMG, imgPatchId});
    //     break;
    // }
    // }

    is_first_run_cond_ = true;
    return true;
}

void LibHelper::setCallback(callback_t callback) {
    this->callback_ = std::move(callback);
}

void LibHelper::run(std::string &input_str, uint8_t *image, unsigned max_step,unsigned int image_length)  {
    // PreProcess
    switch (model_) {
    case LLAMA: {
        if (input_str[0]!=' ') {
            input_str = ' ' + input_str;
        }
        break;
    }
    case FUYU: {
        if (input_str[input_str.length()-1]!='\n') {
            input_str = input_str + '\n';
        }
    }

    }
    auto tokens_id = vector<token_id_t>();
    shared_ptr<Tensor> input = std::make_shared<Tensor>();
    shared_ptr<Tensor> img_patch = std::make_shared<Tensor>();
    shared_ptr<Tensor> img_patch_id = std::make_shared<Tensor>();

    if (image!=nullptr) {
        LOGI("Image Found!");
        if (pre_processor_!=nullptr&&image_length>0) {
            switch (model_) {
            case FUYU:{
                const auto pre_processor = dynamic_cast<FuyuPreProcess*>(pre_processor_);
                pre_processor->PreProcessImages({image}, {image_length});//, 1080, 1920, true, true, true, 0.5, 0.5);
                pre_processor->Process(input_str);
                auto input_ids = pre_processor->image_input_ids_;
                if (input_ids.empty()) {
                    input_ids = pre_processor->text_ids_;
                }

                // input = mllm::Tokenizer::token2Tensor( net_, input_ids[0]);
                UnigramTokenizer::token2Tensor(net_, input_ids[0], input);
                const auto image_patches = pre_processor->image_patches_;
                const auto image_patch_indices = pre_processor->image_patches_indices_ ;
                patches2Tensor(img_patch, net_, image_patches);
                patchIdx2Tensor(img_patch_id, net_, image_patch_indices);
                break;
            }
            default:break;
            }
        }else {
            LOGE("Image Processor is not available!");
            return;
        }

    }else {
        if (model_ == LLAMA) {

            tokenizer_->tokenize(input_str, tokens_id, true);
            LOGI("is_first_run_cond_:%d", is_first_run_cond_);
            if (is_first_run_cond_) {
                is_first_run_cond_ = false;
            }else {
                LOGI("Keep Speaker!");
                if (tokens_id[0] >0) {
                    tokens_id[0] = 13;

                }
            }
            UnigramTokenizer::token2Tensor(net_, tokens_id, input);
            // input =  mllm::Tokenizer::token2Tensor (net_, tokens_id);
        }
    }

    auto out_string = std::string();

    // std::cout << input << std::flush;
    for (int step = 0; step < max_step; step++) {
        if (model_==FUYU) {
            LOGI("Image Patch!");
            executor_->run(net_, {input, img_patch, img_patch_id});
        }else {
            executor_->run(net_, {input});
        }
        auto result = executor_->result();
        auto token_idx = postProcessing(result[0], input);
        if (model_==FUYU) {
            // LOGI("1");
            fullTensor(img_patch, net_, {0, 0, 0, 0},1.0F);
            // LOGI("2");

            fullTensor(img_patch_id, net_, {0, 0, 0, 0},1.0F);
            // LOGI("3");

        }
        const auto out_token = tokenizer_->detokenize({token_idx});
        if (out_token == "</s>" || token_idx == eos_id_) {
            callback_(out_string,true);
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
