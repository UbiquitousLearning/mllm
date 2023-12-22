//
// Created by 咸的鱼 on 2023/12/16.
//

#ifndef LIBHELPER_HPP
#define LIBHELPER_HPP
#include <string>
#ifdef ANDROID_API
#include <android/asset_manager.h>
#include <android/log.h>
#define TAG "MLLM"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,TAG,__VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,TAG,__VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,TAG,__VA_ARGS__)
#include "functional"

namespace mllm {
class Tokenizer;
class PreProcessor;
}

struct Context;

namespace mllm {
class Backend;
class Net;
class Executor;
class Tensor;

enum PreDefinedModel {
    LLAMA = 0,
    FUYU,
};

enum MLLMBackendType {
    CPU = 0,
    GPU,
    NNAPI,
};

typedef std::function<void(std::string, bool)> callback_t;

class LibHelper {
    Context *c = nullptr;
    // AAssetManager* asset_manager_;
    Net *net_ = nullptr;
    Executor *executor_ = nullptr;
    callback_t callback_ = [](std::string, bool) {
    };
    Tokenizer *tokenizer_ = nullptr;
    PreProcessor *pre_processor_ = nullptr;
    unsigned int eos_id_ = 2;
    PreDefinedModel model_ = PreDefinedModel::LLAMA;
    bool is_first_run_cond_ = true;
    unsigned postProcessing(std::shared_ptr<Tensor> result, std::shared_ptr<Tensor> &out_result) const;
public:
    bool setUp(const std::string &base_path, std::string weights_path, std::string vacab_path, PreDefinedModel model, MLLMBackendType backend_type = MLLMBackendType::CPU);
    void setCallback(callback_t callback);
    void run(std::string &input_str, uint8_t *image, unsigned max_step, unsigned image_length) ;
    ~LibHelper();
};
} // mllm
#endif
#endif //LIBHELPER_HPP
