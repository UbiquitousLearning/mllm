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

public:
    bool setUp(const std::string &base_path, std::string weights_path, std::string vacab_path, PreDefinedModel model, MLLMBackendType backend_type = MLLMBackendType::CPU);
    void setCallback(callback_t callback);
    void run(const std::string &input_str, unsigned int max_step) const;
    ~LibHelper();
};
} // mllm
#endif
#endif //LIBHELPER_HPP
