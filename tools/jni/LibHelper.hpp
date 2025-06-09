//
// Created by Xiang Li on 2023/12/16.
//

#ifndef LIBHELPER_HPP
#define LIBHELPER_HPP

#include <memory>
#include <string>
// #ifdef ANDROID_API
#include <android/log.h>
#include <vector>
#define TAG "MLLM"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#include "functional"
namespace mllm {
class Tokenizer;
class PreProcessor;
class Module;
class Tensor;
enum PreDefinedModel {
    QWEN25 = 0,
    FUYU,
    Bert,
    PhoneLM,
    QWEN15,
    QWEN2VL
};

enum MLLMBackendType {
    CPU = 0,
    QNN,
};

typedef std::function<void(std::string, bool, std::vector<double>)> callback_t;

class LibHelper {
    // Context *c = nullptr;
    // Net *net_ = nullptr;
    // Executor *executor_ = nullptr;
    callback_t callback_ = [](std::string, bool, std::vector<double>) {
    };

    std::shared_ptr<Tokenizer> tokenizer_;
    PreProcessor *processor_;
    std::shared_ptr<Module> module_;
    std::shared_ptr<Module> prefill_module_;

    // Tokenizer *tokenizer_ = nullptr;
    unsigned int eos_id_ = 2;
    PreDefinedModel model_ = PreDefinedModel::QWEN25;
    MLLMBackendType backend_ = MLLMBackendType::CPU;
    bool is_first_run_cond_ = true;
    int tokens_limit = 4000;
    unsigned postProcessing(std::shared_ptr<Tensor> result, std::shared_ptr<Tensor> &out_result) const;

public:
    bool setUp(const std::string &base_path, std::string weights_path, std::string qnn_weights_path, std::string vocab_path, std::string merge_path, PreDefinedModel model, MLLMBackendType backend_type = MLLMBackendType::CPU);
    void setCallback(callback_t callback);
    void run(std::string &input_str, uint8_t *image, unsigned max_step, unsigned image_length, bool chat_template = false);
    std::vector<float> runForResult(std::string &input_str);
    ~LibHelper();
};
} // namespace mllm
#endif
// #endif //LIBHELPER_HPP
