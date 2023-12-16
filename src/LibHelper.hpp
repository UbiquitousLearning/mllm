//
// Created by 咸的鱼 on 2023/12/16.
//

#ifndef LIBHELPER_HPP
#define LIBHELPER_HPP
#include <string>
#ifdef ANDROID_API
#include <android/asset_manager.h>
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
typedef  std::function<void(std::string,bool)> callback_t;
class LibHelper {
    Context *c=nullptr;
    AAssetManager* asset_manager_;
    Net *net_;
    Executor *executor_;
    callback_t callback_ = [](std::string,bool){};
    Tokenizer *tokenizer_;
public:
    explicit LibHelper(AAssetManager* asset_manager,std::string weights_path,std::string vacab_path);
    void setUp(PreDefinedModel model,MLLMBackendType backend_type);
    void setCallback(callback_t callback);
    void run(const std::string& input_str,unsigned int max_step) const;
};

} // mllm
#endif
#endif //LIBHELPER_HPP
