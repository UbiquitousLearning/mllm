#include "Backend.hpp"
#include "Types.hpp"
#include <memory>
#include <unordered_map>
#include <mutex>
#include "Layer.hpp"

namespace mllm {
extern void registerCPUBackendCreator();
#ifdef USE_QNN
extern void registerQNNBackendCreator();
#elif defined(MLLM_BUILD_XNNPACK_BACKEND)
extern void registerXNNBackendCreator();
#endif

static std::once_flag s_flag;
void registerBackend() {
    std::call_once(s_flag, [&]() {
        registerCPUBackendCreator();
#ifdef USE_QNN
        registerQNNBackendCreator();
#elif defined(MLLM_BUILD_XNNPACK_BACKEND)
            registerXNNBackendCreator();
#endif
    });
}

static std::unordered_map<BackendType, std::shared_ptr<BackendCreator>> &GetBackendCreatorMap() {
    static std::once_flag gInitFlag;
    static std::unordered_map<BackendType, std::shared_ptr<BackendCreator>> *gBackendCreatorMap;
    std::call_once(gInitFlag,
                   [&]() { gBackendCreatorMap = new std::unordered_map<BackendType, std::shared_ptr<BackendCreator>>; });
    return *gBackendCreatorMap;
}

const std::shared_ptr<BackendCreator> GetBackendCreator(BackendType type) {
    if (type == MLLM_QNN || type == MLLM_XNNPACK) {
        Layer::use_layername_2_tensorname = false;
    }
    registerBackend();

    auto &gExtraCreator = GetBackendCreatorMap();
    auto iter = gExtraCreator.find(type);
    if (iter == gExtraCreator.end()) {
        return nullptr;
    }
    if (nullptr != iter->second) {
        return iter->second;
    }
    return nullptr;
}

bool InsertBackendCreatorMap(BackendType type, shared_ptr<BackendCreator> creator) {
    auto &gBackendCreator = GetBackendCreatorMap();
    if (gBackendCreator.find(type) != gBackendCreator.end()) {
        return false;
    }
    gBackendCreator.emplace(type, creator);
    return true;
}

map<BackendType, Backend *> Backend::global_backends;

} // namespace mllm