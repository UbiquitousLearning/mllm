#include "Context.hpp"

#include "backends/cpu/CPUBackend.hpp"
#include "memory/MemoryPoolManager.hpp"

namespace mllm {
Context &Context::Instance() {
    static Context instance;
    return instance;
}

Context::Context() {
}

// void Context::initBackend(BackendType type) {
//     if (Backend::global_backends.find(type) == Backend::global_backends.end() || Backend::global_backends[type] == nullptr) {
//         switch (type) {
//         case BackendType::MLLM_CPU: {
//             shared_ptr<MemoryManager> mm = nullptr;
//             // mm = std::make_shared<SystemMemoryManager>();
//             mm = std::make_shared<MemoryPoolManager>(); // todomm
//             Backend::global_backends[MLLM_CPU] = new CPUBackend(mm);
//             break;
//         }
// #ifdef USE_QNN
//         case BackendType::MLLM_QNN: {
//             Backend::global_backends.emplace(MLLM_QNN, GetBackendCreator(MLLM_QNN)->create({}));
//             break;
//         }
// #endif
// #ifdef MLLM_BUILD_XNNPACK_BACKEND
//         case BackendType::MLLM_XNNPACK: {
//             Context::Instance().initBackend(MLLM_XNNPACK);
//             break;
//         }
// #endif
//         default: {
//         }
//         }
//     }
// }
} // namespace mllm