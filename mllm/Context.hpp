#pragma once

#include "StateManager.hpp"
#include "Types.hpp"
#include "Backend.hpp"

namespace mllm {

class Context {
public:
    static Context &Instance();

    // Backend *globalBackends(BackendType type) const {
    //     return Backend::global_backends[type];
    // }

    // template<class T>
    // T *globalBackends(BackendType type) const {
    //     auto backend = Backend::global_backends[type];
    //     if (backend == nullptr) {
    //         throw std::runtime_error("Backend not initialized: " + std::to_string(type));
    //     }
    //     return dynamic_cast<T *>(backend);
    // }

    // void initBackend(BackendType type);

    InferenceStateManager &inference_state() {
        return inference_state_;
    }

    SpeculativeDecodingManager &speculative_decoding_state() {
        return speculative_decoding_state_;
    }

private:
    Context();
    ~Context() = default;

    Context(const Context &) = delete;
    Context &operator=(const Context &) = delete;

    InferenceStateManager inference_state_;
    SpeculativeDecodingManager speculative_decoding_state_;
};

} // namespace mllm