#include "Backend.hpp"
#include "QNNBackend.hpp"
#include "QNNMemoryManager.hpp"
#include "memory/SystemMemoryManager.hpp"

namespace mllm {

class QNNBackendRegister : public BackendCreator {
public:
    shared_ptr<Backend> create(BackendConfig config) {
        shared_ptr<MemoryManager> mm = nullptr;
        mm = std::make_shared<QNNMemoryManager>();
        return std::make_shared<QNNBackend>(mm);
    };
};

void registerQNNBackendCreator() {
    InsertBackendCreatorMap(MLLM_QNN, std::make_shared<QNNBackendRegister>());
}

} // namespace mllm