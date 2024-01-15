#ifndef MLLM_QNNMEMORY_SYSTEM_H
#define MLLM_QNNMEMORY_SYSTEM_H

#include "Logger.hpp"
#include "MemoryManager.hpp"
#include "PAL/DynamicLoading.hpp"
#include "DynamicLoadUtil.hpp"
#include <cstddef>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <dlfcn.h>

namespace mllm {

typedef void *(*RpcMemAllocFn_t)(int, uint32_t, int);
typedef void (*RpcMemFreeFn_t)(void *);
typedef int (*RpcMemToFdFn_t)(void *);
typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(const QnnInterface_t ***providerList,
                                                          uint32_t *numProviders);

class QNNMemoryManager : public MemoryManager {
public:
    QNNMemoryManager();
    ~QNNMemoryManager();

    void alloc(void **ptr, size_t size, size_t alignment) override;
    void free(void *ptr) override;

    void setQnnInterfaceAndContext(QNN_INTERFACE_VER_TYPE &qnnInterface, Qnn_ContextHandle_t &context) {
        this->qnnInterface_ = &qnnInterface;
        this->context_ = &context;

        if (context == nullptr) {
            QNN_ERROR("qnnInterface or context is nullptr");
            exit(1);
        }
    }

    Qnn_MemHandle_t getMemHandle(void *ptr) const {
        auto it = qnnMemPtrMap_.find(ptr);
        if (it == qnnMemPtrMap_.end()) {
            std::cerr << "getMemHandle failed" << std::endl;
            exit(1);
        }
        return it->second;
    }

private:
    QNN_INTERFACE_VER_TYPE *qnnInterface_ = nullptr;
    Qnn_ContextHandle_t *context_ = nullptr;

    std::vector<Qnn_MemHandle_t> qnnMemHandleList_;
    std::vector<void *> qnnMemPtrList_;
    // relation between buffer memPointer and qnn memHandle
    std::unordered_map<void *, Qnn_MemHandle_t> qnnMemPtrMap_;

    RpcMemAllocFn_t rpcmem_alloc;
    RpcMemFreeFn_t rpcmem_free;
    RpcMemToFdFn_t rpcmem_to_fd;
};

} // namespace mllm
#endif