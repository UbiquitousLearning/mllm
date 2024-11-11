#ifndef MLLM_QNNMEMORY_SYSTEM_H
#define MLLM_QNNMEMORY_SYSTEM_H
#include "Log.h"
#include "Log/Logger.hpp"
#include "MemoryManager.hpp"
#include "PAL/DynamicLoading.hpp"
#include "Utils/DynamicLoadUtil.hpp"
#include "QnnTypes.h"
#include <cstddef>
#include <iostream>
#include <map>
#include <set>
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

    void setQnnInterfaceAndContext(void *context);

    void registerQnnTensor(void *ptr, Qnn_Tensor_t &qnnTensor);
    void deRegisterQnnTensor();

private:
    QNN_INTERFACE_VER_TYPE qnnInterface_;
    Qnn_ContextHandle_t context_ = nullptr;

    // memHandle set, to check if the ptr is allocted by rpcmem_alloc
    std::set<void *> qnnMemPtrMap_;
    std::map<void*, std::pair<int, Qnn_MemHandle_t>> ptrToFdAndMemHandleMap_;

    RpcMemAllocFn_t rpcmem_alloc;
    RpcMemFreeFn_t rpcmem_free;
    RpcMemToFdFn_t rpcmem_to_fd;
};

} // namespace mllm
#endif