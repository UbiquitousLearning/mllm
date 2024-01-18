#include "QNNMemoryManager.hpp"
#include "Logger.hpp"
#include "QnnTypes.h"
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <iostream>

namespace mllm {

QNNMemoryManager::QNNMemoryManager() {
#ifdef QNN_ARM
    // load libcdsprpc.so
    void *libCdspHandle = pal::dynamicloading::dlOpen("libcdsprpc.so", pal::dynamicloading::DL_NOW | pal::dynamicloading::DL_LOCAL);
    if (nullptr == libCdspHandle) {
        std::cerr << "dlopen libcdsprpc.so failed" << std::endl;
    }

    rpcmem_alloc = (RpcMemAllocFn_t)dlsym(libCdspHandle, "rpcmem_alloc");
    rpcmem_free = (RpcMemFreeFn_t)dlsym(libCdspHandle, "rpcmem_free");
    rpcmem_to_fd = (RpcMemToFdFn_t)dlsym(libCdspHandle, "rpcmem_to_fd");

    if (nullptr == rpcmem_alloc || nullptr == rpcmem_free || nullptr == rpcmem_to_fd) {
        dlclose(libCdspHandle);
        std::cerr << "dlsym failed" << std::endl;
    }
#endif
}

QNNMemoryManager::~QNNMemoryManager() {
#ifdef QNN_ARM
    // free all buffers if it's not being used
    for (auto &memHandle : qnnMemHandleList_) {
        Qnn_ErrorHandle_t deregisterRet = qnnInterface_->memDeRegister(&memHandle, 1);
        if (QNN_SUCCESS != deregisterRet) {
            // handle errors
            std::cerr << "qnnInterface_->memDeRegister failed" << std::endl;
        }
    }
#endif
}

void QNNMemoryManager::alloc(void **ptr, size_t size, size_t alignment) {
    assert(size > 0);

#ifdef QNN_ARM
#define RPCMEM_HEAP_ID_SYSTEM 25
#define RPCMEM_DEFAULT_FLAGS 1
    // Allocate the shared buffer
    uint8_t *memPointer = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size);
    if (nullptr == memPointer) {
        std::cerr << "rpcmem_alloc failed" << std::endl;
    }

#else
    void **origin = (void **)malloc(size + sizeof(void *) + alignment - 1);
    assert(origin != nullptr);
    if (origin == nullptr) {
        *ptr = nullptr;
    }
    void **aligned = (void **)(((size_t)(origin) + sizeof(void *) + alignment - 1) & (~(alignment - 1)));
    // printf("origin = %p, align=%p\n",origin,aligned);
    aligned[-1] = origin;
    *ptr = aligned;
#endif
}

void QNNMemoryManager::registerQnnTensor(void *ptr, Qnn_Tensor_t &qnnTensor) {
#ifdef QNN_ARM
    auto it = qnnMemPtrMap_.find(ptr);
    if (it == qnnMemPtrMap_.end()) {
        std::cerr << "getMemHandle failed" << std::endl;
        return;
    }

    int memFd = rpcmem_to_fd(ptr);
    if (-1 == memFd) {
        std::cerr << "rpcmem_to_fd failed" << std::endl;
        return;
    }

    Qnn_MemDescriptor_t memDescriptor = QNN_MEM_DESCRIPTOR_INIT;
    memDescriptor.memShape = {qnnTensor.v1.rank, qnnTensor.v1.dimensions, nullptr};
    memDescriptor.dataType = qnnTensor.v1.dataType;
    memDescriptor.memType = QNN_MEM_TYPE_ION;
    memDescriptor.ionInfo.fd = memFd;
    qnnTensor.v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
    Qnn_ErrorHandle_t registRet = qnnInterface_->memRegister(this->context_, &memDescriptor, 1u, &(qnnTensor.v1.memHandle));
    if (registRet != QNN_SUCCESS) {
        std::cerr << "qnnInterface memRegister failed" << std::endl;
        return;
    }

    qnnMemHandleList_.push_back(qnnTensor.v1.memHandle);
#endif
}

void QNNMemoryManager::free(void *ptr) {
#ifdef QNN_ARM
    rpcmem_free(ptr);
#else
    ::free(((void **)ptr)[-1]);
#endif
}

} // namespace mllm