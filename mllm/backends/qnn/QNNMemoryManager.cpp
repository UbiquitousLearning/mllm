#include "QNNMemoryManager.hpp"
#include "Log.h"
#include "QnnTypes.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <utility>
#include <dlfcn.h>

namespace mllm {

QNNMemoryManager::QNNMemoryManager() {
#ifdef QNN_ARM
    // load libcdsprpc.so
    void *libCdspHandle = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
    if (nullptr == libCdspHandle) {
        MLLM_LOG_ERROR_STREAM << "dlopen libcdsprpc.so failed" << std::endl;
    }

    rpcmem_alloc = (RpcMemAllocFn_t)dlsym(libCdspHandle, "rpcmem_alloc");
    rpcmem_free = (RpcMemFreeFn_t)dlsym(libCdspHandle, "rpcmem_free");
    rpcmem_to_fd = (RpcMemToFdFn_t)dlsym(libCdspHandle, "rpcmem_to_fd");

    if (nullptr == rpcmem_alloc || nullptr == rpcmem_free || nullptr == rpcmem_to_fd) {
        dlclose(libCdspHandle);
        MLLM_LOG_ERROR_STREAM << "dlsym failed" << std::endl;
    }
#endif
}

QNNMemoryManager::~QNNMemoryManager() {
#ifdef QNN_ARM
    for (auto iter = ptrToFdAndMemHandleMap_.begin(); iter != ptrToFdAndMemHandleMap_.end();) {
        Qnn_ErrorHandle_t deregisterRet = qnnInterface_.memDeRegister(&iter->second.second, 1);
        if (QNN_SUCCESS != deregisterRet) {
            // handle errors
            MLLM_LOG_ERROR_STREAM << "qnnInterface_.memDeRegister failed" << std::endl;
        }
        rpcmem_free(iter->first);
        iter = ptrToFdAndMemHandleMap_.erase(iter);
    }
#endif
}

void QNNMemoryManager::setQnnInterfaceAndContext(QNN_INTERFACE_VER_TYPE qnnInterface, void *context) {
    qnnInterface_ = qnnInterface;
    context_ = context;
    if (context_ == nullptr) {
        MLLM_LOG_ERROR_STREAM << "context is null" << std::endl;
        exit(1);
    }
}

void QNNMemoryManager::alloc(void **ptr, size_t size, size_t alignment) {
    assert(size > 0);
#ifdef DEBUGPRINT
    std::cout << "QNN alloc size: " << size << std::endl;
#endif
#ifdef QNN_ARM
#define RPCMEM_HEAP_ID_SYSTEM 25
#define RPCMEM_DEFAULT_FLAGS 1
    // Allocate the shared buffer
    uint8_t *memPointer = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size);
    if (nullptr == memPointer) {
        MLLM_LOG_ERROR_STREAM << "rpcmem_alloc failed" << std::endl;
    }
    qnnMemPtrMap_.insert(memPointer);
    *ptr = memPointer;
#else
    // void **origin = (void **)malloc(size + sizeof(void *) + alignment - 1);
    void *origin = (void *)malloc(size + sizeof(void *) + alignment - 1);
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
    auto it = qnnMemPtrMap_.find(ptr);
    if (it == qnnMemPtrMap_.end()) {
        MLLM_LOG_ERROR_STREAM << "getMemHandle failed " << ptr << std::endl;
        return;
    }

    // check if the ptr has been registered, if so assign the memHandle
    auto mapIt = ptrToFdAndMemHandleMap_.find(ptr);
    if (mapIt != ptrToFdAndMemHandleMap_.end()) {
        qnnTensor.v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
        qnnTensor.v1.memHandle = mapIt->second.second;
        return;
    }

    int memFd = rpcmem_to_fd(ptr);
    if (-1 == memFd) {
        MLLM_LOG_ERROR_STREAM << "rpcmem_to_fd failed" << std::endl;
        return;
    }

    Qnn_MemDescriptor_t memDescriptor = QNN_MEM_DESCRIPTOR_INIT;
    memDescriptor.memShape = {qnnTensor.v1.rank, qnnTensor.v1.dimensions, nullptr};
    memDescriptor.dataType = qnnTensor.v1.dataType;
    memDescriptor.memType = QNN_MEM_TYPE_ION;
    memDescriptor.ionInfo.fd = memFd;
    qnnTensor.v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
    Qnn_ErrorHandle_t registRet = qnnInterface_.memRegister(context_, &memDescriptor, 1u, &(qnnTensor.v1.memHandle));
    if (registRet != QNN_SUCCESS) {
        MLLM_LOG_ERROR_STREAM << "qnnInterface memRegister failed" << std::endl;
        return;
    }

    ptrToFdAndMemHandleMap_.insert(std::make_pair(ptr, std::make_pair(memFd, qnnTensor.v1.memHandle)));
}

void QNNMemoryManager::free(void *ptr) {
#ifdef QNN_ARM
    // if the ptr has been registered, deregister it
    auto it = ptrToFdAndMemHandleMap_.find(ptr);
    if (it != ptrToFdAndMemHandleMap_.end()) {
        Qnn_ErrorHandle_t deregisterRet = qnnInterface_.memDeRegister(&it->second.second, 1);
        if (QNN_SUCCESS != deregisterRet) {
            // handle errors
            MLLM_LOG_ERROR_STREAM << "qnnInterface_.memDeRegister failed" << std::endl;
        }
        it = ptrToFdAndMemHandleMap_.erase(it);
    }
    rpcmem_free(ptr);
#else
    ::free(((void **)ptr)[-1]);
#endif
}

} // namespace mllm