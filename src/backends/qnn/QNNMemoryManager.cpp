#include "QNNMemoryManager.hpp"
#include "Log.h"
#include "Logger.hpp"
#include "QnnTypes.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <utility>

namespace mllm {

template <class T>
static inline T resolveSymbol(void *libHandle, const char *sym) {
    T ptr = (T)pal::dynamicloading::dlSym(libHandle, sym);
    if (ptr == nullptr) {
        MLLM_LOG_ERROR("Unable to access symbol {}. pal::dynamicloading::dlError(): {}",
                       sym,
                       pal::dynamicloading::dlError());
    }
    return ptr;
}

QNNMemoryManager::QNNMemoryManager() {
#ifdef QNN_ARM
    // load libcdsprpc.so
    void *libCdspHandle = pal::dynamicloading::dlOpen("libcdsprpc.so", pal::dynamicloading::DL_NOW | pal::dynamicloading::DL_LOCAL);
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
    // Get QNN Interface
    void *libBackendHandle = pal::dynamicloading::dlOpen(
        "libQnnHtp.so", pal::dynamicloading::DL_NOW | pal::dynamicloading::DL_GLOBAL);
    QnnInterfaceGetProvidersFn_t getInterfaceProviders{nullptr};
    getInterfaceProviders =
        resolveSymbol<QnnInterfaceGetProvidersFn_t>(libBackendHandle, "QnnInterface_getProviders");
    QnnInterface_t **interfaceProviders{nullptr};
    uint32_t numProviders{0};
    if (QNN_SUCCESS != getInterfaceProviders((const QnnInterface_t ***)&interfaceProviders, &numProviders)) {
        MLLM_LOG_ERROR_STREAM << "Failed to get interface providers." << std::endl;
    }
    for (size_t pIdx = 0; pIdx < numProviders; pIdx++) {
        if (QNN_API_VERSION_MAJOR == interfaceProviders[pIdx]->apiVersion.coreApiVersion.major && QNN_API_VERSION_MINOR <= interfaceProviders[pIdx]->apiVersion.coreApiVersion.minor) {
            qnnInterface_ = interfaceProviders[pIdx]->QNN_INTERFACE_VER_NAME;
            break;
        }
    }
}

QNNMemoryManager::~QNNMemoryManager() {
#ifdef QNN_ARM
    for (auto &mem : ptrToFdAndMemHandleMap_) {
        Qnn_ErrorHandle_t deregisterRet = qnnInterface_.memDeRegister(&mem.second.second, 1);
        if (QNN_SUCCESS != deregisterRet) {
            // handle errors
            MLLM_LOG_ERROR_STREAM << "qnnInterface_.memDeRegister failed" << std::endl;
        }
        rpcmem_free(mem.first);
        ptrToFdAndMemHandleMap_.erase(mem.first);
    }
#endif
}

void QNNMemoryManager::setQnnInterfaceAndContext(void *context) {
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

void QNNMemoryManager::deRegisterQnnTensor() {
#ifdef QNN_ARM
    // free all buffers if it's not being used
    for (auto &mem : ptrToFdAndMemHandleMap_) {
        Qnn_ErrorHandle_t deregisterRet = qnnInterface_.memDeRegister(&mem.second.second, 1);
        if (QNN_SUCCESS != deregisterRet) {
            // handle errors
            MLLM_LOG_ERROR_STREAM << "qnnInterface_.memDeRegister failed" << std::endl;
        }
        // rpcmem_free(mem.first);
        // clear the map outside the loop.
        // ptrToFdAndMemHandleMap_.erase(mem.first);
    }
    ptrToFdAndMemHandleMap_.clear();
#endif
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
        ptrToFdAndMemHandleMap_.erase(it);
    }
    rpcmem_free(ptr);
#else
    ::free(((void **)ptr)[-1]);
#endif
}

} // namespace mllm