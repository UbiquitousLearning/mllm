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
}

QNNMemoryManager::~QNNMemoryManager() {
    // free all buffers if it's not being used
    for (auto &memHandle : qnnMemHandleList_) {
        Qnn_ErrorHandle_t deregisterRet = qnnInterface_->memDeRegister(&memHandle, 1);
        if (QNN_SUCCESS != deregisterRet) {
            // handle errors
            std::cerr << "qnnInterface_->memDeRegister failed" << std::endl;
        }
    }
}

void QNNMemoryManager::alloc(void **ptr, size_t size, size_t alignment) {
    assert(size > 0);

    // Calculate the size base on tensor dimensions and data type ......
#define RPCMEM_HEAP_ID_SYSTEM 25
#define RPCMEM_DEFAULT_FLAGS 1
    // Allocate the shared buffer
    uint8_t *memPointer = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size);
    if (nullptr == memPointer) {
        std::cerr << "rpcmem_alloc failed" << std::endl;
    }
    int memFd = rpcmem_to_fd(memPointer);
    if (-1 == memFd) {
        std::cerr << "rpcmem_to_fd failed" << std::endl;
    }
    // Fill the info of Qnn_MemDescriptor_t and regist the buffer to QNN
    // Qnn_MemDescriptor_t is defined in ${QNN_SDK_ROOT}/include/QNN/QnnMem.h
    Qnn_MemDescriptor_t memDescriptor = QNN_MEM_DESCRIPTOR_INIT;
    Qnn_Tensor_t inputTensor;
    uint32_t dimensions[] = {static_cast<uint32_t>(size)};
    memDescriptor.memShape = {1, dimensions, nullptr};
    memDescriptor.dataType = QNN_DATATYPE_INT_8;
    memDescriptor.memType = QNN_MEM_TYPE_ION;
    memDescriptor.ionInfo.fd = memFd;

    Qnn_ErrorHandle_t registRet = qnnInterface_->memRegister(*context_, &memDescriptor, 1u, &(inputTensor.v1.memHandle));
    if (QNN_SUCCESS != registRet) {
        rpcmem_free(memPointer);
        // handle errors
        std::cerr << "qnnInterface_->memRegister failed" << std::endl;
    }

    qnnMemHandleList_.push_back(inputTensor.v1.memHandle);
    qnnMemPtrMap_[memPointer] = inputTensor.v1.memHandle;
    *ptr = memPointer;
    /**
     * At this place, the allocation and registration of the shared buffer has been complete.
     * On QNN side, the buffer has been bound by memfd
     * On user side, this buffer can be manipulated through memPointer.
     */
    /**
     * Optionally, user can also allocate and register shared buffer for output as adove codes (lines 7-46).
     * And if so the output buffer also should be deregistered and freed as below codes (lines 66-70).
     */
    // Load the input data to memPointer ......
    // Execute QNN graph with input tensor and output tensor ......
    // Get output data ......
    // Deregister and free all buffers if it's not being used
}

void QNNMemoryManager::free(void *ptr) {
    rpcmem_free(ptr);
}

} // namespace mllm