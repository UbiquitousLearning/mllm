// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/qnn/QNNAllocator.hpp"
#include "mllm/backends/qnn/QNNTypeMacros.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/utils/Log.hpp"
#include <dlfcn.h>

namespace mllm::qnn {

// specified in QNN doc
#define RPCMEM_HEAP_ID_SYSTEM 25
#define RPCMEM_DEFAULT_FLAGS 1

QNNAllocator::QNNAllocator() {
  void* libCdspHandle = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
  if (nullptr == libCdspHandle) { MLLM_ERROR_EXIT(1, "dlopen libcdsprpc.so failed"); }

  rpcmem_alloc = (RpcMemAllocFn_t)dlsym(libCdspHandle, "rpcmem_alloc");
  rpcmem_free = (RpcMemFreeFn_t)dlsym(libCdspHandle, "rpcmem_free");
  rpcmem_to_fd = (RpcMemToFdFn_t)dlsym(libCdspHandle, "rpcmem_to_fd");
}

QNNAllocator::QNNAllocator(QNN_INTERFACE_VER_TYPE qnnInterface, void* context)
    : qnnInterface_(qnnInterface), context_(context) {
  MLLM_RT_ASSERT(context_ != nullptr);

  void* libCdspHandle = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
  if (nullptr == libCdspHandle) { MLLM_ERROR_EXIT(1, "dlopen libcdsprpc.so failed"); }

  rpcmem_alloc = (RpcMemAllocFn_t)dlsym(libCdspHandle, "rpcmem_alloc");
  rpcmem_free = (RpcMemFreeFn_t)dlsym(libCdspHandle, "rpcmem_free");
  rpcmem_to_fd = (RpcMemToFdFn_t)dlsym(libCdspHandle, "rpcmem_to_fd");
}

bool QNNAllocator::alloc(Storage* storage) {
  uint8_t* ptr = (uint8_t*)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, allocSize(storage));

  MLLM_RT_ASSERT(ptr != nullptr);

  storage->ptr_ = ptr;
  qnnMemPtrSet_.insert(ptr);

  return true;
}

void QNNAllocator::free(Storage* storage) {
  if (ptrToFdAndMemHandleMap_.count(storage->ptr_)) {
    MLLM_RT_ASSERT_EQ(QNN_SUCCESS,
                      qnnInterface_.memDeRegister(&(ptrToFdAndMemHandleMap_.find(storage->ptr_)->second.second), 1));
  }

  rpcmem_free(storage->ptr_);
}

void QNNAllocator::registerQnnTensorToSharedBuffer(void* ptr, Qnn_Tensor_t& qnn_tensor) {
  // Make sure there has a memory that we can register to.
  MLLM_RT_ASSERT(qnnMemPtrSet_.count(ptr));

  // Make sure this memory space is not registered yet.
  MLLM_RT_ASSERT(ptrToFdAndMemHandleMap_.count(ptr) == 0);

  // Get the file id of this memory space.
  int mem_fd = rpcmem_to_fd(ptr);
  MLLM_RT_ASSERT(mem_fd != -1);

  // Make qnn memory descriptor. Set ION.
  Qnn_MemDescriptor_t mem_descriptor = QNN_MEM_DESCRIPTOR_INIT;
  mem_descriptor.memShape = {
      .numDim = QNN_TENSOR_GET_RANK(qnn_tensor),
      .dimSize = QNN_TENSOR_GET_DIMENSIONS(qnn_tensor),
      .shapeConfig = nullptr,
  };
  mem_descriptor.dataType = QNN_TENSOR_GET_DATA_TYPE(qnn_tensor);
  mem_descriptor.memType = QNN_MEM_TYPE_ION;
  mem_descriptor.ionInfo.fd = mem_fd;
  QNN_TENSOR_SET_MEM_TYPE(qnn_tensor, QNN_TENSORMEMTYPE_MEMHANDLE);

  // Register to QNN memory
  Qnn_MemHandle_t mem_handle = QNN_TENSOR_GET_MEM_HANDLE(qnn_tensor);
  MLLM_RT_ASSERT_EQ(QNN_SUCCESS, qnnInterface_.memRegister(context_, &mem_descriptor, 1u, &mem_handle));

  ptrToFdAndMemHandleMap_.insert({ptr, {mem_fd, mem_handle}});
}

void QNNAllocator::deRegisterQnnTensorFromSharedBuffer(void* ptr) {
  MLLM_RT_ASSERT_EQ(ptrToFdAndMemHandleMap_.count(ptr), 1);
  MLLM_RT_ASSERT_EQ(QNN_SUCCESS, qnnInterface_.memDeRegister(&(ptrToFdAndMemHandleMap_[ptr].second), 1));
  ptrToFdAndMemHandleMap_.erase(ptr);
}

std::shared_ptr<QNNAllocator> createQNNAllocator() { return std::make_shared<QNNAllocator>(); }

}  // namespace mllm::qnn
