// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <set>
#include <map>
#include "QnnCommon.h"
#include "QnnInterface.h"
#include "mllm/backends/base/Allocator.hpp"
#include "mllm/core/Storage.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::qnn {

using RpcMemAllocFn_t = void* (*)(int, uint32_t, int);
using RpcMemFreeFn_t = void (*)(void*);
using RpcMemToFdFn_t = int (*)(void*);

/**
 * @brief QNN Allocator
 *
 * This class implements the QNN memory allocator interface.
 * It ONLY supports QNN shared buffer now. see
 * https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/htp_shared_buffer_tutorial.html
 */

class QNNAllocator final : public Allocator {
 public:
  QNNAllocator();  // need to setQNNPointer afterward
  QNNAllocator(QNN_INTERFACE_VER_TYPE qnnInterface, void* context);

  ~QNNAllocator();

  // Explicitly release all QNN memory resources. Call this for proper cleanup when you
  // want to release memory during normal operation (not program exit).
  // This is SAFE to call and will properly free all QNN resources.
  void shutdown() {
    if (isShutdown_) return;
    isShutdown_ = true;

    // First, deregister all registered memory
    for (auto iter = ptrToFdAndMemHandleMap_.begin(); iter != ptrToFdAndMemHandleMap_.end();) {
      Qnn_ErrorHandle_t deregisterRet = qnnInterface_.memDeRegister(&iter->second.second, 1);
      if (QNN_SUCCESS != deregisterRet) { MLLM_ERROR("QNNAllocator::shutdown: qnnInterface_.memDeRegister failed"); }
      iter = ptrToFdAndMemHandleMap_.erase(iter);
    }

    // Then, free all allocated memory (registered or not)
    MLLM_INFO("QNNAllocator::shutdown: freeing all allocated memory");
    for (void* ptr : qnnMemPtrSet_) { rpcmem_free(ptr); }
    qnnMemPtrSet_.clear();
  }

  // Legacy name for shutdown() - kept for compatibility
  void releaseAllResources() { shutdown(); }

  // Mark the allocator as shut down without actually freeing memory.
  // Use this in destructors to prevent crashes during program exit when
  // QNN library resources might already be destroyed.
  // After this is called, all free() calls become no-ops.
  void markShutdown() { isShutdown_ = true; }

  void setQNNPointer(QNN_INTERFACE_VER_TYPE qnnInterface, void* context) {
    this->qnnInterface_ = qnnInterface;
    this->context_ = context;
  }

  inline bool ctrlByMemManager() override { return false; }

  bool alloc(Storage* storage) override;

  bool alloc(const Storage::ptr_t& storage) override { return alloc(storage.get()); }

  void free(Storage* storage) override;

  void free(const Storage::ptr_t& storage) override { free(storage.get()); }

  // general alloc/free is needed by MemoryManager to alloc memory pools
  bool generalAlloc(void** ptr, size_t cap, size_t align) override {
    MLLM_ERROR("Should not call generalAlloc() for QNNAllocator");
    return false;
  }
  void generalFree(void* ptr) override { MLLM_ERROR("Should not call generalFree() for QNNAllocator"); }

  size_t allocSize(Storage* storage) override {
    size_t align_size = alignSize();
    size_t required_size = storage->size_;
    size_t aligned_size = (required_size + align_size - 1) & ~(align_size - 1);
    return aligned_size;
  }

  size_t allocSize(const Storage::ptr_t& storage) override { return allocSize(storage.get()); }

  // QNN does not specify the alignment requirement, but 16 bytes is a common choice
  [[nodiscard]] size_t alignSize() const override { return 16; };

  // Sharing access in between processing domains in QNN HTP backend. Using shared buffers can
  // eliminate data copy in between client code on the host CPU and HTP accelerator.
  void registerQnnTensorToSharedBuffer(void* ptr, Qnn_Tensor_t& qnn_tensor);

  void deRegisterQnnTensorFromSharedBuffer(void* ptr);

 private:
  QNN_INTERFACE_VER_TYPE qnnInterface_;
  Qnn_ContextHandle_t context_ = nullptr;

  // Hold the library handle to control unload order
  // libcdsprpc.so will only be unloaded when this allocator is destroyed
  void* libCdspHandle_ = nullptr;

  RpcMemAllocFn_t rpcmem_alloc = nullptr;
  RpcMemFreeFn_t rpcmem_free = nullptr;
  RpcMemToFdFn_t rpcmem_to_fd = nullptr;

  // to check if the ptr is allocted by rpcmem_alloc
  std::set<void*> qnnMemPtrSet_;
  std::map<void*, std::pair<int, Qnn_MemHandle_t>> ptrToFdAndMemHandleMap_;

  // Flag to indicate shutdown has been called or destructor is running
  // When true, free() calls become no-ops to avoid crashes during program exit
  bool isShutdown_ = false;
};

std::shared_ptr<QNNAllocator> createQNNAllocator();

}  // namespace mllm::qnn
