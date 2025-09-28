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

  ~QNNAllocator() {
    for (auto iter = ptrToFdAndMemHandleMap_.begin(); iter != ptrToFdAndMemHandleMap_.end();) {
      Qnn_ErrorHandle_t deregisterRet = qnnInterface_.memDeRegister(&iter->second.second, 1);
      if (QNN_SUCCESS != deregisterRet) { MLLM_ERROR("~QNNAllocator: qnnInterface_.memDeRegister failed"); }
      rpcmem_free(iter->first);
      iter = ptrToFdAndMemHandleMap_.erase(iter);
    }
  }

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

  RpcMemAllocFn_t rpcmem_alloc;
  RpcMemFreeFn_t rpcmem_free;
  RpcMemToFdFn_t rpcmem_to_fd;

  // to check if the ptr is allocted by rpcmem_alloc
  std::set<void*> qnnMemPtrSet_;
  std::map<void*, std::pair<int, Qnn_MemHandle_t>> ptrToFdAndMemHandleMap_;
};

std::shared_ptr<QNNAllocator> createQNNAllocator();

}  // namespace mllm::qnn
