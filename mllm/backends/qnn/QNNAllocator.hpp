// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <map>
#include <set>
#include <vector>
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
  bool registerQnnTensorToSharedBuffer(Storage* storage, Qnn_Tensor_t& qnn_tensor);

  void deRegisterQnnTensorFromSharedBuffer(void* ptr);

  // Debug: Get statistics about registered buffers
  struct BufferStats {
    size_t count;
    size_t total_bytes;
  };
  [[nodiscard]] BufferStats getRegisteredBufferStats() const;
  
  // Debug: Check if a ptr is already registered
  bool isRegistered(void* ptr) const;

 private:
  QNN_INTERFACE_VER_TYPE qnnInterface_;
  Qnn_ContextHandle_t context_ = nullptr;

  RpcMemAllocFn_t rpcmem_alloc;
  RpcMemFreeFn_t rpcmem_free;
  RpcMemToFdFn_t rpcmem_to_fd;

  // to check if the ptr is allocted by rpcmem_alloc
  std::set<void*> qnnMemPtrSet_;
  std::map<void*, std::pair<int, Qnn_MemHandle_t>> ptrToFdAndMemHandleMap_;
  // Track buffer sizes for statistics
  std::map<void*, size_t> ptrToSizeMap_;
  // Map tensor name to registered buffer ptr for reuse
  std::map<std::string, void*> tensorNameToPtrMap_;
  // Map tensor ID to registered buffer ptr for reuse (more reliable than name)
  std::map<uint32_t, void*> tensorIdToPtrMap_;

};

std::shared_ptr<QNNAllocator> createQNNAllocator();

}  // namespace mllm::qnn
