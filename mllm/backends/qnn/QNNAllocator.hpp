// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <map>
#include <set>
#include <string_view>
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
  [[nodiscard]] size_t getRegisteredBufferSize(void* ptr) const;

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
  // Map tensor name to registered buffer ptr for reuse (fallback identifier)
  // Used when tensor ID is 0 or unavailable
  std::map<std::string, void*> tensorNameToPtrMap_;
  
  // Map tensor ID to registered buffer ptr for reuse (primary identifier)
  // Tensor ID is more reliable than name and is used as the primary lookup key
  // This enables buffer reuse across prefill and decode phases
  std::map<uint32_t, void*> tensorIdToPtrMap_;

  /**
   * @brief Information about the last successful buffer registration
   * 
   * This structure stores metadata about the most recent successful registration,
   * which is used as a last-resort fallback when:
   * - New registration fails (e.g., memory exhausted)
   * - Exact tensor ID/name matches are not found
   * - The last registered buffer is still valid and matches the tensor
   * 
   * This is particularly useful in decode phase where memory pressure is high.
   */
  struct LastRegistrationInfo {
    uint32_t tensor_id = 0;           // Tensor ID of the registered tensor
    std::string tensor_name;          // Tensor name of the registered tensor
    void* ptr = nullptr;              // Buffer pointer that was successfully registered
    Qnn_MemHandle_t mem_handle = nullptr;  // QNN memory handle from successful registration
    size_t bytes = 0;                 // Size of the registered buffer in bytes
  };

  LastRegistrationInfo lastRegistrationInfo_{};  // Last successful registration info
  bool hasLastRegistrationInfo_ = false;         // Whether last registration info is valid

  /**
   * @brief Erase all tensor ID and name mappings that point to a specific buffer pointer
   * @param ptr The buffer pointer to remove from mappings
   * @param reason Reason for erasure (for debugging/logging purposes)
   */
  void eraseTensorMappingsForPtr(void* ptr, std::string_view reason);
  
  /**
   * @brief Remember the last successful buffer registration for fallback purposes
   * @param tensor_id Tensor ID of the registered tensor
   * @param tensor_name Tensor name of the registered tensor
   * @param ptr Buffer pointer that was successfully registered
   * @param mem_handle QNN memory handle from successful registration
   * @param total_bytes Size of the registered buffer in bytes
   */
  void rememberLastRegistration(uint32_t tensor_id, const std::string& tensor_name, void* ptr,
                                Qnn_MemHandle_t mem_handle, size_t total_bytes);
  
  /**
   * @brief Clear the last registration info if it matches the given pointer
   * @param ptr The buffer pointer to check against
   * @param reason Reason for clearing (for debugging/logging purposes)
   */
  void clearLastRegistrationIfMatches(void* ptr, std::string_view reason);

};

std::shared_ptr<QNNAllocator> createQNNAllocator();

}  // namespace mllm::qnn
