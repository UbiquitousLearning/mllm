// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/backends/base/Allocator.hpp"

namespace mllm::cuda {

class CudaAllocator final : public Allocator {
 public:
  inline bool ctrlByMemManager() override { return true; }

  bool alloc(Storage* storage) override;

  bool alloc(const Storage::ptr_t& storage) override;

  void free(Storage* storage) override;

  void free(const Storage::ptr_t& storage) override;

  bool generalAlloc(void** ptr, size_t cap, size_t align) override;

  void generalFree(void* ptr) override;

  size_t allocSize(Storage* storage) override;

  size_t allocSize(const Storage::ptr_t& storage) override;

  [[nodiscard]] size_t alignSize() const override;
};

std::shared_ptr<CudaAllocator> createCudaAllocator();

}  // namespace mllm::cuda
