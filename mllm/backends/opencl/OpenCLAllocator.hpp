// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/backends/base/Allocator.hpp"
#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"
#include <map>
#include <memory>
#include <mutex>

namespace mllm::opencl {

class OpenCLAllocator final : public Allocator {
 public:
  explicit OpenCLAllocator(std::shared_ptr<OpenCLRuntime> runtime);
  ~OpenCLAllocator();

  inline bool ctrlByMemManager() override {
    // OpenCLAllocator manages its own memory pool
    return false;
  }

  bool alloc(Storage* storage) override;
  bool alloc(const Storage::ptr_t& storage) override;
  void free(Storage* storage) override;
  void free(const Storage::ptr_t& storage) override;
  bool generalAlloc(void** ptr, size_t cap, size_t align) override;
  void generalFree(void* ptr) override;
  size_t allocSize(Storage* storage) override;
  size_t allocSize(const Storage::ptr_t& storage) override;
  [[nodiscard]] size_t alignSize() const override;

 private:
  std::multimap<size_t, cl_mem> memory_pool_;
  std::mutex pool_mutex_;
  std::shared_ptr<OpenCLRuntime> runtime_;
};

}  // namespace mllm::opencl
