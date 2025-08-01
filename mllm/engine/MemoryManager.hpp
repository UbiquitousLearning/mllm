// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "mllm/core/Tensor.hpp"
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/utils/SymbolTable.hpp"
#include "mllm/engine/BuddyMemPool.hpp"
#include "mllm/backends/base/Allocator.hpp"

namespace mllm {

struct MemoryManagerOptions {
  // threshold
  size_t really_large_tensor_threshold = 128 * 1024 * 1024;  // 128 MB
  bool using_buddy_mem_pool = true;
  BuddyMemPoolOptions buddy_mem_pool_options;
};

class MemoryManager {
 public:
  using ptr_t = std::shared_ptr<MemoryManager>;

  ~MemoryManager();

  void registerAllocator(const DeviceTypes& device, const Allocator::ptr_t& allocator, const MemoryManagerOptions& options);

  void alloc(Storage* s);

  void alloc(const std::shared_ptr<Storage>& s);

  void free(Storage* s);

  void free(const std::shared_ptr<Storage>& s);

  void clearAll();

  void report() const;

 private:
  SymbolTable<std::string, Tensor> global_tensors_;
  SymbolTable<DeviceTypes, Allocator::ptr_t> allocators_;
  SymbolTable<DeviceTypes, MemoryManagerOptions> options_;
  SymbolTable<DeviceTypes, BuddyMemPool::ptr_t> mem_pools_;
};

}  // namespace mllm
