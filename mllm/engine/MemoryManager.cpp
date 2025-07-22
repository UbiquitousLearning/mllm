/**
 * @file MemoryManager.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-22
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/utils/Common.hpp"
#include "mllm/engine/MemoryManager.hpp"

namespace mllm {

MemoryManager::~MemoryManager() { clearAll(); }

void MemoryManager::registerAllocator(const DeviceTypes& device, const Allocator::ptr_t& allocator,
                                      const MemoryManagerOptions& options) {
  if (allocators_.has(device)) {
    MLLM_ERROR_EXIT(ExitCode::kMemory, "Allocator already registered for device {}", deviceTypes2Str(device));
  }
  allocators_.reg(device, allocator);
  options_.reg(device, options);
  if (options.using_buddy_mem_pool) { mem_pools_.reg(device, std::make_shared<BuddyMemPool>(options.buddy_mem_pool_options)); }
}

void MemoryManager::alloc(Storage* s) {
  auto& allocator = allocators_[s->device_];
  auto try_to_alloc_size = allocator->allocSize(s);

  if (try_to_alloc_size >= options_[s->device_].really_large_tensor_threshold) {
    allocator->alloc(s);
    return;
  }

  mem_pools_[s->device_]->alloc(s);
}

void MemoryManager::alloc(const std::shared_ptr<Storage>& s) { alloc(s.get()); }

void MemoryManager::free(Storage* s) {
  auto& allocator = allocators_[s->device_];
  auto try_to_alloc_size = allocator->allocSize(s);

  if (try_to_alloc_size >= options_[s->device_].really_large_tensor_threshold) {
    allocator->free(s);
    return;
  }

  mem_pools_[s->device_]->free(s);
}

void MemoryManager::free(const std::shared_ptr<Storage>& s) { free(s.get()); }

void MemoryManager::clearAll() {
  global_tensors_._ref_raw_data().clear();
  mem_pools_._ref_raw_data().clear();
  options_._ref_raw_data().clear();
}

}  // namespace mllm
