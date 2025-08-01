// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/utils/Common.hpp"
#include "mllm/engine/MemoryManager.hpp"
#include "mllm/engine/Context.hpp"

namespace mllm {

MemoryManager::~MemoryManager() { clearAll(); }

void MemoryManager::registerAllocator(const DeviceTypes& device, const Allocator::ptr_t& allocator,
                                      const MemoryManagerOptions& options) {
  if (allocators_.has(device)) {
    MLLM_ERROR_EXIT(ExitCode::kMemory, "Allocator already registered for device {}", deviceTypes2Str(device));
  }
  allocators_.reg(device, allocator);
  options_.reg(device, options);
  if (options.using_buddy_mem_pool) {
    mem_pools_.reg(device, std::make_shared<BuddyMemPool>(options.buddy_mem_pool_options, allocator));
  }
}

void MemoryManager::alloc(Storage* s) {
  auto& allocator = allocators_[s->device_];
  auto try_to_alloc_size = allocator->allocSize(s);

  // Record memory usage
  if (Context::instance().isPerfMode()) {
    Context::instance().getPerfFile()->mem_blobs_.insert(
        {s->custom_32bit_uuid_, PerfMemoryBlob{
                                    .start_time = Context::instance().curTime(),
                                    .end_time = 0,
                                    .memory_usage = try_to_alloc_size,
                                    .device_type = s->device_,
                                }});
  }

  if (try_to_alloc_size >= options_[s->device_].really_large_tensor_threshold) {
    MLLM_WARN("Trying to alloc a really large storage, whose storage size is {}B. The mllm memory manager will alloc a memory "
              "for this storage from OS directly instead of "
              "allocating one from ObjectCachePool/BuddyMemoryPool. If your scenario need to "
              "handle large storage frequently, you can modify the `buddy_first_segment_cap` in "
              "`MemManagerCargo`.",
              try_to_alloc_size);
    allocator->alloc(s);
    return;
  }

  mem_pools_[s->device_]->alloc(s);
}

void MemoryManager::alloc(const std::shared_ptr<Storage>& s) { alloc(s.get()); }

void MemoryManager::free(Storage* s) {
  auto& allocator = allocators_[s->device_];
  auto try_to_alloc_size = allocator->allocSize(s);

  if (Context::instance().isPerfMode()) {
    Context::instance().getPerfFile()->mem_blobs_[s->custom_32bit_uuid_].end_time = Context::instance().curTime();
  }

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

void MemoryManager::report() const {
  fmt::print("+------------------------------------------------------+\n");
  fmt::print("| {:<52} |\n", "MLLM Memory Report");
  fmt::print("+------------------------------------------------------+\n");
  for (auto& [device, mem_pool] : mem_pools_) {
    auto device_name = deviceTypes2Str(device);
    fmt::print("| Device: {:<44} |\n", device_name);
    fmt::print("+------------------------------------------------------+\n");
    mem_pool->report();
    fmt::print("\n");
  }
}

}  // namespace mllm
