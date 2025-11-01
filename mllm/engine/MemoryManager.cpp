// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/utils/Common.hpp"
#include "mllm/engine/MemoryManager.hpp"
#include "mllm/tracy_perf/Tracy.hpp"

#ifdef MLLM_PERFETTO_ENABLE
#include "mllm/engine/Perf.hpp"
#endif

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
  MLLM_TRACY_ZONE_SCOPED;
  auto& allocator = allocators_[s->device_];
  auto try_to_alloc_size = allocator->allocSize(s);

  if (!allocator->ctrlByMemManager()) {
    allocator->alloc(s);
    return;
  }

#ifdef MLLM_PERFETTO_ENABLE
  MLLM_PERF_TRACE_BEGIN("mllm.tensor_lifecycle", "tensor_hold", perfetto::Track(static_cast<uint64_t>(s->custom_32bit_uuid_)),
                        [&](perfetto::EventContext ctx) {
                          ctx.AddDebugAnnotation("bytes", s->size_);
                          ctx.AddDebugAnnotation("device", deviceTypes2Str(s->device_));
                        });
#endif

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
  MLLM_TRACY_ZONE_SCOPED;
  auto& allocator = allocators_[s->device_];
  auto try_to_alloc_size = allocator->allocSize(s);

  if (!allocator->ctrlByMemManager()) {
    allocator->free(s);
    return;
  }

#ifdef MLLM_PERFETTO_ENABLE
  MLLM_PERF_TRACE_END("mllm.tensor_lifecycle", perfetto::Track(static_cast<uint64_t>(s->custom_32bit_uuid_)));
#endif

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
