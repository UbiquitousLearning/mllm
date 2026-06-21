// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <memory>

#include "mllm/engine/prefix_cache/TLB.hpp"
#include "mllm/engine/prefix_cache/ZenFS.hpp"
#include "mllm/engine/prefix_cache/Allocator.hpp"

namespace mllm::prefix_cache {

_HiCPUAllocator::_HiCPUAllocator(const ZenFileSystemOptions& options) : zen_fs_() { zen_fs_.initialize(options); }

vp_addr_t _HiCPUAllocator::alloc() { return zen_fs_.malloc(); }

char* _HiCPUAllocator::physicalAddr(vp_addr_t addr) { return zen_fs_.access(addr); }

void _HiCPUAllocator::free(vp_addr_t addr) { zen_fs_.free(addr); }

void _HiCPUAllocator::prefetch(vp_addr_t addr) { zen_fs_.hintsPrefetch(addr); }

void _HiCPUAllocator::purge(vp_addr_t addr) { zen_fs_.hintsPurge(addr); }

vp_addr_t _GPUAllocator::alloc() { return INVALID_VP_ADDR; }

char* _GPUAllocator::physicalAddr(vp_addr_t addr) {
  // TODO
  return nullptr;
}

void _GPUAllocator::free(vp_addr_t addr) {
  // TODO
}

void _GPUAllocator::prefetch(vp_addr_t addr) {
  // TODO
}

void _GPUAllocator::purge(vp_addr_t addr) {
  // TODO
}

PrefixCacheAllocator::PrefixCacheAllocator(const PrefixCacheAllocatorOptions& options) : options_(options) {
  if (options_.enable_cuda) { allocators_[DeviceTypes::kCUDA] = std::make_shared<_GPUAllocator>(); }
  if (options_.enable_cpu_hierarchy_memory) {
    allocators_[DeviceTypes::kCPU] = std::make_shared<_HiCPUAllocator>(options_.zen_fs_options);
  }
}

vp_addr_t PrefixCacheAllocator::alloc(DeviceTypes device_type) {
  auto ret_addr = allocators_[device_type]->alloc();
  tlb_.insert(ret_addr, allocators_[device_type]->physicalAddr(ret_addr));
  return ret_addr;
}

char* PrefixCacheAllocator::physicalAddr(vp_addr_t addr) { return tlb_.lookup(addr); }

void PrefixCacheAllocator::free(DeviceTypes device_type, vp_addr_t addr) {
  allocators_[device_type]->free(addr);
  tlb_.remove(addr);
}

void PrefixCacheAllocator::prefetch(DeviceTypes device_type, vp_addr_t addr) { allocators_[device_type]->prefetch(addr); }

void PrefixCacheAllocator::purge(DeviceTypes device_type, vp_addr_t addr) { allocators_[device_type]->purge(addr); }

}  // namespace mllm::prefix_cache
