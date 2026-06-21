// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <memory>
#include <unordered_map>

#include "mllm/core/DeviceTypes.hpp"
#include "mllm/engine/prefix_cache/TLB.hpp"
#include "mllm/engine/prefix_cache/ZenFS.hpp"

namespace mllm::prefix_cache {

class _AllocatorImpl {
 public:
  using ptr_t = std::shared_ptr<_AllocatorImpl>;

  virtual ~_AllocatorImpl() = default;

  virtual vp_addr_t alloc() = 0;

  virtual char* physicalAddr(vp_addr_t addr) = 0;

  virtual void free(vp_addr_t addr) = 0;

  virtual void prefetch(vp_addr_t addr) = 0;

  virtual void purge(vp_addr_t addr) = 0;
};

// Hierarchy CPU Allocator
// 1. Memory
// 2. Disk
class _HiCPUAllocator final : public _AllocatorImpl {
 public:
  _HiCPUAllocator() = default;

  ~_HiCPUAllocator() override = default;

  explicit _HiCPUAllocator(const ZenFileSystemOptions& options);

  vp_addr_t alloc() override;

  char* physicalAddr(vp_addr_t addr) override;

  void free(vp_addr_t addr) override;

  void prefetch(vp_addr_t addr) override;

  void purge(vp_addr_t addr) override;

 private:
  ZenFileSystem zen_fs_;
};

// GPU Allocator
// 1. Memory
class _GPUAllocator final : public _AllocatorImpl {
 public:
  _GPUAllocator() = default;

  ~_GPUAllocator() override = default;

  vp_addr_t alloc() override;

  char* physicalAddr(vp_addr_t addr) override;

  void free(vp_addr_t addr) override;

  void prefetch(vp_addr_t addr) override;

  void purge(vp_addr_t addr) override;
};

struct PrefixCacheAllocatorOptions {
  // Normal things.
  size_t per_k_token_ele = 1024;
  size_t per_v_token_ele = 1024;
  DataTypes k_dtype = kFloat16;
  DataTypes v_dtype = kFloat16;

  // CUDA things.
  bool enable_cuda = false;
  vp_addr_t cuda_mem_base = 0x100000;

  // CPU things.
  bool enable_cpu_hierarchy_memory = true;
  ZenFileSystemOptions zen_fs_options;
};

class PrefixCacheAllocator {
 public:
  PrefixCacheAllocator() = default;

  explicit PrefixCacheAllocator(const PrefixCacheAllocatorOptions& options);

  vp_addr_t alloc(DeviceTypes device_type);

  char* physicalAddr(vp_addr_t addr);

  void free(DeviceTypes device_type, vp_addr_t addr);

  void prefetch(DeviceTypes device_type, vp_addr_t addr);

  void purge(DeviceTypes device_type, vp_addr_t addr);

 private:
  PrefixCacheAllocatorOptions options_;

  TLB tlb_;  //< To accelerate lookup. Especially for ZenFS.
  std::unordered_map<DeviceTypes, _AllocatorImpl::ptr_t> allocators_;
};

}  // namespace mllm::prefix_cache
