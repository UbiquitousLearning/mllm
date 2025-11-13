// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#ifndef ASCENDC_CPU_DEBUG
#include <acl/acl.h>
#else
#include <tikicpulib.h>
#endif

#include "mllm/backends/ascend/AscendAllocator.hpp"

namespace mllm::ascend {

bool AscendAllocator::alloc(Storage* storage) {
#ifdef ASCENDC_CPU_DEBUG
  storage->ptr_ = AscendC::GmAlloc(storage->size_);
#else
  aclrtMalloc((void**)&(storage->ptr_), storage->size_, ACL_MEM_MALLOC_HUGE_FIRST);
#endif
  return storage->ptr_ != nullptr;
}

bool AscendAllocator::alloc(const Storage::ptr_t& storage) {
#ifdef ASCENDC_CPU_DEBUG
  storage->ptr_ = AscendC::GmAlloc(storage->size_);
#else
  aclrtMalloc((void**)&(storage->ptr_), storage->size_, ACL_MEM_MALLOC_HUGE_FIRST);
#endif
  return storage->ptr_ != nullptr;
}

void AscendAllocator::free(const Storage::ptr_t& storage) {
#ifdef ASCENDC_CPU_DEBUG
  AscendC::GmFree((void*)storage->ptr_);
#else
  aclrtFree(storage->ptr_);
#endif
}

void AscendAllocator::free(Storage* storage) {
#ifdef ASCENDC_CPU_DEBUG
  AscendC::GmFree((void*)storage->ptr_);
#else
  aclrtFree(storage->ptr_);
#endif
}

bool AscendAllocator::generalAlloc(void** ptr, size_t cap, size_t align) {
#ifdef ASCENDC_CPU_DEBUG
  *ptr = AscendC::GmAlloc(cap);
#else
  aclrtMalloc((void**)ptr, cap, ACL_MEM_MALLOC_HUGE_FIRST);
#endif
  return *ptr != nullptr;
}

void AscendAllocator::generalFree(void* ptr) {
#ifdef ASCENDC_CPU_DEBUG
  AscendC::GmFree((void*)ptr);
#else
  aclrtFree(ptr);
#endif
}

size_t AscendAllocator::allocSize(const Storage::ptr_t& storage) {
  // remember that alloc size should be aligned
  size_t align_size = alignSize();
  size_t required_size = storage->size_;
  size_t aligned_size = (required_size + align_size - 1) & ~(align_size - 1);
  return aligned_size;
}

size_t AscendAllocator::allocSize(Storage* storage) {
  // remember that alloc size should be aligned
  size_t align_size = alignSize();
  size_t required_size = storage->size_;
  size_t aligned_size = (required_size + align_size - 1) & ~(align_size - 1);
  return aligned_size;
}

size_t AscendAllocator::alignSize() const { return 128; }

std::shared_ptr<AscendAllocator> createAscendAllocator() { return std::make_shared<AscendAllocator>(); }

}  // namespace mllm::ascend
