/**
 * @file CPUAllocator.cpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-23
 *
 */
#include "mllm/backends/cpu/CPUAllocator.hpp"
#include "mllm/backends/cpu/kernels/kernels.hpp"

namespace mllm::cpu {

bool CPUAllocator::alloc(Storage* storage) {
  if constexpr (cpu::isX86_64()) {
    void* ptr;
    x86_align_alloc(&ptr, storage->size_, alignSize());
    if (!ptr) return false;
    storage->ptr_ = ptr;
    return true;
  }
  return false;
}

bool CPUAllocator::alloc(const Storage::ptr_t& storage) {
  if constexpr (cpu::isX86_64()) {
    void* ptr;
    x86_align_alloc(&ptr, storage->size_, alignSize());
    if (!ptr) return false;
    storage->ptr_ = ptr;
    return true;
  }
  return false;
}

void CPUAllocator::free(const Storage::ptr_t& storage) {
  if constexpr (cpu::isX86_64()) { x86_align_free(storage->ptr_); }
}

void CPUAllocator::free(Storage* storage) {
  if constexpr (cpu::isX86_64()) { x86_align_free(storage->ptr_); }
}

bool CPUAllocator::generalAlloc(void** ptr, size_t cap, size_t align) {
  if constexpr (cpu::isX86_64()) {
    x86_align_alloc(ptr, cap, align);
    return ptr != nullptr;
  }
  return false;
}

void CPUAllocator::generalFree(void* ptr) {
  if constexpr (cpu::isX86_64()) {
    if (!ptr) return;
    x86_align_free(ptr);
  }
}

size_t CPUAllocator::allocSize(const Storage::ptr_t& storage) {
  // remember that alloc size should be aligned
  size_t align_size = alignSize();
  size_t required_size = storage->size_;
  size_t aligned_size = (required_size + align_size - 1) & ~(align_size - 1);
  return aligned_size;
}

size_t CPUAllocator::allocSize(Storage* storage) {
  // remember that alloc size should be aligned
  size_t align_size = alignSize();
  size_t required_size = storage->size_;
  size_t aligned_size = (required_size + align_size - 1) & ~(align_size - 1);
  return aligned_size;
}

size_t CPUAllocator::alignSize() const {
  if constexpr (cpu::isX86_64()) {
    if constexpr (cpu::hasAVX512BW() || cpu::hasAVX512DQ() || cpu::hasAVX512VL() || cpu::hasAVX512CD() || cpu::hasAVX512F()) {
      return 64;
    } else if constexpr (cpu::hasAVX2() || cpu::hasAVX()) {
      return 32;
    } else if constexpr (cpu::hasSSE4_2() || cpu::hasSSE4_1() || cpu::hasSSE3() || cpu::hasSSE2() || cpu::hasSSE()) {
      return 16;
    }

    // No matter 128, 256, 512 vector size.
    // 64 is fit for all.
    return 64;
  }

  return 64;
}

std::shared_ptr<CPUAllocator> createCPUAllocator() { return std::make_shared<CPUAllocator>(); }

}  // namespace mllm::cpu
