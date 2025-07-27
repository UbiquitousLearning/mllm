/**
 * @file CPUAllocator.cpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @author oreomaker (zh002919@outlook.com)
 * @brief
 * @version 0.1
 * @date 2025-07-23
 *
 */
#include "mllm/backends/cpu/CPUAllocator.hpp"
#include "mllm/backends/cpu/kernels/Kernels.hpp"

namespace mllm::cpu {

void align_alloc(void** ptr, size_t required_bytes, size_t align) {
  if (align == 0 || (align & (align - 1))) {
    *ptr = nullptr;
    return;
  }
  void* p1;
  void** p2;
  size_t offset = align - 1 + sizeof(void*);
  if ((p1 = (void*)malloc(required_bytes + offset)) == nullptr) {
    *ptr = nullptr;
    return;
  }
  p2 = (void**)(((size_t)(p1) + offset) & ~(align - 1));  // NOLINT
  p2[-1] = p1;
  *ptr = p2;
}

void align_free(void* ptr) { free(((void**)ptr)[-1]); }

bool CPUAllocator::alloc(Storage* storage) {
  void* ptr;
  align_alloc(&ptr, storage->size_, alignSize());
  if (!ptr) return false;
  storage->ptr_ = ptr;
  return true;
}

bool CPUAllocator::alloc(const Storage::ptr_t& storage) {
  void* ptr;
  align_alloc(&ptr, storage->size_, alignSize());
  if (!ptr) return false;
  storage->ptr_ = ptr;
  return true;
}

void CPUAllocator::free(const Storage::ptr_t& storage) { align_free(storage->ptr_); }

void CPUAllocator::free(Storage* storage) { align_free(storage->ptr_); }

bool CPUAllocator::generalAlloc(void** ptr, size_t cap, size_t align) {
  align_alloc(ptr, cap, align);
  return ptr != nullptr;
}

void CPUAllocator::generalFree(void* ptr) {
  if (!ptr) return;
  align_free(ptr);
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
    } else if constexpr (cpu::hasSSE4_2() || cpu::hasSSE4_1() || cpu::hasSSE3() || cpu::hasSSE2() || cpu::hasSSE()
                         || cpu::hasSSSE3()) {
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
