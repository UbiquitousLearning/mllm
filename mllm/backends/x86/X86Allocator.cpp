/**
 * @file X86Allocator.cpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-23
 *
 */
#include "mllm/backends/x86/X86Allocator.hpp"
#include "mllm/backends/x86/kernels/mem.hpp"

namespace mllm::x86 {

bool X86Allocator::alloc(Storage* storage) {
  void* ptr;
  x86_align_alloc(&ptr, storage->size_, alignSize());
  if (!ptr) return false;
  storage->ptr_ = ptr;
  return true;
}

bool X86Allocator::alloc(const Storage::ptr_t& storage) {
  void* ptr;
  x86_align_alloc(&ptr, storage->size_, alignSize());
  if (!ptr) return false;
  storage->ptr_ = ptr;
  return true;
}

void X86Allocator::free(const Storage::ptr_t& storage) { x86_align_free(storage->ptr_); }

void X86Allocator::free(Storage* storage) { x86_align_free(storage->ptr_); }

bool X86Allocator::generalAlloc(void** ptr, size_t cap, size_t align) {
  x86_align_alloc(ptr, cap, align);
  return ptr != nullptr;
}

void X86Allocator::generalFree(void* ptr) {
  if (!ptr) return;
  x86_align_free(ptr);
}

size_t X86Allocator::allocSize(const Storage::ptr_t& storage) {
  // remember that alloc size should be aligned
  size_t align_size = alignSize();
  size_t required_size = storage->size_;
  size_t aligned_size = (required_size + align_size - 1) & ~(align_size - 1);
  return aligned_size;
}

size_t X86Allocator::allocSize(Storage* storage) {
  // remember that alloc size should be aligned
  size_t align_size = alignSize();
  size_t required_size = storage->size_;
  size_t aligned_size = (required_size + align_size - 1) & ~(align_size - 1);
  return aligned_size;
}

size_t X86Allocator::alignSize() const { return 64; }

std::shared_ptr<X86Allocator> createX86Allocator() { return std::make_shared<X86Allocator>(); }

}  // namespace mllm::x86
