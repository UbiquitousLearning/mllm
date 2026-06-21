// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cuda/CudaCommons.hpp"
#include "mllm/backends/cuda/CudaAllocator.hpp"
#include <cuda_runtime.h>

namespace mllm::cuda {

bool CudaAllocator::alloc(Storage* storage) {
  void* ptr;
  MLLM_CUDA_CHECK(cudaMalloc(&ptr, storage->size_));
  storage->ptr_ = ptr;
  return true;
}

bool CudaAllocator::alloc(const Storage::ptr_t& storage) {
  void* ptr;
  MLLM_CUDA_CHECK(cudaMalloc(&ptr, storage->size_));
  storage->ptr_ = ptr;
  return true;
}

void CudaAllocator::free(Storage* storage) {
  if (storage->ptr_) {
    MLLM_CUDA_CHECK(cudaFree(storage->ptr_));
    storage->ptr_ = nullptr;
  }
}

void CudaAllocator::free(const Storage::ptr_t& storage) {
  if (storage->ptr_) {
    MLLM_CUDA_CHECK(cudaFree(storage->ptr_));
    storage->ptr_ = nullptr;
  }
}

bool CudaAllocator::generalAlloc(void** ptr, size_t cap, size_t align) {
  // CUDA malloc returns pointers aligned to 256 bytes by default
  // so we don't need special alignment handling like in CPU version
  MLLM_CUDA_CHECK(cudaMalloc(ptr, cap));
  return true;
}

void CudaAllocator::generalFree(void* ptr) {
  if (ptr) { MLLM_CUDA_CHECK(cudaFree(ptr)); }
}

size_t CudaAllocator::allocSize(Storage* storage) {
  // CUDA allocations don't require manual alignment padding
  // since cudaMalloc already provides proper alignment
  return storage->size_;
}

size_t CudaAllocator::allocSize(const Storage::ptr_t& storage) {
  // CUDA allocations don't require manual alignment padding
  // since cudaMalloc already provides proper alignment
  return storage->size_;
}

size_t CudaAllocator::alignSize() const {
  // CUDA malloc returns pointers aligned to 256 bytes by default
  // according to CUDA documentation
  return 256;
}

std::shared_ptr<CudaAllocator> createCudaAllocator() { return std::make_shared<CudaAllocator>(); }

}  // namespace mllm::cuda
