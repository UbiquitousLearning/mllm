// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/opencl/OpenCLAllocator.hpp"
#include "mllm/backends/opencl/runtime/OpenCLLoader.hpp"
#include "mllm/mllm.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::opencl {

OpenCLAllocator::OpenCLAllocator() { runtime_ = OpenCLRuntime::get(); }

OpenCLAllocator::~OpenCLAllocator() {
  std::lock_guard<std::mutex> lock(pool_mutex_);
  for (auto& [size, buffer] : memory_pool_) {
    if (buffer) { OpenCLLoader::instance().clReleaseMemObject(buffer); }
  }
  memory_pool_.clear();
}

bool OpenCLAllocator::alloc(Storage* storage) {
  void* ptr = nullptr;
  size_t size = allocSize(storage);
  if (generalAlloc(&ptr, size, alignSize())) {
    storage->ptr_ = ptr;
    return true;
  }
  return false;
}

bool OpenCLAllocator::alloc(const Storage::ptr_t& storage) { return alloc(storage.get()); }

void OpenCLAllocator::free(Storage* storage) {
  generalFree(storage->ptr_);
  storage->ptr_ = nullptr;
}

void OpenCLAllocator::free(const Storage::ptr_t& storage) { free(storage.get()); }

bool OpenCLAllocator::generalAlloc(void** ptr, size_t cap, size_t align) {
  if (cap == 0) {
    *ptr = nullptr;
    return true;
  }

  std::lock_guard<std::mutex> lock(pool_mutex_);
  auto it = memory_pool_.lower_bound(cap);

  if (it != memory_pool_.end()) {
    // Reuse buffer
    cl_mem buffer = it->second;
    memory_pool_.erase(it);
    *ptr = (void*)buffer;
  } else {
    // Allocate new buffer
    cl_int err;
    cl_mem buffer = OpenCLLoader::instance().clCreateBuffer(runtime_->context()(), CL_MEM_READ_WRITE, cap, nullptr, &err);
    if (err != CL_SUCCESS) {
      MLLM_ERROR("OpenCLAllocator::clCreateBuffer failed with error %d\n", err);
      *ptr = nullptr;
      return false;
    }
    *ptr = (void*)buffer;
  }
  return true;
}

void OpenCLAllocator::generalFree(void* ptr) {
  if (ptr == nullptr) return;

  std::lock_guard<std::mutex> lock(pool_mutex_);
  cl_mem buffer = static_cast<cl_mem>(ptr);

  size_t buffer_size = 0;
  cl_int err = OpenCLLoader::instance().clGetMemObjectInfo(buffer, CL_MEM_SIZE, sizeof(size_t), &buffer_size, nullptr);
  if (err != CL_SUCCESS) {
    MLLM_ERROR("OpenCLAllocator::clGetMemObjectInfo failed with error %d\n", err);
    OpenCLLoader::instance().clReleaseMemObject(buffer);
    return;
  }

  if (buffer_size > 0) {
    memory_pool_.insert({buffer_size, buffer});
  } else {
    OpenCLLoader::instance().clReleaseMemObject(buffer);
  }
}

size_t OpenCLAllocator::allocSize(Storage* storage) {
  size_t align_size = alignSize();
  size_t required_size = storage->size_;
  size_t aligned_size = (required_size + align_size - 1) & ~(align_size - 1);
  return aligned_size;
}

size_t OpenCLAllocator::allocSize(const Storage::ptr_t& storage) { return allocSize(storage.get()); }

size_t OpenCLAllocator::alignSize() const { return 64; }

}  // namespace mllm::opencl
