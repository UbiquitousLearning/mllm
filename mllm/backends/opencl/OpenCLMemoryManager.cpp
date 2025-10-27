

#include "OpenCLMemoryManager.hpp"
#include <cassert>
#include <iostream> // 用于调试输出

namespace mllm {

OpenCLMemoryManager::OpenCLMemoryManager(cl_context context) :
    context_(context) {
    assert(context_ != nullptr);
}

OpenCLMemoryManager::~OpenCLMemoryManager() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    for (auto const &[size, buffer] : memory_pool_) {
        if (buffer) {
            clReleaseMemObject(buffer);
        }
    }
    memory_pool_.clear();
}

void OpenCLMemoryManager::alloc(void **ptr, size_t size, size_t alignment) {
    assert(ptr != nullptr);
    assert(size > 0);

    std::lock_guard<std::mutex> lock(pool_mutex_);
    auto it = memory_pool_.lower_bound(size);

    if (it != memory_pool_.end()) {
        // 找到了合适的内存块
        cl_mem buffer = it->second; // 获取内存句柄
        memory_pool_.erase(it);     // 从池中移除
        *ptr = buffer;              // 将句柄赋给指针
        // std::cout << "[OpenCL Memory Pool] Reused buffer of size " << it->first << " for request of " << size << std::endl;
    } else {
        // 如果池中没有合适的内存块，则分配新的内存
        cl_int err;
        cl_mem buffer = clCreateBuffer(context_, CL_MEM_READ_WRITE, size, nullptr, &err);
        check_cl_error(err, "OpenCLMemoryManager::clCreateBuffer (new allocation)");
        *ptr = buffer;
        // std::cout << "[OpenCL Memory Pool] Allocated new buffer of size " << size << std::endl;
    }
}

void OpenCLMemoryManager::free(void *ptr) {
    if (ptr != nullptr) {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        cl_mem buffer = static_cast<cl_mem>(ptr);

        size_t buffer_size = 0;
        cl_int err = clGetMemObjectInfo(buffer, CL_MEM_SIZE, sizeof(size_t), &buffer_size, nullptr);
        check_cl_error(err, "OpenCLMemoryManager::clGetMemObjectInfo (on free)");

        if (buffer_size > 0) {
            memory_pool_.insert({buffer_size, buffer});
            // std::cout << "[OpenCL Memory Pool] Returned buffer of size " << buffer_size << " to pool." << std::endl;
        } else {
            clReleaseMemObject(buffer);
        }
    }
}

} // namespace mllm