#include "OpenCLMemoryManager.hpp"
#include <cassert>

namespace mllm {

OpenCLMemoryManager::OpenCLMemoryManager(cl_context context) : context_(context) {
    assert(context_ != nullptr);
}

void OpenCLMemoryManager::alloc(void **ptr, size_t size, size_t alignment) {
    assert(ptr != nullptr);
    assert(size > 0);

    cl_int err;
    // 分配一个可供GPU内核读写的设备缓冲区
    cl_mem buffer = clCreateBuffer(context_, CL_MEM_READ_WRITE, size, nullptr, &err);
    check_cl_error(err, "OpenCLMemoryManager::clCreateBuffer");
    *ptr = buffer;
}

void OpenCLMemoryManager::free(void *ptr) {
    if (ptr != nullptr) {
        cl_mem buffer = static_cast<cl_mem>(ptr);
        clReleaseMemObject(buffer);
    }
}

} // namespace mllm