#ifndef OPENCL_MEMORY_MANAGER_H
#define OPENCL_MEMORY_MANAGER_H

#include "MemoryManager.hpp"
#include <string> // For std::string in check_cl_error

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

// 前向声明，实现在 backend.cpp 中
void check_cl_error(cl_int err, const std::string& operation);

namespace mllm {

class OpenCLMemoryManager : public MemoryManager {
public:
    explicit OpenCLMemoryManager(cl_context context);
    ~OpenCLMemoryManager() override = default;

    void alloc(void **ptr, size_t size, size_t alignment) override;
    void free(void *ptr) override;

private:
    cl_context context_; // 需要OpenCL上下文来创建缓冲区
};

} // namespace mllm

#endif // OPENCL_MEMORY_MANAGER_H