#ifndef OPENCL_MEMORY_MANAGER_H
#define OPENCL_MEMORY_MANAGER_H

#include "MemoryManager.hpp"
#include <string>
#include <map>   // 引入map用于内存池
#include <mutex> // 引入mutex用于线程安全

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

void check_cl_error(cl_int err, const std::string &operation);

namespace mllm {

class OpenCLMemoryManager : public MemoryManager {
public:
    explicit OpenCLMemoryManager(cl_context context);

    ~OpenCLMemoryManager() override;

    void alloc(void **ptr, size_t size, size_t alignment) override;
    void free(void *ptr) override;

private:
    cl_context context_; // 需要OpenCL上下文来创建缓冲区

    std::multimap<size_t, cl_mem> memory_pool_;

    std::mutex pool_mutex_;
};

} // namespace mllm

#endif // OPENCL_MEMORY_MANAGER_H