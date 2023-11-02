#ifndef MLLM_MEMORY_H
#define MLLM_MEMORY_H

#include <cstddef>

namespace mllm {
// 这是一个功能和malloc/free相同的分配/释放内存/显存的函数。

/**
 * 内存管理类 mem pool ... TODO 管理HostMemory
 */
class MemoryManager {
public:
    MemoryManager(){}
    virtual ~MemoryManager(){}

    virtual void alloc(void **ptr, size_t size,size_t alignment) = 0;

    virtual void free(void *ptr) = 0;

};

} // namespace mllm
#endif // MLLM_MEMORY_H
