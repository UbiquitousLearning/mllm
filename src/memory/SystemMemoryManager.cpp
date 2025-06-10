#include "memory/SystemMemoryManager.hpp"
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstdio>

// macOS 特定的内存大小查询头文件
#ifdef __APPLE__
#include <malloc/malloc.h>
#else
#include <malloc.h>
#endif

namespace mllm {

static inline void **align(void **ptr, size_t alignment) {
    return (void **)((intptr_t)((unsigned char *)ptr + alignment - 1) & -alignment);
}

void SystemMemoryManager::alloc(void **ptr, size_t size, size_t alignment) {
    assert(size > 0);
    // allocate a block of memory, void* is used to store the original pointer
    // void **origin = (void **)malloc(size + sizeof(void *) + alignment - 1);
    void *origin = (void *)malloc(size + sizeof(void *) + alignment - 1);
    assert(origin != nullptr);
    if (origin == nullptr) {
        *ptr = nullptr;
        return;
    }
    void **aligned = (void **)(((size_t)(origin) + sizeof(void *) + alignment - 1) & (~(alignment - 1)));
    aligned[-1] = origin;
    *ptr = aligned;
}

void SystemMemoryManager::free(void *ptr) {
    if (ptr != nullptr) {
        void *origin = ((void **)ptr)[-1];  // 取出原始指针
#if defined(_WIN32)
        if (_msize(origin) > 0) {
            ::free(origin);
        }
#elif defined(__APPLE__)
        if (malloc_size(origin) > 0) {  // macOS 平台使用 malloc_size
            ::free(origin);
        }
#else  // Linux 和其他类 Unix 系统
        if (malloc_usable_size(origin) > 0) {
            ::free(origin);
        }
#endif
    }
}

} // namespace mllm