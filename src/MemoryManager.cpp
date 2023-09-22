
#include "MemoryManager.hpp"
#include <cassert>

namespace mllm {

static inline void **alignPointer(void **ptr, size_t alignment) {
    return (void **)((intptr_t)((unsigned char *)ptr + alignment - 1) & -alignment);
}

void SystemMemoryManager::Alloc(void **ptr, size_t size,size_t alignment){
    assert(size > 0);

    void **origin = (void **)malloc(size + sizeof(void *) + alignment);
    assert(origin != NULL);
    if (!origin) {
        *ptr = NULL;
    }

    void **aligned = alignPointer(origin + 1, alignment);
    aligned[-1]    = origin;
    *ptr = aligned;
}

void SystemMemoryManager::Free(void **ptr){
    free(*ptr);
}

} // namespace mllm