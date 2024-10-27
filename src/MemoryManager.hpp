#ifndef MLLM_MEMORY_H
#define MLLM_MEMORY_H

#include <cstddef>

namespace mllm {
/**
 * \brief The MemoryManager class provides an interface for memory management.
 * This class is expected to be overridden by specific memory management implementations.
 */
class MemoryManager {
public:
    MemoryManager() = default;
    virtual ~MemoryManager() = default;

    virtual void alloc(void **ptr, size_t size, size_t alignment) = 0;

    virtual void free(void *ptr) = 0;
};

} // namespace mllm
#endif // MLLM_MEMORY_H
