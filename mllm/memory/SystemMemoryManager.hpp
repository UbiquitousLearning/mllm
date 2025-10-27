#ifndef MLLM_MEMORY_SYSTEM_H
#define MLLM_MEMORY_SYSTEM_H

#include "MemoryManager.hpp"
namespace mllm {
    class SystemMemoryManager : public MemoryManager {
    public:
        SystemMemoryManager(){}
        ~SystemMemoryManager(){}

        void alloc(void **ptr, size_t size,size_t alignment) override ;

        void free(void *ptr) override;

    };
}
#endif