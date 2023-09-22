#ifndef MLLM_MEMORY_SYSTEM_H
#define MLLM_MEMORY_SYSTEM_H

#include "MemoryManager.hpp"
namespace mllm {
    class SystemMemoryManager : public MemoryManager {
    public:
        SystemMemoryManager(){}
        ~SystemMemoryManager(){}

        void Alloc(void **ptr, size_t size,size_t alignment) override ;

        void Free(void **ptr) override;

    };
}
#endif