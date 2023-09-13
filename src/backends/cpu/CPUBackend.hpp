#ifndef MLLM_CPUBACKEND_H
#define MLLM_CPUBACKEND_H

#include "Backend.hpp"
namespace mllm
{
    class CPUBackend : public Backend {
    public:
        CPUBackend(shared_ptr<MemoryManager> mm);
        ~CPUBackend() = default;
    };
} // namespace mllm

#endif //MLLM_CPUBACKEND_H