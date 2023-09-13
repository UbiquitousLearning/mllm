#ifndef MLLM_CPUBACKEND_H
#define MLLM_CPUBACKEND_H

#include "Backend.hpp"
namespace mllm
{
    class CPUBackend : public Backend {
    public:
        CPUBackend(shared_ptr<MemoryManager> mm);
        ~CPUBackend() = default;

    private:
        // std::map<int, Op*> mMap
    };
} // namespace mllm

#endif //MLLM_CPUBACKEND_H