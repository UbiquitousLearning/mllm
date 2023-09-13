#include "CPUBackend.hpp"
namespace mllm
{
    CPUBackend::CPUBackend(shared_ptr<MemoryManager> mm): Backend(mm)
    {
    }

} // namespace mllm
