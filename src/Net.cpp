#include "Net.hpp"
#include "MemoryManager.hpp"
#include "Op.hpp"
#include "Types.hpp"
#include "backends/cpu/CPUAdd.hpp"
#include "backends/cpu/CPUBackend.hpp"
namespace mllm {
Net::Net(const NetParameter &param, BackendConfig config): net_param_(param), config_(config) {
  shared_ptr<MemoryManager> mm = nullptr;
  switch (config.memory) {
  case BackendConfig::Memory_High:
    mm = shared_ptr<MemoryManager>(new MemoryManager());
    break;
  default:
    mm = shared_ptr<MemoryManager>(new MemoryManager());
    break;
  }
  backends_.emplace(BackendType::mllm_CPU, new CPUBackend(mm));
}

void Net::Convert() {
    
    for(auto op:this->net_param_.net_ops){
        
    }

  }
} // namespace mllm
