#include "Net.hpp"
#include "MemoryManager.hpp"
#include "Op.hpp"
#include "Types.hpp"
#include "backends/cpu/CPUAdd.hpp"
#include "backends/cpu/CPUBackend.hpp"
#include <map>
#include <vector>
namespace mllm {
Net::Net(const vector<NetParameter> &param, BackendConfig config) :
    net_param_(param), config_(config) {
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
    // auto bn = new CPUBackend(mm);	//TODO
    // backends_["cpu"] = bn;
    // backends_["cpu"]->RegisterOps();
    // TODO
    // auto sub_param_ = net_param_;
    // shared_ptr<Graph> subg_1;
    // subg_1.reset(new Graph(sub_param_, backends_[BackendType::mllm_CPU]));
    // subgraphs_["fp1"] = subg_1;
}
} // namespace mllm
