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
    backends_.emplace(BackendType::MLLM_CPU, new CPUBackend(mm));

    auto *in_tensor = net_param_[0].net_tensors[0];
    tensors_[in_tensor->name] = std::make_shared<Tensor>(backends_[BackendType::MLLM_CPU]);
    tensors_[in_tensor->name]->setName(in_tensor->name);
    tensors_[in_tensor->name]->setByteWidth(sizeof(float));
    // tensors_[in_tensor->name]->setBackend(backends_[BackendType::MLLM_CPU]);
    tensors_[in_tensor->name]->reshape(in_tensor->shape[0], in_tensor->shape[1], in_tensor->shape[2], in_tensor->shape[3]);
    for (auto &sub_param : net_param_) {
        auto net_in_tensor = sub_param.net_inputs;
        for (const auto &out_t : net_in_tensor) {
            tensors_[out_t->name] = std::make_shared<Tensor>(backends_[BackendType::MLLM_CPU]);
            tensors_[out_t->name]->setName(out_t->name);
            // tensors_[in_tensor->name]->SetByteWidth(sizeof(float));
        }
    }
}

void Net::convert() {
    // auto bn = new CPUBackend(mm);	//TODO
    // backends_["cpu"] = bn;
    // backends_["cpu"]->RegisterOps();
    // TODO
    // for (auto &sub_param : net_param_) {
    for (int i = 0; i < (int)net_param_.size(); ++i) {
        auto &sub_param = net_param_[i];
        sub_param.topologySort();
        shared_ptr<Graph> subg_1;
        subg_1.reset(new Graph(sub_param, backends_[BackendType::MLLM_CPU], tensors_));
        subGraphs_["G" + std::to_string(i)] = subg_1;
    }
}
} // namespace mllm
