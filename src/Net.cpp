#include "Net.hpp"
#include "MemoryManager.hpp"
#include "Op.hpp"
#include "Types.hpp"
#include "backends/cpu/CPUBackend.hpp"
#ifdef NNAPI_ENABLED
#include "backends/nnapi/NNAPIBackend.hpp"
#endif
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
#ifdef NNAPI_ENABLED
    backends_.emplace(BackendType::MLLM_NNAPI, new NNAPIBackend(mm));
#endif

    auto *in_tensor = net_param_[0].net_tensors[0];
    tensors_[in_tensor->name] = std::make_shared<Tensor>(backends_[BackendType::MLLM_CPU]);
    tensors_[in_tensor->name]->setName(in_tensor->name);
//    tensors_[in_tensor->name]->setByteWidth(sizeof(float));
    // tensors_[in_tensor->name]->setBackend(backends_[BackendType::MLLM_CPU]);
    //    tensors_[in_tensor->name]->reshape(in_tensor->shape[0], in_tensor->shape[1], in_tensor->shape[2], in_tensor->shape[3]);
    for (auto &sub_param : net_param_) {
        vector<string> names = {};
        auto net_in_tensor = sub_param.net_inputs;
        for (const auto &out_t : net_in_tensor) {
            tensors_[out_t->name] = std::make_shared<Tensor>(backends_[BackendType::MLLM_CPU]);
            tensors_[out_t->name]->setName(out_t->name);
//            tensors_[out_t->name]->setByteWidth(sizeof(float));
            for (auto &tensor_name : tensor_names_) {
                tensor_name.erase(std::remove(tensor_name.begin(), tensor_name.end(), out_t->name), tensor_name.end());

            }
            names.push_back(out_t->name);
        }
        tensor_names_.push_back(names);
    }
    tensor_names_[0].push_back(in_tensor->name);
    printf("Net init\n");
}
/*
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
}*/

void Net::convert(BackendType backend_type) {
    for (int i = 0; i < (int)net_param_.size(); ++i) {
        auto &sub_param = net_param_[i];
        sub_param.topologySort();
        shared_ptr<Graph> subg_1;
        subg_1.reset(new Graph(sub_param, backends_[backend_type], tensors_));
        subGraphs_["G" + std::to_string(i)] = subg_1;
    }
    printf("Net convert\n");
}

void Net::reshapeInput() {
    auto *in_tensor = net_param_[0].net_tensors[0];
    tensors_[in_tensor->name]->reshape(in_tensor->shape[0], in_tensor->shape[1], in_tensor->shape[2], in_tensor->shape[3]);
}
void Net::reshapeInput(vector<int> shape) {
    auto *in_tensor = net_param_[0].net_tensors[0];
    tensors_[in_tensor->name]->reshape(shape[0], shape[1], shape[2], shape[3]);
}
void Net::setInput() {
    auto *in_tensor = net_param_[0].net_tensors[0];
    tensors_[in_tensor->name]->fullData<float>(1);
}
void Net::freeTensors(int graph_idx) {
    auto &graph_ex_tensor = tensor_names_[graph_idx];
    for (auto &name : graph_ex_tensor) {
        tensors_[name]->free();
    }
}
} // namespace mllm
