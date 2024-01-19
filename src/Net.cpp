#include "Net.hpp"
#include "memory/SystemMemoryManager.hpp"
#include "Op.hpp"
#include "Types.hpp"
#include "backends/cpu/CPUBackend.hpp"
#include <memory>
#ifdef NNAPI_ENABLED
#include "backends/nnapi/NNAPIBackend.hpp"
#endif
#ifdef QNN_ENABLED
#include "backends/QNN/QNNBackend.hpp"
#include "backends/QNN/QNNMemoryManager.hpp"
#endif
#include <map>
#include <vector>

namespace mllm {

shared_ptr<CPUBackend> cpuBn;
#ifdef NNAPI_ENABLED
shared_ptr<NNAPIBackend> nnapiBn;
#endif
#ifdef QNN_ENABLED
shared_ptr<QNNBackend> qnnBn;
#endif

Net::Net(const vector<NetParameter> &param, BackendConfig config) :
    config_(config) {
    shared_ptr<MemoryManager> mm = nullptr;
    switch (config.memory) {
    case BackendConfig::Memory_High:
#ifdef QNN_ENABLED
        mm = std::make_shared<QNNMemoryManager>();
#else
        mm = std::make_shared<SystemMemoryManager>();
#endif   
        break;
    default:
#ifdef QNN_ENABLED
        mm = std::make_shared<QNNMemoryManager>();
#else
        mm = std::make_shared<SystemMemoryManager>();
#endif
        break;
    }

    cpuBn.reset(new CPUBackend(mm));
    backends_.emplace(BackendType::MLLM_CPU,  cpuBn);
    //backends_.emplace(BackendType::MLLM_CPU,  new CPUBackend(mm));  //memory lost
#ifdef NNAPI_ENABLED
    nnapiBn.reset(new NNAPIBackend(mm));
    backends_.emplace(BackendType::MLLM_NNAPI,  nnapiBn);
    //backends_.emplace(BackendType::MLLM_NNAPI, new NNAPIBackend(mm));
#endif
#ifdef QNN_ENABLED
    qnnBn.reset(new QNNBackend(mm));
    backends_.emplace(BackendType::MLLM_QNN,  qnnBn);
#endif

    auto *in_tensor = param[0].net_tensors[0];
    tensors_[in_tensor->name] = std::make_shared<Tensor>(backends_[BackendType::MLLM_CPU].get());
    tensors_[in_tensor->name]->setName(in_tensor->name);
    input_name_ = in_tensor->name;
    for (auto &sub_param : param) {
        vector<string> names = {};
        auto net_in_tensor = sub_param.net_inputs;
        for (const auto &out_t : net_in_tensor) {
            tensors_[out_t->name] = std::make_shared<Tensor>(backends_[BackendType::MLLM_CPU].get());
            tensors_[out_t->name]->setName(out_t->name);
            for (auto &tensor_name : tensor_names_) {
                tensor_name.erase(std::remove(tensor_name.begin(), tensor_name.end(), out_t->name), tensor_name.end());
            }
            names.push_back(out_t->name);
        }
        tensor_names_.push_back(names);
    }
    tensor_names_[0].push_back(in_tensor->name);
}

void Net::convert(vector<NetParameter> &param, BackendType backend_type) {
    for (int i = 0; i < (int)param.size(); ++i) {
        param[i].topologySort();
        shared_ptr<Graph> subg_1;
        subg_1.reset(new Graph( param[i], backends_[backend_type].get(), tensors_));
        subGraphs_["G" + std::to_string(i)] = subg_1;
    }
    //printf("Net convert\n");
}

void Net::freeTensors(int graph_idx) {
    auto &graph_ex_tensor = tensor_names_[graph_idx];
    for (auto &name : graph_ex_tensor) {
        tensors_[name]->free();
    }
}
} // namespace mllm
