#include "QNNNet.hpp"
#include "Op.hpp"
#include "QNNExecutor.hpp"
#include "Types.hpp"
#include "Backend.hpp"
#include <vector>
#include "backends/qnn/QNNGraph.hpp"
#include "express/ExpressBase.hpp"

namespace mllm {

QNNNet::QNNNet(BackendConfig config, Context *ctx) :
    Net(config) {
    backends_.emplace(MLLM_QNN, GetBackendCreator(MLLM_QNN)->create(config));
    ctx_ = ctx;
}


void QNNNet::convert(Context* ctx, BackendType backend_type, int threadCount) {
    auto& param = ctx->sub_param_;
    // tensors will all be converted to QNN shared buffer
    for (int ii = 0; ii < (int)param.size(); ++ii) {
        auto &sub_param = param[ii];
        vector<string> names = {};
        auto net_in_tensor = sub_param.net_inputs;
        for (const auto &out_t : net_in_tensor) {
            tensors_[out_t->name] = std::make_shared<Tensor>(backends_[backend_type].get());
            tensors_[out_t->name]->setName(out_t->name);
            tensors_[out_t->name]->setDtype(out_t->type);
            for (auto &tensor_name : tensor_names_) {
                tensor_name.erase(std::remove(tensor_name.begin(), tensor_name.end(), out_t->name), tensor_name.end());
            }
            names.push_back(out_t->name);
        }

        for (auto *t : sub_param.net_tensors) {
            if (t->in == NULL) {
                auto *in_tensor = t;
                tensors_[in_tensor->name] = std::make_shared<Tensor>(backends_[backend_type].get());
                tensors_[in_tensor->name]->setName(in_tensor->name);
                tensors_[in_tensor->name]->setDtype(in_tensor->type);
                input_names_.push_back(in_tensor->name);
                inputname_graphidx_[in_tensor->name] = ii;
                names.push_back(in_tensor->name);
            }
        }
        tensor_names_.push_back(names);
    }

    for (int i = 0; i < (int)param.size(); ++i) {
        auto expectedBackend = ctx->sub_backend_[i];

        param[i].topologySort();
        shared_ptr<Graph> subg_1;

        if(QNNExecutor::graphOffloadRule(expectedBackend, i) == MLLM_CPU){
            subg_1.reset(new Graph(param[i], backends_[MLLM_CPU].get(), tensors_, threadCount));
        } else if (QNNExecutor::graphOffloadRule(expectedBackend, i) == MLLM_QNN) {
            subg_1.reset(new QNNGraph(param[i], backends_[backend_type].get(), tensors_, threadCount, "Prompt_Graph." + std::to_string(i)));
        }

        subGraphs_["Prompt_Graph." + std::to_string(i)] = subg_1;
    }
}

} // namespace mllm
