#include "QNNOptNet.hpp"
#include "Op.hpp"
#include "Types.hpp"
#include "Backend.hpp"
#include "express/Express.hpp"
#include <vector>
#include "backends/QNN/QNNGraph.hpp"

namespace mllm {

QNNOptNet::QNNOptNet(BackendConfig config, Context *ctx) :
    Net(config) {
    ctx_ = ctx;
}

void QNNOptNet::build_new_graph(std::vector<NetTensor *> inputs, NetOp *op) {
    auto opType = op->type;
    switch (opType) {
    case MATMUL:
        _Matmul(inputs, (bool)op->param["transpose0"], (bool)op->param["transpose1"], op->name + "CPU_ds");
        break;

    default:
        std::cout << "no support ops in build new graph" << std::endl;
    }
}

void QNNOptNet::convert(vector<NetParameter> &param, BackendType backend_type, int threadCount) {
    // *** NOTE: this below is for cpu-qnn hybrid execution, which will use cpu for embedding
    /*
    // the first graph is embedding layer, which shoud be executed in CPU
    std::cout << "net convert:" << param.size() << std::endl;
    std::cout << "net convert: 0" << std::endl;
    {
        auto &sub_param = param[0];
        vector<string> names = {};
        auto net_in_tensor = sub_param.net_inputs;
        for (const auto &out_t : net_in_tensor) {
            // DEBUG
            std::cout << out_t->name << std::endl;

            tensors_[out_t->name] = std::make_shared<Tensor>(backends_[MLLM_QNN].get());
            tensors_[out_t->name]->setName(out_t->name);
            for (auto &tensor_name : tensor_names_) {
                tensor_name.erase(std::remove(tensor_name.begin(), tensor_name.end(), out_t->name), tensor_name.end());
            }
            names.push_back(out_t->name);
        }

        for (auto *t : sub_param.net_tensors) {
            if (t->in == NULL) {
                auto *in_tensor = t;
                // DEBUG
                std::cout << in_tensor->name << std::endl;
                tensors_[in_tensor->name] = std::make_shared<Tensor>(backends_[MLLM_QNN].get());
                tensors_[in_tensor->name]->setName(in_tensor->name);
                input_names_.push_back(in_tensor->name);
                inputname_graphidx_[in_tensor->name] = 0;
                names.push_back(in_tensor->name);
            }
        }
        tensor_names_.push_back(names);
    }
    // the second graph is the rest of the model, which should be executed in QNN
    std::cout << "net convert: 1" << std::endl;
    {
        auto &sub_param = param[1];
        vector<string> names = {};
        auto net_in_tensor = sub_param.net_inputs;
        for (const auto &out_t : net_in_tensor) {
            // DEBUG
            std::cout << out_t->name << std::endl;

            tensors_[out_t->name] = std::make_shared<Tensor>(backends_[backend_type].get());
            tensors_[out_t->name]->setName(out_t->name);
            for (auto &tensor_name : tensor_names_) {
                tensor_name.erase(std::remove(tensor_name.begin(), tensor_name.end(), out_t->name), tensor_name.end());
            }
            names.push_back(out_t->name);
        }

        for (auto *t : sub_param.net_tensors) {
            if (t->in == NULL) {
                auto *in_tensor = t;
                // DEBUG
                std::cout << in_tensor->name << std::endl;

                tensors_[in_tensor->name] = std::make_shared<Tensor>(backends_[backend_type].get());
                tensors_[in_tensor->name]->setName(in_tensor->name);
                input_names_.push_back(in_tensor->name);
                inputname_graphidx_[in_tensor->name] = 1;
                names.push_back(in_tensor->name);
            }
        }
        tensor_names_.push_back(names);
    }

    for (int i = 0; i < 1; ++i) {
        param[i].topologySort();
        shared_ptr<Graph> subg_1;
        subg_1.reset(new Graph( param[i], backends_[MLLM_CPU].get(), tensors_, threadCount));
        subGraphs_["G" + std::to_string(i)] = subg_1;
    }
    for (int i = 1; i < 2; ++i) {
        param[i].topologySort();
        shared_ptr<Graph> subg_1;
        subg_1.reset(new Graph(param[i], backends_[MLLM_QNN].get(), tensors_, threadCount));
        subGraphs_["G" + std::to_string(i)] = subg_1;
    }
    */
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
        param[i].topologySort();
        shared_ptr<Graph> subg_1;
        subg_1.reset(new QNNGraph(param[i], backends_[backend_type].get(), tensors_, threadCount));
        subGraphs_["Prompt_Graph." + std::to_string(i)] = subg_1;
    }
}

} // namespace mllm
