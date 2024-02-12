#include "QNNNet.hpp"
#include "Op.hpp"
#include "Types.hpp"
#include "Backend.hpp"
#include "express/Express.hpp"
#include <vector>

namespace mllm {

QNNNet::QNNNet(BackendConfig config, Context *ctx) : Net(config) {
    ctx_ = ctx;
}

void QNNNet::build_new_graph(std::vector<NetTensor *> inputs, NetOp *op) {

    auto opType = op->type;
    switch(opType) {
        case MATMUL:
            _Matmul(inputs,  (bool)op->param["transpose0"], (bool)op->param["transpose1"], op->name + "CPU_ds");
            break;
        
        

        default:
            std::cout << "no support ops in build new graph" << std::endl;
    }
    

}

// QNN dynamic shape use
void QNNNet::convert(vector<NetParameter> &param, BackendType backend_type, int threadCount) {

    // Quantization type =>  QNNOps selection.

    // before generating a correct graph, we first split them for QNN dynamic shape.
    // In QNN scenario, we will build at least three types of graph
    //      1. prompt QNN graph. 
    //      2. autoregressive QNN graph. 
    //      3. autoregressive CPU graph.

    // We assume only one param now.
    for (int ii = 0; ii < (int)param.size(); ++ii) {
        auto &sub_param = param[ii];
        vector<NetOp *> ops = sub_param.net_ops;
        
        vector<int> splitPositions;
        for (int op_i = 0; op_i < ops.size(); op_i++ ) {

            // find where the matmul => split dynamic shape graph point.
            if (ops[op_i]->type == MATMUL) {
                splitPositions.push_back(op_i);
            }
        }

        if (splitPositions.size() % 2 != 0) {
            std::cout << "dynamic shape graph split errors" << std::endl;
            exit(-1);
        }

        _SubgraphBegin(ctx_);
        auto new_sub_param = get_active_subgraph(ctx_);

        // merge all dynamic shape ops to a CPU graph.
        for (int dop_i; dop_i < splitPositions.size(); dop_i+=2) {
            int opBegin = splitPositions[dop_i];
            int opEnd = splitPositions[dop_i + 1];

            // 1.build new graph.
            //  1) add WNOP to CPU graph.


            //  2) build CPU graph
            //      matmul in BSHD => matmul output BSHD.
            //      directly connect CPU and QNN graph. 




            // 2.delete ops in existing graph and add WNOP.

            // 3.connect tensors in different graphs.

        }


    }



    for (int ii = 0; ii < (int)param.size(); ++ii) {
        auto &sub_param = param[ii];
        vector<string> names = {};
        auto net_in_tensor = sub_param.net_inputs;
        for (const auto &out_t : net_in_tensor) {
            tensors_[out_t->name] = std::make_shared<Tensor>(backends_[backend_type].get());
            tensors_[out_t->name]->setName(out_t->name);
            for (auto &tensor_name : tensor_names_) {
                tensor_name.erase(std::remove(tensor_name.begin(), tensor_name.end(), out_t->name), tensor_name.end());
            }
            names.push_back(out_t->name);
        }

        for (auto *t:sub_param.net_tensors) {
            if(t->in == NULL){
                auto *in_tensor = t;
                tensors_[in_tensor->name] = std::make_shared<Tensor>(backends_[backend_type].get());
                tensors_[in_tensor->name]->setName(in_tensor->name);
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
        subg_1.reset(new Graph( param[i], backends_[backend_type].get(), tensors_, threadCount));
        subGraphs_["Prompt_Graph." + std::to_string(i)] = subg_1;
    }
}

} // namespace mllm
