#include "QNNGraph.hpp"
#include "OpDefined.hpp"
#include "Types.hpp"
#include <cstring>
#include <memory>
#ifdef DEBUGPRINT
#include "Timing.hpp"
#endif

#include "QNNBackend.hpp"

namespace mllm {

QNNGraph::QNNGraph(const NetParameter &param, Backend *bn,
                   unordered_map<string, shared_ptr<Tensor>> &external_tensors,
                   int threadCount, string graphName) :
    Graph(param, bn, external_tensors, threadCount), graphName_(graphName) {
}

// TODO: deprecated, remove
void QNNGraph::setUpTensors(std::string name) {

    // change to use merge op output as graph input tensor
    vector<shared_ptr<Tensor>> graph_in_tensors;
    if (ops_[op_names_[0]]->type() == SPLITINPUT) {
        graph_in_tensors = ops_output_tensors_[op_names_[0]];
    } else {
        graph_in_tensors = ops_input_tensors_[op_names_[0]];
    }
    
    // set graph out tensor TensorType
    auto &graph_out_tensors = ops_output_tensors_[op_names_[op_names_.size() - 1]];
    for (auto &t : graph_out_tensors) {
        t->setTtype(GRAPH_OUTPUT);
        t->alloc();
    }

    this->backend_->onSetUpStart(graph_in_tensors, graph_out_tensors, name);

    // set up tensors of ops
    for (const auto &op_name : op_names_) {
        if (ops_not_inputs_empty_[op_name]) {
            ops_[op_name]->setUp(ops_input_tensors_[op_name],
                                 ops_output_tensors_[op_name]);
        } else {
            // std::cout << "op_name:" << op_name << " is not do" << std::endl;
        }
    }

    this->backend_->onSetUpEnd(graph_in_tensors, graph_out_tensors, name);
}

void QNNGraph::setUpTensors() {
    // change to use merge op output as graph input tensor
    vector<shared_ptr<Tensor>> graph_in_tensors;
    if (ops_[op_names_[0]]->type() == SPLITINPUT) {
        graph_in_tensors = ops_output_tensors_[op_names_[0]];
    } else {
        graph_in_tensors = ops_input_tensors_[op_names_[0]];
    }

    // set graph out tensor TensorType
    auto &graph_out_tensors = ops_output_tensors_[op_names_[op_names_.size() - 1]];
    for (auto &t : graph_out_tensors) {
        t->setTtype(GRAPH_OUTPUT);
        t->alloc();
    }

    this->backend_->onSetUpStart(graph_in_tensors, graph_out_tensors, graphName_);

    // set up tensors of ops
    for (const auto &op_name : op_names_) {
        if (ops_not_inputs_empty_[op_name]) {
            ops_[op_name]->setUp(ops_input_tensors_[op_name],
                                 ops_output_tensors_[op_name]);
        } else {
            // std::cout << "op_name:" << op_name << " is not do" << std::endl;
        }
    }

    this->backend_->onSetUpEnd(graph_in_tensors, graph_out_tensors, graphName_);
}

// WARNING: non virtual override function, all features should be merged into the origin function
const vector<shared_ptr<Tensor>> &QNNGraph::forward(std::string graphName) {
    for (const auto &op_name : op_names_) {
        if (ops_not_inputs_empty_[op_name]) {
#ifdef SAVECHECK
            for (auto &t : ops_input_tensors_[op_name]) {
                t->checkData<float>();
                t->saveData<float>();
            }
#endif
#ifdef DEBUGPRINT
            uint64_t t_start = mllm_time_us();
#endif
            if (ops_[op_name]->type() == LINEARINT8SHADOW || ops_[op_name]->type() == ROPE)
                continue;
            ops_[op_name]->execute(ops_input_tensors_[op_name],
                                   ops_output_tensors_[op_name]);

#ifdef SAVECHECK
            for (auto &t : ops_output_tensors_[op_name]) {
                t->checkData<float>();
                t->saveData<float>();
            }
#endif

#ifdef DEBUGPRINT
            uint64_t t_end = mllm_time_us();
            std::cout << "" << op_name
                      << "       exe_time:" << (t_end - t_start) / 1000.0F << " ms"
                      << std::endl;
#endif
        } else {
            //            std::cout<<"op_name:"<<op_name<<" is not do"<<std::endl;
        }
    }
    
    this->backend_->onExecuteStart(ops_input_tensors_[op_names_[0]], ops_output_tensors_[op_names_[op_names_.size() - 1]], graphName);

    return ops_output_tensors_[op_names_[op_names_.size() - 1]];
}

void QNNGraph::free() {
    auto *qnn_backend = dynamic_cast<QNNBackend *>(this->backend_);
    qnn_backend->freeGraphDataStructure(graphName_);
}

void QNNGraph::allFree() {
    auto *qnn_backend = dynamic_cast<QNNBackend *>(this->backend_);
    qnn_backend->afterAllGraphsExecute();
}

const vector<shared_ptr<Tensor>> &QNNGraph::forward(bool autofree) {
    for (const auto &op_name : op_names_) {
        if (ops_not_inputs_empty_[op_name]) {
#ifdef SAVECHECK
            for (auto &t : ops_input_tensors_[op_name]) {
                t->checkData<float>();
                t->saveData<float>();
            }
#endif
#ifdef DEBUGPRINT
            uint64_t t_start = mllm_time_us();
#endif
            if (ops_[op_name]->type() == LINEARINT8SHADOW || ops_[op_name]->type() == ROPE)
                continue;
            ops_[op_name]->execute(ops_input_tensors_[op_name],
                                   ops_output_tensors_[op_name]);

#ifdef SAVECHECK
            for (auto &t : ops_output_tensors_[op_name]) {
                t->checkData<float>();
                t->saveData<float>();
            }
#endif

#ifdef DEBUGPRINT
            uint64_t t_end = mllm_time_us();
            std::cout << "" << op_name
                      << "       exe_time:" << (t_end - t_start) / 1000.0F << " ms"
                      << std::endl;
#endif
        } else {
            //            std::cout<<"op_name:"<<op_name<<" is not do"<<std::endl;
        }
    }

    this->backend_->onExecuteStart(ops_input_tensors_[op_names_[0]], ops_output_tensors_[op_names_[op_names_.size() - 1]], graphName_);

    return ops_output_tensors_[op_names_[op_names_.size() - 1]];
}

} // namespace mllm
