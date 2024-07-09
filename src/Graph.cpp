//
// Created by Rongjie Yi.
//
#include "Graph.hpp"
#include "MemInspect.hpp"
#include "Types.hpp"
#ifdef DEBUGPRINT
#include "Timing.hpp"
#endif

std::string intToStringWithLeadingZero(int num) {
    if (num < 10) {
        return "0" + std::to_string(num);
    }
    return std::to_string(num);
}

namespace mllm {

#ifdef USE_QNN
static unordered_map<string, Op*> kv_cache_map;
#endif

Graph::Graph(const NetParameter &param, Backend *bn,
             unordered_map<string, shared_ptr<Tensor>> &external_tensors,
             int threadCount) {
    backend_ = bn;

    for (auto net_tensor : param.net_tensors) {
        auto it = external_tensors.find(net_tensor->name);
        if (it == external_tensors.end()) { // not in external_tensors
            tensors_[net_tensor->name] = std::make_shared<Tensor>(backend_);
            tensors_[net_tensor->name]->setName(net_tensor->name);
            tensors_[net_tensor->name]->setDtype(net_tensor->type);
        }
    }
    for (auto net_op : param.net_ops) {
    // for QNN prefill & CPU decoding execution, KVCache should be shared for each block
#ifdef USE_QNN
        if (net_op->type == KVCACHE) {
            std::cout << net_op->name << " is KVCache" << std::endl;
            shared_ptr<Op> my_op(nullptr);
            if (kv_cache_map.find(net_op->name) == kv_cache_map.end()) {
                // for the prefill part, we need to create a new op
                auto *new_op = backend_->opCreate(net_op->param, net_op->name, threadCount);
                my_op.reset(new_op);
                my_op->setOpType(net_op->type);
                kv_cache_map[net_op->name] = new_op;
            } else {
                std::cout << net_op->name << " is shared used" << std::endl;
                // for the decoding part, we need to get created op from global container
                my_op.reset(kv_cache_map[net_op->name]);
            }
            ops_[net_op->name] = my_op;
            continue;
        }
#endif
        shared_ptr<Op> my_op(nullptr);
        auto *new_op = backend_->opCreate(net_op->param, net_op->name, threadCount);
        my_op.reset(new_op);
        my_op->setOpType(net_op->type);
        ops_[net_op->name] = my_op;
    }
    for (auto net_op : param.net_ops) {
        bool connect_input = false;
        string op_name = net_op->name;
        op_names_.push_back(op_name);
        auto in_tensors = net_op->in;
        vector<shared_ptr<Tensor>> inTensors;
        for (auto *in_t : in_tensors) {
            if(in_t->in == NULL){
                connect_input = true;
            }
            auto in_t_name = in_t->name;
            auto it = tensors_.find(in_t_name);
            if (it != tensors_.end()) {
                inTensors.push_back(tensors_[in_t_name]);
            } else {
                inTensors.push_back(external_tensors[in_t_name]);
            }
        }
        vector<shared_ptr<Tensor>> outTensors;
        for (int oz = 0; oz < net_op->out_size; oz++) {
            auto out_t_name = "outtensor-" + op_name + "-" + intToStringWithLeadingZero(oz);
            auto it = tensors_.find(out_t_name);
            if (it != tensors_.end()) {
                outTensors.push_back(tensors_[out_t_name]);
            } else {
                outTensors.push_back(external_tensors[out_t_name]);
            }
        }
        ops_input_tensors_[op_name] = inTensors;
        ops_output_tensors_[op_name] = outTensors;
        if (connect_input) { ops_connect_input_.push_back(op_name); }
    }
}

void Graph::reflashInput(
    unordered_map<string, shared_ptr<Tensor>> &external_tensors) {
    for (auto op : ops_connect_input_) {
        vector<string> tmp_name;
        for (auto in_t : ops_input_tensors_[op]) {
            tmp_name.push_back(in_t->name());
        }
        ops_input_tensors_[op].clear();
        for (auto input_tensor_name : tmp_name) {
            if (tensors_.find(input_tensor_name) != tensors_.end()) {
                ops_input_tensors_[op].push_back(tensors_[input_tensor_name]);
            } else {
                ops_input_tensors_[op].push_back(
                    external_tensors[input_tensor_name]);
            }
        }
    }
}
void Graph::reshape() {
    for (const auto &op_name : op_names_) {
        bool do_ = true;
        if(ops_[op_name]->type() == PARAMETER || ops_[op_name]->type() == RANGE|| ops_[op_name]->type() == GATHER|| ops_[op_name]->type() == REPLACE){
            do_ = true;
        }else {
            for (auto &input_tensor : ops_input_tensors_[op_name]) {
                if (input_tensor->count() == 0) {
                    do_ = false;
                }
            }
        }
        ops_not_inputs_empty_[op_name] = do_;
        if(do_) {
            ops_[op_name]->reshape(
                ops_input_tensors_[op_name],
                ops_output_tensors_[op_name]); // tensors_[op_name]:1.reshape
        }else{
//            std::cout<<"op_name:"<<op_name<<" is not do"<<std::endl;
            for (auto &output_tensor : ops_output_tensors_[op_name]) {
                output_tensor->reshape(0, 0, 0, 0);
            }
        }
    }
}

void Graph::setUpTensors() {
    auto &graph_in_tensors = ops_input_tensors_[op_names_[0]];
    // set graph out tensor TensorType
    auto &graph_out_tensors = ops_output_tensors_[op_names_[op_names_.size() - 1]];
    for (auto &t : graph_out_tensors) {
        t->setTtype(OUTPUT_TENSOT);
    }

    this->backend_->onSetUpStart(graph_in_tensors, graph_out_tensors);

    for (auto &t : graph_in_tensors) { t->alloc(); }

    // set up tensors of ops
    for (const auto &op_name : op_names_) {
        if (ops_not_inputs_empty_[op_name] ) {
            ops_[op_name]->setUp(ops_input_tensors_[op_name],
                                 ops_output_tensors_[op_name]);
            PRINT_MEMORY_USAGE((op_name + " setUp").c_str());
        }else{
//            std::cout<<"op_name:"<<op_name<<" is not do"<<std::endl;
        }
    }
}

void Graph::setUpOps(ParamLoader &loader) {
    for (const auto &op_name : op_names_) {
        ops_[op_name]->load(loader);
        PRINT_MEMORY_USAGE((op_name + " load").c_str());
    }
}
//#define SAVECHECK
const vector<shared_ptr<Tensor>> &Graph::forward(bool autofree) {
    // backend event hook
    this->backend_->onExecuteStart(ops_input_tensors_[op_names_[0]], ops_output_tensors_[op_names_[op_names_.size() - 1]]);

    for (const auto &op_name : op_names_) {
        if (ops_not_inputs_empty_[op_name] ) {
#ifdef SAVECHECK
            for (auto &t : ops_input_tensors_[op_name]) {
                t->checkData<float>();
                t->saveData<float>();
            }
#endif
#ifdef DEBUGPRINT
            uint64_t t_start = mllm_time_us();
#endif
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
            if (autofree) {
                ops_[op_name]->free(ops_input_tensors_[op_name],
                                    ops_output_tensors_[op_name]);
            }
        }else{
//            std::cout<<"op_name:"<<op_name<<" is not do"<<std::endl;
        }
    }
    // backend event hook
    this->backend_->onExecuteEnd();
    return ops_output_tensors_[op_names_[op_names_.size() - 1]];
}

void Graph::freeOps() {
    for (const auto &op_name : op_names_) {
        ops_[op_name]->free(ops_input_tensors_[op_name],
                            ops_output_tensors_[op_name]);
    }
}
void Graph::freeTensors(){
    for(auto& t: tensors_){
        t.second->free();
    }
}
void Graph::free() {
    freeOps();
    freeTensors();
}
} // namespace mllm
