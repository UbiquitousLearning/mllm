//
// Created by 30500 on 2020/12/2 0002.
//
#include "Graph.hpp"
#include "OpDefined.hpp"

std::string intToStringWithLeadingZero(int num) {
    if (num < 10) {
        return "0" + std::to_string(num);
    }
    return std::to_string(num);
}

namespace mllm {
// template class Graph;
// template class Graph;

/**
 * @brief 初始化
 * @param in_param
 */

Graph::Graph(const NetParameter &param, Backend *bn, unordered_map<string, shared_ptr<Tensor>> &external_tensors) {
    backend_ = bn;

    for (auto net_tensor : param.net_tensors) {
        auto it = external_tensors.find(net_tensor->name);
        if (it == tensors_.end()) { // not in external_tensors
            tensors_[net_tensor->name] = std::make_shared<Tensor>(backend_);
            tensors_[net_tensor->name]->setName(net_tensor->name);
        }
    }
    for (auto net_op : param.net_ops) {
        shared_ptr<Op> my_op(nullptr);
        auto *new_op = backend_->opCreate(net_op->param, net_op->name);
        my_op.reset(new_op);
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
        if(connect_input){
            ops_connect_input_.push_back(op_name);
        }
    }
#ifdef NNAPI_ENABLED
    auto *nnapiBackend = dynamic_cast<NNAPIBackend *>(backend_);
    nnapiBackend->identifyInputsAndOutputs(inputTensors(), outputTensors());
#endif
}


void Graph::reflashInput(unordered_map<string, shared_ptr<Tensor>> &external_tensors){
    for (auto op :ops_connect_input_) {
        vector<string> tmp_name ;
        for (auto in_t : ops_input_tensors_[op]) {
            tmp_name.push_back(in_t->name());
        }
        ops_input_tensors_[op].clear();
        for (auto input_tensor_name : tmp_name)
        {
            if (tensors_.find(input_tensor_name) != tensors_.end()) {
                ops_input_tensors_[op].push_back(tensors_[input_tensor_name]);
            } else {
                ops_input_tensors_[op].push_back(external_tensors[input_tensor_name]);
            }
        }

    }
//    ops_input_tensors_[op_names_[0]].clear();
////    auto in_tensors = param_.net_ops[0]->in;
//    //    vector<shared_ptr<Tensor>> inTensors;
//    for (auto input_tensor_name : input_tensor_names)
//    {
//        if (tensors_.find(input_tensor_name) != tensors_.end()) {
//            ops_input_tensors_[op_names_[0]].push_back(tensors_[input_tensor_name]);
//        } else {
//            ops_input_tensors_[op_names_[0]].push_back(external_tensors[input_tensor_name]);
//        }
//    }
    //ops_input_tensors_[param_.net_ops[0]->name][0]->printData<float>();
    //std::cout << param_.net_ops[0]->name << std::endl;
    //    ops_input_tensors_[param_.net_ops[0]->name] = inTensors;
}
void Graph::reshape() {
    // RESHAPE
    for (const auto& op_name : op_names_) {
        ops_[op_name]->reshape(ops_input_tensors_[op_name], ops_output_tensors_[op_name]); // tensors_[op_name]:1.reshape
    }
}

void Graph::setUpTensors() {
    auto &graph_in_tensors = ops_input_tensors_[op_names_[0]];
    for (auto &t : graph_in_tensors) {
        t->alloc();
    }
    for (const auto& op_name : op_names_) {
        ops_[op_name]->setUp(ops_input_tensors_[op_name], ops_output_tensors_[op_name]);
    }
}

void Graph::setUpOps(ParamLoader &loader) {
    for (const auto& op_name : op_names_) {
        ops_[op_name]->load(loader);
    }
}

//void Graph::reshapeOutputs() {
//    // RESHAPE
//    for (int i = 0; i < (int)param_.net_ops.size(); ++i) {
//        auto *net_op = param_.net_ops[i];
//        string op_name = net_op->name;
//        ops_[op_name]->reshapeOutputs(ops_input_tensors_[op_name], ops_output_tensors_[op_name]);
//    }
//}

//void Graph::setUp(unordered_map<string, shared_ptr<Tensor>> &external_tensors, bool init, bool reshape, bool graph0) {
//    if (init) {
//        std::cout << "EXE:: Init" << std::endl;
//        this->setUpTensors();
//    } else if (reshape) {
//        std::cout << "EXE:: Reshape" << std::endl;
//        if (graph0) {
//            this->reFlashInput(external_tensors);
//        }
//        this->reshapeOutputs();
//    }
//}


/**
 * @brief 前向传播
 * @param loss
 * @return
 */
// #define DEBUG
const vector<shared_ptr<Tensor>> &Graph::forward(bool autofree) {
    // TODO 改为递归

    for (const auto& op_name : op_names_) {

#ifdef DEBUG
        uint64_t t_start = mllm_time_us();
#endif
        ops_[op_name]->execute(ops_input_tensors_[op_name], ops_output_tensors_[op_name]);

        // for(auto &t: ops_input_tensors_[op_name]){
        //     // t->checkData<float>();
        //     t->saveData<float>();
        // }
        //  for(auto &t: ops_output_tensors_[op_name]){
        //      // t->checkData<float>();
        //      t->saveData<float>();
        //  }

#ifdef DEBUG
        uint64_t t_end = mllm_time_us();
        std::cout<<"\n ====  "<<op_name<<" ====  "<< (t_end - t_start)/1000.0F << " ms" ;
#endif
        if(autofree){
            ops_[op_name]->free(ops_input_tensors_[op_name], ops_output_tensors_[op_name]);
        }
    }
// invoke nnapi model
#ifdef NNAPI_ENABLED
    std::cout << "NNAPI invoke model" << std::endl;
    auto *nnapiBackend = dynamic_cast<NNAPIBackend *>(backend_);
    nnapiBackend->buildModel();
    nnapiBackend->invokeModel();
#endif
    // TODO
    return ops_output_tensors_[op_names_[op_names_.size() - 1]];
}

//const vector<shared_ptr<Tensor>> &Graph::forward(const vector<shared_ptr<Tensor>> &inTensors) {
//    // Copy
//    // for (int i = 0; i < inTensors.size(); ++i) {
//    //     tensors_["Input0"][i]->CopyFrom(*inTensors[i]);
//    // }
//    return forward();
//}

void Graph::freeOps(){
    for (const auto& op_name: op_names_) {
        ops_[op_name]->free(ops_input_tensors_[op_name], ops_output_tensors_[op_name]);
    }
}
void Graph::freeTensors(){
    for(auto& t: tensors_){
        t.second->free();
    }
}
void Graph::free() {
    //TODO update
    freeOps();
    freeTensors();
}
} // namespace mllm
