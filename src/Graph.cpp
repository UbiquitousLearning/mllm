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
    param_ = param;

    for (int i = 0; i < (int)param_.net_tensors.size(); ++i) {
        auto *net_tensor = param_.net_tensors[i];
        auto it = external_tensors.find(net_tensor->name);
        if (it == tensors_.end()) { // not in external_tensors
            tensors_[net_tensor->name] = std::make_shared<Tensor>(backend_);
            tensors_[net_tensor->name]->setName(net_tensor->name);
        }
    }
    for (int i = 0; i < (int)param_.net_ops.size(); ++i) {
        auto *net_op = param_.net_ops[i];
        shared_ptr<Op> my_op(NULL);
        auto *new_op = backend_->opCreate(net_op->param, net_op->name);
        my_op.reset(new_op);
        ops_[net_op->name] = my_op;
    }
    for (int i = 0; i < (int)param_.net_ops.size(); ++i) {
        auto *net_op = param_.net_ops[i];
        string lname = net_op->name;
        auto in_tensors = net_op->in;
        vector<shared_ptr<Tensor>> inTensors;
        for (auto *in_t : in_tensors) {
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
            auto out_t_name = "outtensor-" + lname + "-" + intToStringWithLeadingZero(oz);
            auto it = tensors_.find(out_t_name);
            if (it != tensors_.end()) {
                outTensors.push_back(tensors_[out_t_name]);
            } else {
                outTensors.push_back(external_tensors[out_t_name]);
            }
        }
        ops_input_tensors_[lname] = inTensors;
        ops_output_tensors_[lname] = outTensors;
    }
#ifdef NNAPI_ENABLED
    auto *nnapiBackend = dynamic_cast<NNAPIBackend *>(backend_);
    nnapiBackend->identifyInputsAndOutputs(inputTensors(), outputTensors());
#endif
}


void Graph::reflashInput(unordered_map<string, shared_ptr<Tensor>> &external_tensors){
    ops_input_tensors_[param_.net_ops[0]->name].clear();
    auto in_tensors = param_.net_ops[0]->in;
    //    vector<shared_ptr<Tensor>> inTensors;
    for (auto *in_t : in_tensors) {
        auto in_t_name = in_t->name;
        auto it = tensors_.find(in_t_name);
        if (it != tensors_.end()) {
            ops_input_tensors_[param_.net_ops[0]->name].push_back(tensors_[in_t_name]);
        } else {
            ops_input_tensors_[param_.net_ops[0]->name].push_back(external_tensors[in_t_name]);
        }
    }
    //ops_input_tensors_[param_.net_ops[0]->name][0]->printData<float>();
    //std::cout << param_.net_ops[0]->name << std::endl;
    //    ops_input_tensors_[param_.net_ops[0]->name] = inTensors;
}
void Graph::reshape() {
    // RESHAPE
    for (int i = 0; i < (int)param_.net_ops.size(); ++i) {
        auto *net_op = param_.net_ops[i];
        string lname = net_op->name;
        ops_[lname]->reshape(ops_input_tensors_[lname], ops_output_tensors_[lname]); // tensors_[lname]:1.reshape
    }
}

void Graph::setUpTensors() {
    auto &graph_in_tensors = ops_input_tensors_[param_.net_ops[0]->name];
    for (auto &t : graph_in_tensors) {
            t->alloc();
    }
    for (int i = 0; i < (int)param_.net_ops.size(); ++i) {
        auto *net_op = param_.net_ops[i];
        string lname = net_op->name;
        ops_[lname]->setUp(ops_input_tensors_[lname], ops_output_tensors_[lname]);
    }
}

void Graph::setUpOps(ParamLoader &loader) {
    for (int i = 0; i < (int)param_.net_ops.size(); ++i) {
        auto *net_op = param_.net_ops[i];
        ops_[net_op->name]->load(loader);
    }
}

//void Graph::reshapeOutputs() {
//    // RESHAPE
//    for (int i = 0; i < (int)param_.net_ops.size(); ++i) {
//        auto *net_op = param_.net_ops[i];
//        string lname = net_op->name;
//        ops_[lname]->reshapeOutputs(ops_input_tensors_[lname], ops_output_tensors_[lname]);
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

    for (int i = 0; i < (int)param_.net_ops.size(); ++i) {
        auto *net_op = param_.net_ops[i];
        string lname = net_op->name;
#ifdef DEBUG
        uint64_t t_start = mllm_time_us();
#endif
        ops_[lname]->execute(ops_input_tensors_[lname], ops_output_tensors_[lname]);
#ifdef DEBUG
        uint64_t t_end = mllm_time_us();
        std::cout<<"\n ====  "<<lname<<" ====  "<< (t_end - t_start)/1000.0F << " ms" ;
#endif
        if(autofree){
            ops_[lname]->free(ops_input_tensors_[lname], ops_output_tensors_[lname]);
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
    return ops_output_tensors_[param_.net_ops[param_.net_ops.size() - 1]->name];
}

//const vector<shared_ptr<Tensor>> &Graph::forward(const vector<shared_ptr<Tensor>> &inTensors) {
//    // Copy
//    // for (int i = 0; i < inTensors.size(); ++i) {
//    //     tensors_["Input0"][i]->CopyFrom(*inTensors[i]);
//    // }
//    return forward();
//}

void Graph::freeOps(){
    for (int i = 0; i < (int)param_.net_ops.size(); ++i) {
        auto *net_op = param_.net_ops[i];
        string lname = net_op->name;
        ops_[lname]->free(ops_input_tensors_[lname], ops_output_tensors_[lname]);
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
