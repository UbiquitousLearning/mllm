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
//        string lname = net_op->name;
//        my_op->setName(lname);
        ops_[net_op->name] = my_op;
    }
//    shapeInit(external_tensors);
}

void Graph::shapeInit(unordered_map<string, shared_ptr<Tensor>> &external_tensors) {

    // RESHAPE
    for (int i = 0; i < (int)param_.net_ops.size(); ++i) {
        auto *net_op = param_.net_ops[i];
//        shared_ptr<Op> my_op(NULL);
//        auto *new_op = backend_->opCreate(net_op->param);
//        my_op.reset(new_op);
        string lname = net_op->name;
//        my_op->setName(lname);

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
//        ops_[lname] = my_op;
        ops_[lname]->reshape(ops_input_tensors_[lname], ops_output_tensors_[lname]); // tensors_[lname]:1.shapeInit
    }
}

void Graph::setUp() {
    auto &graph_in_tensors = ops_input_tensors_[param_.net_ops[0]->name];
    for (auto &t : graph_in_tensors) {
        t->alloc();
    }
    for (int i = 0; i < (int)param_.net_ops.size(); ++i) {
        auto *net_op = param_.net_ops[i];
        string lname = net_op->name;                                               // op_names_[i];
        ops_[lname]->setUp(ops_input_tensors_[lname], ops_output_tensors_[lname]); // tensors_[lname]:malloc&memset 0 //TODO: 加入Bachend后改成不同Device的malloc
    }
}

void Graph::load(ParamLoader &loader) {
    for (int i = 0; i < (int)param_.net_ops.size(); ++i) {
        auto *net_op = param_.net_ops[i];
        ops_[net_op->name]->load(loader);
    }

    // if(loader.load())
}

void Graph::reshapeOutputs(unordered_map<string, shared_ptr<Tensor>> &external_tensors) {
    // RESHAPE
    for (int i = 0; i < (int)param_.net_ops.size(); ++i) {
        auto *net_op = param_.net_ops[i];
        string lname = net_op->name;
        ops_[lname]->reshapeOutputs(ops_input_tensors_[lname], ops_output_tensors_[lname]);
    }
}
/**
 * @brief 前向传播
 * @param loss
 * @return
 */

const vector<shared_ptr<Tensor>> &Graph::forward() {
    // TODO 改为递归

    for (int i = 0; i < (int)param_.net_ops.size(); ++i) {
        auto *net_op = param_.net_ops[i];
        string lname = net_op->name;
        // TODO: CHECK一下 inTensors 尤其是[0]
        // vector<string> inames = net_op->in_op; // op_in_names_[i];
        // vector<shared_ptr<Tensor>> inTensors;
        // for (auto name : inames) {
        //     inTensors.push_back(tensors_[name][0]);
        // }
        // auto in_tensors = net_op->in;
        // vector<shared_ptr<Tensor>> inTensors;
        // for (auto in_t : in_tensors) {
        //     inTensors.push_back(tensors_[in_t->in->name][0]);
        // }
        // ops_[lname]->execute(inTensors, tensors_[lname]);
        ops_[lname]->execute(ops_input_tensors_[lname], ops_output_tensors_[lname]);
    }
    // TODO
    return ops_output_tensors_[param_.net_ops[param_.net_ops.size() - 1]->name];
}

const vector<shared_ptr<Tensor>> &Graph::forward(const vector<shared_ptr<Tensor>> &inTensors) {
    // Copy
    // for (int i = 0; i < inTensors.size(); ++i) {
    //     tensors_["Input0"][i]->CopyFrom(*inTensors[i]);
    // }
    return forward();
}

/**
 * @brief 反向传播
 */

void Graph::backward() {
}

} // namespace mllm
