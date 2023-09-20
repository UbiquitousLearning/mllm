//
// Created by 30500 on 2020/12/2 0002.
//
#include "Graph.hpp"
#include "OpDefined.hpp"

void splitTensorName(std::string input) {
    std::vector<std::string> result;
    std::stringstream ss(input);
    std::string segment;

    while (std::getline(ss, segment, '-')) {
        result.push_back(segment);
    }

    // 输出拆分后的字符串
    for (const auto &str : result) {
        std::cout << str << std::endl;
    }
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
    Init(external_tensors);
}

void Graph::Init(unordered_map<string, shared_ptr<Tensor>> &external_tensors) {
    // RESHAPE
    // tensors_["Input0"] = vector<shared_ptr<Tensor>>(1, NULL);
    // for (auto &t : tensors_["Input0"]) {
    //     std::shared_ptr<Tensor> tensor1 = std::make_shared<Tensor>();
    //     t = tensor1;
    //     t->SetByteWidth(sizeof(float));
    //     t->SetBackend(backend_);
    //     t->Reshape(1, 3, 5, 5); // TODO Reshape  tensors_["input"]
    // }
    // for (int i = 0; i < (int)param_.net_ops.size(); ++i) {
    //     // TODO: 3改成不同的数
    //     auto net_op = param_.net_ops[i];
    //     auto op_name_ = net_op->name;
    //     tensors_[op_name_] = vector<shared_ptr<Tensor>>(3, NULL);
    //     for (auto &t : tensors_[op_name_]) {
    //         std::shared_ptr<Tensor> tensor1 = std::make_shared<Tensor>();
    //         t = tensor1;
    //         t->SetByteWidth(sizeof(float));
    //         t->SetBackend(backend_);
    //     }
    // }

    for (int i = 0; i < (int)param_.net_tensors.size(); ++i) {
        auto net_tensor = param_.net_tensors[i];
        auto it = external_tensors.find(net_tensor->name);
        if (it == tensors_.end()) { // not in external_tensors
            auto net_tensor = param_.net_tensors[i];
            tensors_[net_tensor->name] = std::make_shared<Tensor>();
            tensors_[net_tensor->name]->SetName(net_tensor->name);
        }
    }

    for (int i = 0; i < (int)param_.net_ops.size(); ++i) {
        auto net_op = param_.net_ops[i];
        shared_ptr<Op> myOp(NULL);
        auto newOp = backend_->OpCreate(net_op->param);
        myOp.reset(newOp);
        string lname = net_op->name;

        // TODO: CHECK一下 inTensors 尤其是[0]
        // vector<string> inames = net_op->in_op;
        // vector<shared_ptr<Tensor>> inTensors;
        // for (auto name : inames) {
        //     inTensors.push_back(tensors_[name][0]);
        // }
        auto in_tensors = net_op->in;
        vector<shared_ptr<Tensor>> inTensors;
        for (auto in_t : in_tensors) {
            auto in_t_name = in_t->name;
            auto it = tensors_.find(in_t_name);
            if (it != tensors_.end()) {
                inTensors.push_back(tensors_[in_t_name]);
            } else {
                inTensors.push_back(external_tensors[in_t_name]);
            }
        }
        vector<shared_ptr<Tensor>> outTensors;
        auto out_t_name = "outtensor-" + lname + "-00";
        auto it = tensors_.find(out_t_name);
        if (it != tensors_.end()) {
            outTensors.push_back(tensors_[out_t_name]);
        } else {
            outTensors.push_back(external_tensors[out_t_name]);
        }

        // splitTensorName(in_t_name);
        // std::cout << lname << std::endl;
        ops_input_tensors_[lname] = inTensors;
        ops_output_tensors_[lname] = outTensors;
        ops_[lname] = myOp;
        ops_[lname]->Reshape(ops_input_tensors_[lname], ops_output_tensors_[lname]); // tensors_[lname]:1.Reshape
    }
}

void Graph::Setup() {
    // for (auto &t : tensors_["Input0"]) {
    //     t->Alloc(); // to_cpu//malloc&memset 0 TODO
    //     t->SetName("Input0_");
    // }
    for (int i = 0; i < (int)param_.net_ops.size(); ++i) {
        auto net_op = param_.net_ops[i];
        string lname = net_op->name; // op_names_[i];
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
        ops_[lname]->Setup(ops_input_tensors_[lname], ops_output_tensors_[lname]); // tensors_[lname]:malloc&memset 0 //TODO: 加入Bachend后改成不同Device的malloc
    }
}

void Graph::Load(ParamLoader &loader) {
    for (int i = 0; i < (int)param_.net_ops.size(); ++i) {
        auto net_op = param_.net_ops[i];
        ops_[net_op->name]->Load(loader);
    }

    // if(loader.Load())
}

/**
 * @brief 前向传播
 * @param loss
 * @return
 */

const vector<shared_ptr<Tensor>> &Graph::Forward() {
    // TODO 改为递归

    for (int i = 0; i < (int)param_.net_ops.size(); ++i) {
        auto net_op = param_.net_ops[i];
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
        // ops_[lname]->Execute(inTensors, tensors_[lname]);
        ops_[lname]->Execute(ops_input_tensors_[lname], ops_output_tensors_[lname]);
    }
    // TODO
    return ops_output_tensors_[param_.net_ops[param_.net_ops.size() - 1]->name];
}

const vector<shared_ptr<Tensor>> &Graph::Forward(const vector<shared_ptr<Tensor>> &inTensors) {
    // Copy
    // for (int i = 0; i < inTensors.size(); ++i) {
    //     tensors_["Input0"][i]->CopyFrom(*inTensors[i]);
    // }
    return Forward();
}

/**
 * @brief 反向传播
 */

void Graph::Backward() {
}

} // namespace mllm
