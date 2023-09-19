#include <iostream>
#include <string>
#include <vector>
#include "Express.hpp"
#include "NetParameter.hpp"
#include "OP_defined.hpp"
#include <algorithm> // 包含 reverse 函数的头文件
#include <iostream>
using namespace mllm;
#define _STORE_OUT_TENSOR                      \
    ctx->net_tensors.insert(out_tensor);       \
    auto sub_param = get_active_subgraph(ctx); \
    out_tensor->subgraph = sub_param;          \
    sub_param->net_tensors.push_back(out_tensor);

#define _NEW_OP(type_)                            \
    sub_param->net_ops.emplace_back(new NetOp()); \
    auto net_op_ = (sub_param->net_ops.back());   \
    if (name.empty()) {                           \
        name = #type_ + std::to_string(ctx->idx); \
    }                                             \
    net_op_->name = name;                         \
    net_op_->type = type_;                        \
    net_op_->param = OpParam();                   \
    ctx->net_ops.push_back(net_op_);

#define _UPDATE_INPUT_TENSORS                                                                                                 \
    for (auto &input : inputs) {                                                                                              \
        net_op_->in.push_back(input);                                                                                         \
        input->out.push_back(net_op_);                                                                                        \
        if (std::find(sub_param->net_tensors.begin(), sub_param->net_tensors.end(), input) == sub_param->net_tensors.end()) { \
            sub_param->net_tensors.push_back(input);                                                                          \
            if (input->subgraph != nullptr) {                                                                                 \
                input->subgraph->net_outputs.insert(input);                                                                   \
                sub_param->net_inputs.insert(input);                                                                          \
            }                                                                                                                 \
        }                                                                                                                     \
    } // 函数实现 _EOP_
ETENSOR _EOP_(std::string name, OpType type, std::vector<ETENSOR> inputs, OpParam op_param) {
    EOP op(name, type, op_param);
    for (auto &input : inputs)
        op.connectedETensors.push_back(input);
    return ETENSOR(op);
}

// 函数实现 createNetParem
//
// void createNetParem(ETENSOR endT, NetParameter &net_param_) {
//     loopNetParem(endT, net_param_);
//     std::reverse(net_param_.net_ops.begin(), net_param_.net_ops.end());
// }

// int express_test() {
//     ETENSOR x(EOP("input1", Input));
//     // ETENSOR x("ETENSOR0");
//     auto y = _EOP_("silu1", Silu,{x});
//     x = _EOP_("matmul1", Matmul, {x, y});
//     x = _EOP_("scale1", Scale,{x});

// //     // 输出连接的 EOP
// //     NetParameter net_param_;
// //     createNetParem(x, net_param_);

//     return 0;
// }
// get active subgraph
NetParameter *get_active_subgraph(Context *ctx) {
    if (ctx->active_sub >= ctx->sub_param_.size()) {
        ctx->sub_param_.push_back(new NetParameter());
    }
    return ctx->sub_param_[ctx->active_sub];
}
NetTensor *_Input(Context *ctx, vector<int> dims, string name, DataType type) {
    //Ref Count?
    NetTensor *net_tensor = new NetTensor();
    if (name.empty()) {
        name = "input" + std::to_string(ctx->idx);
    }
    net_tensor->name = name;
    net_tensor->shape = dims;
    net_tensor->type = type;
    net_tensor->subgraph = get_active_subgraph(ctx);
    ctx->idx++;
    auto sub_param = get_active_subgraph(ctx);
    sub_param->net_tensors.push_back(net_tensor);
    ctx->net_tensors.insert(net_tensor);
    return net_tensor;
}
NetTensor *_Add(Context *ctx, std::vector<NetTensor *> inputs, string name) {
    //TODO:Check
    NetTensor *out_tensor = new NetTensor();
    if (name.empty()) {
        name = "Add" + std::to_string(ctx->idx);
    }
    out_tensor->name = name;
    //TODO: check Type
    out_tensor->type = inputs[0]->type;
    ctx->idx++;
    _STORE_OUT_TENSOR
    _NEW_OP(mllm::SoftMax)
    _UPDATE_INPUT_TENSORS
    out_tensor->in = net_op_;
    return out_tensor;
}
NetTensor *_SiLU(Context *ctx, std::vector<NetTensor *> inputs, string name) {
    NetTensor *out_tensor = new NetTensor();
    if (name.empty()) {
        name = "Silu" + std::to_string(ctx->idx);
    }
    out_tensor->name = name;
    //TODO: check Type
    out_tensor->type = inputs[0]->type;
    ctx->idx++;
    _STORE_OUT_TENSOR
    _NEW_OP(mllm::Silu)
    // sub_param->net_ops.emplace_back();
    // auto net_op_ = &(sub_param->net_ops.back());
    // if (name.empty()) {
    //     name = "Silu" + std::to_string(ctx->idx);
    // }
    // std::cout << net_op_ << std::endl;
    // net_op_->name = name;
    // net_op_->type = mllm::Silu;
    // net_op_->param = OpParam();
    // ctx->net_ops.push_back(net_op_);
    _UPDATE_INPUT_TENSORS
    out_tensor->in = net_op_;
    return out_tensor;
}
NetTensor *_Softmax(Context *ctx, std::vector<NetTensor *> inputs, int axis, string name) {
    NetTensor *out_tensor = new NetTensor();
    if (name.empty()) {
        name = "Softmax" + std::to_string(ctx->idx);
    }
    out_tensor->name = name;
    //TODO: check Type
    out_tensor->type = inputs[0]->type;
    ctx->idx++;
    _STORE_OUT_TENSOR
    _NEW_OP(mllm::SoftMax)
    net_op_->param["axis"] = axis;
    _UPDATE_INPUT_TENSORS
    out_tensor->in = net_op_;
    return out_tensor;
}
NetTensor *_Matmul(Context *ctx, std::vector<NetTensor *> inputs, string name) {
    NetTensor *out_tensor = new NetTensor();
    if (name.empty()) {
        name = "Matmul" + std::to_string(ctx->idx);
    }
    out_tensor->name = name;
    //TODO: check Type
    out_tensor->type = inputs[0]->type;
    ctx->idx++;
    _STORE_OUT_TENSOR
    _NEW_OP(mllm::Matmul)
    _UPDATE_INPUT_TENSORS
    out_tensor->in = net_op_;
    return out_tensor;
}

void Subgraph_begin(Context *ctx) {
    ctx->active_sub++;
}

// ETENSOR _Input(){
//     ETENSOR x(EOP("input1", Input));
//     return x;
// }
// ETENSOR _Add(std::string name, std::vector<ETENSOR> inputs){
//     return _EOP_(name, Add, inputs);
// }
// ETENSOR _CausalMask(std::string name, std::vector<ETENSOR> inputs){
//     return _EOP_(name, CausalMask, inputs);
// }
// ETENSOR _MatMul(std::string name, std::vector<ETENSOR> inputs){
//     return _EOP_(name, Matmul, inputs);
// }
// ETENSOR _RMSNorm(std::string name, std::vector<ETENSOR> inputs){
//     return _EOP_(name, RMSNorm, inputs);
// }
// ETENSOR _RoPE(std::string name, std::vector<ETENSOR> inputs){
//     return _EOP_(name, RoPE, inputs);
// }
// ETENSOR _Scale(std::string name, std::vector<ETENSOR> inputs){
//     return _EOP_(name, Scale, inputs);
// }
// ETENSOR _SiLU(std::string name, std::vector<ETENSOR> inputs){
//     return _EOP_(name, Silu, inputs);
// }
// ETENSOR _SoftMax(std::string name, std::vector<ETENSOR> inputs){
//     return _EOP_(name, SoftMax, inputs);
// }