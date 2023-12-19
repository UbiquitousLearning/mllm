#include <iostream>
#include <string>
#include <vector>
#include "unordered_map"
#include "Express.hpp"
#include "NetParameter.hpp"
#include <algorithm> // 包含 reverse 函数的头文件
#include <memory>

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
    net_op_->param["type"] = type_;               \
    ctx->net_ops.push_back(net_op_);

// #define _UPDATE_INPUT_TENSORS                                                                                                 \
//     for (auto &input : inputs) {                                                                                              \
//         net_op_->in.push_back(input);                                                                                         \
//         input->out.push_back(net_op_);                                                                                        \
//         if (std::find(sub_param->net_tensors.begin(), sub_param->net_tensors.end(), input) == sub_param->net_tensors.end()) { \
//             sub_param->net_tensors.push_back(input);                                                                          \
//             if (input->subgraph != nullptr) {                                                                                 \
//                 input->subgraph->net_outputs.insert(input);                                                                   \
//                 sub_param->net_inputs.insert(input);                                                                          \
//             }                                                                                                                 \
//         }                                                                                                                     \
//     }
#define _UPDATE_INPUT_TENSORS                                                                                                 \
    for (auto &input : inputs) {                                                                                              \
        net_op_->in.push_back(input);                                                                                         \
        input->out.push_back(net_op_);                                                                                        \
        if (std::find(sub_param->net_tensors.begin(), sub_param->net_tensors.end(), input) == sub_param->net_tensors.end()) { \
            sub_param->net_tensors.push_back(input);                                                                          \
            if (input->subgraph != nullptr) {                                                                                 \
                sub_param->net_inputs.insert(input);                                                                          \
            }                                                                                                                 \
        }                                                                                                                     \
    }
static void topology(const NetParameter *net, vector<NetOp *> &result, NetOp *op, std::unordered_map<NetOp *, bool> &visited) {
    if (visited[op]) {
        return;
    }
    visited[op] = true;
    for (auto *input : op->in) {
        if (input->in != nullptr && std::find(net->net_inputs.begin(), net->net_inputs.end(), input) == net->net_inputs.end()) {
            topology(net, result, input->in, visited);
        }
    }
    result.push_back(op);
}
void NetParameter::topologySort() {
    std::unique_ptr<vector<NetOp *>> result(new vector<NetOp *>());
    std::unordered_map<NetOp *, bool> visited;
    result->reserve(net_ops.size());
    visited.reserve(net_ops.size());
    for (auto *op : net_ops) {
        topology(this, *result, op, visited);
    }
    /*
    for (auto *op : *result) {
        std::cout << op->name << std::endl;
    }
    */
    net_ops = *result;
}
// get active subgraph
NetParameter *get_active_subgraph(Context *ctx) {
    if (ctx->active_sub >= ctx->sub_param_.size()) {
        ctx->sub_param_.emplace_back();
    }
    return &ctx->sub_param_[ctx->active_sub];
}
// NOLINTBEGIN (readability-identifier-naming)
NetTensor *_Input(Context *ctx, vector<int> dims, string name, DataType type) {
    // Ref Count?
    NetTensor *net_tensor = new NetTensor();
    if (name.empty()) {
        name = "input" + std::to_string(ctx->idx);
    }
    net_tensor->name = name + "-00";
    net_tensor->shape = dims;
    net_tensor->type = type;
    net_tensor->subgraph = get_active_subgraph(ctx);
    ctx->idx++;
    auto *sub_param = get_active_subgraph(ctx);
    sub_param->net_tensors.push_back(net_tensor);
    ctx->net_tensors.insert(net_tensor);
    return net_tensor;
}
NetTensor *_Add(Context *ctx, std::vector<NetTensor *> inputs, string name) {
    // TODO:Check
    NetTensor *out_tensor = new NetTensor();
    if (name.empty()) {
        name = "Add" + std::to_string(ctx->idx);
    }
    out_tensor->name = "outtensor-" + name + "-00";
    // TODO: check Type
    out_tensor->type = inputs[0]->type;
    ctx->idx++;
    _STORE_OUT_TENSOR
    _NEW_OP(mllm::ADD)
    _UPDATE_INPUT_TENSORS
    out_tensor->in = net_op_;
    return out_tensor;
}
NetTensor *_Causalmask(Context *ctx, std::vector<NetTensor *> inputs, string name) {
    NetTensor *out_tensor = new NetTensor();
    if (name.empty()) {
        name = "Causalmask" + std::to_string(ctx->idx);
    }
    out_tensor->name = "outtensor-" + name + "-00";
    // TODO: check Type
    out_tensor->type = inputs[0]->type;
    ctx->idx++;
    _STORE_OUT_TENSOR
    _NEW_OP(mllm::CAUSALMASK)
    _UPDATE_INPUT_TENSORS
    out_tensor->in = net_op_;
    return out_tensor;
}
NetTensor *_SiLU(Context *ctx, std::vector<NetTensor *> inputs, string name) {
    NetTensor *out_tensor = new NetTensor();
    if (name.empty()) {
        name = "Silu" + std::to_string(ctx->idx);
    }
    out_tensor->name = "outtensor-" + name + "-00";
    // TODO: check Type
    out_tensor->type = inputs[0]->type;
    ctx->idx++;
    _STORE_OUT_TENSOR
    _NEW_OP(mllm::SILU)
    _UPDATE_INPUT_TENSORS
    out_tensor->in = net_op_;
    return out_tensor;
}
NetTensor *_Softmax(Context *ctx, std::vector<NetTensor *> inputs, int axis, string name) {
    NetTensor *out_tensor = new NetTensor();
    if (name.empty()) {
        name = "Softmax" + std::to_string(ctx->idx);
    }
    out_tensor->name = "outtensor-" + name + "-00";
    // TODO: check Type
    out_tensor->type = inputs[0]->type;
    ctx->idx++;
    _STORE_OUT_TENSOR
    _NEW_OP(mllm::SOFTMAX)
    net_op_->param["axis"] = axis;
    _UPDATE_INPUT_TENSORS
    out_tensor->in = net_op_;
    return out_tensor;
}
NetTensor *_Matmul(Context *ctx, std::vector<NetTensor *> inputs, bool transpose0, bool transpose1, string name) {
    NetTensor *out_tensor = new NetTensor();
    if (name.empty()) {
        name = "Matmul" + std::to_string(ctx->idx);
    }
    out_tensor->name = "outtensor-" + name + "-00";
    // TODO: check Type
    out_tensor->type = inputs[0]->type;
    ctx->idx++;
    _STORE_OUT_TENSOR
    _NEW_OP(mllm::MATMUL)
    net_op_->param["transpose0"] = transpose0;
    net_op_->param["transpose1"] = transpose1;
    _UPDATE_INPUT_TENSORS
    out_tensor->in = net_op_;
    return out_tensor;
}

NetTensor *_RMSNorm(Context *ctx, std::vector<NetTensor *> inputs, string name) {
    NetTensor *out_tensor = new NetTensor();
    if (name.empty()) {
        name = "RMSNorm" + std::to_string(ctx->idx);
    }
    out_tensor->name = "outtensor-" + name + "-00";
    // TODO: check Type
    out_tensor->type = inputs[0]->type;
    ctx->idx++;
    _STORE_OUT_TENSOR
    _NEW_OP(mllm::RMSNORM)
    _UPDATE_INPUT_TENSORS
    out_tensor->in = net_op_;
    return out_tensor;
}

NetTensor *_RoPE(Context *ctx, std::vector<NetTensor *> inputs, int pose_type, string name) {
    NetTensor *out_tensor = new NetTensor();
    if (name.empty()) {
        name = "RoPE" + std::to_string(ctx->idx);
    }
    out_tensor->name = "outtensor-" + name + "-00";
    // TODO: check Type
    out_tensor->type = inputs[0]->type;
    ctx->idx++;
    _STORE_OUT_TENSOR
    _NEW_OP(mllm::ROPE)
    net_op_->param["pose_type"] = pose_type;
    _UPDATE_INPUT_TENSORS
    out_tensor->in = net_op_;
    return out_tensor;
}

NetTensor *_Scale(Context *ctx, std::vector<NetTensor *> inputs, float scale, float bias, bool bias_after_scale,string name) {
    NetTensor *out_tensor = new NetTensor();
    if (name.empty()) {
        name = "Scale" + std::to_string(ctx->idx);
    }
    out_tensor->name = "outtensor-" + name + "-00";
    // TODO: check Type
    out_tensor->type = inputs[0]->type;
    ctx->idx++;
    _STORE_OUT_TENSOR
    _NEW_OP(mllm::SCALE)
    net_op_->param["scale"] = scale;
    net_op_->param["bias"] = bias;
    net_op_->param["bias_after_scale"] = (int)bias_after_scale;
    _UPDATE_INPUT_TENSORS
    out_tensor->in = net_op_;
    return out_tensor;
}

NetTensor *_Linear(Context *ctx, std::vector<NetTensor *> inputs, int in_features, int out_features, bool bias, string name) {
    NetTensor *out_tensor = new NetTensor();
    if (name.empty()) {
        name = "Linear" + std::to_string(ctx->idx);
    }
    out_tensor->name = "outtensor-" + name + "-00";
    // TODO: check Type
    out_tensor->type = inputs[0]->type;
    ctx->idx++;
    _STORE_OUT_TENSOR
    _NEW_OP(mllm::LINEAR)
    net_op_->param["in_features"] = in_features;
    net_op_->param["out_features"] = out_features;
    net_op_->param["bias"] = (int)bias;
    _UPDATE_INPUT_TENSORS
    out_tensor->in = net_op_;
    return out_tensor;
}
NetTensor *_Embedding(Context *ctx, std::vector<NetTensor *> inputs, int vocab_size, int hidden_size, string name) {
    NetTensor *out_tensor = new NetTensor();
    if (name.empty()) {
        name = "Embedding" + std::to_string(ctx->idx);
    }
    out_tensor->name = "outtensor-" + name + "-00";
    // TODO: check Type
    out_tensor->type = inputs[0]->type;
    ctx->idx++;
    _STORE_OUT_TENSOR
    _NEW_OP(mllm::EMBEDDING)
    net_op_->param["hidden_size"] = hidden_size;
    net_op_->param["vocab_size"] = vocab_size;
    _UPDATE_INPUT_TENSORS
    out_tensor->in = net_op_;
    return out_tensor;
}
NetTensor *_Mul(Context *ctx, std::vector<NetTensor *> inputs, string name) {
    // TODO:Check
    NetTensor *out_tensor = new NetTensor();
    if (name.empty()) {
        name = "Dot" + std::to_string(ctx->idx);
    }
    out_tensor->name = "outtensor-" + name + "-00";
    // TODO: check Type
    out_tensor->type = inputs[0]->type;
    ctx->idx++;
    _STORE_OUT_TENSOR
    _NEW_OP(mllm::MUL)
    _UPDATE_INPUT_TENSORS
    out_tensor->in = net_op_;
    return out_tensor;
}
NetTensor *_View(Context *ctx, std::vector<NetTensor *> inputs, vector<int> dims, vector<int>data_dims, string name){
    NetTensor *out_tensor = new NetTensor();
    if (name.empty()) {
        name = "View" + std::to_string(ctx->idx);
    }
    out_tensor->name = "outtensor-" + name + "-00";
    // TODO: check Type
    out_tensor->type = inputs[0]->type;
    ctx->idx++;
    _STORE_OUT_TENSOR
    _NEW_OP(mllm::VIEW)
    net_op_->param["dim0"] = dims[0];
    net_op_->param["dim1"] = dims[1];
    net_op_->param["dim2"] = dims[2];
    net_op_->param["dim3"] = dims[3];
    net_op_->param["data_dim0"] = data_dims[0];
    net_op_->param["data_dim1"] = data_dims[1];
    net_op_->param["data_dim2"] = data_dims[2];
    net_op_->param["data_dim3"] = data_dims[3];
    _UPDATE_INPUT_TENSORS
    out_tensor->in = net_op_;
    return out_tensor;
}
NetTensor *_KVCache(Context *ctx, std::vector<NetTensor *> inputs, bool isK, string name) {
    NetTensor *out_tensor = new NetTensor();
    if (name.empty()) {
        name = "KVCache" + std::to_string(ctx->idx);
    }
    out_tensor->name = "outtensor-" + name + "-00";
    // TODO: check Type
    out_tensor->type = inputs[0]->type;
    ctx->idx++;
    _STORE_OUT_TENSOR
    _NEW_OP(mllm::KVCACHE)
    net_op_->param["isK"] = isK;
    _UPDATE_INPUT_TENSORS
    out_tensor->in = net_op_;
    return out_tensor;
}
NetTensor *_ReLU(Context *ctx, std::vector<NetTensor *> inputs, string name) {
    NetTensor *out_tensor = new NetTensor();
    if (name.empty()) {
        name = "ReLU" + std::to_string(ctx->idx);
    }
    out_tensor->name = "outtensor-" + name + "-00";
    out_tensor->type = inputs[0]->type;
    ctx->idx++;
    _STORE_OUT_TENSOR
    _NEW_OP(mllm::RELU)
    _UPDATE_INPUT_TENSORS
    out_tensor->in = net_op_;
    return out_tensor;
}
NetTensor *_ReLUSquaredActivation(Context *ctx, std::vector<NetTensor *> inputs, string name) {
    NetTensor *out_tensor = new NetTensor();
    if (name.empty()) {
        name = "ReLUSquaredActivation" + std::to_string(ctx->idx);
    }
    out_tensor->name = "outtensor-" + name + "-00";
    out_tensor->type = inputs[0]->type;
    ctx->idx++;
    _STORE_OUT_TENSOR
    _NEW_OP(mllm::RELU2)
    _UPDATE_INPUT_TENSORS
    out_tensor->in = net_op_;
    return out_tensor;
}
NetTensor *_LayerNorm(Context *ctx, std::vector<NetTensor *> inputs, bool bias, string name) {
    NetTensor *out_tensor = new NetTensor();
    if (name.empty()) {
        name = "LayerNorm" + std::to_string(ctx->idx);
    }
    out_tensor->name = "outtensor-" + name + "-00";
    out_tensor->type = inputs[0]->type;
    ctx->idx++;
    _STORE_OUT_TENSOR
    _NEW_OP(mllm::LAYERNORM)
    net_op_->param["bias"] =(int) bias;
    _UPDATE_INPUT_TENSORS
    out_tensor->in = net_op_;
    return out_tensor;
}
vector<NetTensor *> _Split(Context *ctx, std::vector<NetTensor *> inputs, int split_num, Chl split_dim, int split_dim_size, string name){
    if (name.empty()) {
        name = "LayerNorm" + std::to_string(ctx->idx);
    }
    auto sub_param = get_active_subgraph(ctx);
    _NEW_OP(mllm::SPLIT)
    net_op_->param["split_num"] =(int) split_num;
    net_op_->param["split_dim"] =(int) split_dim;
    net_op_->param["split_dim_size"] =(int) split_dim_size;
    _UPDATE_INPUT_TENSORS
    vector<NetTensor *> out_tensors;
    net_op_->out_size = split_num;
    for (int i = 0; i < split_num; ++i) {
        NetTensor *out_tensor = new NetTensor();
        out_tensor->name = "outtensor-" + name + "-0" + std::to_string(i);
        out_tensor->type = inputs[0]->type;
        ctx->idx++;
        ctx->net_tensors.insert(out_tensor);
        out_tensor->subgraph = sub_param;
        sub_param->net_tensors.push_back(out_tensor);
        out_tensor->in = net_op_;
        out_tensors.push_back(out_tensor);
    }
    return out_tensors;

}
NetTensor *_Gather(Context *ctx, std::vector<NetTensor *> inputs, string name) {
    NetTensor *out_tensor = new NetTensor();
    if (name.empty()) {
        name = "Gather" + std::to_string(ctx->idx);
    }
    out_tensor->name = "outtensor-" + name + "-00";
    out_tensor->type = inputs[0]->type;
    ctx->idx++;
    _STORE_OUT_TENSOR
    _NEW_OP(mllm::GATHER)
    _UPDATE_INPUT_TENSORS
    out_tensor->in = net_op_;
    return out_tensor;
}

NetTensor *_Convolution2D(Context *ctx, std::vector<NetTensor *> inputs, int in_channel, int out_channel, vector<int> kernal, vector<int> stride, PaddingType padding, bool bias, string name) {
    NetTensor *out_tensor = new NetTensor();
    if (name.empty()) {
        name = "Convolution2D" + std::to_string(ctx->idx);
    }
    out_tensor->name = "outtensor-" + name + "-00";
    out_tensor->type = inputs[0]->type;
    ctx->idx++;
    _STORE_OUT_TENSOR
    _NEW_OP(mllm::CONVOLUTION2D)
    net_op_->param["in_channel"] =(float) in_channel;
    net_op_->param["out_channel"] =(float) out_channel;
    net_op_->param["kernal_h"] =(float) kernal[0];
    net_op_->param["kernal_w"] =(float) kernal[1];
    net_op_->param["stride_h"] =(float) stride[0];
    net_op_->param["stride_w"] =(float) stride[1];
    net_op_->param["padding"] =(float) padding;
    net_op_->param["bias"] =(float) bias;
    _UPDATE_INPUT_TENSORS
    out_tensor->in = net_op_;
    return out_tensor;
}
NetTensor *_AvgPool2D(Context *ctx, std::vector<NetTensor *> inputs, vector<int> kernal, vector<int> stride, PaddingType padding, string name ){
    NetTensor *out_tensor = new NetTensor();
    if (name.empty()) {
        name = "AvgPool2D" + std::to_string(ctx->idx);
    }
    out_tensor->name = "outtensor-" + name + "-00";
    out_tensor->type = inputs[0]->type;
    ctx->idx++;
    _STORE_OUT_TENSOR
    _NEW_OP(mllm::AVGPOOL2D)
    net_op_->param["kernal_h"] =(float) kernal[0];
    net_op_->param["kernal_w"] =(float) kernal[1];
    net_op_->param["stride_h"] =(float) stride[0];
    net_op_->param["stride_w"] =(float) stride[1];
    net_op_->param["padding"] =(float) padding;
    _UPDATE_INPUT_TENSORS
    out_tensor->in = net_op_;
    return out_tensor;
}
NetTensor *_MaxPool2D(Context *ctx, std::vector<NetTensor *> inputs, vector<int> kernal, vector<int> stride, PaddingType padding, string name ){
    NetTensor *out_tensor = new NetTensor();
    if (name.empty()) {
        name = "MaxPool2D" + std::to_string(ctx->idx);
    }
    out_tensor->name = "outtensor-" + name + "-00";
    out_tensor->type = inputs[0]->type;
    ctx->idx++;
    _STORE_OUT_TENSOR
    _NEW_OP(mllm::MAXPOOL2D)
    net_op_->param["kernal_h"] =(float) kernal[0];
    net_op_->param["kernal_w"] =(float) kernal[1];
    net_op_->param["stride_h"] =(float) stride[0];
    net_op_->param["stride_w"] =(float) stride[1];
    net_op_->param["padding"] =(float) padding;
    _UPDATE_INPUT_TENSORS
    out_tensor->in = net_op_;
    return out_tensor;
}
void _SubgraphBegin(Context *ctx) {
    ctx->active_sub++;
}
// NOLINTEND (readability-identifier-naming)

/***
 *
 * OLD VERSION
 *

static int op_idx = 0;

// 函数实现 _EOP_
ETENSOR _EOP_(std::string name, OpType type, std::vector<ETENSOR> inputs, OpParam op_param) {
    EOP op(name, type, op_param);
    for (auto &input : inputs)
        op.connectedETensors.push_back(input);
    return ETENSOR(op);
}

ETENSOR _Input(vector<int> shape) {
    OpParam opparam;
    opparam["type"] = INPUT;
    ETENSOR x(EOP(OpNames[INPUT] + std::to_string(op_idx), INPUT, opparam));
    return x;
}
ETENSOR _Add(std::vector<ETENSOR> inputs) {
    op_idx++;
    OpParam opparam;
    opparam["type"] = ADD;
    return _EOP_(OpNames[ADD] + std::to_string(op_idx), ADD, inputs, opparam);
}
ETENSOR _Mask(std::vector<ETENSOR> inputs) {
    op_idx++;
    OpParam opparam;
    opparam["type"] = MASK;
    return _EOP_(OpNames[MASK] + std::to_string(op_idx), MASK, inputs, opparam);
}
ETENSOR _MatMul(std::vector<ETENSOR> inputs) {
    op_idx++;
    OpParam opparam;
    opparam["type"] = MATMUL;
    return _EOP_(OpNames[MATMUL] + std::to_string(op_idx), MATMUL, inputs, opparam);
}
ETENSOR _RMSNorm(std::vector<ETENSOR> inputs) {
    op_idx++;
    OpParam opparam;
    opparam["type"] = RMSNORM;
    return _EOP_(OpNames[RMSNORM] + std::to_string(op_idx), RMSNORM, inputs, opparam);
}
ETENSOR _RoPE(std::vector<ETENSOR> inputs) {
    op_idx++;
    OpParam opparam;
    opparam["type"] = ROPE;
    return _EOP_(OpNames[ROPE] + std::to_string(op_idx), ROPE, inputs, opparam);
}
ETENSOR _Scale(std::vector<ETENSOR> inputs) {
    op_idx++;
    OpParam opparam;
    opparam["type"] = SCALE;
    return _EOP_(OpNames[SCALE] + std::to_string(op_idx), SCALE, inputs, opparam);
}
ETENSOR _SiLU(std::vector<ETENSOR> inputs) {
    op_idx++;
    OpParam opparam;
    opparam["type"] = SILU;
    return _EOP_(OpNames[SILU] + std::to_string(op_idx), SILU, inputs, opparam);
}
ETENSOR _SoftMax(std::vector<ETENSOR> inputs, int axis) {
    op_idx++;
    OpParam opparam;
    opparam["type"] = SOFTMAX;
    opparam["axis"] = axis;
    return _EOP_(OpNames[SOFTMAX] + std::to_string(op_idx), SOFTMAX, inputs, opparam);
}

// 函数实现 createNetParem
void loopNetParem(ETENSOR end_t, NetParameter &net_param) {
    auto net_op_ = new NetOp();
    net_op_->name = end_t.op.name;
    net_op_->type = end_t.op.type;
    net_op_->param = end_t.op.op_param;
    for (const auto &ETensor : end_t.op.connectedETensors) {
        net_op_->in_op.push_back(ETensor.op.name);
        auto nt = new NetTensor();
        nt->name = "outtensor-" + ETensor.op.name + "-00";
        net_op_->in.push_back(nt);
    }
    // std::cout << std::endl;
    if (net_op_->in_op.size() || net_op_->in.size()) {
        net_param.net_ops.push_back(net_op_);
    }

    // 递归调用，继续打印连接的 op
    for (const auto &ETENSOR : end_t.op.connectedETensors) {
        loopNetParem(ETENSOR, net_param);
    }
}
void createNetParem(ETENSOR end_t, NetParameter &net_param) {
    loopNetParem(end_t, net_param);
    std::reverse(net_param.net_ops.begin(), net_param.net_ops.end());
}

 */