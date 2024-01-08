#include <iostream>
#include <string>
#include <vector>
#include "unordered_map"
#include "Express.hpp"
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
//NetParameter *get_active_subgraph(Context *ctx) {
//    if (ctx->active_sub >= ctx->sub_param_.size()) {
//        ctx->sub_param_.emplace_back();
//    }
//    return &ctx->sub_param_[ctx->active_sub];
//}
// NOLINTBEGIN (readability-identifier-naming)
NetTensor *_Input(Context *ctx, vector<int> dims, string name, DataType type) {
    // Ref Count?
    NetTensor *net_tensor = new NetTensor();
    if (name.empty()) {
        name = "input" + std::to_string(ctx->idx);
    }
    net_tensor->name = name + "-00";
    net_tensor->shape_ = dims;
    net_tensor->type = type;
    net_tensor->subgraph = get_active_subgraph(ctx);
    ctx->idx++;
    auto *sub_param = get_active_subgraph(ctx);
    sub_param->net_tensors.push_back(net_tensor);
    ctx->net_tensors.insert(net_tensor);
    net_tensor->ctx = ctx;
    return net_tensor;
}

NetTensor *_Parameter(Context *ctx, std::vector<NetTensor *> inputs, int batch, int seq, int head, int dim, string name, DataType type) {
    // TODO:Check
    NetTensor *out_tensor = new NetTensor();
    if (name.empty()) {
        name = "Parameter" + std::to_string(ctx->idx);
    }
    out_tensor->name = "outtensor-" + name + "-00";
    // TODO: check Type
    out_tensor->type = type;
    ctx->idx++;
    _STORE_OUT_TENSOR
    _NEW_OP(mllm::PARAMETER)
    net_op_->param["batch"] = batch;
    net_op_->param["seq"] = seq;
    net_op_->param["head"] = head;
    net_op_->param["dim"] = dim;
    _UPDATE_INPUT_TENSORS
    out_tensor->in = net_op_;
    out_tensor->ctx = ctx;
    return out_tensor;

}
NetTensor *_Add(std::vector<NetTensor *> inputs, string name) {
    Context *ctx = inputs[0]->ctx;
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
    out_tensor->ctx = ctx;
    return out_tensor;
}
NetTensor *_Causalmask(std::vector<NetTensor *> inputs, string name) {
    Context *ctx = inputs[0]->ctx;
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
    out_tensor->ctx = ctx;
    return out_tensor;
}
NetTensor *_SiLU(std::vector<NetTensor *> inputs, string name) {
    Context *ctx = inputs[0]->ctx;
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
    out_tensor->ctx = ctx;
    return out_tensor;
}
NetTensor *_Softmax(std::vector<NetTensor *> inputs, int axis, string name) {
    Context *ctx = inputs[0]->ctx;
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
    out_tensor->ctx = ctx;
    return out_tensor;
}
NetTensor *_Matmul(std::vector<NetTensor *> inputs, bool transpose0, bool transpose1, string name) {
    Context *ctx = inputs[0]->ctx;
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
    out_tensor->ctx = ctx;
    return out_tensor;
}

NetTensor *_RMSNorm(std::vector<NetTensor *> inputs, int norm_size,  float epsilon, string name) {
    Context *ctx = inputs[0]->ctx;
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
    net_op_->param["norm_size"] = (int) norm_size;
    _UPDATE_INPUT_TENSORS
    out_tensor->in = net_op_;
    out_tensor->ctx = ctx;
    return out_tensor;
}

NetTensor *_RoPE(std::vector<NetTensor *> inputs, int pose_type, string name) {
    Context *ctx = inputs[0]->ctx;
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
    out_tensor->ctx = ctx;
    return out_tensor;
}

NetTensor *_Scale(std::vector<NetTensor *> inputs, float scale, float bias, bool bias_after_scale,string name) {
    Context *ctx = inputs[0]->ctx;
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
    out_tensor->ctx = ctx;
    return out_tensor;
}

NetTensor *_Linear(std::vector<NetTensor *> inputs, int in_features, int out_features, bool bias, string name) {
    Context *ctx = inputs[0]->ctx;
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
    out_tensor->ctx = ctx;
    return out_tensor;
}
NetTensor *_Embedding(std::vector<NetTensor *> inputs, int vocab_size, int hidden_size, string name) {
    Context *ctx = inputs[0]->ctx;
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
    out_tensor->ctx = ctx;
    return out_tensor;
}
NetTensor *_Mul(std::vector<NetTensor *> inputs, string name) {
    Context *ctx = inputs[0]->ctx;
    // TODO:Check
    NetTensor *out_tensor = new NetTensor();
    if (name.empty()) {
        name = "Mul" + std::to_string(ctx->idx);
    }
    out_tensor->name = "outtensor-" + name + "-00";
    // TODO: check Type
    out_tensor->type = inputs[0]->type;
    ctx->idx++;
    _STORE_OUT_TENSOR
    _NEW_OP(mllm::MUL)
    _UPDATE_INPUT_TENSORS
    out_tensor->in = net_op_;
    out_tensor->ctx = ctx;
    return out_tensor;
}
// NetTensor *_View(std::vector<NetTensor *> inputs, vector<int> dims, vector<int>data_dims, string name){
//     Context *ctx = inputs[0]->ctx;
//     NetTensor *out_tensor = new NetTensor();
//     if (name.empty()) {
//         name = "View" + std::to_string(ctx->idx);
//     }
//     out_tensor->name = "outtensor-" + name + "-00";
//     out_tensor->type = inputs[0]->type;
//     ctx->idx++;
//     _STORE_OUT_TENSOR
//     _NEW_OP(mllm::VIEW)
//     net_op_->param["dim0"] = dims[0];
//     net_op_->param["dim1"] = dims[1];
//     net_op_->param["dim2"] = dims[2];
//     net_op_->param["dim3"] = dims[3];
//     net_op_->param["data_dim0"] = data_dims[0];
//     net_op_->param["data_dim1"] = data_dims[1];
//     net_op_->param["data_dim2"] = data_dims[2];
//     net_op_->param["data_dim3"] = data_dims[3];
//     _UPDATE_INPUT_TENSORS
//     out_tensor->in = net_op_;
//     out_tensor->ctx = ctx;
//     return out_tensor;
// }
NetTensor *_KVCache(std::vector<NetTensor *> inputs, bool isK, string name) {
    Context *ctx = inputs[0]->ctx;
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
    out_tensor->ctx = ctx;
    return out_tensor;
}
NetTensor *_ReLU(std::vector<NetTensor *> inputs, string name) {
    Context *ctx = inputs[0]->ctx;
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
    out_tensor->ctx = ctx;
    return out_tensor;
}
NetTensor *_ReLUSquaredActivation(std::vector<NetTensor *> inputs, string name) {
    Context *ctx = inputs[0]->ctx;
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
    out_tensor->ctx = ctx;
    return out_tensor;
}
NetTensor *_GELU(std::vector<NetTensor *> inputs, string name) {
    Context *ctx = inputs[0]->ctx;
    NetTensor *out_tensor = new NetTensor();
    if (name.empty()) {
        name = "GELU" + std::to_string(ctx->idx);
    }
    out_tensor->name = "outtensor-" + name + "-00";
    out_tensor->type = inputs[0]->type;
    ctx->idx++;
    _STORE_OUT_TENSOR
    _NEW_OP(mllm::GELU)
    _UPDATE_INPUT_TENSORS
    out_tensor->in = net_op_;
    out_tensor->ctx = ctx;
    return out_tensor;
}
NetTensor *_QuickGELU(std::vector<NetTensor *> inputs, string name) {
    Context *ctx = inputs[0]->ctx;
    NetTensor *out_tensor = new NetTensor();
    if (name.empty()) {
        name = "_QuickGELU" + std::to_string(ctx->idx);
    }
    out_tensor->name = "outtensor-" + name + "-00";
    out_tensor->type = inputs[0]->type;
    ctx->idx++;
    _STORE_OUT_TENSOR
    _NEW_OP(mllm::QUICKGLUE)
    _UPDATE_INPUT_TENSORS
    out_tensor->in = net_op_;
    out_tensor->ctx = ctx;
    return out_tensor;
}
NetTensor *_LayerNorm(std::vector<NetTensor *> inputs, int norm_size, bool bias,  float epsilon, string name) {
    Context *ctx = inputs[0]->ctx;
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
    net_op_->param["norm_size"] = (int) norm_size;
    _UPDATE_INPUT_TENSORS
    out_tensor->in = net_op_;
    out_tensor->ctx = ctx;
    return out_tensor;
}
vector<NetTensor *> _Split(std::vector<NetTensor *> inputs, int split_num, Chl split_dim, int split_dim_size, string name){
    Context *ctx = inputs[0]->ctx;
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
        out_tensor->ctx = ctx;
        out_tensors.push_back(out_tensor);
    }
    return out_tensors;

}
NetTensor *_Gather(std::vector<NetTensor *> inputs, string name) {
    Context *ctx = inputs[0]->ctx;
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
    out_tensor->ctx = ctx;
    return out_tensor;
}

NetTensor *_Convolution2D(std::vector<NetTensor *> inputs, int in_channel, int out_channel, vector<int> kernal, vector<int> stride, PaddingType padding, bool bias, string name) {
    Context *ctx = inputs[0]->ctx;
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
    out_tensor->ctx = ctx;
    return out_tensor;
}
NetTensor *_Convolution3D(std::vector<NetTensor *> inputs, int in_channel, int out_channel, vector<int> kernal, vector<int> stride, PaddingType padding, bool bias, string name) {
    Context *ctx = inputs[0]->ctx;
    NetTensor *out_tensor = new NetTensor();
    if (name.empty()) {
        name = "Convolution3D" + std::to_string(ctx->idx);
    }
    out_tensor->name = "outtensor-" + name + "-00";
    out_tensor->type = inputs[0]->type;
    ctx->idx++;
    _STORE_OUT_TENSOR
    _NEW_OP(mllm::CONVOLUTION3D)
    net_op_->param["in_channel"] =(float) in_channel;
    net_op_->param["out_channel"] =(float) out_channel;
    net_op_->param["kernal_t"] =(float) kernal[0];
    net_op_->param["kernal_h"] =(float) kernal[1];
    net_op_->param["kernal_w"] =(float) kernal[2];
    net_op_->param["stride_t"] =(float) stride[1];
    net_op_->param["stride_h"] =(float) stride[1];
    net_op_->param["stride_w"] =(float) stride[2];
    net_op_->param["padding"] =(float) padding;
    net_op_->param["bias"] =(float) bias;
    _UPDATE_INPUT_TENSORS
    out_tensor->in = net_op_;
    out_tensor->ctx = ctx;
    return out_tensor;
}
NetTensor *_AvgPool2D(std::vector<NetTensor *> inputs, vector<int> kernal, vector<int> stride, PaddingType padding, string name ){
    Context *ctx = inputs[0]->ctx;
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
    out_tensor->ctx = ctx;
    return out_tensor;
}
NetTensor *_MaxPool2D(std::vector<NetTensor *> inputs, vector<int> kernal, vector<int> stride, PaddingType padding, string name ){
    Context *ctx = inputs[0]->ctx;
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
    out_tensor->ctx = ctx;
    return out_tensor;
}
NetTensor *_Cat(std::vector<NetTensor *> inputs, Chl axis, string name) {
    Context *ctx = inputs[0]->ctx;
    NetTensor *out_tensor = new NetTensor();
    if (name.empty()) {
        name = "_Cat" + std::to_string(ctx->idx);
    }
    out_tensor->name = "outtensor-" + name + "-00";
    out_tensor->type = inputs[0]->type;
    ctx->idx++;
    _STORE_OUT_TENSOR
    _NEW_OP(mllm::CAT)
    net_op_->param["axis"] =(float)axis;
    _UPDATE_INPUT_TENSORS
    out_tensor->in = net_op_;
    out_tensor->ctx = ctx;
    return out_tensor;
}
// NetTensor *_Transpose(std::vector<NetTensor *> inputs, string name) {
//     Context *ctx = inputs[0]->ctx;
//     NetTensor *out_tensor = new NetTensor();
//     if (name.empty()) {
//         name = "_Transpose" + std::to_string(ctx->idx);
//     }
//     out_tensor->name = "outtensor-" + name + "-00";
//     out_tensor->type = inputs[0]->type;
//     ctx->idx++;
//     _STORE_OUT_TENSOR
//     _NEW_OP(mllm::TRANSPOSE)
//     // net_op_->param["axis"] =(float)axis;
//     _UPDATE_INPUT_TENSORS
//     out_tensor->in = net_op_;
//     out_tensor->ctx = ctx;
//     return out_tensor;
// }
// NetTensor *_SubDim(std::vector<NetTensor *> inputs, Chl dim, vector<int> interval, string name){
//     Context *ctx = inputs[0]->ctx;
//     NetTensor *out_tensor = new NetTensor();
//     if (name.empty()) {
//         name = "_SubDim" + std::to_string(ctx->idx);
//     }
//     out_tensor->name = "outtensor-" + name + "-00";
//     out_tensor->type = inputs[0]->type;
//     ctx->idx++;
//     _STORE_OUT_TENSOR
//     _NEW_OP(mllm::SUBDIM)
//     net_op_->param["dim"] =(float)dim;
//     net_op_->param["start_i"] =(float)interval[0];
//     net_op_->param["end_i"] =(float)interval[1];
//     _UPDATE_INPUT_TENSORS
//     out_tensor->in = net_op_;
//     out_tensor->ctx = ctx;
//     return out_tensor;
// }
NetTensor *_Division(std::vector<NetTensor *> inputs, string name) {
    Context *ctx = inputs[0]->ctx;
    NetTensor *out_tensor = new NetTensor();
    if (name.empty()) {
        name = "Division" + std::to_string(ctx->idx);
    }
    out_tensor->name = "outtensor-" + name + "-00";
    out_tensor->type = inputs[0]->type;
    ctx->idx++;
    _STORE_OUT_TENSOR
    _NEW_OP(mllm::DIVISION)
    _UPDATE_INPUT_TENSORS
    out_tensor->in = net_op_;
    out_tensor->ctx = ctx;
    return out_tensor;
}
NetTensor *_Norm(std::vector<NetTensor *> inputs, int L_n, string name) {
    Context *ctx = inputs[0]->ctx;
    NetTensor *out_tensor = new NetTensor();
    if (name.empty()) {
        name = "_Norm" + std::to_string(ctx->idx);
    }
    out_tensor->name = "outtensor-" + name + "-00";
    out_tensor->type = inputs[0]->type;
    ctx->idx++;
    _STORE_OUT_TENSOR
    _NEW_OP(mllm::NORM)
    net_op_->param["L_n"] =(float)L_n;
    _UPDATE_INPUT_TENSORS
    out_tensor->in = net_op_;
    out_tensor->ctx = ctx;
    return out_tensor;
}
void _SubgraphBegin(Context *ctx) {
    ctx->active_sub++;
}
// NOLINTEND (readability-identifier-naming)