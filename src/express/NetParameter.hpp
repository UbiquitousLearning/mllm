
#ifndef MLLM_NETPARAMETER_H
#define MLLM_NETPARAMETER_H

#include "Types.hpp"
#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string.h>
#include <string>
#include <vector>
#include <cassert>
using std::string;
using std::vector;
using std::map;

namespace mllm {

typedef struct TNetTensor NetTensor;
typedef struct TNetParameter NetParameter;

typedef struct TNetOp {
    OpType type;
    vector<NetTensor *> in;
    vector<NetTensor *> out;
    vector<string> in_op; // input ops' names;
    string name;
    OpParam param;
    int out_size = 1; // output tensor size

} NetOp;

typedef struct TNetParameter {
    string weights_path;
    string model_path;
    // string input_name;
    // string output_name;
    vector<NetOp *> net_ops;
    vector<NetTensor *> net_tensors;
    std::set<NetTensor *> net_inputs;
    std::set<NetTensor *> net_outputs;
    void topologySort();

} NetParameter;

// 前置声明
struct Context {
    vector<NetParameter> sub_param_;
    vector<NetOp *> net_ops;
    std::set<NetTensor *> net_tensors;
    int idx = 0;
    int active_sub = 0;
};
inline NetParameter *get_active_subgraph(Context *ctx) {
    if (ctx->active_sub >= ctx->sub_param_.size()) {
        ctx->sub_param_.emplace_back();
    }
    return &ctx->sub_param_[ctx->active_sub];
}
typedef struct TNetTensor {
    string name;
    vector<int> shape;
    DataType type;
    NetOp *in;
    vector<NetOp *> out;
    NetParameter *subgraph;
    Context *ctx;

    NetTensor *clip(vector<int> b, vector<int> h, vector<int> s, vector<int> d){
        Context *ctx =this->ctx;
        NetTensor *out_tensor = new NetTensor();
        if (name.empty()) {
            name = this->name + "_clip_"+std::to_string(ctx->idx);
        }
        out_tensor->name = "outtensor-" + name + "-00";
        out_tensor->type = this->type;
        ctx->idx++;
        ctx->net_tensors.insert(out_tensor);
        auto sub_param = get_active_subgraph(ctx);
        out_tensor->subgraph = sub_param;
        sub_param->net_tensors.push_back(out_tensor);
        sub_param->net_ops.emplace_back(new NetOp());
        auto net_op_ = (sub_param->net_ops.back());
        net_op_->name = name;
        net_op_->type = SUBDIM;
        net_op_->param = OpParam();
        net_op_->param["type"] = SUBDIM;
        ctx->net_ops.push_back(net_op_);
        //PARAM
        if((b.size()+h.size()+s.size()+d.size()==2) &&(b.size()*b.size()+h.size()*h.size()+s.size()*s.size()+d.size()*d.size()==4)) {
            if (b.size() == 2) {
                net_op_->param["dim"] = (float)BATCH;
                net_op_->param["start_i"] = (float)b[0];
                net_op_->param["end_i"] = (float)b[1];
            } else if (h.size() == 2) {
                net_op_->param["dim"] = (float)HEAD;
                net_op_->param["start_i"] = (float)h[0];
                net_op_->param["end_i"] = (float)h[1];
            } else if (s.size() == 2) {
                net_op_->param["dim"] = (float)SEQUENCE;
                net_op_->param["start_i"] = (float)s[0];
                net_op_->param["end_i"] = (float)s[1];
            } else if (d.size() == 2) {
                net_op_->param["dim"] = (float)DIMENSION;
                net_op_->param["start_i"] = (float)d[0];
                net_op_->param["end_i"] = (float)d[1];
            } else {
            }
        }else if((b.size()+h.size()+s.size()+d.size()==1) &&(b.size()*b.size()+h.size()*h.size()+s.size()*s.size()+d.size()*d.size()==1)){
            if (b.size() == 1) {
                net_op_->param["dim"] = (float)BATCH;
                net_op_->param["start_i"] = (float)b[0];
                net_op_->param["end_i"] = (float)(b[0]+1);
            } else if (h.size() == 1) {
                net_op_->param["dim"] = (float)HEAD;
                net_op_->param["start_i"] = (float)h[0];
                net_op_->param["end_i"] = (float)(h[0]+1);
            } else if (s.size() == 1) {
                net_op_->param["dim"] = (float)SEQUENCE;
                net_op_->param["start_i"] = (float)s[0];
                net_op_->param["end_i"] = (float)(s[0]+1);
            } else if (d.size() == 1) {
                net_op_->param["dim"] = (float)DIMENSION;
                net_op_->param["start_i"] = (float)d[0];
                net_op_->param["end_i"] = (float)(d[0]+1);
            } else {
            }
        }else{
            std::cout<<"ERROR: "<<name<<" clip"<<std::endl;
        }
        net_op_->in.push_back(this);
        this->out.push_back(net_op_);
        if (std::find(sub_param->net_tensors.begin(), sub_param->net_tensors.end(), this) == sub_param->net_tensors.end()) {
            sub_param->net_tensors.push_back(this);
            if (this->subgraph != nullptr) {
                sub_param->net_inputs.insert(this);
            }
        }

        out_tensor->in = net_op_;
        out_tensor->ctx = ctx;
        return out_tensor;
    }
} NetTensor;

} // namespace mllm

#endif // MLLM_NETPARAMETER_H