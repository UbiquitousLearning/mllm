//
// Created by ey on 24-1-8.
//

#include "ExpressBase.hpp"
namespace mllm {


NetTensor *TNetTensor::clip(vector<int> b, vector<int> h, vector<int> s, vector<int> d){
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

NetTensor *TNetTensor::_clip(intTensor_pair b, intTensor_pair h, intTensor_pair s, intTensor_pair d){
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
    NetTensor *in_1;
    // PARAM
    if (b.end_i != nullptr) {
        net_op_->param["dim"] = (float)BATCH;
        net_op_->param["start_i"] = (float)b.start_i;
        net_op_->param["end_i"] = (float)ANYDIM;
        in_1 = b.end_i;
    } else if (h.end_i != nullptr) {
        net_op_->param["dim"] = (float)HEAD;
        net_op_->param["start_i"] = (float)h.start_i;
        net_op_->param["end_i"] = (float)ANYDIM;
        in_1 = h.end_i;
    } else if (s.end_i != nullptr) {
        net_op_->param["dim"] = (float)SEQUENCE;
        net_op_->param["start_i"] = (float)s.start_i;
        net_op_->param["end_i"] = (float)ANYDIM;
        in_1 = s.end_i;
    } else if (d.end_i != nullptr) {
        net_op_->param["dim"] = (float)DIMENSION;
        net_op_->param["start_i"] = (float)d.start_i;
        net_op_->param["end_i"] = (float)ANYDIM;
        in_1 = d.end_i;
    } else {
    }
    net_op_->in.push_back(this);
    this->out.push_back(net_op_);
    if (std::find(sub_param->net_tensors.begin(), sub_param->net_tensors.end(), this) == sub_param->net_tensors.end()) {
        sub_param->net_tensors.push_back(this);
        if (this->subgraph != nullptr) {
            sub_param->net_inputs.insert(this);
        }
    }
    net_op_->in.push_back(in_1);
    in_1->out.push_back(net_op_);
    if (std::find(sub_param->net_tensors.begin(), sub_param->net_tensors.end(), in_1) == sub_param->net_tensors.end()) {
        sub_param->net_tensors.push_back(in_1);
        if (in_1->subgraph != nullptr) {
            sub_param->net_inputs.insert(in_1);
        }
    }

    out_tensor->in = net_op_;
    out_tensor->ctx = ctx;
    return out_tensor;
}

NetTensor *TNetTensor::_clip(Tensor_pair b, Tensor_pair h, Tensor_pair s, Tensor_pair d){
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
    NetTensor *in_1;
    // PARAM
    net_op_->param["start_i"] = (float)ANYDIM;
    net_op_->param["end_i"] = (float)ANYDIM;
    if (b.end_i != nullptr) {
        net_op_->param["dim"] = (float)BATCH;
        in_1 = b.end_i;
    } else if (h.end_i != nullptr) {
        net_op_->param["dim"] = (float)HEAD;
        in_1 = h.end_i;
    } else if (s.end_i != nullptr) {
        net_op_->param["dim"] = (float)SEQUENCE;
        in_1 = s.end_i;
    } else if (d.end_i != nullptr) {
        net_op_->param["dim"] = (float)DIMENSION;
        in_1 = d.end_i;
    } else {
    }
    net_op_->in.push_back(this);
    this->out.push_back(net_op_);
    if (std::find(sub_param->net_tensors.begin(), sub_param->net_tensors.end(), this) == sub_param->net_tensors.end()) {
        sub_param->net_tensors.push_back(this);
        if (this->subgraph != nullptr) {
            sub_param->net_inputs.insert(this);
        }
    }
    net_op_->in.push_back(in_1);
    in_1->out.push_back(net_op_);
    if (std::find(sub_param->net_tensors.begin(), sub_param->net_tensors.end(), in_1) == sub_param->net_tensors.end()) {
        sub_param->net_tensors.push_back(in_1);
        if (in_1->subgraph != nullptr) {
            sub_param->net_inputs.insert(in_1);
        }
    }

    out_tensor->in = net_op_;
    out_tensor->ctx = ctx;
    return out_tensor;
}

NetTensor *TNetTensor::shape(Chl axis) {
    Context *ctx =this->ctx;
    NetTensor *out_tensor = new NetTensor();
    if (name.empty()) {
        name = this->name + "_shape_"+std::to_string(axis)+" "+std::to_string(ctx->idx);
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
    net_op_->type = SHAPE;
    net_op_->param = OpParam();
    net_op_->param["type"] = SHAPE;
    ctx->net_ops.push_back(net_op_);
    // PARAM
    net_op_->param["axis"] = axis;

    //
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

NetTensor *TNetTensor::view(int b, int h, int s, int d) {
    Context *ctx =this->ctx;
    NetTensor *out_tensor = new NetTensor();
    if (name.empty()) {
        name = this->name + "_view_"+std::to_string(ctx->idx);
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
    net_op_->type = VIEW;
    net_op_->param = OpParam();
    net_op_->param["type"] = VIEW;
    ctx->net_ops.push_back(net_op_);
    // PARAM
    vector<int> dims;
    vector<int> data_dims;
    if (b == -1 & s == -1 & h != -1 & d != -1) { // keep b&s change h&d
        if(h != 1) {
            dims = {b, h, s, -1};
            data_dims = {BATCH, DIMENSION, SEQUENCE, DIMENSION};
        } else {
            dims = {b, -1, s, -1};
            data_dims = {BATCH, -1, SEQUENCE, HEAD + DIMENSION};
        }
    } else if (b == -1 & d == -1 & h != -1 & s != -1) {// keep b&d change h&s
        if(h != 1) {
            dims = {b, h, -1, d};
            data_dims = {BATCH, SEQUENCE, SEQUENCE, DIMENSION};
        } else {
            dims = {b, -1, -1, d};
            data_dims = {BATCH, -1, HEAD+SEQUENCE, DIMENSION};
        }
    }else {
        std::cout<<"ERROR: "<<name<<" view ["<<b<<", "<<h<<", "<<s<<", "<<d<<"]"<<std::endl;
    }

    net_op_->param["dim0"] = dims[0];
    net_op_->param["dim1"] = dims[1];
    net_op_->param["dim2"] = dims[2];
    net_op_->param["dim3"] = dims[3];
    net_op_->param["data_dim0"] = data_dims[0];
    net_op_->param["data_dim1"] = data_dims[1];
    net_op_->param["data_dim2"] = data_dims[2];
    net_op_->param["data_dim3"] = data_dims[3];

    //
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

NetTensor *TNetTensor::flatten(Chl axis_start, Chl axis_end) {
    Context *ctx =this->ctx;
    NetTensor *out_tensor = new NetTensor();
    if (name.empty()) {
        name = this->name + "_transpose_"+std::to_string(ctx->idx);
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
    net_op_->type = VIEW;
    net_op_->param = OpParam();
    net_op_->param["type"] = VIEW;
    ctx->net_ops.push_back(net_op_);
    // PARAM
    vector<int> dims= {-1, -1, -1, -1};
    vector<int> data_dims;
    if (axis_start == BATCH & axis_end == SEQUENCE) {
        data_dims = {-1, HEAD, BATCH+SEQUENCE, DIMENSION};
    }else if (axis_start == HEAD & axis_end == SEQUENCE) {
        data_dims = {BATCH, -1, HEAD+SEQUENCE, DIMENSION};
    }else if (axis_start == HEAD & axis_end == DIMENSION) {
        data_dims = {BATCH, HEAD, -1, SEQUENCE+DIMENSION};
    }else if (axis_start == TIME & axis_end == CHANNLE) {
        data_dims = {BATCH, -1,  TIME + HEIGHT + WIDTH, CHANNLE};
    }else {
        std::cout<<"ERROR: "<<name<<" flatten  "<<axis_start<<"&"<<axis_end<<std::endl;
    }

    net_op_->param["dim0"] = dims[0];
    net_op_->param["dim1"] = dims[1];
    net_op_->param["dim2"] = dims[2];
    net_op_->param["dim3"] = dims[3];
    net_op_->param["data_dim0"] = data_dims[0];
    net_op_->param["data_dim1"] = data_dims[1];
    net_op_->param["data_dim2"] = data_dims[2];
    net_op_->param["data_dim3"] = data_dims[3];

    //
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

NetTensor *TNetTensor::transpose(Chl axis1, Chl axis2) {
    Context *ctx =this->ctx;
    NetTensor *out_tensor = new NetTensor();
    if (name.empty()) {
        name = this->name + "_transpose_"+std::to_string(ctx->idx);
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
    net_op_->param = OpParam();
    if (axis1 == SEQUENCE & axis2 == DIMENSION){
        net_op_->type = TRANSPOSE;
        net_op_->param["type"] = TRANSPOSE;
        ctx->net_ops.push_back(net_op_);
    } else {
        net_op_->type = VIEW;
        net_op_->param["type"] = VIEW;
        ctx->net_ops.push_back(net_op_);
        // PARAM
        vector<int> dims;
        vector<int> data_dims;
        if (axis1 == BATCH & axis2 == SEQUENCE) {
            dims = {-1, -1, -1, -1};
            data_dims = {SEQUENCE, HEAD, BATCH, DIMENSION};
        }else if (axis1 == HEAD & axis2 == SEQUENCE) {
            dims = {-1, -1, -1, -1};
            data_dims = {BATCH, SEQUENCE, HEAD, DIMENSION};
        }else if (axis1 == HEAD & axis2 == DIMENSION) {
            dims = {-1, -1, -1, -1};
            data_dims = {BATCH, SEQUENCE, DIMENSION, HEAD};
        }else {
            std::cout<<"ERROR: "<<name<<" transpose  "<<axis1<<"&"<<axis2<<std::endl;
        }
        net_op_->param["dim0"] = dims[0];
        net_op_->param["dim1"] = dims[1];
        net_op_->param["dim2"] = dims[2];
        net_op_->param["dim3"] = dims[3];
        net_op_->param["data_dim0"] = data_dims[0];
        net_op_->param["data_dim1"] = data_dims[1];
        net_op_->param["data_dim2"] = data_dims[2];
        net_op_->param["data_dim3"] = data_dims[3];
    }

    //
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

NetTensor *TNetTensor::norm(int L_n) {
    Context *ctx =this->ctx;
    NetTensor *out_tensor = new NetTensor();
    if (name.empty()) {
        name = this->name + "_L"+std::to_string(L_n)+"Norm_"+std::to_string(ctx->idx);
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
    net_op_->type = NORM;
    net_op_->param = OpParam();
    net_op_->param["type"] = NORM;
    ctx->net_ops.push_back(net_op_);
    // PARAM
    net_op_->param["L_n"] =(float)L_n;
    //
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

}