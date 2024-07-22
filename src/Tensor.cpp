#include "Tensor.hpp"

#include <cassert>
#include <cstdlib>
#include <exception>
#include <express/ExpressBase.hpp>
#include "Backend.hpp"
#include "OpDefined.hpp"
#include "Types.hpp"
#include "backends/cpu/CPUTensorFunction.hpp"

#include <Module.hpp>
#include <string>
#include <vector>

namespace mllm {

Tensor::Tensor(const int batch, const int head, const int sequence, const int dimension) :
    host_ptr_(), capacity_(0) {
    reshape(batch, head, sequence, dimension);
}
Tensor::Tensor(int batch, int head, int sequence, int dimension, Backend *bn, bool do_alloc) {
    dtype_ = MLLM_TYPE_F32;
    setBackend(bn);
    reshape(batch, head, sequence, dimension);
    if (do_alloc) {
        alloc();
    }
}

Tensor::Tensor(const vector<int> &shape) :
    host_ptr_(), capacity_(0) {
    reshape(shape);
}

bool Tensor::reshape(const int batch, const int head, const int sequence, const int dimension) {
    vector<int> shape(4);
    shape[chls()[BATCH]] = batch;
    shape[chls()[HEAD]] = head;
    shape[chls()[SEQUENCE]] = sequence;
    shape[chls()[DIMENSION]] = dimension;
    return reshape(shape);
}

void Tensor::alloc() {
    if (aggregated_) { return; }
    assert(backend_ != nullptr);
    if (masterTensor() != nullptr) {
        return;
    }
    if (!shape_offset_.empty() & !shape_master_.empty()) {
        return;
    }
    if (allocated_ != count_) {
        if (host_ptr_ != nullptr) {
            backend_->free(host_ptr_);
            host_ptr_ = nullptr;
        }
        if (count_ > 0) {
            backend_->alloc(&host_ptr_, cntSize(), 8);
        }
        allocated_ = count_;
    }
}

bool Tensor::reshape(const int batch, const int channel, const int time, const int height, const int width) {
    if (ctype_ != BTHWC) {
        ctype_ = BCTHW;
    }
    vector<int> shape(5);
    shape[chls()[BATCH]] = batch;
    shape[chls()[CHANNLE]] = channel;
    shape[chls()[TIME]] = time;
    shape[chls()[HEIGHT]] = height;
    shape[chls()[WIDTH]] = width;
    return reshape(shape);
}

map<string, Tensor> Tensor::gph_;


Tensor& Tensor::to(BackendType backend_type){
    if (Module::doLoad) { return *this; }
    if (Module::doToDevice) { return *this; }
    if (backend_type == device()) {
        return *this;
    }
    const std::string next_name = name_ + "-" + "to" + std::to_string(backend_type);
    switch (status_) {
    case TENSOR_STATIC_INIT: {
        Module::initBackend(backend_type);
        auto backend_t = Module::backends[backend_type];  
        if (gph_.find(name_) == gph_.end()) {
            gph_[name_] = *this;
            gph_[name_].status() = status_;
        }
        if (gph_.find(next_name) == gph_.end()) {
            gph_[next_name] = Tensor(backend_t);
            gph_[next_name].setName(next_name);
        }
        reshape_alloc_cross_bn(gph_[name_], gph_[next_name]);
        vector<shared_ptr<Tensor>> shared_inputs{std::shared_ptr<Tensor>(&Tensor::gph_[name_], [](Tensor *) {})};
        vector<shared_ptr<Tensor>> shared_outputs{std::shared_ptr<Tensor>(&Tensor::gph_[next_name], [](Tensor *) {})};
        vector<shared_ptr<Tensor>> empty_v{};
        Module::backends[gph_[name_].device()]->onSetUpEnd(empty_v, shared_inputs);
        Module::backends[backend_type]->onSetUpStart( shared_outputs, empty_v);
        break;
    }
    case TENSOR_STATIC_READY: {
        copy_data_cross_bn(gph_[name_], gph_[next_name]); 
        vector<shared_ptr<Tensor>> shared_inputs{std::shared_ptr<Tensor>(&Tensor::gph_[name_], [](Tensor *) {})};
        vector<shared_ptr<Tensor>> shared_outputs{std::shared_ptr<Tensor>(&Tensor::gph_[next_name], [](Tensor *) {})};
        vector<shared_ptr<Tensor>> empty_v{};
        Module::backends[gph_[name_].device()]->onExecuteEnd();
        Module::backends[backend_type]->onExecuteStart(shared_outputs, empty_v);
        break;
    };
    default: {
    }
    }
    gph_[next_name].status() = status_;
    return gph_[next_name];
}

vector<Tensor> Tensor::toDevice(vector<Tensor> inputs, BackendType backend_type){
    if (Module::doLoad) { return inputs; }
    if (Module::doToDevice) { return inputs; }
    if (backend_type == inputs[0].device()) {
        return inputs;
    }
    Module::initBackend(backend_type);
    vector<string> next_names;
    for (auto &input : inputs) {
        next_names.push_back(input.name() + "-" + "to" + std::to_string(backend_type));
    }
    vector<shared_ptr<Tensor>> shared_inputs{};
    vector<shared_ptr<Tensor>> shared_outputs{};
    vector<shared_ptr<Tensor>> empty_v{};
    switch (inputs[0].status_) {
    case TENSOR_STATIC_INIT: {
        Module::initBackend(backend_type);
        auto backend_t = Module::backends[backend_type];
        for (auto input : inputs) {
            if (gph_.find(input.name()) == gph_.end()) {
                gph_[input.name()] = input;
                gph_[input.name()].status() = input.status_;
            }
        }
        for (auto next_name : next_names) {
            if (gph_.find(next_name) == gph_.end()) {
                gph_[next_name] = Tensor(backend_t);
                gph_[next_name].setName(next_name);
            }
        }
        for (auto &input : inputs) {
            auto next_name = input.name() + "-" + "to" + std::to_string(backend_type);
            reshape_alloc_cross_bn(gph_[input.name()], gph_[next_name]);
            shared_inputs.push_back(std::shared_ptr<Tensor>(&Tensor::gph_[input.name()], [](Tensor *) {}));
            shared_outputs.push_back(std::shared_ptr<Tensor>(&Tensor::gph_[next_name], [](Tensor *) {}));
        }
        Module::backends[inputs[0].device()]->onSetUpEnd(empty_v, shared_inputs);
        Module::backends[backend_type]->onSetUpStart( shared_outputs, empty_v);
        break;
    }
    case TENSOR_STATIC_READY: {
        for (auto &input : inputs) {
            auto next_name = input.name() + "-" + "to" + std::to_string(backend_type);
            copy_data_cross_bn(gph_[input.name()], gph_[next_name]);
            shared_inputs.push_back(std::shared_ptr<Tensor>(&Tensor::gph_[input.name()], [](Tensor *) {}));
            shared_outputs.push_back(std::shared_ptr<Tensor>(&Tensor::gph_[next_name], [](Tensor *) {}));   
        }
        Module::backends[inputs[0].device()]->onExecuteEnd();
        Module::backends[backend_type]->onExecuteStart(shared_outputs, empty_v);
        break;
    };
    default: {
    }
    }
    vector<Tensor> outputs;
    for (auto &input : inputs) {
        auto next_name = input.name() + "-" + "to" + std::to_string(backend_type);
        gph_[next_name].status() = input.status();
        outputs.push_back(gph_[next_name]);
    }
    return outputs;
}

void Tensor::reshape_alloc_cross_bn(Tensor &src_t, Tensor &dst_t){
    auto src_b_type = src_t.backend_->type();
    auto dst_b_type = dst_t.backend_->type();
    //TODO: QNN
#ifdef USE_QNN
    if (dst_b_type == MLLM_QNN) {
        dst_t.reshape(src_t.batch(), src_t.head(), src_t.sequence(), src_t.dimension());
    
    } 
#endif
    if (dst_b_type == MLLM_CPU) {
        dst_t.reshape(src_t.batch(), src_t.head(), src_t.sequence(), src_t.dimension());
    
    }
}

void Tensor::copy_data_cross_bn(Tensor &src_t, Tensor &dst_t){
    auto src_b_type = src_t.backend_->type();
    auto dst_b_type = dst_t.backend_->type();
    //TODO: QNN
#ifdef USE_QNN
    if (src_b_type == MLLM_CPU && dst_b_type == MLLM_QNN) {
    
    } else if (src_b_type == MLLM_QNN && dst_b_type == MLLM_CPU) {
        dst_t.alloc();
    }
#endif
}
#ifdef USE_QNN
Tensor& Tensor::getOp(const std::string& suffix, const OpType type, OpParam param, vector<Tensor *> other_tensors){
    assert(checkgetOps(backend_));
    if (Module::doLoad) { return *this; }
    if (Module::doToDevice) { return *this; }
    const std::string next_name = name_ + "-" + suffix;
    if (Module::tensor_func_ops.find(next_name) == Module::tensor_func_ops.end()) {
        auto op = backend_->opCreate(param, next_name);
        Module::tensor_func_ops[next_name] = std::shared_ptr<Op>(op , [](Op *) {});
    
    }
    auto op_ = Module::tensor_func_ops[next_name];
    switch (status_) {
    case TENSOR_STATIC_INIT: {
        if (gph_.find(name_) == gph_.end()) {
            gph_[name_] = *this;
            gph_[name_].status() = status_;
        }
        if (gph_.find(next_name) == gph_.end()) {
            gph_[next_name] = Tensor(backend_);
            gph_[next_name].setName(next_name);
        }
        vector<shared_ptr<Tensor>> shared_inputs{std::shared_ptr<Tensor>(&Tensor::gph_[name_], [](Tensor *) {})};
        for (auto &other_tensor : other_tensors) {
            shared_inputs.push_back(std::shared_ptr<Tensor>(other_tensor, [](Tensor *) {}));
        }
        vector<shared_ptr<Tensor>> shared_outputs{std::shared_ptr<Tensor>(&Tensor::gph_[next_name], [](Tensor *) {})};
        op_->reshape(shared_inputs, shared_outputs);
        op_->setUp(shared_inputs, shared_outputs);
        break;
    }
    case TENSOR_STATIC_READY: {
        vector<shared_ptr<Tensor>> shared_inputs{std::shared_ptr<Tensor>(&Tensor::gph_[name_], [](Tensor *) {})};
        for (auto &other_tensor : other_tensors) {
            shared_inputs.push_back(std::shared_ptr<Tensor>(other_tensor, [](Tensor *) {}));
        }
        vector<shared_ptr<Tensor>> shared_outputs{std::shared_ptr<Tensor>(&Tensor::gph_[next_name], [](Tensor *) {})};
        op_->execute(shared_inputs, shared_outputs);
        break;
    }
    default: {
    }
    }
    gph_[next_name].status() = status_;
    return gph_[next_name];
}
bool Tensor::checkgetOps(Backend *bn){
    if(bn == nullptr){
        return true;
    }
    return bn->type() == MLLM_QNN;
}
#endif

Tensor& Tensor::getFunc(const std::string& suffix, const TensorFuncType type, vector<float> float_args, vector<Tensor *> other_tensors){
    if (Module::doLoad) { return *this; }
    if (Module::doToDevice) { return *this; }
    TensorFunction *func = backend_->funcCreate(type);
    const std::string next_name = name_ + "-" + suffix;
    switch (status_) {
    case TENSOR_STATIC_INIT: {
        if (gph_.find(name_) == gph_.end()) {
            gph_[name_] = *this;
            gph_[name_].status() = status_;
        }
        if (gph_.find(next_name) == gph_.end()) {
            gph_[next_name] = Tensor(backend_);
            gph_[next_name].setName(next_name);
        }
        std::vector<Tensor*> tensorPtrs = {&gph_[name_]};
        for (auto &other_tensor : other_tensors) {
            tensorPtrs.push_back(other_tensor);
        }
        func->setup(gph_[next_name], tensorPtrs, float_args);
        break;
    }
    case TENSOR_STATIC_READY: {
        std::vector<Tensor*> tensorPtrs = {&gph_[name_]};
        for (auto &other_tensor : other_tensors) {
            tensorPtrs.push_back(other_tensor);
        }
        func->execute(gph_[next_name], tensorPtrs, float_args);
        if(saveNDataFlag){
            Tensor::gph_[next_name].saveData<float>();
        }
        break;
    }
    default: {
    }
    }
    gph_[next_name].status() = status_;
    return gph_[next_name];
}

Tensor &Tensor::operator+(float data) {
#ifdef USE_QNN
    if(checkgetOps(backend_)){
        std::cerr<<"Tensor's Op add not support"<<std::endl;
        exit(1);
    }
#endif
    return getFunc( "add", FUNC_ADD, {data});
}

Tensor &Tensor::operator-(float data) {
#ifdef USE_QNN
    if(checkgetOps(backend_)){
        std::cerr<<"Tensor's Op sub not support"<<std::endl;
        exit(1);
    }
#endif
    return getFunc( "sub", FUNC_SUB, {data});
}

Tensor &Tensor::operator*(float data) {
#ifdef USE_QNN
    if(checkgetOps(backend_)){
        OpParam param;
        param["type"] = OpType::MUL;
        param["scale"] = data;
        param["bias"] = 0.0F;
        param["bias_after_scale"] = (float)false;
        return getOp( "mul", OpType::SCALE, param);
    }
#endif
    return getFunc( "mul", FUNC_MUL, {data});
}

Tensor &Tensor::operator/(float data) {
#ifdef USE_QNN
    if(checkgetOps(backend_)){
        OpParam param;
        param["type"] = OpType::MUL;
        param["scale"] = 1/data;
        param["bias"] = 0.0F;
        param["bias_after_scale"] = (float)false;
        return getOp( "mul", OpType::SCALE, param);
    }
#endif
    return getFunc( "div", FUNC_DIV, {data});
}

Tensor &Tensor::operator/(double data) {
    return operator/((float)data);// return getFunc( "div", FUNC_DIV, {static_cast<float>(data)});
}

Tensor &Tensor::operator+(Tensor &other) {
#ifdef USE_QNN
    if(checkgetOps(backend_)){
        OpParam param;
        param["type"] = OpType::ADD;
        return getOp( "TTadd", OpType::ADD, param, {&other});
    }
#endif
    return getFunc( "TTadd", FUNC_TTADD, {}, {&other});
}

Tensor &Tensor::operator-(Tensor &other) {
#ifdef USE_QNN
    if(checkgetOps(backend_)){
        std::cerr<<"Tensor's Op TTsub not support"<<std::endl;
        exit(1);
    }
#endif
    return getFunc( "TTsub", FUNC_TTSUB, {}, {&other});
}

Tensor &Tensor::operator*(Tensor &other) {
#ifdef USE_QNN
    if(checkgetOps(backend_)){
        OpParam param;
        param["type"] = OpType::MUL;
        return getOp( "TTmul", OpType::MUL, param, {&other});
    }
#endif
    return getFunc( "TTmul", FUNC_TTMUL, {}, {&other});
}

Tensor &Tensor::operator/(Tensor &other) {
#ifdef USE_QNN
    if(checkgetOps(backend_)){
        OpParam param;
        param["type"] = OpType::DIVISION;
        return getOp( "TTdiv", OpType::DIVISION, param, {&other});
    }
#endif
    return getFunc( "TTdiv", FUNC_TTDIV, {}, {&other});
}

Tensor &Tensor::mean(Chl axis) {
#ifdef USE_QNN
    if(checkgetOps(backend_)){
        OpParam param;
        param["type"] = OpType::MEAN;
        param["axis"] = axis;
        return getOp( "mean", OpType::DIVISION, param);
    }
#endif
    return getFunc( "mean", FUNC_MEAN, {(float)axis});
}

Tensor &Tensor::view(int b, int h, int s, int d) {
#ifdef USE_QNN 
    if(checkgetOps(backend_)){
        OpParam param;
        param["type"] = OpType::VIEW;
        vector<int> dims;
        vector<int> data_dims;
        if (b == -1 & s == -1 & h != -1 & d != -1) { // keep b&s change h&d
            if (h != 1) {
                dims = {b, h, s, -1};
                data_dims = {BATCH, DIMENSION, SEQUENCE, DIMENSION};
            } else {
                dims = {b, -1, s, -1};
                data_dims = {BATCH, -1, SEQUENCE, HEAD + DIMENSION};
            }
        } else if (b == -1 & d == -1 & h != -1 & s != -1) { // keep b&d change h&s
            if (h != 1) {
                dims = {b, h, -1, d};
                data_dims = {BATCH, SEQUENCE, SEQUENCE, DIMENSION};
            } else {
                dims = {b, -1, -1, d};
                data_dims = {BATCH, -1, HEAD + SEQUENCE, DIMENSION};
            }
        } else if (h == -1 & d == -1 & b != -1 & s != -1) { // keep h&d change b&s
            if (s != 1) {
                dims = {-1, h, s, d};
                data_dims = {BATCH, HEAD, BATCH, DIMENSION};
            } else {
                dims = {-1, h, -1, d};
                data_dims = {BATCH + SEQUENCE, HEAD, -1, DIMENSION};
            }
        } else {
            std::cout << "ERROR: view [" << b << ", " << h << ", " << s << ", " << d << "]" << std::endl;
        }
        param["dim0"] = dims[0];
        param["dim1"] = dims[1];
        param["dim2"] = dims[2];
        param["dim3"] = dims[3];
        param["data_dim0"] = data_dims[0];
        param["data_dim1"] = data_dims[1];
        param["data_dim2"] = data_dims[2];
        param["data_dim3"] = data_dims[3];
        return getOp( "view", OpType::DIVISION, param);
    }
#endif
    return getFunc("view", FUNC_VIEW, {(float)b, (float)h, (float)s, (float)d});
}

Tensor &Tensor::flatten(Chl axis_start, Chl axis_end) {
#ifdef USE_QNN 
    if(checkgetOps(backend_)){
        OpParam param;
        param["type"] = OpType::VIEW;
        vector<float> dims = {-1, -1, -1, -1};
        vector<float> data_dims;
        if (axis_start == BATCH & axis_end == SEQUENCE) {
            data_dims = {-1, HEAD, BATCH + SEQUENCE, DIMENSION};
        } else if (axis_start == HEAD & axis_end == SEQUENCE) {
            data_dims = {BATCH, -1, HEAD + SEQUENCE, DIMENSION};
        } else if (axis_start == HEAD & axis_end == DIMENSION) {
            data_dims = {BATCH, HEAD, -1, SEQUENCE + DIMENSION};
        } else if (axis_start == TIME & axis_end == WIDTH) {
            data_dims = {BATCH, -1, TIME + HEIGHT + WIDTH, CHANNLE};
        } else {
            std::cout << "ERROR:  flatten  " << axis_start << "&" << axis_end << std::endl;
        }
        param["dim0"] = dims[0];
        param["dim1"] = dims[1];
        param["dim2"] = dims[2];
        param["dim3"] = dims[3];
        param["data_dim0"] = data_dims[0];
        param["data_dim1"] = data_dims[1];
        param["data_dim2"] = data_dims[2];
        param["data_dim3"] = data_dims[3];
        return getOp( "flatten", OpType::VIEW, param);
    }
#endif
    return getFunc( "flatten", FUNC_FLATTEN, {(float)axis_start, (float)axis_end});
}

Tensor &Tensor::transpose(vector<std::pair<Chl, Chl>> axiss) {
#ifdef USE_QNN 
    if(checkgetOps(backend_)){
        OpParam param;
        OpType type;
        auto axis1 = axiss[0].first;
        auto axis2 = axiss[0].second;
        if (axis1 == SEQUENCE & axis2 == DIMENSION) {
            type = TRANSPOSE;
            param["type"] = TRANSPOSE;
            param["axis0"] = axis1;
            param["axis1"] = axis2;
        } else if (axis1 == THW & axis2 == CHANNLE) {
            type = TRANSPOSE;
            param["type"] = TRANSPOSE;
            param["axis0"] = axis1;
            param["axis1"] = axis2;
        } else if (axis1 == BATCH & axis2 == SEQUENCE) {
            type = TRANSPOSE;
            param["type"] = TRANSPOSE;
            param["axis0"] = axis1;
            param["axis1"] = axis2;
        } else {
            type = VIEW;
            param["type"] = VIEW;
            vector<float> dims;
            vector<float> data_dims;
            if (axis1 == BATCH & axis2 == SEQUENCE) {
                dims = {-1, -1, -1, -1};
                data_dims = {SEQUENCE, HEAD, BATCH, DIMENSION};
            } else if (axis1 == HEAD & axis2 == SEQUENCE) {
                dims = {-1, -1, -1, -1};
                data_dims = {BATCH, SEQUENCE, HEAD, DIMENSION};
            } else if (axis1 == HEAD & axis2 == DIMENSION) {
                dims = {-1, -1, -1, -1};
                data_dims = {BATCH, SEQUENCE, DIMENSION, HEAD};
            } else {
                std::cout << "ERROR:  transpose  " << axis1 << "&" << axis2 << std::endl;
            }
            param["dim0"] = dims[0];
            param["dim1"] = dims[1];
            param["dim2"] = dims[2];
            param["dim3"] = dims[3];
            param["data_dim0"] = data_dims[0];
            param["data_dim1"] = data_dims[1];
            param["data_dim2"] = data_dims[2];
            param["data_dim3"] = data_dims[3];
        }
        return getOp( "transpose", type, param);
    }
#endif
    vector<float> axis_s;
    for (auto &axis : axiss) {
        axis_s.push_back((float)axis.first);
        axis_s.push_back((float)axis.second);
    }
    return getFunc( "transpose", FUNC_TRANPOSE, axis_s);
}

Tensor &Tensor::clip(vector<int> b, vector<int> h, vector<int> s, vector<int> d) {
#ifdef USE_QNN
    if(checkgetOps(backend_)){
        std::cerr<<"Tensor's Op clip not support"<<std::endl;
        exit(1);
    }
#endif
    vector<float> axis_s;
    axis_s.push_back(b.size());
    axis_s.push_back(h.size());
    axis_s.push_back(s.size());
    axis_s.push_back(d.size());
    for (auto &axis : b) {
        axis_s.push_back((float)axis);
    }
    for (auto &axis : h) {
        axis_s.push_back((float)axis);
    }
    for (auto &axis : s) {
        axis_s.push_back((float)axis);
    }
    for (auto &axis : d) {
        axis_s.push_back((float)axis);
    }
    return getFunc( "clip", FUNC_CLIP, axis_s);
}

Tensor &Tensor::clip(Chl keep_axis, vector<int> b, vector<int> h, vector<int> s, vector<int> d) {
#ifdef USE_QNN
    if(checkgetOps(backend_)){
        std::cerr<<"Tensor's Op clipaxis not support"<<std::endl;
        exit(1);
    }
#endif
    vector<float> axis_s = {(float)keep_axis};
    axis_s.push_back(b.size());
    axis_s.push_back(h.size());
    axis_s.push_back(s.size());
    axis_s.push_back(d.size());
    for (auto &axis : b) {
        axis_s.push_back((float)axis);
    }
    for (auto &axis : h) {
        axis_s.push_back((float)axis);
    }
    for (auto &axis : s) {
        axis_s.push_back((float)axis);
    }
    for (auto &axis : d) {
        axis_s.push_back((float)axis);
    }
    return getFunc( "clipaxis", FUNC_CLIPAXIS, axis_s);
}

Tensor &Tensor::norm(int L_n) {
#ifdef USE_QNN
    if(checkgetOps(backend_)){
        OpParam param;
        param["type"] = OpType::NORM;
        param["L_n"] = (float)L_n;
        return getOp( "norm", OpType::NORM, param);
    }
#endif
    return getFunc("norm", FUNC_NORM, {(float)L_n});
}

Tensor &Tensor::where(float value, Chl axis) {
#ifdef USE_QNN
    if(checkgetOps(backend_)){
        OpParam param;
        param["type"] = OpType::WHERE;
        param["data"] = value;
        param["axis"] = axis;
        return getOp( "where", OpType::WHERE, param);
    }
#endif
    return getFunc("where", FUNC_WHERE, {(float)value, (float)axis});
}




/**
 * static function
 */
 #ifdef USE_QNN
 Tensor& Tensor::getStaticOp(const std::string& suffix, const OpType type,  OpParam param, vector<Tensor *> other_tensors){
    if (Module::doLoad) { return Tensor::gph_["0"]; }
    if (Module::doToDevice) { return Tensor::gph_["0"];}
    const std::string next_name = suffix;
    auto backend_ = Module::backends[MLLM_QNN];
    if (Module::tensor_func_ops.find(next_name) == Module::tensor_func_ops.end()) {
        auto op = backend_->opCreate(param, next_name);
        Module::tensor_func_ops[next_name] = std::shared_ptr<Op>(op , [](Op *) {});
    
    }
    auto op_ = Module::tensor_func_ops[next_name];
    switch (Module::tensor_status) {
    case TENSOR_STATIC_INIT: {
        if (gph_.find(next_name) == gph_.end()) {
            gph_[next_name] = Tensor(backend_);
            gph_[next_name].setName(next_name);
        }
        vector<shared_ptr<Tensor>> shared_inputs{};
        for (auto &other_tensor : other_tensors) {
            shared_inputs.push_back(std::shared_ptr<Tensor>(other_tensor, [](Tensor *) {}));
        }
        vector<shared_ptr<Tensor>> shared_outputs{std::shared_ptr<Tensor>(&Tensor::gph_[next_name], [](Tensor *) {})};
        op_->reshape(shared_inputs, shared_outputs);
        op_->setUp(shared_inputs, shared_outputs);
        break;
    }
    case TENSOR_STATIC_READY: {
        vector<shared_ptr<Tensor>> shared_inputs{};
        for (auto &other_tensor : other_tensors) {
            shared_inputs.push_back(std::shared_ptr<Tensor>(other_tensor, [](Tensor *) {}));
        }
        vector<shared_ptr<Tensor>> shared_outputs{std::shared_ptr<Tensor>(&Tensor::gph_[next_name], [](Tensor *) {})};
        op_->execute(shared_inputs, shared_outputs);
        break;
    }
    default: {
    }
    }
    gph_[next_name].status() = Module::tensor_status;
    return gph_[next_name];
}
#endif

Tensor& Tensor::getStaticFunc(const std::string& suffix, const TensorFuncType type, vector<float> float_args, vector<Tensor *> other_tensors){
    if (Module::doLoad) { return Tensor::gph_["0"]; }
    if (Module::doToDevice) { return Tensor::gph_["0"];}
    auto backend_h = Module::backends[MLLM_CPU];
    if(!other_tensors.empty() && other_tensors[0]->backend_!= nullptr){
        backend_h = other_tensors[0]->backend();
    }
    TensorFunction *func = backend_h->funcCreate(type);
    const std::string next_name = suffix;
    switch (Module::tensor_status) {
    case TENSOR_STATIC_INIT: {
        if (gph_.find(next_name) == gph_.end()) {
            gph_[next_name] = Tensor(backend_h);
            gph_[next_name].setName(next_name);
        }
        func->setup(gph_[next_name], other_tensors, float_args);
        break;
    }
    case TENSOR_STATIC_READY: {
        func->execute(gph_[next_name], other_tensors, float_args);
        if(saveNDataFlag){
            Tensor::gph_[next_name].saveData<float>();
        }
        break;
    }
    default: {
    }
    }
    gph_[next_name].status() = Module::tensor_status;
    return gph_[next_name];
}

Tensor &Tensor::cat(vector<Tensor> input_tensors, Chl axis) {
    const std::string next_name = input_tensors[0].name() + "-cat";
#ifdef USE_QNN
    if(checkgetOps(input_tensors[0].backend())){
        OpParam param;
        param["type"] = OpType::CAT;
        param["axis"] =(float)axis;
        return getStaticOp( next_name, OpType::CAT, param);
    }
#endif
    vector<Tensor *> inputs = {};
    for (const auto &input_tensor : input_tensors) {
        inputs.push_back(&gph_[input_tensor.name()]);
    }
    return getStaticFunc(next_name, FUNC_CAT, {(float)axis}, inputs);
}

Tensor &Tensor::mm(Tensor &input0, Tensor &input1) {
    const std::string next_name = input0.name() + "-mm-" + input1.name();
#ifdef USE_QNN
    if(checkgetOps(input0.backend_)){
        OpParam param;
        param["type"] = OpType::MATMUL;
        param["transpose0"] = false;
        param["transpose1"] = false;
        return getStaticOp( next_name, OpType::MATMUL, param);
    }
#endif
    return getStaticFunc(next_name, FUNC_MM, {}, {&gph_[input0.name()], &gph_[input1.name()]});
}

Tensor &Tensor::range(int start, int end) {
    const std::string next_name = "range-" + std::to_string(start) + "-" + std::to_string(end);
    return getStaticFunc(next_name, FUNC_RANGE, {(float)start, (float)end});
}

std::vector<Tensor> Tensor::getStaticFuncOupts(vector<std::string> out_names, const TensorFuncType type, vector<float> float_args, 
                                    vector<Tensor *> input_tensors){
    if (Module::doLoad) { 
        std::vector<Tensor> outPtrs;
        for (auto out_name: out_names) {
            outPtrs.push_back(Tensor::gph_["0"]);
        }
        return outPtrs; 
    }
    auto backend_h = Module::backends[MLLM_CPU];
    if(!input_tensors.empty() && input_tensors[0]->backend_!= nullptr){
        backend_h = input_tensors[0]->backend();
    }
    TensorFunction *func = backend_h->funcCreate(type);
    switch (Module::tensor_status) {
    case TENSOR_STATIC_INIT: {
        std::vector<Tensor*> outPtrs;
        for (auto out_name: out_names) {
            if (gph_.find(out_name) == gph_.end()) {
                gph_[out_name] = Tensor(backend_h);
                gph_[out_name].setName(out_name);
            }
            outPtrs.push_back(&gph_[out_name]);
        }
        
        func->setup(outPtrs, input_tensors, float_args);
        break;
    }
    case TENSOR_STATIC_READY: {
        std::vector<Tensor*> outPtrs;
        for (auto out_name: out_names) {
            outPtrs.push_back(&gph_[out_name]);
        }
        func->execute(outPtrs, input_tensors, float_args);
        if(saveNDataFlag){
            for (auto out_name: out_names) {
                Tensor::gph_[out_name].saveData<float>();
            }
        }
        break;
    }
    default: {
    }
    }
    std::vector<Tensor> results;
    for (auto out_name: out_names) {
        gph_[out_name].status() = Module::tensor_status;
        results.push_back(gph_[out_name]);
    }
    return results;
}

vector<Tensor> Tensor::split(Tensor& input, std::vector<int> each_dims, Chl split_dim, int head_size){
    vector<std::string> next_names;
    std::vector<float> args;
    for (int i = 0; i < each_dims.size(); ++i) {
        args.push_back(each_dims[i]);
        next_names.push_back(input.name() + "-split-" + std::to_string(i) + "-"+ std::to_string(each_dims[i]));
    }
    args.push_back(split_dim);
    args.push_back(head_size);
    std::vector<Tensor*> input_tensors = {&gph_[input.name()]};
    return getStaticFuncOupts(next_names, FUNC_SPLIT, args, input_tensors);
}


} // namespace mllm