#include "Tensor.hpp"

#include <express/ExpressBase.hpp>
#include "OpDefined.hpp"
#include "Timing.hpp"
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

map<string, shared_ptr<Tensor>> Tensor::graphs;
TensorStatus Tensor::tensor_status;

Tensor& Tensor::getFunc(const std::string& suffix, const TensorFuncType type, vector<float> float_args, vector<Tensor *> other_tensors){
    const std::string next_name = name_ + "-" + suffix;
    if (Tensor::graphs.find(name_) == Tensor::graphs.end()) {
        Tensor::graphs[name_] = std::shared_ptr<Tensor>(this, [](Tensor *) {});
    }
    if (Tensor::graphs.find(next_name) == Tensor::graphs.end()) {
        Tensor::graphs[next_name] = std::make_shared<Tensor>(backend_);
        Tensor::graphs[next_name]->setName(next_name);
    }
    if (Module::doLoad) { 
        return  *Tensor::graphs[next_name];
    }
    TensorFunction *func = backend_->funcCreate(type);
    std::vector<Tensor*> tensorPtrs = {Tensor::graphs[name_].get()};
    for (auto &other_tensor : other_tensors) {
        tensorPtrs.push_back(other_tensor);
    }
#ifdef DEBUGOPTIME
    auto start_t = mllm_time_us();
#endif
    switch (Tensor::tensor_status) {
    case TENSOR_STATIC_INIT: {
        func->setup({Tensor::graphs[next_name].get()}, tensorPtrs, float_args);
        break;
    }
    case TENSOR_STATIC_READY: {
        func->execute({Tensor::graphs[next_name].get()},tensorPtrs, float_args);
        break;
    }
    default: {
    }
    }
#ifdef DEBUGOPTIME
    auto end_t = mllm_time_us();
    std::cout<<next_name << " | "<<Tensor::tensor_status<<" time: " << (end_t - start_t)/1000.0F <<"ms"<< std::endl;
#endif
#ifdef DEBUGSAVETENSOR
    Tensor::graphs[next_name]->saveNData<float>();
#endif
    return  *Tensor::graphs[next_name];
}

Tensor &Tensor::operator+(float data) {
    return getFunc( "add", FUNC_ADD, {data});
}

Tensor &Tensor::operator-(float data) {
    return getFunc( "sub", FUNC_SUB, {data});
}

Tensor &Tensor::operator*(float data) {
    return getFunc( "mul", FUNC_MUL, {data});
}

Tensor &Tensor::operator/(float data) {
    return getFunc( "div", FUNC_DIV, {data});
}

Tensor &Tensor::operator/(double data) {
    return getFunc( "div", FUNC_DIV, {static_cast<float>(data)});
}

Tensor &Tensor::operator+(Tensor &other) {
    return getFunc( "TTadd", FUNC_TTADD, {}, {&other});
}

Tensor &Tensor::operator-(Tensor &other) {
    return getFunc( "TTsub", FUNC_TTSUB, {}, {&other});
}

Tensor &Tensor::operator*(Tensor &other) {
    return getFunc( "TTmul", FUNC_TTMUL, {}, {&other});
}

Tensor &Tensor::operator/(Tensor &other) {
    return getFunc( "TTdiv", FUNC_TTDIV, {}, {&other});
}

Tensor &Tensor::mean(Chl axis) {
    return getFunc( "mean", FUNC_MEAN, {(float)axis});
}

Tensor &Tensor::view(int b, int h, int s, int d) {
    return getFunc("view", FUNC_VIEW, {(float)b, (float)h, (float)s, (float)d});
}

Tensor &Tensor::flatten(Chl axis_start, Chl axis_end) {
    return getFunc( "flatten", FUNC_FLATTEN, {(float)axis_start, (float)axis_end});
}

Tensor &Tensor::transpose(vector<std::pair<Chl, Chl>> axiss) {
    vector<float> axis_s;
    for (auto &axis : axiss) {
        axis_s.push_back((float)axis.first);
        axis_s.push_back((float)axis.second);
    }
    return getFunc( "transpose", FUNC_TRANPOSE, axis_s);
}

Tensor &Tensor::clip(vector<int> b, vector<int> h, vector<int> s, vector<int> d) {
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
    return getFunc("norm", FUNC_NORM, {(float)L_n});
}

Tensor &Tensor::where(float value, Chl axis) {
    return getFunc("where", FUNC_WHERE, {(float)value, (float)axis});
}




/**
 * static function
 */

Tensor& Tensor::getStaticFunc(const std::string& suffix, const TensorFuncType type, vector<float> float_args, vector<Tensor *> other_tensors){
    auto backend_h = Module::backends[MLLM_CPU];
    if(!other_tensors.empty() && other_tensors[0]->backend_!= nullptr){
        backend_h = other_tensors[0]->backend();
    }
    TensorFunction *func = backend_h->funcCreate(type);
    const std::string next_name = suffix;
    if (Tensor::graphs.find(next_name) == Tensor::graphs.end()) {
        Tensor::graphs[next_name] = std::make_shared<Tensor>(backend_h);
        Tensor::graphs[next_name]->setName(next_name);
    }
    if (Module::doLoad) { 
        return  *Tensor::graphs[next_name];
    }
#ifdef DEBUGOPTIME
    auto start_t = mllm_time_us();
#endif
    switch (Tensor::tensor_status) {
    case TENSOR_STATIC_INIT: {
        func->setup({Tensor::graphs[next_name].get()}, other_tensors, float_args);
        break;
    }
    case TENSOR_STATIC_READY: {
        func->execute({Tensor::graphs[next_name].get()}, other_tensors, float_args);
        break;
    }
    default: {
    }
    }
#ifdef DEBUGOPTIME
    auto end_t = mllm_time_us();
    std::cout<<next_name << " | "<<Tensor::tensor_status<<" time: " << (end_t - start_t)/1000.0F <<"ms"<< std::endl;
#endif
#ifdef DEBUGSAVETENSOR
    Tensor::graphs[next_name]->saveNData<float>();
#endif
    return *Tensor::graphs[next_name];
}

Tensor &Tensor::cat(vector<Tensor> input_tensors, Chl axis) {
    vector<Tensor *> inputs = {};
    for (const auto &input_tensor : input_tensors) {
        inputs.push_back(Tensor::graphs[input_tensor.name()].get());
    }
    const std::string next_name = input_tensors[0].name() + "-cat";
    return getStaticFunc(next_name, FUNC_CAT, {(float)axis}, inputs);
}

Tensor &Tensor::mm(Tensor &input0, Tensor &input1) {
    const std::string next_name = input0.name() + "-mm-" + input1.name();
    return getStaticFunc(next_name, FUNC_MM, {}, {Tensor::graphs[input0.name()].get(), Tensor::graphs[input1.name()].get()});
}

Tensor &Tensor::range(int start, int end) {
    const std::string next_name = "range-" + std::to_string(start) + "-" + std::to_string(end);
    return getStaticFunc(next_name, FUNC_RANGE, {(float)start, (float)end});
}

std::vector<Tensor> Tensor::getStaticFuncOupts(vector<std::string> out_names, const TensorFuncType type, vector<float> float_args, 
                                    vector<Tensor *> input_tensors){
    auto backend_h = Module::backends[MLLM_CPU];
    if(!input_tensors.empty() && input_tensors[0]->backend_!= nullptr){
        backend_h = input_tensors[0]->backend();
    }
    for (auto out_name: out_names) {
        if (Tensor::graphs.find(out_name) == Tensor::graphs.end()) {
            Tensor::graphs[out_name] = std::make_shared<Tensor>(backend_h);
            Tensor::graphs[out_name]->setName(out_name);
        }
    }
    if (Module::doLoad) {
        std::vector<Tensor> results;
        for (auto out_name: out_names) {
            results.push_back(*Tensor::graphs[out_name]);
        }
        return results;
    }
    TensorFunction *func = backend_h->funcCreate(type);
    std::vector<Tensor*> outPtrs;
    for (auto out_name: out_names) {
        outPtrs.push_back(Tensor::graphs[out_name].get());
    }
#ifdef DEBUGOPTIME
    auto start_t = mllm_time_us();
#endif
    switch (Tensor::tensor_status) {
    case TENSOR_STATIC_INIT: {
        func->setup(outPtrs, input_tensors, float_args);
        break;
    }
    case TENSOR_STATIC_READY: {
        func->execute(outPtrs, input_tensors, float_args);
        break;
    }
    default: {
    }
    }
#ifdef DEBUGOPTIME
    auto end_t = mllm_time_us();
    std::cout<<out_names[0] << " | "<<Tensor::tensor_status<<" time: " << (end_t - start_t)/1000.0F <<"ms"<< std::endl;
#endif
#ifdef DEBUGSAVETENSOR
    for (auto out_name: out_names) {
        Tensor::graphs[out_name]->saveNData<float>();
    }
#endif
    std::vector<Tensor> results;
    for (auto out_name: out_names) {
        results.push_back(*Tensor::graphs[out_name]);
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
    std::vector<Tensor*> input_tensors = {Tensor::graphs[input.name()].get()};
    return getStaticFuncOupts(next_names, FUNC_SPLIT, args, input_tensors);
}


} // namespace mllm