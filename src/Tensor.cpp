#include "Tensor.hpp"

#include <express/ExpressBase.hpp>
#include "backends/cpu/CPUTensorFunction.hpp"

#include <Module.hpp>
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

template <typename Func, typename... Args>
Tensor &Tensor::applyFunc(const std::string &suffix, Func func, Args... args) {
    if (Module::doLoad) { return *this; }
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
        func.setup(gph_[name_], gph_[next_name], args...);
        break;
    }
    case TENSOR_STATIC_READY: {
        func.execute(gph_[name_], gph_[next_name], args...);
        break;
    }
    default: {
    }
    }
    gph_[next_name].status() = status_;
    return gph_[next_name];
}

template <typename Func>
Tensor &Tensor::binaryCompute(Func operation, string append_s, float data) {
    return applyFunc(append_s, CPUbinaryFunction(), operation, data);
}

Tensor &Tensor::operator+(float data) {
    return binaryCompute(std::plus<float>(), "-TDadd", data);
}
Tensor &Tensor::operator-(float data) {
    return binaryCompute(std::minus<float>(), "-TDsub", data);
}
Tensor &Tensor::operator*(float data) {
    return binaryCompute(std::multiplies<float>(), "-TDmul", data);
}
Tensor &Tensor::operator/(float data) {
    return binaryCompute(std::divides<float>(), "-TDdiv", data);
}
Tensor &Tensor::operator/(double data) {
    return binaryCompute(std::divides<float>(), "-TDdiv", static_cast<float>(data));
}

template <typename Func>
Tensor &Tensor::binaryTwoCompute(Func operation, string append_s, Tensor &other) {
    return applyFunc(append_s, CPUbinaryTwoFunction(), other, operation);
}

Tensor &Tensor::operator+(Tensor &other) {
    return binaryTwoCompute(std::plus<float>(), "-TTadd", other);
}
Tensor &Tensor::operator-(Tensor &other) {
    return binaryTwoCompute(std::minus<float>(), "-TTsub", other);
}
Tensor &Tensor::operator*(Tensor &other) {
    return binaryTwoCompute(std::multiplies<float>(), "-TTmul", other);
}
Tensor &Tensor::operator/(Tensor &other) {
    return binaryTwoCompute(std::divides<float>(), "-TTdiv", other);
}

Tensor &Tensor::mean(Chl axis) {
    return applyFunc("mean", CPUmeanFunction(), axis);
}

Tensor &Tensor::view(int b, int h, int s, int d) {
    return applyFunc("view", CPUviewFunction(), b, h, s, d);
}

Tensor &Tensor::flatten(Chl axis_start, Chl axis_end) {
    return applyFunc("flatten", CPUflattenFunction(), axis_start, axis_end);
}

Tensor &Tensor::transpose(vector<std::pair<Chl, Chl>> axiss) {
    return applyFunc("transpose", CPUtransposeFunction(), axiss);
}

Tensor &Tensor::clip(vector<int> b, vector<int> h, vector<int> s, vector<int> d) {
    return applyFunc("clip", CPUclipFunction(), b, h, s, d);
}

Tensor &Tensor::clip(Chl keep_axis, vector<int> b, vector<int> h, vector<int> s, vector<int> d) {
    return applyFunc("clip", CPUclipaxisFunction(), keep_axis, b, h, s, d);
}

Tensor &Tensor::norm(int L_n) {
    return applyFunc("norm", CPUnormFunction(), L_n);
}

Tensor &Tensor::where(float value, Chl axis) {
    return applyFunc("where", CPUwhereFunction(), value, axis);
}
/**
 * static function
 */

template <typename Func, typename... Args>
Tensor &Tensor::applyStaticFunc(const std::string &suffix, Func func, Args... args) {
    if (Module::doLoad) { return Tensor::gph_["0"]; }
    const std::string next_name = suffix;
    switch (Module::tensor_status) {
    case TENSOR_DYNAMIC: {
        std::cout << "[TODO] not support dynamic tensor view" << std::endl;
        break;
    }
    case TENSOR_STATIC_INIT: {
        if (gph_.find(next_name) == gph_.end()) {
            gph_[next_name] = Tensor(Module::backends[MLLM_CPU]);
            gph_[next_name].setName(next_name);
        }
        func.setup(gph_[next_name], args...);
        break;
    }
    case TENSOR_STATIC_READY: {
        func.execute(gph_[next_name], args...);
        break;
    }
    default: {
    }
    }
    gph_[next_name].status() = Module::tensor_status;
    return gph_[next_name];
}

Tensor &Tensor::cat(vector<Tensor> input_tensors, Chl axis) {
    vector<Tensor *> inputs = {};
    for (const auto &input_tensor : input_tensors) {
        inputs.push_back(&gph_[input_tensor.name()]);
    }
    const std::string next_name = input_tensors[0].name() + "-cat";
    return applyStaticFunc(next_name, CPUcatFunction(), inputs, axis);
}

Tensor &Tensor::mm(Tensor &input0, Tensor &input1) {
    const std::string next_name = input0.name() + "-mm-" + input1.name();
    return applyStaticFunc(next_name, CPUmmFunction(), gph_[input0.name()], gph_[input1.name()]);
}

Tensor &Tensor::range(int start, int end) {
    const std::string next_name = "range-" + std::to_string(start) + "-" + std::to_string(end);
    return applyStaticFunc(next_name, CPURangeFunction(), start, end);
}

} // namespace mllm