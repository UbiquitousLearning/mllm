#include "Tensor.hpp"

#include <express/ExpressBase.hpp>
#include "backends/cpu/CPUTensorFunction.hpp"

#include <Module.hpp>

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
    shape[chls_[BATCH]] = batch;
    shape[chls_[HEAD]] = head;
    shape[chls_[SEQUENCE]] = sequence;
    shape[chls_[DIMENSION]] = dimension;

    // shape[0] = batch;
    // switch (ctype_) {
    // case BSHD:
    //     shape[1] = sequence;
    //     shape[2] = head;
    //     shape[3] = dimension;
    //     break;
    // case BHDS:
    //     shape[1] = head;
    //     shape[2] = dimension;
    //     shape[3] = sequence;
    //     break;
    // case SBHD:
    //     shape[0] = sequence;
    //     shape[1] = batch;
    //     shape[2] = head;
    //     shape[3] = dimension;
    // default:
    //     break;
    // }

    // vector<int> shape1(4);
    // shape1[chls_[BATCH]] = batch;
    // shape1[chls_[HEAD]] = head;
    // shape1[chls_[SEQUENCE]] = sequence;
    // shape1[chls_[DIMENSION]] = dimension;
    // bool isSame = std::equal(shape.begin(), shape.end(), shape1.begin());
    // if(!isSame) {
    //     std::cout<<"";
    // }
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
    shape[chls_[BATCH]] = batch;
    shape[chls_[CHANNLE]] = channel;
    shape[chls_[TIME]] = time;
    shape[chls_[HEIGHT]] = height;
    shape[chls_[WIDTH]] = width;
    return reshape(shape);
    // if (ctype_ != BTHWC) {
    //     ctype_ = BCTHW;
    //     vector<int> shape(5);
    //     shape[0] = batch;
    //     shape[1] = channel;
    //     shape[2] = time;
    //     shape[3] = height;
    //     shape[4] = width;
    //     return reshape(shape);
    // } else {
    //     vector<int> shape(5);
    //     shape[0] = batch;
    //     shape[1] = time;
    //     shape[2] = height;
    //     shape[3] = width;
    //     shape[4] = channel;
    //     return reshape(shape);
    // }
}

map<string, Tensor> Tensor::gph_;

template <typename Func>
Tensor &Tensor::binaryCompute(Func operation, string append_s, float data) {
    const std::string next_name = name_ + append_s;
    switch (status_) {
    case TENSOR_DYNAMIC: {
        if (gph_.find(name_) == gph_.end()) {
            gph_[name_] = *this;
            gph_[name_].status() = status_;
        }
        if (gph_.find(next_name) == gph_.end()) {
            gph_[next_name] = Tensor(backend_);
            gph_[next_name].setName(next_name);
        }
        CPUbinaryFunction::reshape(gph_[name_], gph_[next_name]);
        CPUbinaryFunction::setup(gph_[name_], gph_[next_name]);
        CPUbinaryFunction::execute(gph_[name_], gph_[next_name], operation, data);
        break;
    }
    case TENSOR_STATIC_INIT: {
        if (gph_.find(name_) == gph_.end()) {
            gph_[name_] = *this;
            gph_[name_].status() = status_;
        }
        if (gph_.find(next_name) == gph_.end()) {
            gph_[next_name] = Tensor(backend_);
            gph_[next_name].setName(next_name);
        }
        CPUbinaryFunction::reshape(gph_[name_], gph_[next_name]);
        break;
    }
    case TENSOR_STATIC_SHAPED: {
        CPUbinaryFunction::setup(gph_[name_], gph_[next_name]);
        break;
    }
    case TENSOR_STATIC_ALLOCED: {
        CPUbinaryFunction::execute(gph_[name_], gph_[next_name], operation, data);
        break;
    }
    default: {
    }
    }
    gph_[next_name].status() = status_;
    return gph_[next_name];
}
Tensor &Tensor::operator+(float data) {
    return binaryCompute(std::plus<float>(), "-TDadd",  data);
}
Tensor &Tensor::operator-(float data) {
    return binaryCompute(std::minus<float>(), "-TDsub",  data);
}
Tensor &Tensor::operator*(float data) {
    return binaryCompute(std::multiplies<float>(), "-TDmul",  data);
}
Tensor &Tensor::operator/(float data) {
    return binaryCompute(std::divides<float>(), "-TDdiv",  data);
}
Tensor &Tensor::operator/(double data) {
    return binaryCompute(std::divides<float>(), "-TDdiv",  static_cast<float>(data));
}
template <typename Func>
Tensor &Tensor::binaryTwoCompute(Func operation, string append_s, Tensor& other) {
    int thread_count = 4;
    const std::string next_name = name_ + append_s;
    switch (status_) {
    case TENSOR_DYNAMIC: {
        if (gph_.find(name_) == gph_.end()) {
            gph_[name_] = *this;
            gph_[name_].status() = status_;
        }
        if (gph_.find(next_name) == gph_.end()) {
            gph_[next_name] = Tensor(backend_);
            gph_[next_name].setName(next_name);
        }
        CPUbinaryTwoFunction::reshape(gph_[name_], gph_[other.name_], gph_[next_name]);
        CPUbinaryTwoFunction::setup(gph_[name_], gph_[other.name_], gph_[next_name]);
        CPUbinaryTwoFunction::execute(gph_[name_], gph_[other.name_], gph_[next_name], operation);
        break;
    }
    case TENSOR_STATIC_INIT: {
        if (gph_.find(name_) == gph_.end()) {
            gph_[name_] = *this;
            gph_[name_].status() = status_;
        }
        if (gph_.find(next_name) == gph_.end()) {
            gph_[next_name] = Tensor(backend_);
            gph_[next_name].setName(next_name);
        }
        CPUbinaryTwoFunction::reshape(gph_[name_], gph_[other.name_], gph_[next_name]);
        break;
    }
    case TENSOR_STATIC_SHAPED: {
        CPUbinaryTwoFunction::setup(gph_[name_], gph_[other.name_], gph_[next_name]);
        break;
    }
    case TENSOR_STATIC_ALLOCED: {
        CPUbinaryTwoFunction::execute(gph_[name_], gph_[other.name_], gph_[next_name], operation);
        break;
    }
    default: {
    }
    }
    gph_[next_name].status() = status_;
    return gph_[next_name];
}
Tensor& Tensor::operator+(Tensor& other) {
    return binaryTwoCompute(std::plus<float>(), "-TTadd", other);
}
Tensor& Tensor::operator-(Tensor& other){
    return binaryTwoCompute(std::minus<float>(), "-TTsub", other);
}
Tensor& Tensor::operator*(Tensor& other){
    return binaryTwoCompute(std::multiplies<float>(), "-TTmul", other);
}
Tensor& Tensor::operator/(Tensor& other){
    return binaryTwoCompute(std::divides<float>(), "-TTdiv", other);
}

Tensor& Tensor::mean(Chl axis) {
    const std::string next_name = name_ + "-mean";
    switch (status_) {
    case TENSOR_DYNAMIC: {
        std::cout<<"[TODO] not support dynamic tensor view"<<std::endl;
        break;
    }
    case TENSOR_STATIC_INIT: {
        if (gph_.find(name_) == gph_.end()) {
            gph_[name_] = *this;
            gph_[name_].status() = status_;
        }
        if (gph_.find(next_name) == gph_.end()) {
            gph_[next_name] = Tensor( backend_);
            gph_[next_name].setName(next_name);
        }
        CPUmeanFunction::reshape(gph_[name_], gph_[next_name], axis);
        break;
    }
    case TENSOR_STATIC_SHAPED: {
        CPUmeanFunction::setup(gph_[name_], gph_[next_name], axis);
        break;
    }
    case TENSOR_STATIC_ALLOCED: {
        CPUmeanFunction::execute(gph_[name_], gph_[next_name], axis);
        break;
    }
    default: {
    }
    }
    gph_[next_name].status() = status_;
    return gph_[next_name];
}

Tensor& Tensor::view(int b, int h, int s, int d) {
    const std::string next_name = name_ + "-view";
    switch (status_) {
    case TENSOR_DYNAMIC: {
        std::cout<<"[TODO] not support dynamic tensor view"<<std::endl;
        break;
    }
    case TENSOR_STATIC_INIT: {
        if (gph_.find(name_) == gph_.end()) {
            gph_[name_] = *this;
            gph_[name_].status() = status_;
        }
        if (gph_.find(next_name) == gph_.end()) {
            gph_[next_name] = Tensor( backend_);
            gph_[next_name].setName(next_name);
        }
        CPUviewFunction::reshape(gph_[name_], gph_[next_name], b, h, s, d);
        break;
    }
    case TENSOR_STATIC_SHAPED: {
        CPUviewFunction::setup(gph_[name_], gph_[next_name], b, h, s, d);
        break;
    }
    case TENSOR_STATIC_ALLOCED: {
        CPUviewFunction::execute(gph_[name_], gph_[next_name]);
        break;
    }
    default: {
    }
    }
    gph_[next_name].status() = status_;
    return gph_[next_name];
}

Tensor& Tensor::flatten(Chl axis_start, Chl axis_end) {
    const std::string next_name = name_ + "-flatten";
    switch (status_) {
    case TENSOR_DYNAMIC: {
        std::cout << "[TODO] not support dynamic tensor view" << std::endl;
        break;
    }
    case TENSOR_STATIC_INIT: {
        if (gph_.find(name_) == gph_.end()) {
            gph_[name_] = *this;
            gph_[name_].status() = status_;
        }
        if (gph_.find(next_name) == gph_.end()) {
            gph_[next_name] = Tensor(backend_);
            gph_[next_name].setName(next_name);
        }
        CPUflattenFunction::reshape(gph_[name_], gph_[next_name], axis_start, axis_end);
        break;
    }
    case TENSOR_STATIC_SHAPED: {
        CPUflattenFunction::setup(gph_[name_], gph_[next_name], axis_start, axis_end);
        break;
    }
    case TENSOR_STATIC_ALLOCED: {
        CPUflattenFunction::execute(gph_[name_], gph_[next_name]);
        break;
    }
    default: {
    }
    }
    gph_[next_name].status() = status_;
    return gph_[next_name];
}

Tensor &Tensor::transpose(Chl axis0, Chl axis1) {
    return transpose({{axis0, axis1}});
}
Tensor &Tensor::transpose(vector<std::pair<Chl, Chl>> axiss) {
    const std::string next_name = name_ + "-transpose";
    if (next_name.find(".X.") != std::string::npos && Module::runlistIdx > 0) {}
    else {
        switch (status_) {
        case TENSOR_DYNAMIC: {
            std::cout << "[TODO] not support dynamic tensor view" << std::endl;
            break;
        }
        case TENSOR_STATIC_INIT: {
            if (gph_.find(name_) == gph_.end()) {
                gph_[name_] = *this;
                gph_[name_].status() = status_;
            }
            // reshape
            if (gph_.find(next_name) == gph_.end()) {
                gph_[next_name] = Tensor(backend_);
                gph_[next_name].setName(next_name);
            }
            gph_[next_name].trans_copy_shape(gph_[name_].shape());
            std::map<Chl, int> origin_chls = {{BATCH, 0}, {SEQUENCE, 1}, {HEAD, 2}, {DIMENSION, 3},
                                {CHANNLE, 1}, {TIME, 2}, {HEIGHT, 3}, {WIDTH, 4}};
            if(std::equal(gph_[next_name].chls_.begin(), gph_[next_name].chls_.end(), origin_chls.begin())) {
                gph_[next_name].chls_ = gph_[name_].chls_;
                for (auto axis : axiss) {
                    auto axis0 = axis.first;
                    auto axis1 = axis.second;
                    auto ori_0_idx = gph_[next_name].chls_[axis0];
                    auto ori_1_idx = gph_[next_name].chls_[axis1];
                    gph_[next_name].chls_[axis0] = ori_1_idx;
                    gph_[next_name].chls_[axis1] = ori_0_idx;
                }
                gph_[next_name].changeCtype(gph_[name_].shape().size());
                gph_[next_name].undiffusion_ = true;
            }
            break;
        }
        case TENSOR_STATIC_SHAPED: {
            if(gph_[name_].masterTensor() != nullptr) {
                if (gph_[next_name].master_tensor_ == nullptr) {
                    gph_[next_name].setDtype(gph_[name_].dtype());
                    gph_[next_name].deepCopyFrom(gph_[name_], false);
                }
            }else {
                if(gph_[name_].masterTensor() == nullptr) {
                    gph_[name_].free();
                }
                gph_[next_name].setDtype(gph_[name_].dtype());
                gph_[next_name].alloc();
                gph_[name_].undiffusion_ = true;
                gph_[name_].deepCopyFrom(gph_[next_name], false);
                gph_[next_name].trans_from_ = axiss;
            }
            break;
        }
        case TENSOR_STATIC_ALLOCED: {
            break;
        }
        default: {
        }
        }
    }
    gph_[next_name].status() = status_;
    return gph_[next_name];
}

Tensor &Tensor::clip(vector<int> b, vector<int> h, vector<int> s, vector<int> d) {
    const std::string next_name = name_ + "-clip";
    switch (status_) {
    case TENSOR_DYNAMIC: {
        std::cout << "[TODO] not support dynamic tensor view" << std::endl;
        break;
    }
    case TENSOR_STATIC_INIT: {
        if (gph_.find(name_) == gph_.end()) {
            gph_[name_] = *this;
            gph_[name_].status() = status_;
        }
        if (gph_.find(next_name) == gph_.end()) {
            gph_[next_name] = Tensor(backend_);
            gph_[next_name].setName(next_name);
        }
        CPUclipFunction::reshape(gph_[name_], gph_[next_name], b, h, s, d);
        break;
    }
    case TENSOR_STATIC_SHAPED: {
        CPUclipFunction::setup(gph_[name_], gph_[next_name], b, h, s, d);
        break;
    }
    case TENSOR_STATIC_ALLOCED: {
        CPUclipFunction::execute(gph_[name_], gph_[next_name], b, h, s, d);
        break;
    }
    default: {
    }
    }
    gph_[next_name].status() = status_;
    return gph_[next_name];
}


Tensor &Tensor::clip(Chl keep_axis, vector<int> b, vector<int> h, vector<int> s, vector<int> d) {
    const std::string next_name = name_ + "-clip";
    switch (status_) {
    case TENSOR_DYNAMIC: {
        std::cout << "[TODO] not support dynamic tensor view" << std::endl;
        break;
    }
    case TENSOR_STATIC_INIT: {
        if (gph_.find(name_) == gph_.end()) {
            gph_[name_] = *this;
            gph_[name_].status() = status_;
        }
        if (gph_.find(next_name) == gph_.end()) {
            gph_[next_name] = Tensor(backend_);
            gph_[next_name].setName(next_name);
        }
        CPUclipaxisFunction::reshape(gph_[name_], gph_[next_name], keep_axis, b, h, s, d);
        break;
    }
    case TENSOR_STATIC_SHAPED: {
        CPUclipaxisFunction::setup(gph_[name_], gph_[next_name],  keep_axis, b, h, s, d);
        break;
    }
    case TENSOR_STATIC_ALLOCED: {
        CPUclipaxisFunction::execute(gph_[name_], gph_[next_name],  keep_axis, b, h, s, d);
        break;
    }
    default: {
    }
    }
    gph_[next_name].status() = status_;
    return gph_[next_name];
}

Tensor &Tensor::cat(vector<Tensor> input_tensors, Chl axis) {
    const std::string next_name = input_tensors[0].name() + "-cat";
    int expd_batch_ = input_tensors[0].batch();
    int expd_batch_input_idx = 0;
    for (int ii = 0; ii < input_tensors.size(); ++ii) {
        auto input = input_tensors[ii];
        if (input.batch() > expd_batch_) {
            expd_batch_ = input.batch();
            expd_batch_input_idx = ii;
        }
    }
    vector<Tensor*> inputs = {};
    for (const auto& input_tensor : input_tensors) {
        inputs.push_back(&gph_[input_tensor.name()]);
    }
    switch (input_tensors[0].status()) {
    case TENSOR_DYNAMIC: {
        std::cout << "[TODO] not support dynamic tensor view" << std::endl;
        break;
    }
    case TENSOR_STATIC_INIT: {
        if (gph_.find(next_name) == gph_.end()) {
            gph_[next_name] = Tensor(input_tensors[0].backend());
            gph_[next_name].setName(next_name);
        }
        CPUcatFunction::reshape(inputs, gph_[next_name], axis, expd_batch_, expd_batch_input_idx);
        break;
    }
    case TENSOR_STATIC_SHAPED: {
        CPUcatFunction::setup(inputs, gph_[next_name], axis, expd_batch_, expd_batch_input_idx);
        break;
    }
    case TENSOR_STATIC_ALLOCED: {
        CPUcatFunction::execute(inputs, gph_[next_name], axis, expd_batch_, expd_batch_input_idx);
        break;
    }
    default: {
    }
    }
    gph_[next_name].status() = input_tensors[0].status();
    return gph_[next_name];
}

Tensor &Tensor::mm(Tensor& input0, Tensor& input1) {
    const std::string next_name = input0.name() + "-mm-" + input1.name();
    switch (input0.status()) {
    case TENSOR_DYNAMIC: {
        std::cout << "[TODO] not support dynamic tensor view" << std::endl;
        break;
    }
    case TENSOR_STATIC_INIT: {
        if (gph_.find(next_name) == gph_.end()) {
            gph_[next_name] = Tensor(input0.backend());
            gph_[next_name].setName(next_name);
        }
        if (input0.name().find(".X.") != std::string::npos && input1.name().find(".X.") != std::string::npos && next_name.find(".X.") != std::string::npos
            && Module::runlistIdx > 0) {
        } else {
            CPUmmFunction::reshape(gph_[input0.name()], gph_[input1.name()], gph_[next_name]);
        }
        break;
    }
    case TENSOR_STATIC_SHAPED: {
        CPUmmFunction::setup(gph_[input0.name()], gph_[input1.name()], gph_[next_name]);
        break;
    }
    case TENSOR_STATIC_ALLOCED: {
        CPUmmFunction::execute(gph_[input0.name()], gph_[input1.name()], gph_[next_name]);
        break;
    }
    default: {
    }
    }
    gph_[next_name].status() = input0.status();
    return gph_[next_name];
}

Tensor& Tensor::norm(int L_n) {
    // int thread_count = 4;
    assert(L_n ==1 || L_n ==2);
    const std::string next_name = name_ + "-norm";
    switch (status_) {
    case TENSOR_DYNAMIC: {
        std::cout << "[TODO] not support dynamic tensor view" << std::endl;
        break;
    }
    case TENSOR_STATIC_INIT: {
        if (gph_.find(name_) == gph_.end()) {
            gph_[name_] = *this;
            gph_[name_].status() = status_;
        }
        if (gph_.find(next_name) == gph_.end()) {
            gph_[next_name] = Tensor(backend_);
            gph_[next_name].setName(next_name);
        }
        CPUnormFunction::reshape(gph_[name_], gph_[next_name], L_n);
        break;
    }
    case TENSOR_STATIC_SHAPED: {
        CPUnormFunction::setup(gph_[name_], gph_[next_name], L_n);
        break;
    }
    case TENSOR_STATIC_ALLOCED: {
        CPUnormFunction::execute(gph_[name_], gph_[next_name], L_n);
        break;
    }
    default: {
    }
    }
    gph_[next_name].status() = status_;
    return gph_[next_name];
}
} // namespace mllm