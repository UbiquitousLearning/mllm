#include "Tensor.hpp"

#include <express/ExpressBase.hpp>

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
    shape[0] = batch;
    switch (ctype_) {
    case BSHD:
        shape[1] = sequence;
        shape[2] = head;
        shape[3] = dimension;
        break;
    case BHDS:
        shape[1] = head;
        shape[2] = dimension;
        shape[3] = sequence;
        break;
    case SBHD:
        shape[0] = sequence;
        shape[1] = batch;
        shape[2] = head;
        shape[3] = dimension;
    default:
        break;
    }
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
        vector<int> shape(5);
        shape[0] = batch;
        shape[1] = channel;
        shape[2] = time;
        shape[3] = height;
        shape[4] = width;
        return reshape(shape);
    } else {
        vector<int> shape(5);
        shape[0] = batch;
        shape[1] = time;
        shape[2] = height;
        shape[3] = width;
        shape[4] = channel;
        return reshape(shape);
    }
}

map<string, Tensor> Tensor::gph_;

template <typename Func>
void Tensor::binaryTensorCompute(Tensor &input, Tensor &output, Func operation, float data, int thread_count) {
    if (input.masterTensor() == nullptr && output.masterTensor() == nullptr && input.ctype() == output.ctype()) {
#pragma omp parallel for num_threads(thread_count)
        for (int is = 0; is < input.batch() * input.head() * input.sequence() * input.dimension(); ++is) {
            output.hostPtr<float>()[is] = operation(input.hostPtr<float>()[is], data);
        }
    } else {
        for (int n = 0; n < input.batch(); ++n) {
            for (int c = 0; c < input.head(); ++c) {
                for (int h = 0; h < input.sequence(); ++h) {
#pragma omp parallel for num_threads(thread_count)
                    for (int w = 0; w < input.dimension(); ++w) {
                        output.ptrAt<float>(n, c, h, w)[0] = operation(input.ptrAt<float>(n, c, h, w)[0], data);
                    }
                }
            }
        }
    }
}
template <typename Func>
Tensor &Tensor::binaryCompute(Func operation, string append_s, float data) {
    int thread_count = 4;
    const std::string next_name = name_ + append_s;
    switch (status_) {
    case TENSOR_DYNAMIC: {
        if (gph_.find(name_) == gph_.end()) {
            gph_[name_] = *this;
            gph_[name_].status() = status_;
        }
        gph_[next_name] = Tensor(gph_[name_].batch(), gph_[name_].head(), gph_[name_].sequence(), gph_[name_].dimension(), backend_, false);
        gph_[next_name].setName(next_name);
        gph_[next_name].alloc();
        binaryTensorCompute(gph_[name_], gph_[next_name], operation, data, thread_count);
        break;
    }
    case TENSOR_STATIC_INIT: {
        if (gph_.find(name_) == gph_.end()) {
            gph_[name_] = *this;
            gph_[name_].status() = status_;
        }
        if (gph_.find(next_name) == gph_.end()) {
            gph_[next_name] = Tensor(gph_[name_].batch(), gph_[name_].head(), gph_[name_].sequence(), gph_[name_].dimension(), backend_, false);
            gph_[next_name].setName(next_name);
        } else {
            gph_[next_name].reshape(gph_[name_].batch(), gph_[name_].head(), gph_[name_].sequence(), gph_[name_].dimension());
        }
        //     break;
        // }
        // case TENSOR_STATIC_SHAPED: {
        gph_[next_name].alloc();
        /*
        if (gph_[name_].masterTensor() == nullptr) {
            gph_[name_].free();
        }
        gph_[name_].deepCopyFrom(gph_[next_name], false);
        */
        break;
    }
    case TENSOR_STATIC_ALLOCED: {
        binaryTensorCompute(gph_[name_], gph_[next_name], operation, data, thread_count);
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

template <typename Func>
void Tensor::binaryTensorsCompute(Tensor &input0,Tensor &input1, Tensor &output, Func operation, int thread_count){
    if (input0.masterTensor() == nullptr && output.masterTensor() == nullptr && input0.ctype() == output.ctype()) {
#pragma omp parallel for num_threads(thread_count)
        for (int is = 0; is < input0.batch() * input0.head() * input0.sequence() * input0.dimension(); ++is) {
            output.hostPtr<float>()[is] = operation(input0.hostPtr<float>()[is], input1.hostPtr<float>()[is]);
        }
    } else {
        for (int n = 0; n < input0.batch(); ++n) {
            for (int c = 0; c < input0.head(); ++c) {
                for (int h = 0; h < input0.sequence(); ++h) {
#pragma omp parallel for num_threads(thread_count)
                    for (int w = 0; w < input0.dimension(); ++w) {
                        output.ptrAt<float>(n, c, h, w)[0] =
                            operation(input0.ptrAt<float>(n, c, h, w)[0],
                                input1.ptrAt<float>(n, c, h, w)[0]);
                    }
                }
            }
        }
    }
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
        gph_[next_name] = Tensor(gph_[name_].batch(), gph_[name_].head(), gph_[name_].sequence(), gph_[name_].dimension(), backend_, false);
        gph_[next_name].setName(next_name);
        gph_[next_name].alloc();
        binaryTensorsCompute(gph_[name_],gph_[other.name_], gph_[next_name], operation, thread_count);
        break;
    }
    case TENSOR_STATIC_INIT: {
        if (gph_.find(name_) == gph_.end()) {
            gph_[name_] = *this;
            gph_[name_].status() = status_;
        }
        if (gph_.find(next_name) == gph_.end()) {
            gph_[next_name] = Tensor(gph_[name_].batch(), gph_[name_].head(), gph_[name_].sequence(), gph_[name_].dimension(), backend_, false);
            gph_[next_name].setName(next_name);
        } else {
            gph_[next_name].reshape(gph_[name_].batch(), gph_[name_].head(), gph_[name_].sequence(), gph_[name_].dimension());
        }
        //     break;
        // }
        // case TENSOR_STATIC_SHAPED: {
        gph_[next_name].setDtype(gph_[name_].dtype());
        gph_[next_name].alloc();
        /*
        if (gph_[name_].masterTensor() == nullptr) {
            gph_[name_].free();
        }
        gph_[name_].deepCopyFrom(gph_[next_name], false);
        */
        break;
    }
    case TENSOR_STATIC_ALLOCED: {
        binaryTensorsCompute(gph_[name_], gph_[other.name_], gph_[next_name], operation, thread_count);
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
        //reshape
        int dim_b = gph_[name_].batch();
        int dim_h = gph_[name_].head();
        int dim_s = gph_[name_].sequence();
        int dim_d = gph_[name_].dimension();
        if(b == -1 && h != -1 && s == -1 && d != -1) { // head & dimension
            if (h != ANYDIM && d != ANYDIM) {
                assert(gph_[name_].dimension() * gph_[name_].head() == h * d);
                dim_h = h;
                dim_d = d;
            } else if (h != ANYDIM) {
                dim_h = h;
                dim_d = gph_[name_].dimension()* gph_[name_].head()/ h;
            } else if (d != ANYDIM) {
                dim_h =  gph_[name_].dimension()* gph_[name_].head()/ d;
                dim_d = d;
            } else {
                std::cout<<"[TODO]Tensor.View not support!!!!"<<std::endl;
            }
        } else if(b == -1 && h != -1 && s != -1 && d == -1){ // head & sequence
            if (h != ANYDIM && s != ANYDIM) {
                assert(gph_[name_].sequence() * gph_[name_].head() == h * s);
                dim_h = h;
                dim_s = s;
            } else if (h != ANYDIM ) {
                dim_h = h;
                dim_s = gph_[name_].sequence() * gph_[name_].head()/ h;
            } else if (s != ANYDIM) {
                dim_h = gph_[name_].sequence() * gph_[name_].head()/ s;
                dim_s = s;
            } else {
                std::cout<<"[TODO]Tensor.View not support!!!!"<<std::endl;
            }
        } else if (b != -1 && h == -1 && s != -1 && d == -1) { // batch & sequence
            if (b != ANYDIM && s != ANYDIM) {
                assert(gph_[name_].sequence() * gph_[name_].batch() == b * s);
                dim_b = b;
                dim_s = s;
            } else if (b != ANYDIM) {
                dim_b = b;
                dim_s = gph_[name_].sequence() * gph_[name_].batch()/ b;
            } else if (s != ANYDIM) {
                dim_b = gph_[name_].sequence() * gph_[name_].batch()/ s;
                dim_s = s;
            } else {
                std::cout<<"[TODO]Tensor.View not support!!!!"<<std::endl;
            }
        } else {
            std::cout<<"[TODO]Tensor.View not support!!!!"<<std::endl;
        }

        if (gph_.find(next_name) == gph_.end()) {
            gph_[next_name] = Tensor( backend_);
            gph_[next_name].reshape(dim_b, dim_h, dim_s, dim_d);
            gph_[next_name].setName(next_name);
        } else {
            gph_[next_name].reshape(dim_b, dim_h, dim_s, dim_d);
        }
        //alloc
        if (   (b == -1 && s == -1 && gph_[name_].ctype()!=BCTHW)  // head & dimension
            || (b == -1 && d == -1 && gph_[name_].ctype()==BSHD) // head & sequence
            || (h == -1 && d == -1 && gph_[name_].ctype()==BSHD) // batch & sequence
        ){
            if(gph_[name_].masterTensor() == nullptr) {
                gph_[name_].free();
            }
            gph_[next_name].setDtype(gph_[name_].dtype());
            gph_[next_name].alloc();
            gph_[name_].deepCopyFrom(gph_[next_name], false);
        }else {
            std::cout<<"[TODO]Tensor.View not support!!!!"<<std::endl;
        }
        break;
    }
    case TENSOR_STATIC_ALLOCED: {
        break;
    }
    default: {
    }
    }
    gph_[next_name].status() = status_;
    return gph_[next_name];
}

} // namespace mllm