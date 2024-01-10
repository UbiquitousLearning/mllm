#include "Tensor.hpp"

namespace mllm {

Tensor::Tensor(const int num, const int channels, const int height, const int width) :
    host_ptr_(), capacity_(0) {
    reshape(num, channels, height, width);
    // TODO
}

Tensor::Tensor(const vector<int> &shape) :
    host_ptr_(), capacity_(0) {
    reshape(shape);
}

bool Tensor::reshape(const int batch, const int head, const int sequence, const int dimension) {
    vector<int> shape(4);
    shape[0] = batch;
    switch (ctype_) {
    // case BHSD:
    //     shape[1] = head;
    //     shape[2] = sequence;
    //     shape[3] = dimension;
    //     break;
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
    if(aggregated_){return;}
    assert(backend_ != nullptr);
    if(masterTensor() != nullptr) {
        return;
    }
    if(!shape_offset_.empty() & !shape_master_.empty()) {
        return;
    }
    if (allocated_ != count_) {
        // 如果原有内存已经分配，则释放它
        if (host_ptr_ != nullptr) {
            //::free(host_ptr_);
            backend_->free(host_ptr_);
            host_ptr_ = nullptr;
        }
        if(count_ >0) {
            // host_ptr_ = malloc(cntSize());
            backend_->alloc(&host_ptr_, cntSize(), 8);
        }
        allocated_ = count_;
    }
}


bool Tensor::reshape(const int batch, const int channel, const int time, const int height, const int width) {
    if(ctype_ != BTHWC) {
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
} // namespace mllm