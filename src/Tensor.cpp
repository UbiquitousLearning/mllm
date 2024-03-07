#include "Tensor.hpp"

namespace mllm {

Tensor::Tensor(const int batch, const int head, const int sequence, const int dimension) :
    host_ptr_(), capacity_(0) {
    reshape(batch, head, sequence, dimension);
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
    if(aggregated_){return;}
    assert(backend_ != nullptr);
    if(masterTensor() != nullptr) {
        return;
    }
    if(!shape_offset_.empty() & !shape_master_.empty()) {
        return;
    }
    if (allocated_ != count_) {
        if (host_ptr_ != nullptr) {
            backend_->free(host_ptr_);
            host_ptr_ = nullptr;
        }
        if(count_ >0) {
            backend_->alloc(&host_ptr_, cntSize(), 8);
        }
        allocated_ = count_;
    }
}

void Tensor::alloc(vector<uint> alloc_size) {
    if(aggregated_){return;}
    assert(backend_ != nullptr);
    if(masterTensor() != nullptr) {
        return;
    }
    if(!shape_offset_.empty() & !shape_master_.empty()) {
        return;
    }
    // alloc size is different from shape size
    size_t qnn_alloc_size = alloc_size[0] * alloc_size[1] * alloc_size[2] * alloc_size[3];
    
    if (allocated_ != qnn_alloc_size) {
        if (host_ptr_ != nullptr) {
            backend_->free(host_ptr_);
            host_ptr_ = nullptr;
        }

        
        if(qnn_alloc_size > 0) {
            backend_->alloc(&host_ptr_, DataTypeSize(dtype_, qnn_alloc_size), 8);
        }
        allocated_ = qnn_alloc_size;
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