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

//bool Tensor::reshape(const int num, const int channels, const int height, const int width) {
//    vector<int> shape(4);
//    shape[0] = num;
//    shape[1] = channels;
//    shape[2] = height;
//    shape[3] = width;
//    return reshape(shape);
//}
bool Tensor::reshape(const int batch, const int head, const int sequence, const int dimension){
    vector<int> shape(4);
    shape[0] = batch;
    shape[1] = head;
    shape[2] = sequence;
    shape[3] = dimension;
    return reshape(shape);
}

bool Tensor::reshape(const vector<int> &shape) {
    CHECK_LE(shape.size(), KMaxAxes); // 维数不能超过kMaxBlobAxes
    count_ = 1;                       // num*channels*height*width 赋值为1，为了相乘
    shape_.resize(shape.size());
    // if (!shape_data_ || shape_data_->size() < shape.size() * sizeof(int)) {
    //     shape_data_.reset(new HostMemory(shape.size() * sizeof(int)));
    // }
    for (int i = 0; i < shape.size(); ++i) {
        CHECK_GE(shape[i], 0);
        if (count_ != 0) {
            CHECK_LE(shape[i], INT_MAX / count_);
        }
        count_ *= shape[i]; // 记录数据大小
        shape_[i] = shape[i];
    }
    if (count_ > capacity_) { // capactity不小于count
        capacity_ = count_;
        // data_.reset(new  HostMemory(capacity_ * sizeof(Dtype)));
        // diff_.reset(new  HostMemory(capacity_ * sizeof(Dtype)));
        return true;
    }
    return false;
}

void Tensor::alloc() {
    if (host_ptr_ != nullptr && allocated_) {
        // 如果原有内存已经分配，则释放它
        backend_->free(host_ptr_);
    }
    backend_->alloc(&host_ptr_, CntSize());
    allocated_ = true;
}

// const float *Tensor::cpu_data() const {
//     return (const float*)host_ptr_;
// }

//
// const Dtype *Tensor::cpu_diff() const {
//     CHECK(diff_);
//     return (const Dtype*)diff_->cpu_data();
// }

//
// void Tensor::set_cpu_data(Dtype *data) { //外部指针
//     CHECK(data);
//     size_t size = count_ * sizeof(Dtype);
//     if(size != data_->size()){
//         data_.reset(new HostMemory(size));
//         diff_.reset(new HostMemory(size));
//     }
//     data_->set_cpu_data(data);
// }

//
// void Tensor::set_cpu_diff(Dtype *diff) { //外部指针
//     CHECK(diff);
//     diff_->set_cpu_data(diff);
// }

void Tensor::copyFrom(const Tensor &source, bool copy_diff, bool reshape) {
    CHECK_EQ(source.dtype(), dtype());
    CHECK_EQ(source.count(), count());
    // copy
    memcpy(host_ptr_, source.host_ptr_, CntSize());
}
void Tensor::copyFrom(const shared_ptr<Tensor> &source, bool reshape) {
    CHECK_EQ(source->dtype(), dtype());
    CHECK_EQ(source->count(), count());
    // copy
    memcpy(host_ptr_, source->host_ptr_, CntSize());
}
void Tensor::permute(int axis0, int axis1, int axis2, int axis3, bool copy) {
    // 检查轴的合法性
    CHECK_GE(axis0, 0);
    CHECK_LT(axis0, 4);
    CHECK_GE(axis1, 0);
    CHECK_LT(axis1, 4);
    CHECK_GE(axis2, 0);
    CHECK_LT(axis2, 4);
    CHECK_GE(axis3, 0);
    CHECK_LT(axis3, 4);

    // 计算新的形状
    vector<int> new_shape = {shape_[axis0], shape_[axis1], shape_[axis2], shape_[axis3]};
    // 使用临时存储来保存重新排列后的数据
    vector<float> temp_data(count_);
    if (copy) {
        // 对数据进行重新排列
        for (int n = 0; n < num(); ++n) {
            for (int c = 0; c < channels(); ++c) {
                for (int h = 0; h < height(); ++h) {
                    for (int w = 0; w < width(); ++w) {
                        int old_idx = offset(n, c, h, w);
                        int new_idx = ((n * new_shape[1] + c) * new_shape[2] + h) * new_shape[3] + w;
                        temp_data[new_idx] = dataAt<float>(n, c, h, w);
                    }
                }
            }
        }
    }

    // 更新形状和数据
    shape_ = new_shape;
    if (copy) {
        for (int n = 0; n < num(); ++n) {
            for (int c = 0; c < channels(); ++c) {
                for (int h = 0; h < height(); ++h) {
                    for (int w = 0; w < width(); ++w) {
                        int new_idx = ((n * new_shape[1] + c) * new_shape[2] + h) * new_shape[3] + w;
                        setDataAt<float>(n, c, h, w, temp_data[new_idx]);
                    }
                }
            }
        }
    }
}
} // namespace mllm