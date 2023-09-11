#include "Tensor.hpp"

namespace mllm{

    template class Tensor<float>;
    template class Tensor<int8_t>;

    template<typename Dtype>
    Tensor<Dtype>::Tensor(const int num, const int channels, const int height, const int width)
    :capacity_(0){
        Reshape(num, channels, height, width);
        //TODO
    }

    template<typename Dtype>
    Tensor<Dtype>::Tensor(const vector<int> &shape)
    :capacity_(0){
        Reshape(shape);
    }

    template<typename Dtype>
    bool Tensor<Dtype>::Reshape(const int num, const int channels, const int height,const int width) {
        vector<int> shape(4);
        shape[0] = num;
        shape[1] = channels;
        shape[2] = height;
        shape[3] = width;
        return Reshape(shape);
    }
    
    template<typename Dtype>
    bool Tensor<Dtype>::Reshape(const vector<int> &shape) {
        CHECK_LE(shape.size(), kMaxAxes); //维数不能超过kMaxBlobAxes
        count_ = 1; //num*channels*height*width 赋值为1，为了相乘
        shape_.resize(shape.size());
        if (!shape_data_ || shape_data_->size() < shape.size() * sizeof(int)) {
            shape_data_.reset(new HostMemory(shape.size() * sizeof(int)));
        }
        for(int i=0; i<shape.size(); ++i){
            CHECK_GE(shape[i], 0);
            if (count_ != 0) {
                CHECK_LE(shape[i], INT_MAX / count_);
            }
            count_*=shape[i]; //记录数据大小
            shape_[i] = shape[i];
        }
        if(count_ > capacity_){ //capactity不小于count
            capacity_ = count_;
            data_.reset(new  HostMemory(capacity_ * sizeof(Dtype)));
            diff_.reset(new  HostMemory(capacity_ * sizeof(Dtype)));
            return true;
        } else{
            return false;
        }
    }

    
    template<typename Dtype>
    const Dtype *Tensor<Dtype>::cpu_data() const {
        CHECK(data_);
        return (const Dtype*)data_->cpu_data();
    }

    template<typename Dtype>
    const Dtype *Tensor<Dtype>::cpu_diff() const {
        CHECK(diff_);
        return (const Dtype*)diff_->cpu_data();
    }

    template<typename Dtype>
    void Tensor<Dtype>::set_cpu_data(Dtype *data) { //外部指针
        CHECK(data);
        size_t size = count_ * sizeof(Dtype);
        if(size != data_->size()){
            data_.reset(new HostMemory(size));
            diff_.reset(new HostMemory(size));
        }
        data_->set_cpu_data(data);
    }

    template<typename Dtype>
    void Tensor<Dtype>::set_cpu_diff(Dtype *diff) { //外部指针
        CHECK(diff);
        diff_->set_cpu_data(diff);
    }
// template<typename Dtype>
// Dtype Tensor<Dtype>::asum_data() const
// {
// return Dtype();
// }


    template <typename Dtype>
    void Tensor<Dtype>::CopyFrom(const Tensor<Dtype> &source, bool copy_diff, bool reshape)
    {
    }
}