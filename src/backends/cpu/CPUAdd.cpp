
#include "CPUAdd.hpp"

namespace mllm{
    
    template class CPUAdd<float>;
    template class CPUAdd<int8_t>;
    template <typename Dtype>
    CPUAdd<Dtype>::CPUAdd(Backend *bn, bool multiThread) : Op<Dtype>(bn)
    {
    }

    template <typename Dtype>
    inline ErrorCode CPUAdd<Dtype>::Setup(vector<shared_ptr<Tensor<Dtype>>> &inputs, vector<shared_ptr<Tensor<Dtype>>> &outputs)
    {
        std::cout<<"CPUAdd  Setup"<<std::endl;
        return NO_ERROR;
    }

    template <typename Dtype>
    inline ErrorCode CPUAdd<Dtype>::Execute(vector<shared_ptr<Tensor<Dtype>>> &inputs, vector<shared_ptr<Tensor<Dtype>>> &outputs)
    {
        std::cout<<"CPUAdd()"<<std::endl;
        return NO_ERROR;
    }
} // namespace mllm


