
#include "CPUCausalMask.hpp"

namespace mllm{
    
    template class CPUCausalMask<float>;
    template class CPUCausalMask<int8_t>;
    template <typename Dtype>
    CPUCausalMask<Dtype>::CPUCausalMask(Backend *bn, bool multiThread) : Op<Dtype>(bn)
    {
    }

    template <typename Dtype>
    inline ErrorCode CPUCausalMask<Dtype>::Setup(vector<shared_ptr<Tensor<Dtype>>> &inputs, vector<shared_ptr<Tensor<Dtype>>> &outputs)
    {
        std::cout<<"CPUCausalMask  Setup"<<std::endl;
        return NO_ERROR;
    }

    template <typename Dtype>
    inline ErrorCode CPUCausalMask<Dtype>::Execute(vector<shared_ptr<Tensor<Dtype>>> &inputs, vector<shared_ptr<Tensor<Dtype>>> &outputs)
    {
        std::cout<<"CPUCausalMask()"<<std::endl;
        return NO_ERROR;
    }
} // namespace mllm


