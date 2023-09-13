
#include "CPURoPE.hpp"

namespace mllm{
    
    template class CPURoPE<float>;
    template class CPURoPE<int8_t>;
    template <typename Dtype>
    CPURoPE<Dtype>::CPURoPE(Backend *bn, bool multiThread) : Op<Dtype>(bn)
    {
    }

    template <typename Dtype>
    inline ErrorCode CPURoPE<Dtype>::Setup(vector<shared_ptr<Tensor<Dtype>>> &inputs, vector<shared_ptr<Tensor<Dtype>>> &outputs)
    {
        std::cout<<"CPURoPE  Setup"<<std::endl;
        return NO_ERROR;
    }

    template <typename Dtype>
    inline ErrorCode CPURoPE<Dtype>::Execute(vector<shared_ptr<Tensor<Dtype>>> &inputs, vector<shared_ptr<Tensor<Dtype>>> &outputs)
    {
        std::cout<<"CPURoPE()"<<std::endl;
        return NO_ERROR;
    }
} // namespace mllm


