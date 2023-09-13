
#include "CPURMSNorm.hpp"

namespace mllm{
    
    template class CPURMSNorm<float>;
    template class CPURMSNorm<int8_t>;
    template <typename Dtype>
    CPURMSNorm<Dtype>::CPURMSNorm(Backend *bn, bool multiThread) : Op<Dtype>(bn)
    {
    }

    template <typename Dtype>
    inline ErrorCode CPURMSNorm<Dtype>::Setup(vector<shared_ptr<Tensor<Dtype>>> &inputs, vector<shared_ptr<Tensor<Dtype>>> &outputs)
    {
        std::cout<<"CPURMSNorm  Setup"<<std::endl;
        return NO_ERROR;
    }

    template <typename Dtype>
    inline ErrorCode CPURMSNorm<Dtype>::Execute(vector<shared_ptr<Tensor<Dtype>>> &inputs, vector<shared_ptr<Tensor<Dtype>>> &outputs)
    {
        std::cout<<"CPURMSNorm()"<<std::endl;
        return NO_ERROR;
    }
} // namespace mllm


