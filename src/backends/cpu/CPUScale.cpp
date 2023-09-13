
#include "CPUScale.hpp"

namespace mllm{
    
    template class CPUScale<float>;
    template class CPUScale<int8_t>;
    template <typename Dtype>
    CPUScale<Dtype>::CPUScale(Backend *bn, bool multiThread) : Op<Dtype>(bn)
    {
    }

    template <typename Dtype>
    inline ErrorCode CPUScale<Dtype>::Setup(vector<shared_ptr<Tensor<Dtype>>> &inputs, vector<shared_ptr<Tensor<Dtype>>> &outputs)
    {
        std::cout<<"CPUScale  Setup"<<std::endl;
        return NO_ERROR;
    }

    template <typename Dtype>
    inline ErrorCode CPUScale<Dtype>::Execute(vector<shared_ptr<Tensor<Dtype>>> &inputs, vector<shared_ptr<Tensor<Dtype>>> &outputs)
    {
        std::cout<<"CPUScale()"<<std::endl;
        return NO_ERROR;
    }
} // namespace mllm


