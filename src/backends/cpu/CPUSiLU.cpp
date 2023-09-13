
#include "CPUSiLU.hpp"

namespace mllm{
    
    template class CPUSiLU<float>;
    template class CPUSiLU<int8_t>;
    template <typename Dtype>
    CPUSiLU<Dtype>::CPUSiLU(Backend *bn, bool multiThread) : Op<Dtype>(bn)
    {
    }

    template <typename Dtype>
    inline ErrorCode CPUSiLU<Dtype>::Setup(vector<shared_ptr<Tensor<Dtype>>> &inputs, vector<shared_ptr<Tensor<Dtype>>> &outputs)
    {
        std::cout<<"CPUSiLU  Setup"<<std::endl;
        return NO_ERROR;
    }

    template <typename Dtype>
    inline ErrorCode CPUSiLU<Dtype>::Execute(vector<shared_ptr<Tensor<Dtype>>> &inputs, vector<shared_ptr<Tensor<Dtype>>> &outputs)
    {
        std::cout<<"CPUSiLU()"<<std::endl;
        return NO_ERROR;
    }
} // namespace mllm


