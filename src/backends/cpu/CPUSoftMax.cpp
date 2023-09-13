
#include "CPUSoftMax.hpp"

namespace mllm{
    
    template class CPUSoftMax<float>;
    template class CPUSoftMax<int8_t>;
    template <typename Dtype>
    CPUSoftMax<Dtype>::CPUSoftMax(Backend *bn, bool multiThread) : Op<Dtype>(bn)
    {
    }

    template <typename Dtype>
    inline ErrorCode CPUSoftMax<Dtype>::Setup(vector<shared_ptr<Tensor<Dtype>>> &inputs, vector<shared_ptr<Tensor<Dtype>>> &outputs)
    {
        std::cout<<"CPUSoftMax  Setup"<<std::endl;
        return NO_ERROR;
    }

    template <typename Dtype>
    inline ErrorCode CPUSoftMax<Dtype>::Execute(vector<shared_ptr<Tensor<Dtype>>> &inputs, vector<shared_ptr<Tensor<Dtype>>> &outputs)
    {
        std::cout<<"CPUSoftMax()"<<std::endl;
        return NO_ERROR;
    }
} // namespace mllm


