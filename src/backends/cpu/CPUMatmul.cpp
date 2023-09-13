
#include "CPUMatmul.hpp"

namespace mllm{
    
    template class CPUMatmul<float>;
    template class CPUMatmul<int8_t>;
    template <typename Dtype>
    CPUMatmul<Dtype>::CPUMatmul(Backend *bn, bool transposeA, bool transposeB, bool transposeC, bool multiThread) : Op<Dtype>(bn)
    {
    }

    template <typename Dtype>
    inline ErrorCode CPUMatmul<Dtype>::Setup(vector<shared_ptr<Tensor<Dtype>>> &inputs, vector<shared_ptr<Tensor<Dtype>>> &outputs)
    {
        std::cout<<"CPUMatmul  Setup"<<std::endl;
        return NO_ERROR;
    }

    template <typename Dtype>
    inline ErrorCode CPUMatmul<Dtype>::Execute(vector<shared_ptr<Tensor<Dtype>>> &inputs, vector<shared_ptr<Tensor<Dtype>>> &outputs)
    {
        std::cout<<"CPUMatmul()"<<std::endl;
        return NO_ERROR;
    }
} // namespace mllm


