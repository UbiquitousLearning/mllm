
#include "CPUMatmul.hpp"

namespace mllm{
    
    template class CPUMatmul<float>;
    template <typename Dtype>
    inline CPUMatmul<Dtype>::CPUMatmul(const BackendType betype, bool transposeA, bool transposeB, bool transposeC, bool multiThread)
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


