
#include "CPUCausalMask.hpp"

namespace mllm{
    
    // template class CPUCausalMask;
    // template class CPUCausalMask;
    
    CPUCausalMask::CPUCausalMask(Backend *bn, bool multiThread) : Op(bn)
    {
    }

    
    inline ErrorCode CPUCausalMask::Setup(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs)
    {
        std::cout<<"CPUCausalMask  Setup"<<std::endl;
        return NO_ERROR;
    }

    
    inline ErrorCode CPUCausalMask::Execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs)
    {
        std::cout<<"CPUCausalMask()"<<std::endl;
        return NO_ERROR;
    }
} // namespace mllm


