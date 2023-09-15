
#include "CPUMatmul.hpp"

namespace mllm{
    
    // template class CPUMatmul;
    // template class CPUMatmul;
    
    CPUMatmul::CPUMatmul(Backend *bn, bool transposeA, bool transposeB, bool transposeC, bool multiThread) : Op(bn)
    {
    }

    
    inline ErrorCode CPUMatmul::Setup(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs)
    {
        std::cout<<"CPUMatmul  Setup"<<std::endl;
        return NO_ERROR;
    }

    
    inline ErrorCode CPUMatmul::Execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs)
    {
        std::cout<<"CPUMatmul()"<<std::endl;
        return NO_ERROR;
    }

        
} // namespace mllm


