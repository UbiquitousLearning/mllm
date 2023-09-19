
#include "CPUMatmul.hpp"

namespace mllm{
    
    // template class CPUMatmul;
    // template class CPUMatmul;
    
    CPUMatmul::CPUMatmul(Backend *bn, bool transposeA, bool transposeB, bool transposeC, bool multiThread) : Op(bn)
    {
    }
    
    ErrorCode CPUMatmul::Reshape(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs)
    {
        std::cout<<"CPUMatmul  Reshape"<<std::endl;
        return NO_ERROR;
    }


    
    ErrorCode CPUMatmul::Setup(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs)
    {
        std::cout<<"CPUMatmul  Setup"<<std::endl;
        return NO_ERROR;
    }

    
    ErrorCode CPUMatmul::Execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs)
    {
        std::cout<<"CPUMatmul()"<<std::endl;
        return NO_ERROR;
    }



    ErrorCode CPUMatmul::Load(ParamLoader& loader)
    {
        std::cout<<"CPUMatmul load"<<std::endl;
        return NO_ERROR;
    }

        
} // namespace mllm


