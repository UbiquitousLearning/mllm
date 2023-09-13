
#include "CPUSiLU.hpp"

namespace mllm{
    
    // template class CPUSiLU;
    // template class CPUSiLU;
    
    CPUSiLU::CPUSiLU(Backend *bn, bool multiThread) : Op(bn)
    {
    }

    
    inline ErrorCode CPUSiLU::Setup(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs)
    {
        std::cout<<"CPUSiLU  Setup"<<std::endl;
        return NO_ERROR;
    }

    
    inline ErrorCode CPUSiLU::Execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs)
    {
        std::cout<<"CPUSiLU()"<<std::endl;
        return NO_ERROR;
    }
} // namespace mllm


