
#include "CPURoPE.hpp"

namespace mllm{
    
    // template class CPURoPE;
    // template class CPURoPE;
    
    CPURoPE::CPURoPE(Backend *bn, bool multiThread) : Op(bn)
    {
    }
    
    ErrorCode CPURoPE::Reshape(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs)
    {
        std::cout<<"CPURoPE  Reshape"<<std::endl;
        return NO_ERROR;
    }

    
    ErrorCode CPURoPE::Setup(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs)
    {
        std::cout<<"CPURoPE  Setup"<<std::endl;
        return NO_ERROR;
    }

    
    ErrorCode CPURoPE::Execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs)
    {
        std::cout<<"CPURoPE()"<<std::endl;
        return NO_ERROR;
    }


    ErrorCode CPURoPE::Load(ParamLoader& loader)
    {
        std::cout<<"CPURoPE load"<<std::endl;
        return NO_ERROR;
    }
} // namespace mllm


