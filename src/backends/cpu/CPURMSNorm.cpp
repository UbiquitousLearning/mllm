
#include "CPURMSNorm.hpp"

namespace mllm{
    
    // template class CPURMSNorm;
    // template class CPURMSNorm;
    
    CPURMSNorm::CPURMSNorm(Backend *bn, bool multiThread) : Op(bn)
    {
    }
    
    ErrorCode CPURMSNorm::Reshape(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs)
    {
        std::cout<<"CPURMSNorm  Reshape"<<std::endl;
        return NO_ERROR;
    }

    
    ErrorCode CPURMSNorm::Setup(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs)
    {
        std::cout<<"CPURMSNorm  Setup"<<std::endl;
        return NO_ERROR;
    }

    
    ErrorCode CPURMSNorm::Execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs)
    {
        std::cout<<"CPURMSNorm()"<<std::endl;
        return NO_ERROR;
    }


    ErrorCode CPURMSNorm::Load(ParamLoader& loader)
    {
        std::cout<<"CPURMSNorm load"<<std::endl;
        return NO_ERROR;
    }
} // namespace mllm


