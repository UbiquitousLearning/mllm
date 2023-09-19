
#include "CPUScale.hpp"

namespace mllm{
    
    // template class CPUScale;
    // template class CPUScale;
    
    CPUScale::CPUScale(Backend *bn, bool multiThread) : Op(bn)
    {
    }
    
    ErrorCode CPUScale::Reshape(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs)
    {
        std::cout<<"CPUScale  Reshape"<<std::endl;
        return NO_ERROR;
    }


    
    ErrorCode CPUScale::Setup(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs)
    {
        std::cout<<"CPUScale  Setup"<<std::endl;
        return NO_ERROR;
    }

    
    ErrorCode CPUScale::Execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs)
    {
        std::cout<<"CPUScale()"<<std::endl;
        return NO_ERROR;
    }

    ErrorCode CPUScale::Load(ParamLoader& loader)
    {
        std::cout<<"CPUScale load"<<std::endl;
        return NO_ERROR;
    }
} // namespace mllm


