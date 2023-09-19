
#include "CPUAdd.hpp"

namespace mllm{
    
    // template class CPUAdd;
    // template class CPUAdd;
    
    CPUAdd::CPUAdd(Backend *bn, bool multiThread) : Op(bn)
    {
    }

    ErrorCode CPUAdd::Reshape(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs)
    {
        std::cout<<"CPUAdd  Reshape"<<std::endl;
        return NO_ERROR;
    }

    
    ErrorCode CPUAdd::Setup(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs)
    {
        std::cout<<"CPUAdd  Setup"<<std::endl;
        return NO_ERROR;
    }

    
    ErrorCode CPUAdd::Execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs)
    {
        std::cout<<"CPUAdd()"<<std::endl;
        return NO_ERROR;
    }


    ErrorCode CPUAdd::Load(ParamLoader& loader)
    {
        std::cout<<"CPUAdd load"<<std::endl;
        return NO_ERROR;
    }
} // namespace mllm


