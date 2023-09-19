
#include "CPUSoftMax.hpp"

namespace mllm{
    
    // template class CPUSoftMax;
    // template class CPUSoftMax;
    
    CPUSoftMax::CPUSoftMax(Backend *bn, bool multiThread) : Op(bn)
    {
    }
    
    ErrorCode CPUSoftMax::Reshape(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs)
    {
        std::cout<<"CPUSoftMax  Reshape"<<std::endl;
        return NO_ERROR;
    }

    
    ErrorCode CPUSoftMax::Setup(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs)
    {
        std::cout<<"CPUSoftMax  Setup"<<std::endl;
        return NO_ERROR;
    }

    
    ErrorCode CPUSoftMax::Execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs)
    {
        std::cout<<"CPUSoftMax()"<<std::endl;
        return NO_ERROR;
    }


    ErrorCode CPUSoftMax::Load(ParamLoader& loader)
    {
        std::cout<<"CPUSoftMax load"<<std::endl;
        return NO_ERROR;
    }
} // namespace mllm


