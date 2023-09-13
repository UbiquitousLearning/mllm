
#include "CPUSoftMax.hpp"

namespace mllm{
    
    // template class CPUSoftMax;
    // template class CPUSoftMax;
    
    CPUSoftMax::CPUSoftMax(Backend *bn, bool multiThread) : Op(bn)
    {
    }

    
    inline ErrorCode CPUSoftMax::Setup(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs)
    {
        std::cout<<"CPUSoftMax  Setup"<<std::endl;
        return NO_ERROR;
    }

    
    inline ErrorCode CPUSoftMax::Execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs)
    {
        std::cout<<"CPUSoftMax()"<<std::endl;
        return NO_ERROR;
    }
} // namespace mllm


