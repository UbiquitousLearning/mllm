
#include "CPUCausalMask.hpp"

namespace mllm {

// template class CPUCausalMask;
// template class CPUCausalMask;

CPUCausalMask::CPUCausalMask(Backend *bn, bool multiThread) :
    Op(bn) {
}

ErrorCode CPUCausalMask::Reshape(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPUCausalMask  Reshape" << std::endl;
    return NO_ERROR;
}

ErrorCode CPUCausalMask::Setup(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPUCausalMask  Setup" << std::endl;
    return NO_ERROR;
}

ErrorCode CPUCausalMask::Execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPUCausalMask()" << std::endl;
    return NO_ERROR;
}

ErrorCode CPUCausalMask::Load(ParamLoader &loader) {
    std::cout << "CPUCausalMask load" << std::endl;
    return NO_ERROR;
}
} // namespace mllm
