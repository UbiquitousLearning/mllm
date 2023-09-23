
#include "CPUCausalMask.hpp"

namespace mllm {

// template class CPUCausalMask;
// template class CPUCausalMask;

CPUCausalMask::CPUCausalMask(Backend *bn, bool multiThread) :
    Op(bn) {
}

ErrorCode CPUCausalMask::reshape(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPUCausalMask  reshape" << std::endl;
    return NO_ERROR;
}

ErrorCode CPUCausalMask::setUp(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPUCausalMask  setUp" << std::endl;
    return NO_ERROR;
}

ErrorCode CPUCausalMask::execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPUCausalMask()" << std::endl;
    return NO_ERROR;
}

ErrorCode CPUCausalMask::load(ParamLoader &loader) {
    std::cout << "CPUCausalMask load" << std::endl;
    return NO_ERROR;
}
} // namespace mllm
