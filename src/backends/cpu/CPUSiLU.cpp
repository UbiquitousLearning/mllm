
#include "CPUSiLU.hpp"

namespace mllm {

// template class CPUSiLU;
// template class CPUSiLU;

CPUSiLU::CPUSiLU(Backend *bn, bool multiThread) :
    Op(bn) {
}

ErrorCode CPUSiLU::reshape(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPUSiLU  reshape" << std::endl;
    return NO_ERROR;
}

ErrorCode CPUSiLU::setUp(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPUSiLU  setUp" << std::endl;
    return NO_ERROR;
}

ErrorCode CPUSiLU::execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPUSiLU()" << std::endl;
    return NO_ERROR;
}

ErrorCode CPUSiLU::load(ParamLoader &loader) {
    std::cout << "CPUSiLU load" << std::endl;
    return NO_ERROR;
}
} // namespace mllm
