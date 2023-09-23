
#include "CPURoPE.hpp"

namespace mllm {

// template class CPURoPE;
// template class CPURoPE;

CPURoPE::CPURoPE(Backend *bn, bool multiThread) :
    Op(bn) {
}

ErrorCode CPURoPE::reshape(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPURoPE  reshape" << std::endl;
    return NO_ERROR;
}

ErrorCode CPURoPE::setUp(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPURoPE  setUp" << std::endl;
    return NO_ERROR;
}

ErrorCode CPURoPE::execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPURoPE()" << std::endl;
    return NO_ERROR;
}

ErrorCode CPURoPE::load(ParamLoader &loader) {
    std::cout << "CPURoPE load" << std::endl;
    return NO_ERROR;
}
} // namespace mllm
