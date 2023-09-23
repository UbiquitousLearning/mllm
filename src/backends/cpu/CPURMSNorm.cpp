
#include "CPURMSNorm.hpp"

namespace mllm {

// template class CPURMSNorm;
// template class CPURMSNorm;

CPURMSNorm::CPURMSNorm(Backend *bn, bool multiThread) :
    Op(bn) {
}

ErrorCode CPURMSNorm::reshape(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPURMSNorm  reshape" << std::endl;
    return NO_ERROR;
}

ErrorCode CPURMSNorm::setUp(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPURMSNorm  setUp" << std::endl;
    return NO_ERROR;
}

ErrorCode CPURMSNorm::execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPURMSNorm()" << std::endl;
    return NO_ERROR;
}

ErrorCode CPURMSNorm::load(ParamLoader &loader) {
    std::cout << "CPURMSNorm load" << std::endl;
    return NO_ERROR;
}
} // namespace mllm
