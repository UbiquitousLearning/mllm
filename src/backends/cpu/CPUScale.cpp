
#include "CPUScale.hpp"

namespace mllm {

// template class CPUScale;
// template class CPUScale;

CPUScale::CPUScale(Backend *bn, bool multiThread) :
    Op(bn) {
}

ErrorCode CPUScale::reshape(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPUScale  reshape" << std::endl;
    return NO_ERROR;
}

ErrorCode CPUScale::setUp(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPUScale  setUp" << std::endl;
    return NO_ERROR;
}

ErrorCode CPUScale::execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPUScale()" << std::endl;
    return NO_ERROR;
}

ErrorCode CPUScale::load(ParamLoader &loader) {
    std::cout << "CPUScale load" << std::endl;
    return NO_ERROR;
}
} // namespace mllm
