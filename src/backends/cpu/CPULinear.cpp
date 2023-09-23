
#include "CPULinear.hpp"

namespace mllm {

CPULinear::CPULinear(Backend *bn, int in_features, int out_features, bool bias, bool multiThread) :
    Op(bn) {
}

ErrorCode CPULinear::reshape(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPULinear  reshape" << std::endl;
    return NO_ERROR;
}

ErrorCode CPULinear::setUp(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPULinear  setUp" << std::endl;
    return NO_ERROR;
}

ErrorCode CPULinear::execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPULinear()" << std::endl;
    return NO_ERROR;
}

ErrorCode CPULinear::load(ParamLoader &loader) {
    std::cout << "CPULinear load" << std::endl;
    return NO_ERROR;
}

} // namespace mllm
