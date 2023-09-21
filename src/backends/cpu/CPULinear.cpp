
#include "CPULinear.hpp"

namespace mllm {

CPULinear::CPULinear(Backend *bn, int in_features, int out_features, bool bias, bool multiThread) :
    Op(bn) {
}

ErrorCode CPULinear::Reshape(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPULinear  Reshape" << std::endl;
    return NO_ERROR;
}

ErrorCode CPULinear::Setup(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPULinear  Setup" << std::endl;
    return NO_ERROR;
}

ErrorCode CPULinear::Execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPULinear()" << std::endl;
    return NO_ERROR;
}

ErrorCode CPULinear::Load(ParamLoader &loader) {
    std::cout << "CPULinear load" << std::endl;
    return NO_ERROR;
}

} // namespace mllm
