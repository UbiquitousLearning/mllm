
#include "CPUAdd.hpp"

namespace mllm {

// template class CPUAdd;
// template class CPUAdd;

CPUAdd::CPUAdd(Backend *bn, bool multiThread) :
    Op(bn) {
}

ErrorCode CPUAdd::reshape(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPUAdd  reshape" << std::endl;
    return NO_ERROR;
}

ErrorCode CPUAdd::setUp(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPUAdd  setUp" << std::endl;
    return NO_ERROR;
}

ErrorCode CPUAdd::execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPUAdd()" << std::endl;
    return NO_ERROR;
}

ErrorCode CPUAdd::load(ParamLoader &loader) {
    std::cout << "CPUAdd load" << std::endl;
    return NO_ERROR;
}
} // namespace mllm
