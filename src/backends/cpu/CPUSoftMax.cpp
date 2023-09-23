
#include "CPUSoftMax.hpp"

namespace mllm {

// template class CPUSoftMax;
// template class CPUSoftMax;

CPUSoftMax::CPUSoftMax(Backend *bn, bool multiThread) :
    Op(bn) {
}

ErrorCode CPUSoftMax::reshape(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPUSoftMax  reshape" << std::endl;
    return NO_ERROR;
}

ErrorCode CPUSoftMax::setUp(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPUSoftMax  setUp" << std::endl;
    return NO_ERROR;
}

ErrorCode CPUSoftMax::execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPUSoftMax()" << std::endl;
    return NO_ERROR;
}

ErrorCode CPUSoftMax::load(ParamLoader &loader) {
    std::cout << "CPUSoftMax load" << std::endl;
    return NO_ERROR;
}
} // namespace mllm
