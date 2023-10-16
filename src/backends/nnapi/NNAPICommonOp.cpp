#include "NNAPICommonOp.hpp"
#include "NNAPINeuralNetworks.h"
#include <sys/types.h>

namespace mllm {

// template class NNAPICommonOp;
// template class NNAPICommonOp;

NNAPICommonOp::NNAPICommonOp(Backend *bn) :
    Op(bn) {
    nnapiBackend_ = dynamic_cast<NNAPIBackend *>(bn);
}

ErrorCode NNAPICommonOp::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    std::cout << "NNAPICommonOp reshape" << std::endl;
    return NO_ERROR;
}

ErrorCode NNAPICommonOp::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    std::cout << "NNAPICommonOp()" << std::endl;
    // do nothing, should be implemented by NNAPI
    return NO_ERROR;
}

ErrorCode NNAPICommonOp::load(ParamLoader &loader) {
    std::cout << "NNAPICommonOp load" << std::endl;
    return NO_ERROR;
}

std::vector<uint32_t> NNAPICommonOp::getTensorIdxs(const vector<shared_ptr<Tensor>> &tensors) {
    std::vector<uint32_t> idxs(tensors.size());
    for (int i = 0; i < tensors.size(); i++) {
        idxs[i] = nnapiBackend_->getTensorIdx(tensors[i].get(), true);
    }
    return idxs;
}

ErrorCode NNAPICommonOp::buildOperation(int op, const std::vector<uint32_t> &inputs, const std::vector<uint32_t> &outputs) {
    auto name = this->name();
    return nnapiBackend_->buildOperation(op, inputs, outputs, name);
}

int NNAPICommonOp::formatAxis(int axis, const Tensor *t) {
    // NCHW -> NHWC
    const int axisChange[4] = {0, 3, 1, 2};
    return axisChange[axis];
}
} // namespace mllm
