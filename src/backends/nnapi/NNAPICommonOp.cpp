#include "NNAPICommonOp.hpp"
#include "NNAPINeuralNetworks.h"
#include "Types.hpp"
#include <sys/types.h>

namespace mllm {

NNAPICommonOp::NNAPICommonOp(Backend *bn, string name) :
    Op(bn, name) {
    nnapiBackend_ = dynamic_cast<NNAPIBackend *>(bn);
}

ErrorCode NNAPICommonOp::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
#ifdef DEBUG
    std::cout << "*NNAPI " << name() << " reshape*" << std::endl;
#endif
    return MLLM_NO_ERROR;
}

ErrorCode NNAPICommonOp::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
#ifdef DEBUG
    std::cout << "*NNAPI " << name() << " setUp*" << std::endl;
#endif
    return MLLM_NO_ERROR;
}

ErrorCode NNAPICommonOp::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
#ifdef DEBUG
    std::cout << "*NNAPI " << name() << " execute*" << std::endl;
#endif
    // do nothing, should be implemented by NNAPI
    return MLLM_NO_ERROR;
}

ErrorCode NNAPICommonOp::load(AbstructLoader &loader) {
#ifdef DEBUG
    std::cout << "*NNAPI " << name() << " load*" << std::endl;
#endif
    return MLLM_NO_ERROR;
}

ErrorCode NNAPICommonOp::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
#ifdef DEBUG
    std::cout << "*" << name() << " free*" << std::endl;
#endif
    return MLLM_NO_ERROR;
}

std::vector<uint32_t> NNAPICommonOp::getTensorIdxs(const vector<shared_ptr<Tensor>> &tensors) {
    std::vector<uint32_t> idxs(tensors.size());
    for (int i = 0; i < tensors.size(); i++) {
        idxs[i] = nnapiBackend_->getTensorIdx(tensors[i].get(), false);
    }
    return idxs;
}

uint32_t NNAPICommonOp::getTensorIdx(const Tensor *t, bool isReshape, std::vector<uint32_t> dims) {
    return nnapiBackend_->getTensorIdx(t, false, isReshape, dims);
}

uint32_t NNAPICommonOp::buildConstant(const void *data, size_t size, OperandCode dtype, std::vector<uint32_t> dims, const float *scales, int zero) {
    return nnapiBackend_->buildOperand(data, size, dtype, dims, scales, zero);
}

uint32_t NNAPICommonOp::buildTensor(OperandCode dtype, std::vector<int> dims) {
    std::vector<uint32_t> udims(dims.begin(), dims.end());
    return nnapiBackend_->buildOperand(nullptr, 0, dtype, udims);
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
