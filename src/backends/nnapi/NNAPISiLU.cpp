
#include "NNAPISiLU.hpp"
#include "NNAPICommonOp.hpp"
#include "NNAPINeuralNetworks.h"
#include <cmath>

namespace mllm {

// template class NNAPISiLU;
// template class NNAPISiLU;

NNAPISiLU::NNAPISiLU(Backend *bn, string opName) :
    NNAPICommonOp(bn, opName) {
}

ErrorCode NNAPISiLU::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    outputs[0]->reshape(inputs[0]->num(), inputs[0]->channels(), inputs[0]->height(), inputs[0]->width());
    outputs[0]->setDtype(activationDtype());
    std::cout << name() << "  NNAPISiLU  reshape" << std::endl;
    return NO_ERROR;
}

ErrorCode NNAPISiLU::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    if (!inputs[0]->allocted()) {
        inputs[0]->alloc();
    }
    outputs[0]->alloc();
    std::cout << name() << "  NNAPISiLU  setUp" << std::endl;

    auto middleIdx = buildTensor(ANEURALNETWORKS_TENSOR_FLOAT32, inputs[0]->shape());
    buildOperation(ANEURALNETWORKS_LOGISTIC, getTensorIdxs(inputs), {middleIdx});
    return buildOperation(ANEURALNETWORKS_MUL,
                          {middleIdx, getTensorIdxs(inputs)[0], buildScalar(ANEURALNETWORKS_FUSED_NONE)},
                          getTensorIdxs(outputs));
}

ErrorCode NNAPISiLU::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    std::cout << name() << " NNAPISiLU execute do nothing" << std::endl;
    return NO_ERROR;
}

ErrorCode NNAPISiLU::load(ParamLoader &loader) {
    std::cout << name() << " NNAPISiLU load" << std::endl;
    return NO_ERROR;
}
} // namespace mllm
