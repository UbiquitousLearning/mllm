
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
#ifdef DEBUG
    std::cout << "*NNAPI " << name() << " reshape*" << std::endl;
#endif
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return MLLM_NO_ERROR;
}

ErrorCode NNAPISiLU::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
#ifdef DEBUG
    std::cout << "*NNAPI " << name() << " setUp*" << std::endl;
#endif
    outputs[0]->alloc();

    auto middleIdx = buildTensor(ANEURALNETWORKS_TENSOR_FLOAT32, inputs[0]->shape());
    buildOperation(ANEURALNETWORKS_LOGISTIC, getTensorIdxs(inputs), {middleIdx});
    return buildOperation(ANEURALNETWORKS_MUL,
                          {middleIdx, getTensorIdxs(inputs)[0], buildScalar(ANEURALNETWORKS_FUSED_NONE)},
                          getTensorIdxs(outputs));
}
} // namespace mllm
