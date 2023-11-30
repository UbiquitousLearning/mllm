#include "NNAPISoftmax.hpp"
#include "NNAPICommonOp.hpp"
#include "NNAPINeuralNetworks.h"

namespace mllm {

// template class NNAPISoftMax;
// template class NNAPISoftMax;

NNAPISoftMax::NNAPISoftMax(Backend *bn, string opName, int axis) :
    NNAPICommonOp(bn, opName) {
    axis_ = axis;
}

ErrorCode NNAPISoftMax::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
#ifdef DEBUG
    std::cout << "*NNAPI " << name() << " reshape*" << std::endl;
#endif
    CHECK_EQ(inputs.size(), 1);
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return NO_ERROR;
}

ErrorCode NNAPISoftMax::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
#ifdef DEBUG
    std::cout << "*NNAPI " << name() << " setUp*" << std::endl;
#endif
    outputs[0]->alloc();

    // NNAPI Softmax inputs: [input, beta, axis]
    auto inputIdxs = getTensorIdxs(inputs);
    float beta = 1.0;
    inputIdxs.push_back(buildScalar(beta));
    bool needAxis = false;
    auto dims = inputs[0]->shape();
    for (int i = 0; i < dims.size(); i++) {
        if (i != axis_ && dims[i] > 1) {
            needAxis = true;
            break;
        }
    }
    if (needAxis) {
        inputIdxs.push_back(buildScalar(formatAxis(axis_, inputs[0].get())));
    }
    return buildOperation(ANEURALNETWORKS_SOFTMAX, inputIdxs, getTensorIdxs(outputs));
}
} // namespace mllm
