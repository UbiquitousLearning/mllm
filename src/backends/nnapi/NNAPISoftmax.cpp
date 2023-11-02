#include "NNAPISoftMax.hpp"
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
    std::cout << name() << "  NNAPISoftMax  reshape" << std::endl;
    CHECK_EQ(inputs.size(), 1);
    outputs[0]->reshape(inputs[0]->shape(0), inputs[0]->shape(1), inputs[0]->shape(2), inputs[0]->shape(3));
    outputs[0]->setDtype(activationDtype());
    return NO_ERROR;
}

ErrorCode NNAPISoftMax::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    std::cout << name() << "  NNAPISoftMax  setUp" << std::endl;
    if (!inputs[0]->allocted()) {
        inputs[0]->alloc(); // TODO remove
    }
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

ErrorCode NNAPISoftMax::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    std::cout << name() << "NNAPISoftMax execute do nothing" << std::endl;
    return NO_ERROR;
}

ErrorCode NNAPISoftMax::load(ParamLoader &loader) {
    std::cout << name() << "NNAPISoftMax load" << std::endl;
    return NO_ERROR;
}
} // namespace mllm
