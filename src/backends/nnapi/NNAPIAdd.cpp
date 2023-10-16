
#include "NNAPIAdd.hpp"
#include "NNAPICommonOp.hpp"
#include "NNAPINeuralNetworks.h"
#include <sys/types.h>

namespace mllm {

// template class NNAPIAdd;
// template class NNAPIAdd;

NNAPIAdd::NNAPIAdd(Backend *bn) :
    NNAPICommonOp(bn) {
}

ErrorCode NNAPIAdd::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    std::cout << "NNAPIAdd reshape" << std::endl;
    CHECK_EQ(inputs.size(), 2);
    CHECK_EQ(outputs.size(), 1);
    CHECK_EQ(inputs[0]->shape(0), inputs[1]->shape(0));
    CHECK_EQ(inputs[0]->shape(1), inputs[1]->shape(1));
    CHECK_EQ(inputs[0]->shape(2), inputs[1]->shape(2));
    CHECK_EQ(inputs[0]->shape(3), inputs[1]->shape(3));

    outputs[0]->reshape(inputs[0]->shape(0), inputs[0]->shape(1), inputs[0]->shape(2), inputs[0]->shape(3));

    auto inputIdxs = getTensorIdxs(inputs);
    inputIdxs.push_back(buildScalar(ANEURALNETWORKS_FUSED_NONE));
    return buildOperation(ANEURALNETWORKS_ADD, inputIdxs, getTensorIdxs(outputs));
}

ErrorCode NNAPIAdd::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    std::cout << "NNAPIAdd setUp" << std::endl;
    // TODO: bulid nnapi operation
    buildOperation(ANEURALNETWORKS_ADD, getTensorIdxs(inputs), getTensorIdxs(outputs));
    return NO_ERROR;
}
} // namespace mllm
