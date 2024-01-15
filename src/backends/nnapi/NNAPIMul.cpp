
#include "NNAPIMul.hpp"
#include "NNAPICommonOp.hpp"

namespace mllm {

NNAPIMul::NNAPIMul(Backend *bn, string opName) :
    NNAPICommonOp(bn, opName) {
}

ErrorCode NNAPIMul::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
#ifdef DEBUG
    std::cout << "*NNAPI " << name() << " reshape*" << std::endl;
#endif
    CHECK_EQ(inputs.size(), 2);
    CHECK_EQ(outputs.size(), 1);
    CHECK_EQ(inputs[0]->batch(), inputs[1]->batch());
    CHECK_EQ(inputs[0]->head(), inputs[1]->head());
    CHECK_EQ(inputs[0]->sequence(), inputs[1]->sequence());
    CHECK_EQ(inputs[0]->dimension(), inputs[1]->dimension());
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return MLLM_NO_ERROR;
}

ErrorCode NNAPIMul::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    std::cout << "*NNAPI " << name() << " setUp" << std::endl;
    outputs[0]->alloc();

    auto inputIdxs = getTensorIdxs(inputs);
    inputIdxs.push_back(buildScalar(ANEURALNETWORKS_FUSED_NONE));

    return buildOperation(ANEURALNETWORKS_MUL, inputIdxs, getTensorIdxs(outputs));
}
} // namespace mllm
