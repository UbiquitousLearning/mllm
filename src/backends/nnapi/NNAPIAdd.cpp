#include "NNAPIAdd.hpp"
#include "NNAPICommonOp.hpp"
#include "Types.hpp"

namespace mllm {

NNAPIAdd::NNAPIAdd(Backend *bn, string opName) :
    NNAPICommonOp(bn, opName) {
}

ErrorCode NNAPIAdd::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
#ifdef DEBUG
    std::cout << "*NNAPI " << name() << " reshape*" << std::endl;
#endif
    CHECK_EQ(inputs.size(), 2);
    CHECK_EQ(outputs.size(), 1);
    CHECK_EQ(inputs[0]->batch(), inputs[1]->batch());
    CHECK_EQ(inputs[0]->head(), inputs[1]->head());
    CHECK_EQ(inputs[0]->sequence(), inputs[1]->sequence());
    CHECK_EQ(inputs[0]->dimension(), inputs[1]->dimension());

    outputs[0]->reshape(inputs[0]->batch(),
                        inputs[0]->head(),
                        inputs[0]->sequence(),
                        inputs[0]->dimension());

    return NO_ERROR;
}

ErrorCode NNAPIAdd::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
#ifdef DEBUG
    std::cout << "*NNAPI " << name() << " setUp*" << std::endl;
#endif
    outputs[0]->alloc();

    auto inputIdxs = getTensorIdxs(inputs);
    inputIdxs.push_back(buildScalar(ANEURALNETWORKS_FUSED_NONE));
    return buildOperation(ANEURALNETWORKS_ADD, inputIdxs, getTensorIdxs(outputs));
}
} // namespace mllm
