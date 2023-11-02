#include "NNAPIAdd.hpp"
#include "NNAPICommonOp.hpp"
#include "Types.hpp"

namespace mllm {

NNAPIAdd::NNAPIAdd(Backend *bn, string opName) :
    NNAPICommonOp(bn, opName) {
}

ErrorCode NNAPIAdd::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    std::cout << "NNAPIAdd reshape" << std::endl;
    CHECK_EQ(inputs.size(), 2);
    CHECK_EQ(outputs.size(), 1);
    CHECK_EQ(inputs[0]->shape(0), inputs[1]->shape(0));
    CHECK_EQ(inputs[0]->shape(1), inputs[1]->shape(1));
    CHECK_EQ(inputs[0]->shape(2), inputs[1]->shape(2));
    CHECK_EQ(inputs[0]->shape(3), inputs[1]->shape(3));

    outputs[0]->reshape(inputs[0]->shape(0),
                        inputs[0]->shape(1),
                        inputs[0]->shape(2),
                        inputs[0]->shape(3));

    return NO_ERROR;
}

ErrorCode NNAPIAdd::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    std::cout << "NNAPIAdd setUp" << std::endl;
    outputs[0]->alloc();

    auto inputIdxs = getTensorIdxs(inputs);
    inputIdxs.push_back(buildScalar(ANEURALNETWORKS_FUSED_NONE));
    return buildOperation(ANEURALNETWORKS_ADD, inputIdxs, getTensorIdxs(outputs));
}

ErrorCode NNAPIAdd::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    std::cout << "NNAPIAdd execute do nothing" << std::endl;
    return NO_ERROR;
}
} // namespace mllm
