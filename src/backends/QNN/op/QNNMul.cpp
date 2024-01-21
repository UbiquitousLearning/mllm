
#include "QNNMul.hpp"
#include "Types.hpp"
#include "QNNCommonOp.hpp"

namespace mllm {
QNNMul::QNNMul(Backend *bn, string opName) :
    QNNCommonOp(bn, opName) {
}

ErrorCode QNNMul::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    CHECK_EQ(inputs.size(), 2);
    CHECK_EQ(outputs.size(), 1);
    CHECK_EQ(inputs[0]->shape(0), inputs[1]->shape(0));
    CHECK_EQ(inputs[0]->shape(1), inputs[1]->shape(1));
    CHECK_EQ(inputs[0]->shape(2), inputs[1]->shape(2));
    CHECK_EQ(inputs[0]->shape(3), inputs[1]->shape(3));
    outputs[0]->reshape(inputs[0]->shape(0), inputs[0]->shape(1), inputs[0]->shape(2), inputs[0]->shape(3));
    return Op::reshape(inputs, outputs);
}

ErrorCode QNNMul::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return graphAddNode(name(), "LLaMAMul", inputs, outputs, {}, "LLaMAPackage");
}
} // namespace mllm

