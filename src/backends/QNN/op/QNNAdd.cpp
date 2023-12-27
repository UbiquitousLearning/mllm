#include "QNNAdd.hpp"
#include "Types.hpp"
#include "QNNCommonOp.hpp"

namespace mllm {
QNNAdd::QNNAdd(Backend *bn, string opName) :
    QNNCommonOp(bn, opName) {
}

ErrorCode QNNAdd::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
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

ErrorCode QNNAdd::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // graph add node
    return graphAddNode(name(), "Add", inputs, outputs);
}
} // namespace mllm