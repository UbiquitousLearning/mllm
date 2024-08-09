#include "QNNAdd.hpp"
#include "Types.hpp"
#include "QNNCommonOp.hpp"

namespace mllm {
QNNAdd::QNNAdd(Backend *bn, string opName) :
    QNNCommonOp(bn, opName) {
}

ErrorCode QNNAdd::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 2);
    assert(outputs.size() == 1);
    if (inputs[0]->batch() == 1 || inputs[1]->batch() == 1) {
    } else {
        assert(inputs[0]->batch() == inputs[1]->batch());
    }
    assert(inputs[0]->head() == inputs[1]->head());
    assert(inputs[0]->sequence() == inputs[1]->sequence());
    assert(inputs[0]->dimension() == inputs[1]->dimension());

    outputs[0]->reshape(std::max(inputs[0]->batch(), inputs[1]->batch()), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());

    return Op::reshape(inputs, outputs);
}

ErrorCode QNNAdd::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // graph add node
    return graphAddNode(name(), "LLaMAAdd", inputs, outputs, {}, "LLaMAPackage");
    // return graphAddNode(name(), "ElementWiseAdd", inputs, outputs, {});
}
} // namespace mllm