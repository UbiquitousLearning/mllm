
#include "QNNSiLU.hpp"
#include "Types.hpp"
#include "QNNCommonOp.hpp"

namespace mllm {
QNNSiLU::QNNSiLU(Backend *bn, string opName) :
    QNNCommonOp(bn, opName) {
}

ErrorCode QNNSiLU::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode QNNSiLU::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return graphAddNode(name(), "SiLU", inputs, outputs, {}, "LLaMAPackage");
}
} // namespace mllm

