
#include "QNNSiLU.hpp"
#include "Types.hpp"
#include "QNNCommonOp.hpp"

namespace mllm {
QNNSiLU::QNNSiLU(Backend *bn, string opName) :
    QNNCommonOp(bn, opName) {
}

ErrorCode QNNSiLU::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return NO_ERROR;
}

ErrorCode QNNSiLU::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return graphAddNode(name(), "SiLU", inputs, outputs);
}
} // namespace mllm

