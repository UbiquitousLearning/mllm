
#include "QNNRope.hpp"
#include "Types.hpp"
#include "QNNCommonOp.hpp"

namespace mllm {
QNNRope::QNNRope(Backend *bn, string opName) :
    QNNCommonOp(bn, opName) {
}

ErrorCode QNNRope::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return NO_ERROR;
}

ErrorCode QNNRope::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return graphAddNode(name(), "Add", inputs, outputs);
}
} // namespace mllm

