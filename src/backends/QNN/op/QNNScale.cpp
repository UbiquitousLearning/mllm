
#include "QNNScale.hpp"
#include "Types.hpp"
#include "QNNCommonOp.hpp"

namespace mllm {
QNNScale::QNNScale(Backend *bn, string opName) :
    QNNCommonOp(bn, opName) {
}

ErrorCode QNNScale::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return NO_ERROR;
}

ErrorCode QNNScale::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return graphAddNode(name(), "Add", inputs, outputs);
}
} // namespace mllm

