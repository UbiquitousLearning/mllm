
#include "QNNView.hpp"
#include "Types.hpp"
#include "QNNCommonOp.hpp"

namespace mllm {
QNNView::QNNView(Backend *bn, string opName) :
    QNNCommonOp(bn, opName) {
}

ErrorCode QNNView::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return NO_ERROR;
}

ErrorCode QNNView::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return graphAddNode(name(), "Add", inputs, outputs);
}
} // namespace mllm

