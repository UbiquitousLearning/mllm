
#include "QNNRoPE.hpp"
#include "Types.hpp"
#include "QNNCommonOp.hpp"

namespace mllm {
QNNRoPE::QNNRoPE(Backend *bn, string opName) :
    QNNCommonOp(bn, opName) {
}

ErrorCode QNNRoPE::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return NO_ERROR;
}

ErrorCode QNNRoPE::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return graphAddNode(name(), "RoPE", inputs, outputs, {}, "LLaMAOpPackageHtp");
}
} // namespace mllm

