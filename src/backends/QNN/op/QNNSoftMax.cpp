
#include "QNNSoftMax.hpp"
#include "QnnTypes.h"
#include "Types.hpp"
#include "QNNCommonOp.hpp"

namespace mllm {
QNNSoftMax::QNNSoftMax(Backend *bn, string opName, int axis) :
    QNNCommonOp(bn, opName) {
    axis_ = axis;
}

ErrorCode QNNSoftMax::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    CHECK_EQ(inputs.size(), 1);
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode QNNSoftMax::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    vector<Qnn_Param_t> params = {
        {.paramType = QNN_PARAMTYPE_SCALAR,
         .name = "axis",
         {.scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_UINT_32, {.uint32Value = static_cast<uint32_t>(axis_)}}}},
        {.paramType = QNN_PARAMTYPE_SCALAR,
         .name = "beta",
         {.scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_FLOAT_32, {.floatValue = 1.000000000000f}}}}};
    return graphAddNode(name(), "Softmax", inputs, outputs);
}
} // namespace mllm
