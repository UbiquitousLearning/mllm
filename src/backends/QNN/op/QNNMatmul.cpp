
#include "QNNMatmul.hpp"
#include "Types.hpp"
#include "QNNCommonOp.hpp"

namespace mllm {
QNNMatmul::QNNMatmul(Backend *bn, string opName) :
    QNNCommonOp(bn, opName) {
}

ErrorCode QNNMatmul::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    CHECK_EQ(inputs.size(), 2);
    CHECK_EQ(outputs.size(), 1);
    CHECK_EQ(inputs[0]->head(), inputs[1]->head());
    //    CHECK_EQ(inputs[0]->head(), 1);
    CHECK_EQ(inputs[0]->batch(), inputs[1]->batch());
    if (!transpose0_ && !transpose1_) {
        /*
         N     |    C       |   H                   |  W
         -----------------------------------------------
         batch |out_channel | in_channel            |  1
         -----------------------------------------------
         batch |in_channel  | seq_len               |  1
         -----------------------------------------------
         batch |out_channel | seq_len               |  1
         */
        CHECK_EQ(inputs[0]->dimension(), inputs[1]->sequence());
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[1]->dimension());
    } else if (transpose1_) {
        /*
         N     |    C       |   H                   |  W
         -----------------------------------------------
         batch |in_channel | out_channel            |  1
         -----------------------------------------------
         batch |in_channel  | seq_len               |  1
         -----------------------------------------------
         batch |out_channel | seq_len               |  1
         */
        CHECK_EQ(inputs[0]->dimension(), inputs[1]->dimension());
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[1]->sequence());
    } else {
        /*
         N     |    C       |   H                   |  W
         -----------------------------------------------
         batch |out_channel | in_channel            |  1
         -----------------------------------------------
         batch |seq_len     | in_channel            |  1
         -----------------------------------------------
         batch |out_channel | seq_len               |  1
         */
        CHECK_EQ(inputs[0]->sequence(), inputs[1]->sequence());
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->dimension(), inputs[1]->dimension());
    }
    // outputs[0]->setDtype(activationDtype());
    return Op::reshape(inputs, outputs);
}

ErrorCode QNNMatmul::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    vector<Qnn_Param_t> paramsMatmul = {
        {.paramType = QNN_PARAMTYPE_SCALAR,
         .name = "transpose_in0",
         {.scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_BOOL_8, {.bool8Value = transpose0_}}}},
        {.paramType = QNN_PARAMTYPE_SCALAR,
         .name = "transpose_in1",
         {.scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_BOOL_8, {.bool8Value = transpose1_}}}}};
    return graphAddNode(name(), "MatMul", inputs, outputs, paramsMatmul);
}
} // namespace mllm
