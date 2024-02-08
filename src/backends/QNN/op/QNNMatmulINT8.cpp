
#include "QNNMatmulINT8.hpp"
#include "QnnTypes.h"
#include "Types.hpp"
#include "QNNCommonOp.hpp"

namespace mllm {
QNNMatmulINT8::QNNMatmulINT8(Backend *bn, string opName, bool transpose0, bool transpose1) :
    QNNCommonOp(bn, opName), transpose0_(transpose0), transpose1_(transpose1) {
}

ErrorCode QNNMatmulINT8::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 2);
    assert(outputs.size() == 1);
    assert(inputs[0]->head() == inputs[1]->head());

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

        assert(inputs[0]->dimension() == inputs[1]->sequence());
        inputs[1]->transShape(SEQUENCE, DIMENSION);
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

        assert(inputs[0]->dimension() == inputs[1]->dimension());
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
        assert(inputs[0]->sequence() == inputs[1]->sequence());
        inputs[0]->transShape(SEQUENCE, DIMENSION);
        inputs[1]->transShape(SEQUENCE, DIMENSION);
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->dimension(), inputs[1]->dimension());
    }
    // outputs[0]->setDtype(activationDtype());
    return Op::reshape(inputs, outputs);
}

ErrorCode QNNMatmulINT8::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    if (name().find("qkv") != string::npos) {
        // QKV matmul only transpose v
        uint32_t dimVTranspose[4];
        dimVTranspose[0] = inputs[1]->batch();
        dimVTranspose[1] = inputs[1]->head();
        dimVTranspose[2] = inputs[1]->sequence();
        dimVTranspose[3] = inputs[1]->dimension();

        uint32_t transposeParamsDimension[4] = {4};
        uint32_t transposeParamsValue[4] = {0, 2, 1, 3};

        auto paramsTransposeName = inputs[0]->name() + "transpose_params";
        vector<Qnn_Param_t> paramsTranspose = {
            {.paramType = QNN_PARAMTYPE_TENSOR,
             .name = "perm",
             {.tensorParam =
                  (Qnn_Tensor_t){.version = QNN_TENSOR_VERSION_1,
                                 {.v1 = {
                                      .id = 0,
                                      .name = paramsTransposeName.c_str(),
                                      .type = QNN_TENSOR_TYPE_STATIC,
                                      .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                      .dataType = QNN_DATATYPE_UINT_32,
                                      .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                         QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                         {.scaleOffsetEncoding = {.scale = 0.0000000000000000f,
                                                                                  .offset = 0}}},
                                      .rank = 1,
                                      .dimensions = transposeParamsDimension,
                                      .memType = QNN_TENSORMEMTYPE_RAW,
                                      {.clientBuf = {.data = (uint8_t *)transposeParamsValue,
                                                     .dataSize = 4 * sizeof(uint32_t)}}}}}}}};

        auto outVTransposeName = inputs[1]->name() + ".v_transpose_out";
        vector<Qnn_Tensor_t> outVTranspose = {
            (Qnn_Tensor_t){
                .version = QNN_TENSOR_VERSION_1,
                {.v1 = {
                     .id = 0,
                     .name = outVTransposeName.c_str(),
                     .type = QNN_TENSOR_TYPE_NATIVE,
                     .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                     .dataType = QNN_DATATYPE_UFIXED_POINT_8,
                     .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                        QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                        {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                     .rank = 4,
                     .dimensions = dimVTranspose,
                     .memType = QNN_TENSORMEMTYPE_RAW,
                     {.clientBuf = {.data = nullptr,
                                    .dataSize = 0}}}}}};
        graphAddNode(name() + ".v_transpose", "Transpose", {inputs[1]->name()}, outVTranspose, paramsTranspose);

        vector<Qnn_Param_t> paramsMatmulINT8 = {
            {.paramType = QNN_PARAMTYPE_SCALAR,
             .name = "transpose_in0",
             {.scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_BOOL_8, {.bool8Value = transpose0_}}}},
            {.paramType = QNN_PARAMTYPE_SCALAR,
             .name = "transpose_in1",
             {.scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_BOOL_8, {.bool8Value = transpose1_}}}}};

        uint32_t dimOutMatmulINT8[4];
        dimOutMatmulINT8[0] = outputs[0]->batch();
        dimOutMatmulINT8[1] = outputs[0]->head();
        dimOutMatmulINT8[2] = outputs[0]->sequence();
        dimOutMatmulINT8[3] = outputs[0]->dimension();
        auto outMatmulINT8Name = outputs[0]->name() + ".matmul";
        vector<Qnn_Tensor_t> outMatmulINT8 = {
            (Qnn_Tensor_t){
                .version = QNN_TENSOR_VERSION_1,
                {.v1 = {
                     .id = 0,
                     .name = outMatmulINT8Name.c_str(),
                     .type = QNN_TENSOR_TYPE_NATIVE,
                     .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                     .dataType = QNN_DATATYPE_UFIXED_POINT_8,
                     .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                        QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                        {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                     .rank = 4,
                     .dimensions = dimOutMatmulINT8,
                     .memType = QNN_TENSORMEMTYPE_RAW,
                     {.clientBuf = {.data = nullptr,
                                    .dataSize = 0}}}}}};

        graphAddNode(name(), "MatMul", {inputs[0]->name(), outVTransposeName}, outMatmulINT8, paramsMatmulINT8);

        uint32_t dimOut[4];
        dimOut[0] = outputs[0]->batch();
        dimOut[1] = outputs[0]->sequence();
        dimOut[2] = outputs[0]->head();
        dimOut[3] = outputs[0]->dimension();

        auto outTransposeName = outputs[0]->name();
        vector<Qnn_Tensor_t> outTranspose = {
            (Qnn_Tensor_t){
                .version = QNN_TENSOR_VERSION_1,
                {.v1 = {
                     .id = 0,
                     .name = outTransposeName.c_str(),
                     .type = getOutputTensorType(outputs[0]),
                     .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                     .dataType = QNN_DATATYPE_UFIXED_POINT_8,
                     .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                        QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                        {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                     .rank = 4,
                     .dimensions = dimOut,
                     .memType = QNN_TENSORMEMTYPE_RAW,
                     {.clientBuf = {.data = nullptr,
                                    .dataSize = 0}}}}}};
        return graphAddNode(name() + ".out_transpose", "Transpose", {outMatmulINT8Name}, outTranspose, paramsTranspose);

    } else {
        // QK matmul transpose q and k

        uint32_t dimQTranspose[4];
        dimQTranspose[0] = inputs[0]->batch();
        dimQTranspose[1] = inputs[0]->head();
        dimQTranspose[2] = inputs[0]->sequence();
        dimQTranspose[3] = inputs[0]->dimension();

        uint32_t transposeParamsDimension[4] = {4};
        uint32_t transposeParamsValue[4] = {0, 2, 1, 3};

        auto paramsTransposeName = inputs[0]->name() + "transpose_params";
        vector<Qnn_Param_t> paramsTranspose = {
            {.paramType = QNN_PARAMTYPE_TENSOR,
             .name = "perm",
             {.tensorParam =
                  (Qnn_Tensor_t){.version = QNN_TENSOR_VERSION_1,
                                 {.v1 = {
                                      .id = 0,
                                      .name = paramsTransposeName.c_str(),
                                      .type = QNN_TENSOR_TYPE_STATIC,
                                      .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                      .dataType = QNN_DATATYPE_UINT_32,
                                      .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                         QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                         {.scaleOffsetEncoding = {.scale = 0.0000000000000000f,
                                                                                  .offset = 0}}},
                                      .rank = 1,
                                      .dimensions = transposeParamsDimension,
                                      .memType = QNN_TENSORMEMTYPE_RAW,
                                      {.clientBuf = {.data = (uint8_t *)transposeParamsValue,
                                                     .dataSize = 4 * sizeof(uint32_t)}}}}}}}};

        auto outQTransposeName = inputs[0]->name() + ".q_transpose_out";
        vector<Qnn_Tensor_t> outQTranspose = {
            (Qnn_Tensor_t){
                .version = QNN_TENSOR_VERSION_1,
                {.v1 = {
                     .id = 0,
                     .name = outQTransposeName.c_str(),
                     .type = QNN_TENSOR_TYPE_NATIVE,
                     .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                     .dataType = QNN_DATATYPE_UFIXED_POINT_8,
                     .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                        QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                        {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                     .rank = 4,
                     .dimensions = dimQTranspose,
                     .memType = QNN_TENSORMEMTYPE_RAW,
                     {.clientBuf = {.data = nullptr,
                                    .dataSize = 0}}}}}};
        graphAddNode(name() + ".q_transpose", "Transpose", {inputs[0]->name()}, outQTranspose, paramsTranspose);

        uint32_t dimKTranspose[4];
        dimKTranspose[0] = inputs[1]->batch();
        dimKTranspose[1] = inputs[1]->head();
        dimKTranspose[2] = inputs[1]->sequence();
        dimKTranspose[3] = inputs[1]->dimension();

        auto outKTransposeName = inputs[1]->name() + ".k_transpose_out";
        vector<Qnn_Tensor_t> outKTranspose = {
            (Qnn_Tensor_t){
                .version = QNN_TENSOR_VERSION_1,
                {.v1 = {
                     .id = 0,
                     .name = outKTransposeName.c_str(),
                     .type = QNN_TENSOR_TYPE_NATIVE,
                     .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                     .dataType = QNN_DATATYPE_UFIXED_POINT_8,
                     .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                        QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                        {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                     .rank = 4,
                     .dimensions = dimKTranspose,
                     .memType = QNN_TENSORMEMTYPE_RAW,
                     {.clientBuf = {.data = nullptr,
                                    .dataSize = 0}}}}}};
        graphAddNode(name() + ".v_transpose", "Transpose", {inputs[1]->name()}, outKTranspose, paramsTranspose);

        vector<Qnn_Param_t> paramsMatmulINT8 = {
            {.paramType = QNN_PARAMTYPE_SCALAR,
             .name = "transpose_in0",
             {.scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_BOOL_8, {.bool8Value = transpose0_}}}},
            {.paramType = QNN_PARAMTYPE_SCALAR,
             .name = "transpose_in1",
             {.scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_BOOL_8, {.bool8Value = transpose1_}}}}};

        uint32_t dimOut[4];
        dimOut[0] = outputs[0]->batch();
        dimOut[1] = outputs[0]->head();
        dimOut[2] = outputs[0]->sequence();
        dimOut[3] = outputs[0]->dimension();

        auto outName = outputs[0]->name();
        vector<Qnn_Tensor_t> out = {
            (Qnn_Tensor_t){
                .version = QNN_TENSOR_VERSION_1,
                {.v1 = {
                     .id = 0,
                     .name = outName.c_str(),
                     .type = getOutputTensorType(outputs[0]),
                     .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                     .dataType = QNN_DATATYPE_UFIXED_POINT_8,
                     .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                        QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                        {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                     .rank = 4,
                     .dimensions = dimOut,
                     .memType = QNN_TENSORMEMTYPE_RAW,
                     {.clientBuf = {.data = nullptr,
                                    .dataSize = 0}}}}}};

        return graphAddNode(name(), "MatMul", {outQTransposeName, outKTransposeName}, out, paramsMatmulINT8);
    }
}
} // namespace mllm
