
#include "QNNLinearFP.hpp"
#include "QnnTypes.h"
#include "Types.hpp"
#include "QNNCommonOp.hpp"
#include <cstdint>
#include <memory>

namespace mllm {
QNNLinearFP::QNNLinearFP(Backend *bn, string opName, int in_features, int out_features, bool bias) :
    QNNCommonOp(bn, opName), in_features_(in_features), out_features_(out_features), support_bias_(bias) {
}

ErrorCode QNNLinearFP::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(outputs.size(), 1);
    // N     |    C       |   H                   |  W
    // -----------------------------------------------
    // 1     |out_channel | in_channel            |  1
    //       |out_features| in_features           |
    // -----------------------------------------------
    // batch |in_channel  | seq_len               |  1
    //       |in_features | inputs[0]->sequence()   |
    // -----------------------------------------------
    // batch |out_channel | seq_len               |  1
    //       |out_features|  inputs[0]->sequence()  |
    CHECK_EQ(inputs[0]->head(), 1);
    CHECK_EQ(in_features_, inputs[0]->dimension());
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), out_features_);
    return Op::reshape(inputs, outputs);
}

ErrorCode QNNLinearFP::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // add matmul param to qnn
    vector<Qnn_Param_t> paramsMatmul = {
        {.paramType = QNN_PARAMTYPE_SCALAR,
         .name = "transpose_in0",
         {.scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
        {.paramType = QNN_PARAMTYPE_SCALAR,
         .name = "transpose_in1",
         {.scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_BOOL_8, {.bool8Value = 1}}}}};

    // add weight tensor to qnn
    uint32_t dimensionsWeight[4];
    for (int i = 0; i < 4; i++) {
        dimensionsWeight[i] = weight_.shape()[i];
    }
    auto weightName = weight_.name();
    auto weightQuantName = weightName + ".quantized";
    qnnBackend_->modelAddTensor(weight_.name(), (Qnn_Tensor_t){
                                                    .version = QNN_TENSOR_VERSION_1,
                                                    {.v1 = {
                                                         .id = 0,
                                                         .name = weightName.c_str(),
                                                         .type = QNN_TENSOR_TYPE_STATIC,
                                                         .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                                         .dataType = QNN_DATATYPE_UFIXED_POINT_8,
                                                         .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                                            QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                                            {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                                         .rank = 4,
                                                         .dimensions = dimensionsWeight,
                                                         .memType = QNN_TENSORMEMTYPE_RAW,
                                                         {.clientBuf = {.data = weight_.hostPtr<void>(),
                                                                        .dataSize = (uint32_t)weight_.cntSize()}}}}});
    // output of dequantized result of weight
    vector<Qnn_Tensor_t> weightQuantOut = {{QNN_TENSOR_VERSION_1,
                                            {.v1 = {
                                                 .id = 0,
                                                 .name = weightQuantName.c_str(),
                                                 .type = getOutputTensorType(outputs[0]),
                                                 .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                                 .dataType = QNN_DATATYPE_FLOAT_32,
                                                 .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                                    {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                                 .rank = 4,
                                                 .dimensions = dimensionsWeight,
                                                 .memType = QNN_TENSORMEMTYPE_RAW,
                                                 {.clientBuf = {.data = nullptr,
                                                                .dataSize = 0}}}}}};
    // dequantize weight to float and matmul
    // TODO: implement llama op for dequantize using group quantize
    graphAddNode(name() + ".dequantize", "Dequantize", {weightName}, weightQuantOut);
    // dimensions of matmul output and bias
    uint32_t dimensionsOutput[4];
    for (int i = 0; i < 4; i++) {
        dimensionsOutput[i] = outputs[0]->shape()[i];
    }
    auto outName = outputs[0]->name();

    // if don't support bias, just execute matmul
    if (!support_bias_) {
        vector<Qnn_Tensor_t> matmulOut = {{QNN_TENSOR_VERSION_1,
                                           {.v1 = {
                                                .id = 0,
                                                .name = outName.c_str(),
                                                .type = QNN_TENSOR_TYPE_NATIVE,
                                                .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                                .dataType = QNN_DATATYPE_FLOAT_32,
                                                .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                                   QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                                   {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                                .rank = 4,
                                                .dimensions = dimensionsOutput,
                                                .memType = QNN_TENSORMEMTYPE_RAW,
                                                {.clientBuf = {.data = nullptr,
                                                               .dataSize = 0}}}}}};
        return graphAddNode(name() + ".matmul", "MatMul", {inputs[0]->name(), weightQuantName}, matmulOut, paramsMatmul);
    }

    string matmulOutName = name() + ".matmul";
    vector<Qnn_Tensor_t> matmulOut = {{QNN_TENSOR_VERSION_1,
                                       {.v1 = {
                                            .id = 0,
                                            .name = matmulOutName.c_str(),
                                            .type = QNN_TENSOR_TYPE_NATIVE,
                                            .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                            .dataType = QNN_DATATYPE_FLOAT_32,
                                            .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                               {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                            .rank = 4,
                                            .dimensions = dimensionsOutput,
                                            .memType = QNN_TENSORMEMTYPE_RAW,
                                            {.clientBuf = {.data = nullptr,
                                                           .dataSize = 0}}}}}};
    graphAddNode(name() + ".matmul", "MatMul", {inputs[0]->name(), weightQuantName}, matmulOut, paramsMatmul);
    // add bias tensor to qnn
    uint32_t dimensionsBias[4] = {1, 1, 1, (uint32_t)out_features_};
    qnnBackend_->modelAddTensor(bias_.name(), (Qnn_Tensor_t){
                                                  .version = QNN_TENSOR_VERSION_1,
                                                  {.v1 = {
                                                       .id = 0,
                                                       .name = bias_.name().c_str(),
                                                       .type = QNN_TENSOR_TYPE_STATIC,
                                                       .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                                       .dataType = QNN_DATATYPE_FLOAT_32,
                                                       .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                                          QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                                          {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                                       .rank = 4,
                                                       .dimensions = dimensionsBias,
                                                       .memType = QNN_TENSORMEMTYPE_RAW,
                                                       {.clientBuf = {.data = bias_.hostPtr<void>(),
                                                                      .dataSize = (uint32_t)bias_.cntSize()}}}}});
    // final output
    vector<Qnn_Tensor_t> biasOutput = {{QNN_TENSOR_VERSION_1,
                                        {.v1 = {
                                             .id = 0,
                                             .name = outName.c_str(),
                                             .type = getOutputTensorType(outputs[0]),
                                             .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                             .dataType = QNN_DATATYPE_FLOAT_32,
                                             .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                                QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                                {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                             .rank = 4,
                                             .dimensions = dimensionsOutput,
                                             .memType = QNN_TENSORMEMTYPE_RAW,
                                             {.clientBuf = {.data = nullptr,
                                                            .dataSize = 0}}}}}};
    return graphAddNode(name() + ".add", "ElementWiseAdd", {matmulOutName, bias_.name()}, biasOutput);
}

ErrorCode QNNLinearFP::load(AbstructLoader &loader) {
    weight_.setName(name() + ".weight");
    weight_.reshape(1, 1, out_features_, in_features_);
    weight_.setDtype(loader.getDataType(weight_.name()));
    weight_.setBackend(qnnBackend_);
    weight_.alloc();
    loader.load(&weight_);
    if (support_bias_) {
        bias_.setName(name() + ".bias");
        bias_.reshape(1, 1, 1, out_features_);
        bias_.setDtype(loader.getDataType(bias_.name()));
        bias_.setBackend(qnnBackend_);
        bias_.alloc();
        loader.load(&bias_);
    }
    return Op::load(loader);
}

ErrorCode QNNLinearFP::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    weight_.free();
    if (support_bias_) {
        bias_.free();
    }
    return Op::free(inputs, outputs);
}
} // namespace mllm
