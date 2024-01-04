
#include "QNNLinear.hpp"
#include "QnnTypes.h"
#include "Types.hpp"
#include "QNNCommonOp.hpp"
#include <memory>

namespace mllm {
QNNLinear::QNNLinear(Backend *bn, string opName, int in_features, int out_features, bool bias) :
    QNNCommonOp(bn, opName), in_features_(in_features), out_features_(out_features), support_bias_(bias) {
}

ErrorCode QNNLinear::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
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

ErrorCode QNNLinear::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    vector<Qnn_Param_t> paramsMatmul = {
        {.paramType = QNN_PARAMTYPE_SCALAR,
         .name = "transpose_in0",
         {.scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
        {.paramType = QNN_PARAMTYPE_SCALAR,
         .name = "transpose_in1",
         {.scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_BOOL_8, {.bool8Value = 1}}}}};
    // add input tensor to qnn
    uint32_t dimensionsInput[4];
    for (int i = 0; i < 4; i++) {
        dimensionsInput[i] = inputs[0]->shape()[i];
    }
    qnnBackend_->modelAddTensor(inputs[0]->name().c_str(), (Qnn_Tensor_t){
                                                               .version = QNN_TENSOR_VERSION_1,
                                                               {.v1 = {
                                                                    .id = 0,
                                                                    .name = inputs[0]->name().c_str(),
                                                                    .type = QNN_TENSOR_TYPE_APP_WRITE,
                                                                    .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                                                    .dataType = QNN_DATATYPE_FLOAT_32,
                                                                    .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                                                       QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                                                       {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                                                    .rank = 4,
                                                                    .dimensions = dimensionsInput,
                                                                    .memType = QNN_TENSORMEMTYPE_RAW,
                                                                    {.clientBuf = {.data = nullptr,
                                                                                   .dataSize = 0}}}}});
    // add weight tensor to qnn
    uint32_t dimensionsWeight[4];
    for (int i = 0; i < 4; i++) {
        dimensionsWeight[i] = weight_.shape()[i];
    }
    qnnBackend_->modelAddTensor(weight_.name().c_str(), (Qnn_Tensor_t){
                                                            .version = QNN_TENSOR_VERSION_1,
                                                            {.v1 = {
                                                                 .id = 0,
                                                                 .name = weight_.name().c_str(),
                                                                 .type = QNN_TENSOR_TYPE_STATIC,
                                                                 .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                                                 .dataType = QNN_DATATYPE_FLOAT_32,
                                                                 .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                                                    {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                                                 .rank = 4,
                                                                 .dimensions = dimensionsWeight,
                                                                 .memType = QNN_TENSORMEMTYPE_RAW,
                                                                 {.clientBuf = {.data = weight_.hostPtr<void>(),
                                                                                .dataSize = (uint32_t)weight_.cntSize()}}}}});

    uint32_t dimensionsBias[4];
    for (int i = 0; i < 4; i++) {
        dimensionsBias[i] = outputs[0]->shape()[i];
    }
    auto outString = outputs[0]->name();
    if (!support_bias_) { // if don't support bias, just matmul and write to outputs[0]
        vector<Qnn_Tensor_t> qnnOutputTensor = {{QNN_TENSOR_VERSION_1,
                                                 {.v1 = {
                                                      .id = 0,
                                                      .name = outString.c_str(),
                                                      .type = QNN_TENSOR_TYPE_APP_READ,
                                                      .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                                      .dataType = QNN_DATATYPE_FLOAT_32,
                                                      .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                                         QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                                         {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                                      .rank = 4,
                                                      .dimensions = dimensionsBias,
                                                      .memType = QNN_TENSORMEMTYPE_RAW,
                                                      {.clientBuf = {.data = nullptr,
                                                                     .dataSize = 0}}}}}};
        return graphAddNode(name(), "MatMul", {inputs[0]->name().c_str(), weight_.name().c_str()}, qnnOutputTensor, paramsMatmul);
    }

    // add bias tensor to qnn
    qnnBackend_->modelAddTensor(bias_.name().c_str(), (Qnn_Tensor_t){
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
    // add intermediate output of matmul
    vector<Qnn_Tensor_t> intermediateOutput = {
        {.version = QNN_TENSOR_VERSION_1,
         {.v1 = {
              .id = 0,
              .name = (name() + ".intermediate").c_str(),
              .type = QNN_TENSOR_TYPE_NATIVE,
              .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
              .dataType = QNN_DATATYPE_FLOAT_32,
              .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                 QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                 {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
              .rank = 4,
              .dimensions = dimensionsBias,
              .memType = QNN_TENSORMEMTYPE_RAW,
              {.clientBuf = {.data = nullptr,
                             .dataSize = 0}}}}}};

    graphAddNode(name(), "MatMul", {inputs[0]->name().c_str(), weight_.name().c_str()}, intermediateOutput, paramsMatmul);

    vector<Qnn_Tensor_t> biasOutput = {{QNN_TENSOR_VERSION_1,
                                        {.v1 = {
                                             .id = 0,
                                             .name = outString.c_str(),
                                             .type = QNN_TENSOR_TYPE_APP_READ,
                                             .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                             .dataType = QNN_DATATYPE_FLOAT_32,
                                             .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                                QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                                {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                             .rank = 4,
                                             .dimensions = dimensionsBias,
                                             .memType = QNN_TENSORMEMTYPE_RAW,
                                             {.clientBuf = {.data = nullptr,
                                                            .dataSize = 0}}}}}};
    return graphAddNode(name(), "ElementWiseAdd", {(name() + ".intermediate").c_str(), bias_.name().c_str()},
                        biasOutput);
}

ErrorCode QNNLinear::load(AbstructLoader &loader) {
    weight_.setName(name() + ".weight");
    weight_.reshape(1, 1, out_features_, in_features_);
    // weight_.setDtype(loader.getDataType(weight_.name()));
    weight_.setBackend(qnnBackend_);
    weight_.alloc();
    // loader.load(&weight_);

    if (support_bias_) {
        bias_.setName(name() + ".bias");
        bias_.reshape(1, 1, 1, out_features_);
        // bias_.setDtype(loader.getDataType(bias_.name()));
        bias_.alloc();
        // loader.load(&bias_);
    }
    return Op::load(loader);
}
} // namespace mllm
