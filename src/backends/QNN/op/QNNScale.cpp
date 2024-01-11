
#include "QNNScale.hpp"
#include "QnnTypes.h"
#include "Types.hpp"
#include "QNNCommonOp.hpp"
#include <cstdint>

namespace mllm {
QNNScale::QNNScale(Backend *bn, string opName, float scale, float bias, bool bias_after_scale) :
    QNNCommonOp(bn, opName) {
    scale_ = scale;
    bias_ = bias;
    bias_after_scale_ = bias_after_scale;
}

ErrorCode QNNScale::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(outputs.size(), 1);
    outputs[0]->reshape(inputs[0]->shape(0), inputs[0]->shape(1), inputs[0]->shape(2), inputs[0]->shape(3));
    return NO_ERROR;
}

ErrorCode QNNScale::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // add intermediate output of matmul
    uint32_t dimensions[4];
    for (int i = 0; i < 4; i++) {
        dimensions[i] = inputs[0]->shape(i);
    }
    vector<Qnn_Tensor_t>
        intermediateOutput = {
            {.version = QNN_TENSOR_VERSION_1,
             {.v1 = {
                  .id = 0,
                  .name = (name() + ".intermediate").c_str(),
                  .type = QNN_TENSOR_TYPE_NATIVE,
                  .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                  .dataType = QNN_DATATYPE_FLOAT_16,
                  .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                     QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                     {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                  .rank = 4,
                  .dimensions = dimensions,
                  .memType = QNN_TENSORMEMTYPE_RAW,
                  {.clientBuf = {.data = nullptr,
                                 .dataSize = 0}}}}}};
    // add scale and bias tensor
    uint32_t scalarDimensions[1] = {1};
    float biasData[] = {bias_};
    float scaleData[] = {scale_};
    qnnBackend_->modelAddTensor((name() + ".scale").c_str(), (Qnn_Tensor_t){
                                                                 .version = QNN_TENSOR_VERSION_1,
                                                                 {.v1 = {
                                                                      .id = 0,
                                                                      .name = inputs[0]->name().c_str(),
                                                                      .type = QNN_TENSOR_TYPE_STATIC,
                                                                      .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                                                      .dataType = QNN_DATATYPE_FLOAT_16,
                                                                      .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                                                         QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                                                         {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                                                      .rank = 1,
                                                                      .dimensions = scalarDimensions,
                                                                      .memType = QNN_TENSORMEMTYPE_RAW,
                                                                      {.clientBuf = {.data = scaleData,
                                                                                     .dataSize = 4}}}}});
    qnnBackend_->modelAddTensor((name() + ".bias").c_str(), (Qnn_Tensor_t){
                                                                .version = QNN_TENSOR_VERSION_1,
                                                                {.v1 = {
                                                                     .id = 0,
                                                                     .name = inputs[0]->name().c_str(),
                                                                     .type = QNN_TENSOR_TYPE_STATIC,
                                                                     .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                                                     .dataType = QNN_DATATYPE_FLOAT_16,
                                                                     .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                                                        QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                                                        {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                                                     .rank = 1,
                                                                     .dimensions = scalarDimensions,
                                                                     .memType = QNN_TENSORMEMTYPE_RAW,
                                                                     {.clientBuf = {.data = biasData,
                                                                                    .dataSize = 4}}}}});
    // convert output to qnn tensor
    vector<Qnn_Tensor_t> outputTensors = {
        {.version = QNN_TENSOR_VERSION_1,
         {.v1 = {
              .id = 0,
              .name = outputs[0]->name().c_str(),
              .type = QNN_TENSOR_TYPE_APP_READ,
              .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
              .dataType = QNN_DATATYPE_FLOAT_16,
              .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                 QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                 {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
              .rank = 4,
              .dimensions = dimensions,
              .memType = QNN_TENSORMEMTYPE_RAW,
              {.clientBuf = {.data = nullptr,
                             .dataSize = 0}}}}}};
    if (bias_after_scale_) {
        graphAddNode(name(), "ElementWiseMultiply", {inputs[0]->name().c_str(), (name() + ".scale").c_str()}, intermediateOutput);
        return graphAddNode(name(), "ElementWiseAdd", {(name() + ".intermediate").c_str(), (name() + ".bias").c_str()}, outputTensors);
    } else {
        graphAddNode(name(), "ElementWiseAdd", {inputs[0]->name().c_str(), (name() + ".bias").c_str()}, intermediateOutput);
        return graphAddNode(name(), "ElementWiseMultiply", {(name() + ".intermediate").c_str(), (name() + ".scale").c_str()}, outputTensors);
    }
}
} // namespace mllm
