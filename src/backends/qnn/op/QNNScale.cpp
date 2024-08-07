
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
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode QNNScale::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // add intermediate output of matmul
    uint32_t dimensions[4] = {static_cast<uint32_t>(inputs[0]->batch()),
                              static_cast<uint32_t>(inputs[0]->sequence()),
                              static_cast<uint32_t>(inputs[0]->head()),
                              static_cast<uint32_t>(inputs[0]->dimension())};
    auto interName = name() + ".intermediate";
    vector<Qnn_Tensor_t>
        intermediateOutput = {
            {.version = QNN_TENSOR_VERSION_1,
             {.v1 = {
                  .id = 0,
                  .name = interName.c_str(),
                  .type = QNN_TENSOR_TYPE_NATIVE,
                  .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                  .dataType = QNN_DATATYPE_FLOAT_32,
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
    auto scaleName = name() + ".scale";
    auto biasName = name() + ".bias";
    qnnBackend_->modelAddTensor(scaleName, (Qnn_Tensor_t){
                                               .version = QNN_TENSOR_VERSION_1,
                                               {.v1 = {
                                                    .id = 0,
                                                    .name = scaleName.c_str(),
                                                    .type = QNN_TENSOR_TYPE_STATIC,
                                                    .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                                    .dataType = QNN_DATATYPE_FLOAT_32,
                                                    .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                                       QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                                       {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                                    .rank = 1,
                                                    .dimensions = scalarDimensions,
                                                    .memType = QNN_TENSORMEMTYPE_RAW,
                                                    {.clientBuf = {.data = scaleData,
                                                                   .dataSize = 4}}}}});
    qnnBackend_->modelAddTensor(biasName, (Qnn_Tensor_t){
                                              .version = QNN_TENSOR_VERSION_1,
                                              {.v1 = {
                                                   .id = 0,
                                                   .name = biasName.c_str(),
                                                   .type = QNN_TENSOR_TYPE_STATIC,
                                                   .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                                   .dataType = QNN_DATATYPE_FLOAT_32,
                                                   .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                                      QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                                      {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                                   .rank = 1,
                                                   .dimensions = scalarDimensions,
                                                   .memType = QNN_TENSORMEMTYPE_RAW,
                                                   {.clientBuf = {.data = biasData,
                                                                  .dataSize = 4}}}}});
    // convert output to qnn tensor
    auto outName = outputs[0]->name();
    vector<Qnn_Tensor_t> outputTensors = {
        {.version = QNN_TENSOR_VERSION_1,
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
              .dimensions = dimensions,
              .memType = QNN_TENSORMEMTYPE_RAW,
              {.clientBuf = {.data = nullptr,
                             .dataSize = 0}}}}}};
    if (bias_after_scale_) {
        graphAddNode(name(), "ElementWiseMul", {inputs[0]->name(), scaleName}, intermediateOutput, {}, "LLaMAPackage");
        return graphAddNode(name(), "ElementWiseAdd", {interName, biasName}, outputTensors);
    } else {
        graphAddNode(name(), "ElementWiseMul", {inputs[0]->name(), biasName}, intermediateOutput);
        return graphAddNode(name(), "LLaMAMul", {interName, scaleName}, outputTensors, {}, "LLaMAPackage");
    }
}
} // namespace mllm
