
#include "QNNScale.hpp"
#include "QnnTypes.h"
#include "Types.hpp"
#include "QNNCommonOp.hpp"
#include <arm_neon.h>
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
    outputs[0]->setDtype(inputs[0]->dtype());
    // add intermediate output of matmul
    bool isHaveBias = (bias_ != 0.0f);
    if (!isHaveBias) {
        // if no bias and bias_after_scale is false, then we don't need intermediate output
        // add scale and bias tensor
        uint32_t scalarDimensions[1] = {1};
        float scaleData[] = {scale_};
        float16_t scaleDataF16[] = {static_cast<float16_t>(scale_)};
        auto scaleName = name() + ".scale";
        auto qnnDtype = QNN_DATATYPE_FLOAT_32;

        switch (outputs[0]->dtype()) {
        case MLLM_TYPE_F32:
            qnnDtype = QNN_DATATYPE_FLOAT_32;
            qnnBackend_->modelAddTensor(scaleName, (Qnn_Tensor_t){
                                                       .version = QNN_TENSOR_VERSION_1,
                                                       .v1 = {
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
                                                           .clientBuf = {.data = scaleData,
                                                                         .dataSize = 4}}});
            break;
        case MLLM_TYPE_F16:
            qnnDtype = QNN_DATATYPE_FLOAT_16;
            qnnBackend_->modelAddTensor(scaleName, (Qnn_Tensor_t){
                                                       .version = QNN_TENSOR_VERSION_1,
                                                       .v1 = {
                                                           .id = 0,
                                                           .name = scaleName.c_str(),
                                                           .type = QNN_TENSOR_TYPE_STATIC,
                                                           .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                                           .dataType = QNN_DATATYPE_FLOAT_16,
                                                           .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                                              QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                                              {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                                           .rank = 1,
                                                           .dimensions = scalarDimensions,
                                                           .memType = QNN_TENSORMEMTYPE_RAW,
                                                           .clientBuf = {.data = scaleDataF16,
                                                                         .dataSize = 2}}});
            break;
        default:
            MLLM_LOG_ERROR_STREAM << "[ERROR] QNNScale not support dtype: " << outputs[0]->dtype() << std::endl;
            return ErrorCode::NOT_SUPPORT;
        }

        // the scale is used after q*k in qnn graph, dimension should be BHSD
        uint32_t dimensions[4] = {static_cast<uint32_t>(inputs[0]->batch()),
                                  static_cast<uint32_t>(inputs[0]->head()),
                                  static_cast<uint32_t>(inputs[0]->sequence()),
                                  static_cast<uint32_t>(inputs[0]->dimension())};
        // convert output to qnn tensor
        auto outName = outputs[0]->name();
        vector<Qnn_Tensor_t> outputTensors = {
            {.version = QNN_TENSOR_VERSION_1,
             .v1 = {
                 .id = 0,
                 .name = outName.c_str(),
                 .type = getOutputTensorType(outputs[0]),
                 .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                 .dataType = qnnDtype,
                 .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                    {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                 .rank = 4,
                 .dimensions = dimensions,
                 .memType = QNN_TENSORMEMTYPE_RAW,
                 .clientBuf = {.data = nullptr,
                               .dataSize = 0}}}};
        return graphAddNode(name(), "ElementWiseMultiply", {inputs[0]->name(), scaleName}, outputTensors);
    }

    // TODO: below should deprecated
    uint32_t dimensions[4] = {static_cast<uint32_t>(inputs[0]->batch()),
                              static_cast<uint32_t>(inputs[0]->head()),
                              static_cast<uint32_t>(inputs[0]->sequence()),
                              static_cast<uint32_t>(inputs[0]->dimension())};
    auto interName = name() + ".intermediate";
    vector<Qnn_Tensor_t>
        intermediateOutput = {
            {.version = QNN_TENSOR_VERSION_1,
             .v1 = {
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
                 .clientBuf = {.data = nullptr,
                               .dataSize = 0}}}};
    // add scale and bias tensor
    uint32_t scalarDimensions[1] = {1};
    float biasData[] = {bias_};
    float scaleData[] = {scale_};
    auto scaleName = name() + ".scale";
    auto biasName = name() + ".bias";
    qnnBackend_->modelAddTensor(scaleName, (Qnn_Tensor_t){
                                               .version = QNN_TENSOR_VERSION_1,
                                               .v1 = {
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
                                                   .clientBuf = {.data = scaleData,
                                                                 .dataSize = 4}}});
    qnnBackend_->modelAddTensor(biasName, (Qnn_Tensor_t){
                                              .version = QNN_TENSOR_VERSION_1,
                                              .v1 = {
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
                                                  .clientBuf = {.data = biasData,
                                                                .dataSize = 4}}});
    // convert output to qnn tensor
    auto outName = outputs[0]->name();
    vector<Qnn_Tensor_t> outputTensors = {
        {.version = QNN_TENSOR_VERSION_1,
         .v1 = {
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
             .clientBuf = {.data = nullptr,
                           .dataSize = 0}}}};
    if (bias_after_scale_) {
        graphAddNode(name(), "ElementWiseMul", {inputs[0]->name(), scaleName}, intermediateOutput, {}, "LLaMAPackage");
        return graphAddNode(name(), "ElementWiseAdd", {interName, biasName}, outputTensors);
    } else {
        graphAddNode(name(), "ElementWiseMul", {inputs[0]->name(), biasName}, intermediateOutput);
        return graphAddNode(name(), "LLaMAMul", {interName, scaleName}, outputTensors, {}, "LLaMAPackage");
    }
}
} // namespace mllm
