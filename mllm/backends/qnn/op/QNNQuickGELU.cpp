
#include "QNNQuickGELU.hpp"
#include "QnnTypes.h"
#include "Types.hpp"
#include "QNNCommonOp.hpp"
#include "Context.hpp"
#include <cstdint>

namespace mllm {
QNNQuickGELU::QNNQuickGELU(Backend *bn, string opName) :
    QNNCommonOp(bn, opName) {
}

ErrorCode QNNQuickGELU::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode QNNQuickGELU::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    outputs[0]->setDtype(inputs[0]->dtype());

    auto outName = outputs[0]->name();
    
    uint32_t scalarDimensions[1] = {1};
    float scaleData[] = {1.702f};
    mllm_fp16_t scaleDataF16[] = {static_cast<mllm_fp16_t>(1.702f)};
    auto scaleName = name() + ".gelu_scale";
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
    uint32_t dimensions[4] = {static_cast<uint32_t>(outputs[0]->batch()),
                                static_cast<uint32_t>(outputs[0]->sequence()),
                                static_cast<uint32_t>(outputs[0]->head()),
                                static_cast<uint32_t>(outputs[0]->dimension())};
    // convert output to qnn tensor
    auto scaleOutName = outputs[0]->name() + "-multiply";
    vector<Qnn_Tensor_t> outputTensors = {
        {.version = QNN_TENSOR_VERSION_1,
            .v1 = {
                .id = 0,
                .name = scaleOutName.c_str(),
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
    graphAddNode(name() + "-multiply", "LLaMAMul", {inputs[0]->name(), scaleName}, outputTensors, {}, "LLaMAPackage");

    uint32_t dimensionsOutput[4];

    dimensionsOutput[0] = static_cast<uint32_t>(outputs[0]->batch());
    dimensionsOutput[1] = static_cast<uint32_t>(outputs[0]->sequence());
    dimensionsOutput[2] = static_cast<uint32_t>(outputs[0]->head());
    dimensionsOutput[3] = static_cast<uint32_t>(outputs[0]->dimension());

    auto type = QNN_DATATYPE_FLOAT_32;
    outputs[0]->setDtype(MLLM_TYPE_F32);

    if (inputs[0]->dtype() == MLLM_TYPE_F16) {
        type = QNN_DATATYPE_FLOAT_16;
        outputs[0]->setDtype(MLLM_TYPE_F16);
    }

    // add sigmoid node
    auto sigmoidName = name() + "-sigmoid";
    vector<Qnn_Tensor_t> outputSigmoid{
        {QNN_TENSOR_VERSION_1,
         {.v1 = {
              .id = 0,
              .name = sigmoidName.c_str(),
              .type = QNN_TENSOR_TYPE_NATIVE,
              .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
              .dataType = type,
              .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                 QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                 {.scaleOffsetEncoding = {.scale = 0, .offset = 0}}},
              .rank = 4,
              .dimensions = dimensionsOutput,
              .memType = QNN_TENSORMEMTYPE_RAW,
              .clientBuf = {.data = nullptr,
                            .dataSize = 0}}}}};
    graphAddNode(name() + "-sigmoid", "Sigmoid", {scaleOutName}, outputSigmoid);

    vector<Qnn_Tensor_t> outputTensor = {{QNN_TENSOR_VERSION_1,
                                          {.v1 = {
                                               .id = 0,
                                               .name = outName.c_str(),
                                               .type = getOutputTensorType(outputs[0]),
                                               .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                               .dataType = type,
                                               .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                                  QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                                  {.scaleOffsetEncoding = {.scale = 0.0000000000000000f,
                                                                                           .offset = 0}}},
                                               .rank = 4,
                                               .dimensions = dimensionsOutput,
                                               .memType = QNN_TENSORMEMTYPE_RAW,
                                               .clientBuf = {.data = nullptr,
                                                             .dataSize = 0}}}}};
    return graphAddNode(name(), "LLaMAMul", {sigmoidName, inputs[0]->name()}, outputTensor, {}, "LLaMAPackage");
}

} // namespace mllm

