
#include "QNNQuantize.hpp"
#include "QnnTypes.h"
#include "Types.hpp"
#include "QNNCommonOp.hpp"
#include <cassert>
#include <cmath>

namespace mllm {
QNNQuantize::QNNQuantize(Backend *bn, string opName, DataType type, bool isNSHD) :
    QNNCommonOp(bn, opName) {
    isNSHD_ = isNSHD;
    assert(type == MLLM_TYPE_I8 || type == MLLM_TYPE_I16);
    activation_dtype_ = type;
    scale_.setBackend(bn);
}

ErrorCode QNNQuantize::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(outputs.size() == 1);
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode QNNQuantize::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    switch (activation_dtype_) {
    case MLLM_TYPE_I8:
        return setUpI8(inputs, outputs);
    case MLLM_TYPE_I16:
        return setUpI16(inputs, outputs);
    default:
        return NOT_SUPPORT;
    }
}

ErrorCode QNNQuantize::setUpI8(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    outputs[0]->setDtype(MLLM_TYPE_I8);
    auto outName = outputs[0]->name();

    uint32_t dimensionsOutput[4];

    if (isNSHD_) {
        dimensionsOutput[0] = static_cast<uint32_t>(outputs[0]->batch());
        dimensionsOutput[1] = static_cast<uint32_t>(outputs[0]->sequence());
        dimensionsOutput[2] = static_cast<uint32_t>(outputs[0]->head());
        dimensionsOutput[3] = static_cast<uint32_t>(outputs[0]->dimension());
    } else {
        dimensionsOutput[0] = static_cast<uint32_t>(outputs[0]->batch());
        dimensionsOutput[1] = static_cast<uint32_t>(outputs[0]->head());
        dimensionsOutput[2] = static_cast<uint32_t>(outputs[0]->sequence());
        dimensionsOutput[3] = static_cast<uint32_t>(outputs[0]->dimension());
    }

    float quantScale = 0;
    quantScale = scale_.hostPtr<float>()[0] / (pow(2, 7) - 1);
    // quantScale = roundf(quantScale * 100000) / 100000;

    uint32_t paramsQuantizeDimension[1] = {1};
    auto paramsQuantizeName = name() + "quantize_params";
    vector<Qnn_Param_t> paramsQuantize = {
        {.paramType = QNN_PARAMTYPE_TENSOR,
         .name = "scale",
         .tensorParam =
             (Qnn_Tensor_t){.version = QNN_TENSOR_VERSION_1,
                            .v1 = {
                                .id = 0,
                                .name = paramsQuantizeName.c_str(),
                                .type = QNN_TENSOR_TYPE_STATIC,
                                .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                .dataType = QNN_DATATYPE_FLOAT_32,
                                .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                   QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                   {.scaleOffsetEncoding = {.scale = 0.0000000000000000f,
                                                                            .offset = 0}}},
                                .rank = 1,
                                .dimensions = paramsQuantizeDimension,
                                .memType = QNN_TENSORMEMTYPE_RAW,
                                .clientBuf = {.data = (uint8_t *)&quantScale,
                                              .dataSize = sizeof(float)}}}}};

    vector<Qnn_Tensor_t> outputTensor = {{QNN_TENSOR_VERSION_1,
                                          {.v1 = {
                                               .id = 0,
                                               .name = outName.c_str(),
                                               .type = getOutputTensorType(outputs[0]),
                                               .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                               .dataType = QNN_DATATYPE_SFIXED_POINT_8,
                                               .quantizeParams = {QNN_DEFINITION_DEFINED,
                                                                  QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                                  {.scaleOffsetEncoding = {.scale = quantScale, .offset = 0}}},
                                               .rank = 4,
                                               .dimensions = dimensionsOutput,
                                               .memType = QNN_TENSORMEMTYPE_RAW,
                                               .clientBuf = {.data = nullptr,
                                                             .dataSize = 0}}}}};
    return graphAddNode(name(), "LLaMAQuantize", {inputs[0]->name()}, outputTensor, paramsQuantize, "LLaMAPackage");
}

ErrorCode QNNQuantize::setUpI16(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    outputs[0]->setDtype(MLLM_TYPE_I16);
    auto outName = outputs[0]->name();

    uint32_t dimensionsOutput[4];

    if (isNSHD_) {
        dimensionsOutput[0] = static_cast<uint32_t>(outputs[0]->batch());
        dimensionsOutput[1] = static_cast<uint32_t>(outputs[0]->sequence());
        dimensionsOutput[2] = static_cast<uint32_t>(outputs[0]->head());
        dimensionsOutput[3] = static_cast<uint32_t>(outputs[0]->dimension());
    } else {
        dimensionsOutput[0] = static_cast<uint32_t>(outputs[0]->batch());
        dimensionsOutput[1] = static_cast<uint32_t>(outputs[0]->head());
        dimensionsOutput[2] = static_cast<uint32_t>(outputs[0]->sequence());
        dimensionsOutput[3] = static_cast<uint32_t>(outputs[0]->dimension());
    }

    float quantScale = 0;
    quantScale = scale_.hostPtr<float>()[0] / (pow(2, 15) - 1);
    // quantScale = roundf(quantScale * 100000) / 100000;

    uint32_t paramsQuantizeDimension[1] = {1};
    auto paramsQuantizeName = name() + "quantize_params";
    vector<Qnn_Param_t> paramsQuantize = {
        {.paramType = QNN_PARAMTYPE_TENSOR,
         .name = "scale",
         .tensorParam =
             (Qnn_Tensor_t){.version = QNN_TENSOR_VERSION_1,
                            .v1 = {
                                .id = 0,
                                .name = paramsQuantizeName.c_str(),
                                .type = QNN_TENSOR_TYPE_STATIC,
                                .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                .dataType = QNN_DATATYPE_FLOAT_32,
                                .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                   QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                   {.scaleOffsetEncoding = {.scale = 0.0000000000000000f,
                                                                            .offset = 0}}},
                                .rank = 1,
                                .dimensions = paramsQuantizeDimension,
                                .memType = QNN_TENSORMEMTYPE_RAW,
                                .clientBuf = {.data = (uint8_t *)&quantScale,
                                              .dataSize = sizeof(float)}}}}};

    vector<Qnn_Tensor_t> outputTensor = {{QNN_TENSOR_VERSION_1,
                                          {.v1 = {
                                               .id = 0,
                                               .name = outName.c_str(),
                                               .type = getOutputTensorType(outputs[0]),
                                               .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                               .dataType = QNN_DATATYPE_SFIXED_POINT_16,
                                               .quantizeParams = {QNN_DEFINITION_DEFINED,
                                                                  QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                                  {.scaleOffsetEncoding = {.scale = quantScale, .offset = 0}}},
                                               .rank = 4,
                                               .dimensions = dimensionsOutput,
                                               .memType = QNN_TENSORMEMTYPE_RAW,
                                               .clientBuf = {.data = nullptr,
                                                             .dataSize = 0}}}}};
    return graphAddNode(name(), "LLaMAQuantize", {inputs[0]->name()}, outputTensor, paramsQuantize, "LLaMAPackage");
}

ErrorCode QNNQuantize::load(AbstructLoader &loader) {
    string scaleName = name();

    std::string wordToRemove = "quantize";
    int pos = scaleName.find(wordToRemove);
    if (pos != -1) {
        scaleName.erase(pos, wordToRemove.length());
    }

    scale_.setName(scaleName + "input_scale");
    scale_.reshape(1, 1, 1, 1);
    scale_.setDtype(MLLM_TYPE_F32);
    scale_.alloc();
    loader.load(&scale_);

    return Op::load(loader);
}
} // namespace mllm
