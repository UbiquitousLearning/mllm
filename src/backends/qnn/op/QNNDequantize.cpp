
#include "QNNDequantize.hpp"
#include "QnnTypes.h"
#include "Types.hpp"
#include "QNNCommonOp.hpp"
#include <cassert>

namespace mllm {
QNNDequantize::QNNDequantize(Backend *bn, string opName, bool isNSHD, bool isFP32) :
    QNNCommonOp(bn, opName) {
    isNSHD_ = isNSHD;
    isFP32_ = isFP32;
    scale_.setBackend(bn);
}

ErrorCode QNNDequantize::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(outputs.size() == 1);
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode QNNDequantize::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {

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

    float dequantScale = 0;
    dequantScale = scale_.hostPtr<float>()[0] / 127.0;
    dequantScale = roundf(dequantScale * 100000) / 100000;

    if (name().find("q_proj") != -1) {
        dequantScale = dequantScale / std::sqrt(outputs[0]->dimension());
    }

    if (isFP32_) {
        uint32_t paramsDeQuantizeDimension[1] = {1};
        auto paramsDeQuantizeName = name() + "dequantize_params";
        vector<Qnn_Param_t> paramsDeQuantize = {
            {.paramType = QNN_PARAMTYPE_TENSOR,
             .name = "scale",
             .tensorParam =
                 (Qnn_Tensor_t){.version = QNN_TENSOR_VERSION_1,
                                .v1 = {
                                    .id = 0,
                                    .name = paramsDeQuantizeName.c_str(),
                                    .type = QNN_TENSOR_TYPE_STATIC,
                                    .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                    .dataType = QNN_DATATYPE_FLOAT_32,
                                    .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                       QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                       {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                    .rank = 1,
                                    .dimensions = paramsDeQuantizeDimension,
                                    .memType = QNN_TENSORMEMTYPE_RAW,
                                    .clientBuf = {.data = (uint8_t *)&dequantScale,
                                                  .dataSize = sizeof(float)}}}}};

        vector<Qnn_Tensor_t> outputTensor = {{.version = QNN_TENSOR_VERSION_1,
                                              .v1 = {
                                                  .id = 0,
                                                  .name = outName.c_str(),
                                                  .type = getOutputTensorType(outputs[0]),
                                                  .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                                  .dataType = QNN_DATATYPE_FLOAT_32,
                                                  .quantizeParams = {QNN_DEFINITION_DEFINED,
                                                                     QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                                     {.scaleOffsetEncoding = {.scale = dequantScale, .offset = 0}}},
                                                  .rank = 4,
                                                  .dimensions = dimensionsOutput,
                                                  .memType = QNN_TENSORMEMTYPE_RAW,
                                                  .clientBuf = {.data = nullptr,
                                                                .dataSize = 0}}}};
        return graphAddNode(name(), "LLaMADequantize", {inputs[0]->name()}, outputTensor, paramsDeQuantize, "LLaMAPackage");
    } else {
        outputs[0]->setDtype(MLLM_TYPE_F16);
        uint32_t paramsDeQuantizeDimension[1] = {1};
        auto paramsDeQuantizeName = name() + "dequantize_params";
        vector<Qnn_Param_t> paramsDeQuantize = {
            {.paramType = QNN_PARAMTYPE_TENSOR,
             .name = "scale",
             .tensorParam =
                 (Qnn_Tensor_t){.version = QNN_TENSOR_VERSION_1,
                                .v1 = {
                                    .id = 0,
                                    .name = paramsDeQuantizeName.c_str(),
                                    .type = QNN_TENSOR_TYPE_STATIC,
                                    .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                    .dataType = QNN_DATATYPE_FLOAT_32,
                                    .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                       QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                       {.scaleOffsetEncoding = {.scale = 0.0000000000000000f,
                                                                                .offset = 0}}},
                                    .rank = 1,
                                    .dimensions = paramsDeQuantizeDimension,
                                    .memType = QNN_TENSORMEMTYPE_RAW,
                                    .clientBuf = {.data = (uint8_t *)&dequantScale,
                                                  .dataSize = sizeof(float)}}}}};

        vector<Qnn_Tensor_t> outputTensor = {{QNN_TENSOR_VERSION_1,
                                              {.v1 = {
                                                   .id = 0,
                                                   .name = outName.c_str(),
                                                   .type = getOutputTensorType(outputs[0]),
                                                   .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                                   .dataType = QNN_DATATYPE_FLOAT_16,
                                                   .quantizeParams = {QNN_DEFINITION_DEFINED,
                                                                      QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                                      {.scaleOffsetEncoding = {.scale = dequantScale, .offset = 0}}},
                                                   .rank = 4,
                                                   .dimensions = dimensionsOutput,
                                                   .memType = QNN_TENSORMEMTYPE_RAW,
                                                   .clientBuf = {.data = nullptr,
                                                                 .dataSize = 0}}}}};
        return graphAddNode(name(), "LLaMADequantize", {inputs[0]->name()}, outputTensor, paramsDeQuantize, "LLaMAPackage");
    }
}

ErrorCode QNNDequantize::load(AbstructLoader &loader) {
    string scaleName = name();
    string scaleTypeName = "output_scale";

    std::string wordToRemove = "dequantize";
    int pos = scaleName.find(wordToRemove);
    if (pos != -1) {
        scaleName.erase(pos, wordToRemove.length());
    }

    wordToRemove = ".x.";
    pos = scaleName.find(wordToRemove);
    if (pos != -1) {
        scaleName.erase(pos, wordToRemove.length());
        scaleTypeName = ".q_proj.input_scale";
    }

    scale_.setName(scaleName + scaleTypeName);
    scale_.reshape(1, 1, 1, 1);
    scale_.setDtype(MLLM_TYPE_F32);
    scale_.alloc();
    loader.load(&scale_);

    return Op::load(loader);
}
} // namespace mllm
