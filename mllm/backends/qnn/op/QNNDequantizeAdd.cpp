
#include "QNNDequantizeAdd.hpp"
#include "QNNActivationScaleOverride.hpp"
#include "QnnTypes.h"
#include "Types.hpp"
#include "QNNCommonOp.hpp"
#include "Context.hpp"
#include <cassert>

namespace mllm {
QNNDequantizeAdd::QNNDequantizeAdd(Backend *bn, string opName, bool isNSHD, int out_features, bool isFP32, DataType type) :
    QNNCommonOp(bn, opName) {
    isNSHD_ = isNSHD;
    isFP32_ = isFP32;
    out_features_ = out_features;
    activation_dtype_ = type;
    scale_.setBackend(Backend::global_backends[MLLM_CPU].get());
    bias_.setBackend(Backend::global_backends[MLLM_CPU].get());
}

ErrorCode QNNDequantizeAdd::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(outputs.size() == 1);
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode QNNDequantizeAdd::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
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
    switch (activation_dtype_) {
    case MLLM_TYPE_I8:
        dequantScale = scale_.hostPtr<float>()[0] / (pow(2, 7) - 1);
        break;
    case MLLM_TYPE_I16:
        dequantScale = scale_.hostPtr<float>()[0] / (pow(2, 15) - 1);
        break;
    default:
        return NOT_SUPPORT;
    }

    if (isFP32_) {
        uint32_t paramsDequantizeAddDimension[1] = {1};
        auto paramsDequantizeAddName = name() + "DequantizeAdd_params";

        vector<Qnn_Param_t> paramsDequantizeAdd = {
            {.paramType = QNN_PARAMTYPE_TENSOR,
             .name = "scale",
             .tensorParam =
                 (Qnn_Tensor_t){.version = QNN_TENSOR_VERSION_1,
                                .v1 = {
                                    .id = 0,
                                    .name = paramsDequantizeAddName.c_str(),
                                    .type = QNN_TENSOR_TYPE_STATIC,
                                    .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                    .dataType = QNN_DATATYPE_FLOAT_32,
                                    .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                       QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                       {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                    .rank = 1,
                                    .dimensions = paramsDequantizeAddDimension,
                                    .memType = QNN_TENSORMEMTYPE_RAW,
                                    .clientBuf = {.data = (uint8_t *)&dequantScale,
                                                  .dataSize = sizeof(float)}}}}};

        uint32_t dimensionsBias[4] = {1, 1, 1, static_cast<uint32_t>(bias_.dimension())};
        qnnBackend_->modelAddTensor(bias_.name(), (Qnn_Tensor_t){
                                                      .version = QNN_TENSOR_VERSION_1,
                                                      .v1 = {
                                                          .id = 0,
                                                          .name = bias_.name().c_str(),
                                                          .type = QNN_TENSOR_TYPE_STATIC,
                                                          .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                                          .dataType = QNN_DATATYPE_FLOAT_32,
                                                          .rank = 4,
                                                          .dimensions = dimensionsBias,
                                                          .memType = QNN_TENSORMEMTYPE_RAW,
                                                          .clientBuf = {.data = bias_.hostPtr<void>(),
                                                                        .dataSize = (uint32_t)bias_.cntSize()}}});

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
        return graphAddNode(name(), "LLaMADequantizeAdd", {inputs[0]->name(), bias_.name()}, outputTensor, paramsDequantizeAdd, "LLaMAPackage");
    } else {
        outputs[0]->setDtype(MLLM_TYPE_F16);
        uint32_t paramsDequantizeAddDimension[1] = {1};
        auto paramsDequantizeAddName = name() + "DequantizeAdd_params";

        vector<Qnn_Param_t> paramsDequantizeAdd = {
            {.paramType = QNN_PARAMTYPE_TENSOR,
             .name = "scale",
             .tensorParam =
                 (Qnn_Tensor_t){.version = QNN_TENSOR_VERSION_1,
                                .v1 = {
                                    .id = 0,
                                    .name = paramsDequantizeAddName.c_str(),
                                    .type = QNN_TENSOR_TYPE_STATIC,
                                    .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                    .dataType = QNN_DATATYPE_FLOAT_32,
                                    .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                       QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                       {.scaleOffsetEncoding = {.scale = 0.0000000000000000f,
                                                                                .offset = 0}}},
                                    .rank = 1,
                                    .dimensions = paramsDequantizeAddDimension,
                                    .memType = QNN_TENSORMEMTYPE_RAW,
                                    .clientBuf = {.data = (uint8_t *)&dequantScale,
                                                  .dataSize = sizeof(float)}}}}};

        uint32_t dimensionsBias[4] = {1, 1, 1, static_cast<uint32_t>(bias_.dimension())};
        qnnBackend_->modelAddTensor(bias_.name(), (Qnn_Tensor_t){
                                                      .version = QNN_TENSOR_VERSION_1,
                                                      .v1 = {
                                                          .id = 0,
                                                          .name = bias_.name().c_str(),
                                                          .type = QNN_TENSOR_TYPE_STATIC,
                                                          .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                                          .dataType = QNN_DATATYPE_FLOAT_32,
                                                          .rank = 4,
                                                          .dimensions = dimensionsBias,
                                                          .memType = QNN_TENSORMEMTYPE_RAW,
                                                          .clientBuf = {.data = bias_.hostPtr<void>(),
                                                                        .dataSize = (uint32_t)bias_.cntSize()}}});

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
        return graphAddNode(name(), "LLaMADequantizeAdd", {inputs[0]->name(), bias_.name()}, outputTensor, paramsDequantizeAdd, "LLaMAPackage");
    }
}

ErrorCode QNNDequantizeAdd::load(AbstructLoader &loader) {
    string scaleName = name();
    string scaleTypeName = "output_scale";

    std::string wordToRemove = "dequantize";
    int pos = scaleName.find(wordToRemove);
    if (pos != -1) {
        scaleName.erase(pos, wordToRemove.length());
    }

    scale_.setName(scaleName + scaleTypeName);
    scale_.reshape(1, 1, 1, 1);
    scale_.setDtype(MLLM_TYPE_F32);
    scale_.alloc();
    loader.load(&scale_);
    if (const auto value = qnnActivationScaleOverride(
            scale_.name(), scale_.hostPtr<float>()[0])) {
        MLLM_LOG_INFO_STREAM << "ACTIVATION_SCALE_OVERRIDE tensor="
                             << scale_.name() << " old="
                             << scale_.hostPtr<float>()[0] << " new="
                             << *value << std::endl;
        qnnSetPrivateActivationScale(scale_, *value);
    }

    string biasName = name();
    wordToRemove = "dequantize";
    string biasTypeName = "bias";

    pos = biasName.find(wordToRemove);
    if (pos != -1) {
        biasName.erase(pos, wordToRemove.length());
    }

    bias_.setName(biasName + biasTypeName);
    bias_.reshape(1, 1, 1, out_features_);
    bias_.setDtype(MLLM_TYPE_F32);
    bias_.alloc();

    const DataType stored_bias_dtype = loader.getDataType(bias_.name());
    if (stored_bias_dtype == MLLM_TYPE_I8
        || stored_bias_dtype == MLLM_TYPE_I32) {
        Tensor quantized_bias(Backend::global_backends[MLLM_CPU].get());
        quantized_bias.setName(bias_.name());
        quantized_bias.reshape(1, 1, 1, out_features_);
        quantized_bias.setDtype(stored_bias_dtype);
        quantized_bias.alloc();
        if (!loader.load(&quantized_bias)) return ::INVALID_VALUE;

        Tensor bias_scale(Backend::global_backends[MLLM_CPU].get());
        bias_scale.setName(bias_.name() + ".scale");
        bias_scale.reshape(1, 1, 1, 1);
        bias_scale.setDtype(MLLM_TYPE_F32);
        bias_scale.alloc();
        if (!loader.load(&bias_scale)) return ::INVALID_VALUE;

        const float scale = bias_scale.hostPtr<float>()[0];
        for (int i = 0; i < out_features_; ++i) {
            const float value = stored_bias_dtype == MLLM_TYPE_I8
                ? static_cast<float>(
                      quantized_bias.dataAt<int8_t>(0, 0, 0, i))
                : static_cast<float>(
                      quantized_bias.dataAt<int32_t>(0, 0, 0, i));
            bias_.setDataAt<float>(0, 0, 0, i, value * scale);
        }
    } else if (!loader.load(&bias_)) {
        return ::INVALID_VALUE;
    }

    return Op::load(loader);
}
} // namespace mllm
