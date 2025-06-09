
#include "QNNDequantize.hpp"
#include "QnnTypes.h"
#include "Types.hpp"
#include "QNNCommonOp.hpp"
#include <cassert>

namespace mllm {
QNNDequantize::QNNDequantize(Backend *bn, string opName, bool isNSHD, bool isFP32, DataType type) :
    QNNCommonOp(bn, opName) {
    isNSHD_ = isNSHD;
    isFP32_ = isFP32;
    activation_dtype_ = type;
    scale_.setBackend(bn);
    bias_.setBackend(bn);
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
    // dequantScale = roundf(dequantScale * 100000) / 100000;

    // if (name().find("q_proj") != -1) {
    //     dequantScale = dequantScale / std::sqrt(outputs[0]->dimension());
    // }

    if (name().find("q_proj") != -1 || name().find("k_proj") != -1 || name().find("v_proj") != -1 ) {
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
            return graphAddNode(name(), "LLaMADequantizeAdd", {inputs[0]->name(), bias_.name()}, outputTensor, paramsDeQuantize, "LLaMAPackage");
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
            return graphAddNode(name(), "LLaMADequantizeAdd", {inputs[0]->name(), bias_.name()}, outputTensor, paramsDeQuantize, "LLaMAPackage");
        }
    } else {

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


    if (name().find("q_proj") != -1 || name().find("k_proj") != -1 || name().find("v_proj") != -1 ) {

        // std::cout << name() << std::endl;

        string biasName = name();
        wordToRemove = "dequantize";
        string biasTypeName = "bias";

        int pos = biasName.find(wordToRemove);
        if (pos != -1) {
            biasName.erase(pos, wordToRemove.length());
        }

        // std::cout << biasName + biasTypeName << std::endl;

        int hidden_size = 1536;
        if (name().find("k_proj") != -1 || name().find("v_proj") != -1 )
            hidden_size = 256;

        bias_.setName(biasName + biasTypeName);
        bias_.reshape(1, 1, 1, hidden_size);
        bias_.setDtype(MLLM_TYPE_F32);
        bias_.alloc();
        loader.load(&bias_);

        // bias_.printData<float>();

    }

    

    return Op::load(loader);
}
} // namespace mllm
