
#include "QNNRMSNorm.hpp"
#include "Types.hpp"
#include "QNNCommonOp.hpp"
#include <cstdint>

namespace mllm {
QNNRMSNorm::QNNRMSNorm(Backend *bn, string opName, int normSize, float epsilon, bool isFP32) :
    QNNCommonOp(bn, opName), normSize_(normSize), epsilon_(epsilon), isFP32_(isFP32) {
    weight_.setBackend(bn);
    scale_.setBackend(bn);
}

ErrorCode QNNRMSNorm::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    normSize_ = inputs[0]->dimension();
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode QNNRMSNorm::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    float quantScale = 0;
    quantScale = scale_.hostPtr<float>()[0] / 127.0;
    // quantScale = roundf(quantScale * 100000) / 100000;

    uint32_t dimWeight[4] = {(uint32_t)normSize_};
    qnnBackend_->modelAddTensor(weight_.name(), (Qnn_Tensor_t){
                                                    .version = QNN_TENSOR_VERSION_1,
                                                    .v1 = {
                                                        .id = 0,
                                                        .name = weight_.name().c_str(),
                                                        .type = QNN_TENSOR_TYPE_STATIC,
                                                        .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                                        .dataType = QNN_DATATYPE_FLOAT_32,
                                                        .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                                           QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                                           {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                                        .rank = 1,
                                                        .dimensions = dimWeight,
                                                        .memType = QNN_TENSORMEMTYPE_RAW,
                                                        .clientBuf = {.data = weight_.hostPtr<void>(),
                                                                      .dataSize = static_cast<uint32_t>(weight_.cntSize())}}});
    // free weight_
    weight_.free();

    uint32_t dimOut[] = {(uint32_t)outputs[0]->batch(), (uint32_t)outputs[0]->sequence(), (uint32_t)outputs[0]->head(), (uint32_t)outputs[0]->dimension()};
    auto outName = outputs[0]->name();

    if (isFP32_) {
        outputs[0]->setDtype(MLLM_TYPE_F32);
        vector<Qnn_Tensor_t>
            out = {
                (Qnn_Tensor_t){
                    .version = QNN_TENSOR_VERSION_1,
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
                        .dimensions = dimOut,
                        .memType = QNN_TENSORMEMTYPE_RAW,
                        .clientBuf = {.data = nullptr,
                                      .dataSize = 0}}}};
        return graphAddNode(name(), "RMSNorm", {inputs[0]->name(), weight_.name()}, out, {}, "LLaMAPackage");

    } else {
        outputs[0]->setDtype(MLLM_TYPE_I8);
        vector<Qnn_Tensor_t>
            out = {
                (Qnn_Tensor_t){
                    .version = QNN_TENSOR_VERSION_1,
                    .v1 = {
                        .id = 0,
                        .name = outName.c_str(),
                        .type = getOutputTensorType(outputs[0]),
                        .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                        .dataType = QNN_DATATYPE_SFIXED_POINT_8,
                        .quantizeParams = {QNN_DEFINITION_DEFINED,
                                           QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                           {.scaleOffsetEncoding = {.scale = quantScale, .offset = 0}}},
                        .rank = 4,
                        .dimensions = dimOut,
                        .memType = QNN_TENSORMEMTYPE_RAW,
                        .clientBuf = {.data = nullptr,
                                      .dataSize = 0}}}};
        return graphAddNode(name(), "RMSNorm", {inputs[0]->name(), weight_.name()}, out, {}, "LLaMAPackage");
    }
}

ErrorCode QNNRMSNorm::load(AbstructLoader &loader) {
    weight_.setName(name() + ".weight");
    weight_.reshape(1, 1, 1, normSize_);
    if (loader.getDataType(weight_.name()) != MLLM_TYPE_COUNT) {
        weight_.setDtype(loader.getDataType(weight_.name()));
        weight_.alloc();
        // auto l = loader.length(weight_.name());
        loader.load(&weight_);
    } else {
        weight_.setDtype(MLLM_TYPE_F32);
        weight_.alloc();
    }

    string scaleName = name();

    std::string wordToRemove = "post_attention_layernorm";
    int pos = scaleName.find(wordToRemove);
    if (pos != -1) {
        scaleName.erase(pos, wordToRemove.length());
    }

    scale_.setName(scaleName + "mlp.up_proj.input_scale");
    scale_.reshape(1, 1, 1, 1);
    scale_.setDtype(MLLM_TYPE_F32);
    scale_.alloc();
    loader.load(&scale_);

    return Op::load(loader);
}
} // namespace mllm
