
#include "QNNRMSNorm.hpp"
#include "Types.hpp"
#include "QNNCommonOp.hpp"
#include <cstdint>

namespace mllm {
QNNRMSNorm::QNNRMSNorm(Backend *bn, string opName) :
    QNNCommonOp(bn, opName) {
    weight_.setBackend(bn);
}

ErrorCode QNNRMSNorm::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    normSize_ = inputs[0]->dimension();
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->shape(1), inputs[0]->shape(2), inputs[0]->shape(3));
    return Op::reshape(inputs, outputs);
}

ErrorCode QNNRMSNorm::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto inName = inputs[0]->name();
    auto weightName = weight_.name();
    uint32_t dimWeight[4] = {(uint32_t)normSize_};
    qnnBackend_->modelAddTensor(weightName.c_str(), (Qnn_Tensor_t){
                                                        .version = QNN_TENSOR_VERSION_1,
                                                        {.v1 = {
                                                             .id = 0,
                                                             .name = weightName.c_str(),
                                                             .type = QNN_TENSOR_TYPE_STATIC,
                                                             .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                                             .dataType = QNN_DATATYPE_FLOAT_32,
                                                             .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                                                QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                                                {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                                             .rank = 1,
                                                             .dimensions = dimWeight,
                                                             .memType = QNN_TENSORMEMTYPE_RAW,
                                                             {.clientBuf = {.data = weight_.hostPtr<void>(),
                                                                            .dataSize = static_cast<uint32_t>(weight_.cntSize())}}}}});
    auto outString = outputs[0]->name();
    uint32_t dimOut[] = {(uint32_t)outputs[0]->batch(), (uint32_t)outputs[0]->head(), (uint32_t)outputs[0]->sequence(), (uint32_t)outputs[0]->dimension()};
    vector<Qnn_Tensor_t>
        out = {
            (Qnn_Tensor_t){
                .version = QNN_TENSOR_VERSION_1,
                {.v1 = {
                     .id = 0,
                     .name = outString.c_str(),
                     .type = getOutputTensorType(outputs[0]),
                     .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                     .dataType = QNN_DATATYPE_FLOAT_32,
                     .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                        QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                        {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                     .rank = 4,
                     .dimensions = dimOut,
                     .memType = QNN_TENSORMEMTYPE_RAW,
                     {.clientBuf = {.data = nullptr,
                                    .dataSize = 0}}}}}};
    return graphAddNode(name(), "RMSNorm", {inName.c_str(), weightName.c_str()}, out, {}, "LLaMAPackage");
}

ErrorCode QNNRMSNorm::load(AbstructLoader &loader) {
    weight_.setName(name() + ".weight");
    weight_.reshape(1, 1, 1, normSize_);
    weight_.setDtype(loader.getDataType(weight_.name()));
    weight_.alloc();
    loader.load(&weight_);
    return Op::load(loader);
}
} // namespace mllm
