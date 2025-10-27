#include "QNNSiLU.hpp"
#include "Types.hpp"
#include "QNNCommonOp.hpp"

namespace mllm {
QNNSiLU::QNNSiLU(Backend *bn, string opName) :
    QNNCommonOp(bn, opName) {
}

ErrorCode QNNSiLU::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode QNNSiLU::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto outName = outputs[0]->name();

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
    return graphAddNode(name(), "SiLU", {inputs[0]->name()}, outputTensor, {}, "LLaMAPackage");
}
} // namespace mllm