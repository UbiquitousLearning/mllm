
#include "QNNMul.hpp"
#include "Types.hpp"
#include "QNNCommonOp.hpp"

namespace mllm {
QNNMul::QNNMul(Backend *bn, string opName) :
    QNNCommonOp(bn, opName) {
}

ErrorCode QNNMul::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 2);
    assert(outputs.size() == 1);
    assert(inputs[0]->batch() == inputs[1]->batch());
    assert(inputs[0]->head() == inputs[1]->head());
    assert(inputs[0]->sequence() == inputs[1]->sequence());
    assert(inputs[0]->dimension() == inputs[1]->dimension());
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode QNNMul::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {

    auto outName = outputs[0]->name();

    uint32_t dimensionsOutput[4];


    dimensionsOutput[0] = static_cast<uint32_t>(outputs[0]->batch());
    dimensionsOutput[1] = static_cast<uint32_t>(outputs[0]->sequence());
    dimensionsOutput[2] = static_cast<uint32_t>(outputs[0]->head());
    dimensionsOutput[3] = static_cast<uint32_t>(outputs[0]->dimension());

    vector<Qnn_Tensor_t> outputTensor = {{QNN_TENSOR_VERSION_1,
                                          {.v1 = {
                                               .id = 0,
                                               .name = outName.c_str(),
                                               .type = getOutputTensorType(outputs[0]),
                                               .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                               .dataType = QNN_DATATYPE_FLOAT_16,
                                               .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding = {.scale  = 0.0000000000000000f,
                                                                            .offset = 0}}},
                                               .rank = 4,
                                               .dimensions = dimensionsOutput,
                                               .memType = QNN_TENSORMEMTYPE_RAW,
                                               {.clientBuf = {.data = nullptr,
                                                              .dataSize = 0}}}}}};
    return graphAddNode(name(), "LLaMAMul", {inputs[0]->name(), inputs[1]->name()}, outputTensor, {}, "LLaMAPackage");


    // return graphAddNode(name(), "LLaMAMul", inputs, outputs, {}, "LLaMAPackage");
    // return graphAddNode(name(), "ElementWiseMul", inputs, outputs, {});
}
} // namespace mllm

