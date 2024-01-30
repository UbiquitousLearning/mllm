#include "QNNAdd.hpp"
#include "Types.hpp"
#include "QNNCommonOp.hpp"

namespace mllm {
QNNAdd::QNNAdd(Backend *bn, string opName) :
    QNNCommonOp(bn, opName) {
}

ErrorCode QNNAdd::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 2);
    assert(outputs.size() == 1);
    if (inputs[0]->batch() == 1 || inputs[1]->batch() == 1) {
    } else {
        assert(inputs[0]->batch() == inputs[1]->batch());
    }
    assert(inputs[0]->head() == inputs[1]->head());
    assert(inputs[0]->sequence() == inputs[1]->sequence());
    assert(inputs[0]->dimension() == inputs[1]->dimension());

    outputs[0]->reshape(std::max(inputs[0]->batch(), inputs[1]->batch()), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());

    return Op::reshape(inputs, outputs);
}

ErrorCode QNNAdd::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // graph add node
    auto inputName0 = inputs[0]->name();
    auto inputName1 = inputs[1]->name();
    auto outString = outputs[0]->name();
    uint32_t dimensions[4];
    for (int i = 0; i < outputs[0]->shape().size(); i++) {
        dimensions[i] = outputs[0]->shape()[i];
    }
    vector<Qnn_Tensor_t> out = {{QNN_TENSOR_VERSION_1,
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
                                      .dimensions = dimensions,
                                      .memType = QNN_TENSORMEMTYPE_RAW,
                                      {.clientBuf = {.data = nullptr,
                                                     .dataSize = 0}}}}}};
    return graphAddNode(name(), "ElementWiseAdd", {inputName0, inputName1}, out);
}
} // namespace mllm