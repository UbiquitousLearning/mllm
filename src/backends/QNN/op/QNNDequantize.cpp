
#include "QNNDequantize.hpp"
#include "QnnTypes.h"
#include "Types.hpp"
#include "QNNCommonOp.hpp"
#include <cassert>

namespace mllm {
QNNDequantize::QNNDequantize(Backend *bn, string opName) :
    QNNCommonOp(bn, opName) {
}

ErrorCode QNNDequantize::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(outputs.size() == 1);
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode QNNDequantize::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto outName = outputs[0]->name();
    uint32_t dimensionsOutput[4] = {static_cast<uint32_t>(outputs[0]->batch()),
                                    static_cast<uint32_t>(outputs[0]->sequence()),
                                    static_cast<uint32_t>(outputs[0]->head()),
                                    static_cast<uint32_t>(outputs[0]->dimension())};
    vector<Qnn_Tensor_t> outputTensor = {{QNN_TENSOR_VERSION_1,
                                          {.v1 = {
                                               .id = 0,
                                               .name = outName.c_str(),
                                               .type = getOutputTensorType(outputs[0]),
                                               .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                               .dataType = QNN_DATATYPE_FLOAT_32,
                                               .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                                  QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                                  {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                               .rank = 4,
                                               .dimensions = dimensionsOutput,
                                               .memType = QNN_TENSORMEMTYPE_RAW,
                                               {.clientBuf = {.data = nullptr,
                                                              .dataSize = 0}}}}}};
    return graphAddNode(name(), "Dequantize", {inputs[0]->name()}, outputTensor);
}
} // namespace mllm
