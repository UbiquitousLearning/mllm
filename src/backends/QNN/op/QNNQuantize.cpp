
#include "QNNQuantize.hpp"
#include "QnnTypes.h"
#include "Types.hpp"
#include "QNNCommonOp.hpp"
#include <cassert>

namespace mllm {
QNNQuantize::QNNQuantize(Backend *bn, string opName, bool isNSHD) :
    QNNCommonOp(bn, opName) {
        isNSHD_ = isNSHD;
}

ErrorCode QNNQuantize::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(outputs.size() == 1);
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode QNNQuantize::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
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
    vector<Qnn_Tensor_t> outputTensor = {{QNN_TENSOR_VERSION_1,
                                          {.v1 = {
                                               .id = 0,
                                               .name = outName.c_str(),
                                               .type = getOutputTensorType(outputs[0]),
                                               .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                               .dataType = QNN_DATATYPE_UFIXED_POINT_8,
                                               .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                                  QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                                  {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                               .rank = 4,
                                               .dimensions = dimensionsOutput,
                                               .memType = QNN_TENSORMEMTYPE_RAW,
                                               {.clientBuf = {.data = nullptr,
                                                              .dataSize = 0}}}}}};
    return graphAddNode(name(), "Quantize", {inputs[0]->name()}, outputTensor);
}
} // namespace mllm
