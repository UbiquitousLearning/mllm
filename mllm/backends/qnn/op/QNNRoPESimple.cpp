
#include "QNNRoPESimple.hpp"
#include "Types.hpp"
#include "QNNCommonOp.hpp"
#include <cstdint>

namespace mllm {

QNNRoPESimple::QNNRoPESimple(Backend *bn, string opName) :
    QNNCommonOp(bn, opName) {
}

ErrorCode QNNRoPESimple::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 3);
    assert(outputs.size() == 1);
    int partial_dimension = inputs[0]->dimension();
    assert(partial_dimension % 2 == 0);

    assert(inputs[0]->batch() == inputs[1]->batch() && inputs[0]->batch() == inputs[2]->batch());
    assert(inputs[0]->head() == inputs[1]->head() && inputs[0]->head() == inputs[2]->head());
    assert(inputs[0]->sequence() == inputs[1]->sequence() && inputs[0]->sequence() == inputs[2]->sequence());
    assert(inputs[0]->dimension() == inputs[1]->dimension() * 2 && inputs[0]->dimension() == inputs[2]->dimension() * 2);

    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());

    return Op::reshape(inputs, outputs);
}

ErrorCode QNNRoPESimple::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto type = QNN_DATATYPE_FLOAT_32;
    if (inputs[0]->dtype() == MLLM_TYPE_F16) {
        type = QNN_DATATYPE_FLOAT_16;
        outputs[0]->setDtype(MLLM_TYPE_F16);
    } else if (inputs[0]->dtype() == MLLM_TYPE_F32) {
        type = QNN_DATATYPE_FLOAT_32;
        outputs[0]->setDtype(MLLM_TYPE_F32);
    } else {
        return ErrorCode::NOT_SUPPORT;
    }

    uint32_t dimOut[4] = {static_cast<uint32_t>(inputs[0]->batch()),
                          static_cast<uint32_t>(inputs[0]->sequence()),
                          static_cast<uint32_t>(inputs[0]->head()),
                          static_cast<uint32_t>(inputs[0]->dimension())};
    auto outName = outputs[0]->name();
    vector<Qnn_Tensor_t> out = {
        (Qnn_Tensor_t){
            .version = QNN_TENSOR_VERSION_1,
            .v1 = {
                .id = 0,
                .name = outName.c_str(),
                .type = getOutputTensorType(outputs[0]),
                .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                .dataType = type,
                .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                   QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                   {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                .rank = 4,
                .dimensions = dimOut,
                .memType = QNN_TENSORMEMTYPE_RAW,
                .clientBuf = {.data = nullptr,
                              .dataSize = 0}}}};

    return graphAddNode(name(), "RoPESimple", {inputs[0]->name(), inputs[1]->name(), inputs[2]->name()}, out, {}, "LLaMAPackage");
}

} // namespace mllm
