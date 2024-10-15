
#include "QNNTranspose.hpp"
#include "Types.hpp"
#include "QNNCommonOp.hpp"

namespace mllm {
QNNTranspose::QNNTranspose(Backend *bn, int perm0, int perm1, int perm2, int perm3, string opName) :
    QNNCommonOp(bn, opName) {
    perm[0] = perm0;
    perm[1] = perm1;
    perm[2] = perm2;
    perm[3] = perm3;
}

ErrorCode QNNTranspose::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    if (perm[0] == 0 && perm[1] == 2 && perm[2] == 3 && perm[3] == 1)
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->dimension(), inputs[0]->sequence());

    return Op::reshape(inputs, outputs);
}

ErrorCode QNNTranspose::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    if (getOutputTensorType(outputs[0]) == QNN_TENSOR_TYPE_APP_READ) {
        outputs[0]->setBackend(qnnBackend_);
        outputs[0]->setDtype(MLLM_TYPE_F32);
        outputs[0]->alloc();

        qnnBackend_->pushOutputBuffers(outputs[0]->hostPtr<uint8_t>());
    }

    uint32_t transposeParamsDimension[4] = {4};

    auto paramsTransposeName = name() + "transpose_params";
    vector<Qnn_Param_t> paramsTranspose = {
        {.paramType = QNN_PARAMTYPE_TENSOR,
         .name = "perm",
         .tensorParam =
             (Qnn_Tensor_t){.version = QNN_TENSOR_VERSION_1,
                            .v1 = {
                                .id = 0,
                                .name = paramsTransposeName.c_str(),
                                .type = QNN_TENSOR_TYPE_STATIC,
                                .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                .dataType = QNN_DATATYPE_UINT_32,
                                .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                   QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                   {.scaleOffsetEncoding = {.scale = 0.0000000000000000f,
                                                                            .offset = 0}}},
                                .rank = 1,
                                .dimensions = transposeParamsDimension,
                                .memType = QNN_TENSORMEMTYPE_RAW,
                                .clientBuf = {.data = (uint8_t *)perm,
                                              .dataSize = 4 * sizeof(uint32_t)}}}}};

    uint32_t dimVTranspose[4];
    dimVTranspose[0] = outputs[0]->batch();
    dimVTranspose[1] = outputs[0]->head();
    dimVTranspose[2] = outputs[0]->sequence();
    dimVTranspose[3] = outputs[0]->dimension();

    auto outVTransposeName = outputs[0]->name();
    vector<Qnn_Tensor_t> outVTranspose = {
        (Qnn_Tensor_t){
            .version = QNN_TENSOR_VERSION_1,
            .v1 = {
                .id = 0,
                .name = outVTransposeName.c_str(),
                .type = getOutputTensorType(outputs[0]),
                .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                .dataType = QNN_DATATYPE_FLOAT_32,
                .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                   QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                   {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                .rank = 4,
                .dimensions = dimVTranspose,
                .memType = QNN_TENSORMEMTYPE_RAW,
                .clientBuf = {.data = nullptr,
                              .dataSize = 0}}}};

    return graphAddNode(name() + ".v_transpose", "Transpose", {inputs[0]->name()}, outVTranspose, paramsTranspose);
}
} // namespace mllm
