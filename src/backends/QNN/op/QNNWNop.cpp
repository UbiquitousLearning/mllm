
#include "QNNWNop.hpp"

namespace mllm {

QNNWNop::QNNWNop(Backend *bn,  string opName, int sync_type) : sync_type_(sync_type),
    QNNCommonOp(bn, opName) {
}

ErrorCode QNNWNop::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {

    assert(inputs.size() == 1);
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    
    return Op::reshape(inputs, outputs);
}

ErrorCode QNNWNop::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {

    // sync logic.
    std::cout << "sync now." << std::endl;
    return Op::execute(inputs, outputs);
}

ErrorCode QNNWNop::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {

    outputs[0]->setDtype(MLLM_TYPE_F32);
    outputs[0]->setBackend(qnnBackend_);
    outputs[0]->alloc();

    // output for net shape and QNN name index
    // cache_ for QNN shared buffer storage
    qnnBackend_->pushOutputBuffers(outputs[0]->hostPtr<uint8_t>());

    syncVar_.setBackend(qnnBackend_);
    syncVar_.reshape(1,1,1,1);
    syncVar_.setDtype(MLLM_TYPE_I32);
    syncVar_.alloc();
    syncVar_.setDataAt<uint32_t>(0, 0, 0, 0, 0);

    qnnBackend_->pushOutputBuffers(syncVar_.hostPtr<uint8_t>());


    vector<Qnn_Param_t> paramsWNop = {
        {.paramType = QNN_PARAMTYPE_SCALAR,
         .name = "sync_type",
         {.scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_UINT_32, {.uint32Value = static_cast<uint32_t>(sync_type_)}}}},};


    uint32_t dimensionsActivation[4] = {static_cast<uint32_t>(outputs[0]->batch()),
                                   static_cast<uint32_t>(outputs[0]->sequence()),
                                   static_cast<uint32_t>(outputs[0]->head()),
                                   static_cast<uint32_t>(outputs[0]->dimension())};

    auto outName = outputs[0]->name();

    uint32_t dimensionsSync[4] = {1, 1, 1, 1};
    auto outSyncName = outputs[0]->name() + ".sync";

    vector<Qnn_Tensor_t> activation_output = {
                                            (Qnn_Tensor_t){QNN_TENSOR_VERSION_1,
                                            {.v1 = {
                                                 .id = 0,
                                                 .name = outName.c_str(),
                                                 .type = QNN_TENSOR_TYPE_APP_READ,
                                                 .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                                 .dataType = QNN_DATATYPE_FLOAT_32,
                                                 .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                                    {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                                 .rank = 4,
                                                 .dimensions = dimensionsActivation,
                                                 .memType = QNN_TENSORMEMTYPE_RAW,
                                                 {.clientBuf = {.data = nullptr,
                                                                .dataSize = 0}}}}},            
                                            (Qnn_Tensor_t){QNN_TENSOR_VERSION_1,
                                            {.v1 = {
                                                 .id = 1,
                                                 .name = outSyncName.c_str(),
                                                 .type = QNN_TENSOR_TYPE_APP_READ,
                                                 .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                                 .dataType = QNN_DATATYPE_UINT_32,
                                                 .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                                    {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                                 .rank = 4,
                                                 .dimensions = dimensionsSync,
                                                 .memType = QNN_TENSORMEMTYPE_RAW,
                                                 {.clientBuf = {.data = nullptr,
                                                                .dataSize = 0}}}}} };

    


    return graphAddNode(name(), "WNop", {inputs[0]->name()}, activation_output, paramsWNop, "LLaMAPackage");
}

} // namespace mllm
