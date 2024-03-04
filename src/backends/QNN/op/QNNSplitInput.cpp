
#include "QNNSplitInput.hpp"
#include "QnnTypes.h"
#include "Types.hpp"
#include "QNNCommonOp.hpp"
#include <cstdint>
#include <memory>

namespace mllm {
QNNSplitInput::QNNSplitInput(Backend *bn, string opName, bool isPrompt) :
    QNNCommonOp(bn, opName) {
    
    isPrompt_ = isPrompt;

}

ErrorCode QNNSplitInput::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 3);

    if (isPrompt_) {

        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence() / 3, inputs[0]->dimension());
        outputs[1]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence() / 3, inputs[0]->dimension());
        outputs[2]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence() / 3, inputs[0]->dimension());

    } else {

        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), 1, inputs[0]->dimension());
        outputs[1]->reshape(inputs[0]->batch(), inputs[0]->head(), (inputs[0]->sequence() - 1) / 2, inputs[0]->dimension());
        outputs[2]->reshape(inputs[0]->batch(), inputs[0]->head(), (inputs[0]->sequence() - 1) / 2, inputs[0]->dimension());

    }
    

    return Op::reshape(inputs, outputs);
}

ErrorCode QNNSplitInput::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {  

    seqs_.setName(name() + ".seqs");
    seqs_.reshape(1, 1, 1, 3);
    seqs_.setDtype(MLLM_TYPE_I32);
    seqs_.setBackend(qnnBackend_);
    seqs_.alloc();

    seqs_.setDataAt<uint32_t>(0, 0, 0, 0, outputs[0]->sequence());
    seqs_.setDataAt<uint32_t>(0, 0, 0, 1, outputs[1]->sequence());
    seqs_.setDataAt<uint32_t>(0, 0, 0, 2, outputs[2]->sequence());

    uint32_t dimensionsSeqs[1] = {3};
    qnnBackend_->modelAddTensor(seqs_.name(), (Qnn_Tensor_t){
                                                     .version = QNN_TENSOR_VERSION_1,
                                                     {.v1 = {
                                                          .id = 0,
                                                          .name = seqs_.name().c_str(),
                                                          .type = QNN_TENSOR_TYPE_STATIC,
                                                          .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                                          .dataType = QNN_DATATYPE_UINT_32,
                                                          .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                                             QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                                             {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                                          .rank = 1,
                                                          .dimensions = dimensionsSeqs,
                                                          .memType = QNN_TENSORMEMTYPE_RAW,
                                                          {.clientBuf = {.data = seqs_.hostPtr<uint8_t>(),
                                                                         .dataSize = 4 * 3}}}}});
    auto data_type = QNN_DATATYPE_FLOAT_32;
    if (outputs[0]->dtype() == MLLM_TYPE_I8) {
        std::cout << "QNN INT8 op" << std::endl;
        data_type = QNN_DATATYPE_UFIXED_POINT_8;
    }

    auto outName_0 = outputs[0]->name();
    auto outName_1 = outputs[1]->name();
    auto outName_2 = outputs[2]->name();
    uint32_t dimensionsOut_0[4] = {static_cast<uint32_t>(outputs[0]->batch()),
                                   static_cast<uint32_t>(outputs[0]->sequence()),
                                   static_cast<uint32_t>(outputs[0]->head()),
                                   static_cast<uint32_t>(outputs[0]->dimension())};

    uint32_t dimensionsOut_1[4] = {static_cast<uint32_t>(outputs[1]->batch()),
                                    static_cast<uint32_t>(outputs[1]->sequence()),
                                    static_cast<uint32_t>(outputs[1]->head()),
                                    static_cast<uint32_t>(outputs[1]->dimension())};

    uint32_t dimensionsOut_2[4] = {static_cast<uint32_t>(outputs[2]->batch()),
                                    static_cast<uint32_t>(outputs[2]->sequence()),
                                    static_cast<uint32_t>(outputs[2]->head()),
                                    static_cast<uint32_t>(outputs[2]->dimension())};
    vector<Qnn_Tensor_t> activation_outputs = {
                                            (Qnn_Tensor_t){QNN_TENSOR_VERSION_1,
                                            {.v1 = {
                                                 .id = 0,
                                                 .name = outName_0.c_str(),
                                                 .type = QNN_TENSOR_TYPE_NATIVE,
                                                 .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                                 .dataType = data_type,
                                                 .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                                    {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                                 .rank = 4,
                                                 .dimensions = dimensionsOut_0,
                                                 .memType = QNN_TENSORMEMTYPE_RAW,
                                                 {.clientBuf = {.data = nullptr,
                                                                .dataSize = 0}}}}},            
                                            (Qnn_Tensor_t){QNN_TENSOR_VERSION_1,
                                            {.v1 = {
                                                 .id = 1,
                                                 .name = outName_1.c_str(),
                                                 .type = QNN_TENSOR_TYPE_NATIVE,
                                                 .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                                 .dataType = data_type,
                                                 .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                                    {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                                 .rank = 4,
                                                 .dimensions = dimensionsOut_1,
                                                 .memType = QNN_TENSORMEMTYPE_RAW,
                                                 {.clientBuf = {.data = nullptr,
                                                                .dataSize = 0}}}}},
                                            (Qnn_Tensor_t){QNN_TENSOR_VERSION_1,
                                            {.v1 = {
                                                 .id = 1,
                                                 .name = outName_2.c_str(),
                                                 .type = QNN_TENSOR_TYPE_NATIVE,
                                                 .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                                 .dataType = data_type,
                                                 .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                                    {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                                 .rank = 4,
                                                 .dimensions = dimensionsOut_2,
                                                 .memType = QNN_TENSORMEMTYPE_RAW,
                                                 {.clientBuf = {.data = nullptr,
                                                                .dataSize = 0}}}}}};

    return graphAddNode(name() + ".split", "SplitInput", {inputs[0]->name(), seqs_.name()}, activation_outputs, {}, "LLaMAPackage");
}

ErrorCode QNNSplitInput::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    
    seqs_.free();

    return MLLM_NO_ERROR;
}

} // namespace mllm
