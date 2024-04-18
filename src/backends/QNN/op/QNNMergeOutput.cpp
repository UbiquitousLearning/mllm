
#include "QNNMergeOutput.hpp"
#include "QnnTypes.h"
#include "Types.hpp"
#include "QNNCommonOp.hpp"
#include <cstdint>
#include <memory>

#define DYNAMICBUFFER 32

namespace mllm {
QNNMergeOutput::QNNMergeOutput(Backend *bn, string opName) :
    QNNCommonOp(bn, opName) {
    
}

ErrorCode QNNMergeOutput::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 4);
    assert(outputs.size() == 1);

    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence() + inputs[1]->sequence() + inputs[2]->sequence() + inputs[3]->sequence() * 4, inputs[0]->dimension());

    return Op::reshape(inputs, outputs);
}

ErrorCode QNNMergeOutput::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {  

    vector<string> inputTensorNames;
    for (auto &input : inputs) {
        inputTensorNames.push_back(input->name());
    }

    vector<Qnn_Param_t> paramsMerge = {
        {.paramType = QNN_PARAMTYPE_SCALAR,
         .name = "num",
         {.scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_UINT_32, {.uint32Value = static_cast<uint32_t>(inputs.size())}}}},
    };

    // outputs[0]->setBackend(qnnBackend_);
    // outputs[0]->alloc();
    // qnnBackend_->pushOutputBuffers(outputs[0]->hostPtr<uint8_t>());

    std::cout << getOutputTensorType(outputs[0]) << std::endl;

    uint32_t dimensions[4] = {static_cast<uint32_t>(outputs[0]->batch()),
                                  static_cast<uint32_t>(outputs[0]->sequence()),
                                  static_cast<uint32_t>(outputs[0]->head()),
                                  static_cast<uint32_t>(outputs[0]->dimension())};

    std::cout << dimensions[0] << " " << dimensions[1] << " " << dimensions[2] << " " << dimensions[3] << std::endl;

    // TODO tensor type = MLLM_TYPE_I8
    auto data_type = QNN_DATATYPE_FLOAT_32;
    if (outputs[0]->dtype() == MLLM_TYPE_I8) {
        std::cout << "QNN INT8 op" << std::endl;
        data_type = QNN_DATATYPE_SFIXED_POINT_8;
    }

    auto outName = outputs[0]->name();

    vector<Qnn_Tensor_t> outputTensors = {
        (Qnn_Tensor_t){QNN_TENSOR_VERSION_1,
                                 {.v1 = {
                                      .id = 0,
                                      .name = outName.c_str(),
                                      .type = getOutputTensorType(outputs[0]),
                                      .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                      .dataType = data_type,
                                      .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                         QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                         {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                      .rank = 4,
                                      .dimensions = dimensions,
                                      .memType = QNN_TENSORMEMTYPE_RAW,
                                      {.clientBuf = {.data = nullptr,
                                                     .dataSize = 0}}}}}

    };

    return graphAddNode(name() + ".mergeoutput", "MergeOutput", inputTensorNames, outputTensors, paramsMerge, "LLaMAPackage");
}

ErrorCode QNNMergeOutput::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {

    outputs[0]->free();
    
    return MLLM_NO_ERROR;
}

} // namespace mllm
