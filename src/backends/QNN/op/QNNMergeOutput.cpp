
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
    assert(inputs.size() == 3);
    assert(outputs.size() == 1);

    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence() + inputs[1]->sequence() + inputs[2]->sequence(), inputs[0]->dimension());

    return Op::reshape(inputs, outputs);
}

ErrorCode QNNMergeOutput::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {  

    vector<string> inputTensorNames;
    for (auto &input : inputs) {
        inputTensorNames.push_back(input->name());
    }

    outputs[0]->setBackend(qnnBackend_);
    outputs[0]->alloc();
    qnnBackend_->pushOutputBuffers(outputs[0]->hostPtr<uint8_t>());

    vector<Qnn_Tensor_t> outputTensors;
    for (auto &output : outputs) {
        uint32_t dimensions[4] = {static_cast<uint32_t>(output->batch()),
                                  static_cast<uint32_t>(output->sequence()),
                                  static_cast<uint32_t>(output->head()),
                                  static_cast<uint32_t>(output->dimension())};

        std::cout << dimensions[0] << " " << dimensions[1] << " " << dimensions[2] << " " << dimensions[3] << std::endl;

        // TODO tensor type = MLLM_TYPE_I8
        auto data_type = QNN_DATATYPE_FLOAT_32;
        if (output->dtype() == MLLM_TYPE_I8) {
            std::cout << "QNN INT8 op" << std::endl;
            data_type = QNN_DATATYPE_UFIXED_POINT_8;
        }
            

        auto outName = new string(output->name());
        outputTensors.push_back({QNN_TENSOR_VERSION_1,
                                 {.v1 = {
                                      .id = 0,
                                      .name = outName->c_str(),
                                      .type = QNN_TENSOR_TYPE_APP_READ,
                                      .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                      .dataType = data_type,
                                      .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                         QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                         {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                      .rank = 4,
                                      .dimensions = dimensions,
                                      .memType = QNN_TENSORMEMTYPE_RAW,
                                      {.clientBuf = {.data = nullptr,
                                                     .dataSize = 0}}}}});
    }

    return graphAddNode(name() + ".mergeoutput", "MergeOutput", inputTensorNames, outputTensors, {}, "LLaMAPackage");
}

ErrorCode QNNMergeOutput::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {

    outputs[0]->free();
    
    return MLLM_NO_ERROR;
}

} // namespace mllm
