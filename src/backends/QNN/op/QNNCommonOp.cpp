#include "QNNCommonOp.hpp"
#include "QnnTypes.h"
#include "QnnWrapperUtils.hpp"
#include "Types.hpp"

namespace mllm {

QNNCommonOp::QNNCommonOp(Backend *bn, string opName) :
    Op(bn, opName) {
    qnnBackend_ = dynamic_cast<QNNBackend *>(bn);
}

ErrorCode QNNCommonOp::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
#if DEBUG
    std::cout << "*QNN" << name() << " reshape*" << std::endl;
#endif
    return NO_ERROR;
}

ErrorCode QNNCommonOp::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
#if DEBUG
    std::cout << "*QNN" << name() << " execute*" << std::endl;
#endif
    return NO_ERROR;
}

ErrorCode QNNCommonOp::load(AbstructLoader &loader) {
#if DEBUG
    std::cout << "*QNN" << name() << " *" << std::endl;
#endif
    return NO_ERROR;
}

ErrorCode QNNCommonOp::graphAddNode(string name, string nodeType, vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs, string packageName) {
    vector<const char *> inputTensorNames;
    for (auto &input : inputs) {
        inputTensorNames.push_back(input->name().c_str());
    }

    vector<Qnn_Tensor_t> outputTensors;
    for (auto &output : outputs) {
        uint32_t dimensions[4];
        for (int i = 0; i < output->shape().size(); i++) {
            dimensions[i] = output->shape()[i];
        }
        outputTensors.push_back({QNN_TENSOR_VERSION_1,
                                 {.v1 = {
                                      .id = 0,
                                      .name = output->name().c_str(),
                                      .type = QNN_TENSOR_TYPE_APP_READ,
                                      .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                      .dataType = QNN_DATATYPE_FLOAT_32,
                                      .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                         QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                         {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                      .rank = 4,
                                      .dimensions = dimensions,
                                      .memType = QNN_TENSORMEMTYPE_RAW,
                                      {.clientBuf = {.data = nullptr,
                                                     .dataSize = 0}}}}});
    }

    if (qnn_wrapper_api::ModelError_t::MODEL_NO_ERROR != qnnBackend_->graphAddNode(name, nodeType, inputTensorNames, outputTensors, packageName)) {
        return ErrorCode::INVALID_VALUE;
    }
    return NO_ERROR;
}

} // namespace mllm
