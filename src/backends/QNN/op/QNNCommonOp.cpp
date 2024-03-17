#include "QNNCommonOp.hpp"
#include "OpDefined.hpp"
#include "QnnTypes.h"
#include "WrapperUtils/QnnWrapperUtils.hpp"
#include "Types.hpp"
#include <memory>
#include <string>

namespace mllm {

QNNCommonOp::QNNCommonOp(Backend *bn, string opName) :
    Op(bn, opName) {
    qnnBackend_ = dynamic_cast<QNNBackend *>(bn);
}

ErrorCode QNNCommonOp::graphAddNode(string name, string nodeType, vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs, vector<Qnn_Param_t> params, string packageName, bool isNSHD) {
    vector<string> inputTensorNames;
    for (auto &input : inputs) {
        inputTensorNames.push_back(input->name());
    }
// DEBUGLOG
// #ifdef DEBUGPRINT
//     std::cout << "node name:" << name << std::endl;
//     std::cout << "node type:" << nodeType << std::endl;
//     for (auto &inputTensorName : inputTensorNames) {
//         std::cout << "input:" << inputTensorName << std::endl;
//     }
//     for (auto &output : outputs) {
//         std::cout << "output:" << output->name() << std::endl;
//     }
// #endif

    vector<Qnn_Tensor_t> outputTensors;
    for (auto &output : outputs) {
        uint32_t dimensions[4] = {static_cast<uint32_t>(output->batch()),
                                  static_cast<uint32_t>(output->sequence()),
                                  static_cast<uint32_t>(output->head()),
                                  static_cast<uint32_t>(output->dimension())};
        if (!isNSHD) {
            dimensions[1] = static_cast<uint32_t>(output->head());
            dimensions[2] = static_cast<uint32_t>(output->sequence());
        }

        // TODO tensor type = MLLM_TYPE_I8
        auto data_type = QNN_DATATYPE_UFIXED_POINT_8;
        if (output->dtype() == MLLM_TYPE_I8) {
            std::cout << "QNN INT8 op" << std::endl;
            data_type = QNN_DATATYPE_UFIXED_POINT_8;
        }
            

        inputTensorNames_.push_back(new string(output->name()));
        outputTensors.push_back({QNN_TENSOR_VERSION_1,
                                 {.v1 = {
                                      .id = 0,
                                      .name = inputTensorNames_.back()->c_str(),
                                      .type = getOutputTensorType(output),
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

    if (qnn_wrapper_api::ModelError_t::MODEL_NO_ERROR != qnnBackend_->graphAddNode(name, nodeType, inputTensorNames, outputTensors, params, packageName)) {
        exit(1);
        return ErrorCode::INVALID_VALUE;
    }
    return MLLM_NO_ERROR;
}

ErrorCode QNNCommonOp::graphAddNode(string name, string nodeType, vector<string> inputTensorNames, vector<Qnn_Tensor_t> outputs, vector<Qnn_Param_t> params, string packageName) {
// DEBUGLOG
// #ifdef DEBUGPRINT
//     std::cout << "node name:" << name << std::endl;
//     std::cout << "node type:" << nodeType << std::endl;
//     for (auto &inputTensorName : inputTensorNames) {
//         std::cout << "input:" << inputTensorName << std::endl;
//     }
//     for (auto &output : outputs) {
//         std::cout << "output:" << output.v1.name << std::endl;
//     }
// #endif
    if (qnn_wrapper_api::ModelError_t::MODEL_NO_ERROR != qnnBackend_->graphAddNode(name, nodeType, inputTensorNames, outputs, params, packageName)) {
        exit(1);
        return ErrorCode::INVALID_VALUE;
    }
    return MLLM_NO_ERROR;
}

Qnn_TensorType_t QNNCommonOp::getOutputTensorType(shared_ptr<mllm::Tensor> tensor) const {
    if (tensor->tensorType() == GRAPH_OUTPUT)
        return QNN_TENSOR_TYPE_APP_READ;
    else
        return QNN_TENSOR_TYPE_NATIVE; // qnn input is set APP_WRITE by backend
}

} // namespace mllm
