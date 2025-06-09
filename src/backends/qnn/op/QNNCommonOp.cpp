#include "QNNCommonOp.hpp"
#include "OpDefined.hpp"
#include "QnnTypes.h"
#include "WrapperUtils/QnnWrapperUtils.hpp"
#include "Types.hpp"
#include <string>

namespace mllm {

QNNCommonOp::QNNCommonOp(Backend *bn, string opName) :
    Op(bn, opName) {
    qnnBackend_ = dynamic_cast<QNNBackend *>(bn);
}

ErrorCode QNNCommonOp::graphAddNode(string name, string nodeType, vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs, vector<Qnn_Param_t> params, string packageName, bool isNSHD, Tensor *scale) {
    vector<string> inputTensorNames;
    for (auto &input : inputs) {
        inputTensorNames.push_back(input->name());
    }

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

        float quantScale = 0.0f;
        auto quantDefine = QNN_DEFINITION_UNDEFINED;
        auto quantType = QNN_QUANTIZATION_ENCODING_UNDEFINED;
        auto data_type = QNN_DATATYPE_FLOAT_32;
        switch (output->dtype()) {
        case MLLM_TYPE_I8:
            data_type = QNN_DATATYPE_SFIXED_POINT_8;
            quantScale = scale->hostPtr<float>()[0] / (pow(2, 7) - 1);
            // quantScale = roundf(quantScale * 100000) / 100000;
            quantDefine = QNN_DEFINITION_DEFINED;
            quantType = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
            break;
        case MLLM_TYPE_I16:
            data_type = QNN_DATATYPE_SFIXED_POINT_16;
            quantScale = scale->hostPtr<float>()[0] / (pow(2, 15) - 1);
            // quantScale = roundf(quantScale * 100000) / 100000;
            quantDefine = QNN_DEFINITION_DEFINED;
            quantType = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
            break;
        case MLLM_TYPE_F16:
            data_type = QNN_DATATYPE_FLOAT_16;
            break;
        default:
            data_type = QNN_DATATYPE_FLOAT_32;
            break;
        }

        inputTensorNames_.push_back(new string(output->name()));
        outputTensors.push_back({QNN_TENSOR_VERSION_1,
                                 {.v1 = {
                                      .id = 0,
                                      .name = inputTensorNames_.back()->c_str(),
                                      .type = getOutputTensorType(output),
                                      .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                      .dataType = data_type,
                                      .quantizeParams = {quantDefine,
                                                         quantType,
                                                         {.scaleOffsetEncoding = {.scale = quantScale, .offset = 0}}},
                                      .rank = 4,
                                      .dimensions = dimensions,
                                      .memType = QNN_TENSORMEMTYPE_RAW,
                                      .clientBuf = {.data = nullptr,
                                                    .dataSize = 0}}}});
    }

    if (qnn_wrapper_api::ModelError_t::MODEL_NO_ERROR != qnnBackend_->graphAddNode(name, nodeType, inputTensorNames, outputTensors, params, packageName)) {
        exit(1);
        return ErrorCode::INVALID_VALUE;
    }
    return MLLM_NO_ERROR;
}

ErrorCode QNNCommonOp::graphAddNode(string name, string nodeType, vector<string> inputTensorNames, vector<Qnn_Tensor_t> outputs, vector<Qnn_Param_t> params, string packageName) {
    if (qnn_wrapper_api::ModelError_t::MODEL_NO_ERROR != qnnBackend_->graphAddNode(name, nodeType, inputTensorNames, outputs, params, packageName)) {
        exit(1);
        return ErrorCode::INVALID_VALUE;
    }
    return MLLM_NO_ERROR;
}

Qnn_TensorType_t QNNCommonOp::getOutputTensorType(shared_ptr<mllm::Tensor> tensor) const {
    if (tensor->ttype() == GRAPH_OUTPUT) {
        // in Module API, the outputs of a graph is not allocated before setUp, alloc here
        if (tensor->allocted() == 0) {
            tensor->alloc();
        }
        qnnBackend_->pushOutputBuffers(tensor->hostPtr<uint8_t>());
        return QNN_TENSOR_TYPE_APP_READ;
    } else {
        if (tensor->childTensors().size() > 0 && tensor->childTensors()[0]->ttype() == GRAPH_OUTPUT) {
            if (tensor->allocted() == 0) {
                tensor->alloc();
            }
            qnnBackend_->pushOutputBuffers(tensor->hostPtr<uint8_t>());
            return QNN_TENSOR_TYPE_APP_READ;
        }

        return QNN_TENSOR_TYPE_NATIVE; // qnn input is set APP_WRITE by backend
    }
}

} // namespace mllm
