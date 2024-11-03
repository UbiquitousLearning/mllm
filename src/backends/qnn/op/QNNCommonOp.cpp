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

        // TODO tensor type = MLLM_TYPE_I8
        auto data_type = QNN_DATATYPE_FLOAT_32;
        if (output->dtype() == MLLM_TYPE_I8) {
            data_type = QNN_DATATYPE_SFIXED_POINT_8;
        }

        if (output->dtype() == MLLM_TYPE_F16) {
            data_type = QNN_DATATYPE_FLOAT_16;
        }


        float quantScale = 0.0f;
        auto quantDefine = QNN_DEFINITION_UNDEFINED;
        auto quantType = QNN_QUANTIZATION_ENCODING_UNDEFINED;

        if (scale != nullptr) {
            quantScale = scale->hostPtr<float>()[0] / 127.0;
            quantScale = roundf(quantScale * 100000) / 100000;
            quantDefine = QNN_DEFINITION_DEFINED;
            quantType = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
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
        if(tensor->allocted() == 0) {
            tensor->alloc();
        }
        qnnBackend_->pushOutputBuffers(tensor->hostPtr<uint8_t>());
        return QNN_TENSOR_TYPE_APP_READ;
    } else {
        auto name = tensor->name();

        // qwen 1.5-1.8B
//         if (name.find("q_proj.rope") != -1 || name.find("k_proj.rope") != -1 || name.find("v_proj.transpose") != -1) {
// #ifdef DEBUGPRINT
//             std::cout << "view output" << std::endl;
// #endif
//             return QNN_TENSOR_TYPE_APP_READ;
//         }

// // SHADOW
//         if (name == "outtensor-model.layers.1.mlp.down_proj-00" || name == "outtensor-model.layers.1.mlp.silu-00_mul_-00" || name == "outtensor-model.layers.1.mlp.down_proj.dequantize-00_view_-00_add_-00") {
// #ifdef DEBUGPRINT
//             std::cout << "shadow output" << std::endl;
// #endif
//             return QNN_TENSOR_TYPE_APP_READ;
//         }

//                 if (name == "outtensor-model.layers.2.mlp.down_proj-00" || name == "outtensor-model.layers.2.mlp.silu-00_mul_-00" || name == "outtensor-model.layers.2.mlp.down_proj.dequantize-00_view_-00_add_-00") {
// #ifdef DEBUGPRINT
//             std::cout << "shadow output" << std::endl;
// #endif
//             return QNN_TENSOR_TYPE_APP_READ;
//         }

//         if (name == "outtensor-model.layers.6.mlp.down_proj-00" || name == "outtensor-model.layers.6.mlp.silu-00_mul_-00" || name == "outtensor-model.layers.6.mlp.down_proj.dequantize-00_view_-00_add_-00") {
// #ifdef DEBUGPRINT
//             std::cout << "shadow output" << std::endl;
// #endif
//             return QNN_TENSOR_TYPE_APP_READ;
//         }

        // PhoneLM 1.5B
        if (name.find("q_proj.dequantize") != -1 || name.find("k_proj.dequantize") != -1 || name.find("v_proj.transpose") != -1) {
#ifdef DEBUGPRINT
            std::cout << "view output" << std::endl;
#endif
            return QNN_TENSOR_TYPE_APP_READ;
        }

// SHADOW
        if (name == "outtensor-model.layers.1.mlp.down_proj-00" || name == "outtensor-model.layers.1.mlp.relu-00_mul_-00" || name == "outtensor-model.layers.1.mlp.down_proj.dequantize-00_view_-00_add_-00") {
#ifdef DEBUGPRINT
            std::cout << "shadow output" << std::endl;
#endif
            return QNN_TENSOR_TYPE_APP_READ;
        }

                if (name == "outtensor-model.layers.3.mlp.down_proj-00" || name == "outtensor-model.layers.3.mlp.relu-00_mul_-00" || name == "outtensor-model.layers.3.mlp.down_proj.dequantize-00_view_-00_add_-00") {
#ifdef DEBUGPRINT
            std::cout << "shadow output" << std::endl;
#endif
            return QNN_TENSOR_TYPE_APP_READ;
        }

        if (name == "outtensor-model.layers.4.mlp.down_proj-00" || name == "outtensor-model.layers.4.mlp.relu-00_mul_-00" || name == "outtensor-model.layers.4.mlp.down_proj.dequantize-00_view_-00_add_-00") {
#ifdef DEBUGPRINT
            std::cout << "shadow output" << std::endl;
#endif
            return QNN_TENSOR_TYPE_APP_READ;
        }

        return QNN_TENSOR_TYPE_NATIVE; // qnn input is set APP_WRITE by backend
    }
}

} // namespace mllm
