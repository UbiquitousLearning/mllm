// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/qnn/op/QNNLinearOp.hpp"
#include "QnnTypes.h"
#include "mllm/backends/qnn/QNNBackend.hpp"
#include "mllm/backends/qnn/QNNUtils.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/aops/LinearOp.hpp"
#include "mllm/mllm.hpp"
#include "mllm/utils/Log.hpp"
#include <cstring>
#include <cmath>

namespace mllm::qnn {

/*
QNN Linear in mllm uses W8A8/W8A16 per-tensor quantization scheme.
The weight and the activation is using offline profiled scale to quantize the float32 data to int8/int16.
In QNN, the quantizeParam is set to QNN_QUANTIZATION_ENCODING_SCALE_OFFSET (with offset=0)

FIXME: Due to history reason, the bias is stored in int8 format.
In w8a8 case, the bias is converted to uint8 with zero-point 128.
In w8a16 case, the bias is needed to convert to int32.
Additionally, when using rotation quant and fused dequantize-add, the bias is not needed here and in float32 format.

TODO: per-channel quantization support, use FullyConnected to replace Conv2d
*/

QNNLinearOp::QNNLinearOp(const aops::LinearOpOptions& options) : LinearOp(options) {}

void QNNLinearOp::load(const ParameterFile::ptr_t& ploader) {
  switch (ploader->version()) {
    case ModelFileVersion::kV1: {
      weight_ = ploader->pull(getName() + ".weight");
      // using Conv2d in QNN, need to reshape the weight to 4D
      weight_ = weight_.view({1, 1, options_.out_channels, options_.in_channels});
      if (options_.bias) {
        bias_ = ploader->pull(getName() + ".bias");
        bias_ = bias_.view({options_.out_channels});

        biasScale_ = ploader->pull(getName() + ".bias.scale");
        biasScale_ = biasScale_.view({1});
      }

      weightScale_ = ploader->pull(getName() + ".weight.scale");
      weightScale_ = weightScale_.view({1});

      outputScale_ = ploader->pull(getName() + ".output_scale");
      outputScale_ = outputScale_.view({1});
      break;
    }
    case ModelFileVersion::kUserTemporary:
    case ModelFileVersion::kV2: {
      weight_ = ploader->pull(getName() + ".weight");
      // using Conv2d in QNN, need to reshape the weight to 4D
      weight_ = weight_.view({1, 1, options_.out_channels, options_.in_channels});
      if (options_.bias) {
        bias_ = ploader->pull(getName() + ".bias");
        bias_ = bias_.view({options_.out_channels});

        biasScale_ = ploader->pull(getName() + ".bias.scale");
      }

      weightScale_ = ploader->pull(getName() + ".weight.scale");

      outputScale_ = ploader->pull(getName() + ".output_scale");
      break;
    }
    default: NYI("Unsupported model file version")
  }

  // set quant scale to tensor's attached view
  setQuantScale(weight_, weightScale_.at<float>({0}));
  if (options_.bias) { setQuantScale(biasScale_, biasScale_.at<float>({0})); }

  // handle bias conversion for history reason
  if (options_.bias) {
    switch (options_.qnn_impl_type) {
      case aops::LinearImplTypes::kQNN_tensor_symm_w8a8: {
        // convert int8 bias to uint8 with zero-point 128
        bias_ = bias_.to(kUInt8);
        break;
      }
      case aops::LinearImplTypes::kQNN_tensor_symm_w8a16: {
        // convert int8 bias to int32
        bias_ = bias_.to(kInt32);
        break;
      }
      default: MLLM_ERROR_EXIT(1, "QNN not support other linear impl");
    }
  }
}

void QNNLinearOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& i = inputs[0];
  auto i_shape = i.shape();

  MLLM_RT_ASSERT_EQ(i_shape[i_shape.size() - 1], options_.in_channels);

  auto o_shape = i_shape;
  o_shape[o_shape.size() - 1] = options_.out_channels;

  DataTypes o_dtype = i.dtype();

  switch (options_.qnn_impl_type) {
    case aops::LinearImplTypes::kQNN_tensor_symm_w8a8: {
      o_dtype = kInt8;
      break;
    }
    case aops::LinearImplTypes::kQNN_tensor_symm_w8a16: {
      o_dtype = kInt16;
      break;
    }
    default: MLLM_ERROR_EXIT(1, "QNN not support other linear impl");
  }

  outputs.emplace_back(Tensor::empty(o_shape, o_dtype, i.device()));
}

bool QNNLinearPattern::addNode(const std::string& graphName, const ir::op_ptr_t& op,
                               const std::vector<ir::tensor::TensorValue::ptr_t>& inputs,
                               const std::vector<ir::tensor::TensorValue::ptr_t>& outputs) {
  // Get the LinearOp from the IR
  auto* linalgOp = dynamic_cast<ir::linalg::LinearOp*>(op.get());
  if (!linalgOp) {
    MLLM_ERROR("Failed to cast to linalg::LinearOp");
    return false;
  }

  auto* qnnLinearOp = dynamic_cast<QNNLinearOp*>(linalgOp->getAOp());
  if (!qnnLinearOp) {
    MLLM_ERROR("Failed to cast to QNNLinearOp");
    return false;
  }

  const auto& options = qnnLinearOp->options();

  auto qnnBackend = std::dynamic_pointer_cast<QNNBackend>(Context::instance().getBackend(kQNN));
  if (!qnnBackend) {
    MLLM_ERROR("Failed to get QNN backend.");
    return false;
  }

  auto& weight = qnnLinearOp->weight();
  auto& bias = qnnLinearOp->bias();

  std::vector<std::string> inputTensorNames;
  inputTensorNames.reserve(inputs.size());
  for (const auto& input : inputs) { inputTensorNames.push_back(input->name()); }
  inputTensorNames.push_back(weight.name());

  // Create stride and pad parameters using QNNParamTensorWrapper
  auto strideParam = mllm::qnn::QNNParamTensorWrapper::create("stride", qnnLinearOp->getName() + ".stride",
                                                              QNN_DATATYPE_UINT_32, std::vector<uint32_t>{2});
  uint32_t* strideData = static_cast<uint32_t*>(strideParam->alloc());
  strideData[0] = 1;
  strideData[1] = 1;

  auto padParam = mllm::qnn::QNNParamTensorWrapper::create("pad_amount", qnnLinearOp->getName() + ".pad", QNN_DATATYPE_UINT_32,
                                                           std::vector<uint32_t>{2, 2});
  uint32_t* padData = static_cast<uint32_t*>(padParam->alloc());
  padData[0] = 0;
  padData[1] = 0;
  padData[2] = 0;
  padData[3] = 0;

  // Add weight tensor using qnnBackend interface
  float weightScale = mllm::qnn::getQuantScale(weight);
  Qnn_QuantizeParams_t weightQuantizeParams = {QNN_DEFINITION_DEFINED,
                                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                               {.scaleOffsetEncoding = {.scale = weightScale, .offset = 0}}};

  if (!qnnBackend->addStaticTensor(graphName, weight.name(), weight, weightQuantizeParams)) {
    MLLM_ERROR("Failed to add weight tensor {} to graph {}", weight.name(), graphName);
    return false;
  }

  // Add bias tensor if needed
  if (options.bias) {
    float biasScale = mllm::qnn::getQuantScale(bias);
    Qnn_QuantizeParams_t biasQuantizeParams = {QNN_DEFINITION_DEFINED,
                                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                               {.scaleOffsetEncoding = {.scale = biasScale, .offset = 0}}};

    if (!qnnBackend->addStaticTensor(graphName, bias.name(), bias, biasQuantizeParams)) {
      MLLM_ERROR("Failed to add bias tensor {} to graph {}", bias.name(), graphName);
      return false;
    }

    inputTensorNames.push_back(bias.name());
  }

  // Add output tensor using qnnBackend interface
  float outputScale = mllm::qnn::getQuantScale(outputs[0]->tensor_);
  // output tensor is not in int8/int16 range, need to normalize it
  switch (options.qnn_impl_type) {
    case aops::LinearImplTypes::kQNN_tensor_symm_w8a8: outputScale /= (std::pow(2, 7) - 1); break;
    case aops::LinearImplTypes::kQNN_tensor_symm_w8a16: outputScale /= (std::pow(2, 15) - 1); break;
    default: MLLM_ERROR("Unsupported QNN linear implementation type"); return false;
  }
  Qnn_QuantizeParams_t outputQuantizeParams = {QNN_DEFINITION_DEFINED,
                                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                               {.scaleOffsetEncoding = {.scale = outputScale, .offset = 0}}};

  // Determine output tensor type using helper function
  auto qnn_output_tensor_type = mllm::qnn::getQnnOutputTensorType(outputs[0]);

  if (!qnnBackend->addTensor(graphName, outputs[0]->name(), qnn_output_tensor_type, outputs[0]->tensor_,
                             outputQuantizeParams)) {
    MLLM_ERROR("Failed to add output tensor {} to graph {}", outputs[0]->name(), graphName);
    return false;
  }

  qnnBackend->graphAddNode(graphName, qnnLinearOp->getName(), "Conv2d", inputTensorNames, {outputs[0]->name()},
                           {strideParam, padParam}, {});

  return true;
}

}  // namespace mllm::qnn
