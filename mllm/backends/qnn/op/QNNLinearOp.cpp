// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/qnn/op/QNNLinearOp.hpp"
#include "QnnTypes.h"
#include "mllm/backends/qnn/QNNBackend.hpp"
#include "mllm/backends/qnn/QNNUtils.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/DeviceTypes.hpp"
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

This implementation uses QNN FullyConnected operator instead of Conv2d.
For FullyConnected, weights have shape [m, n] where n is input channels and m is output channels.
Bias is required to be int32 type for proper quantization handling.

TODO: per-channel quantization support
*/

QNNLinearOp::QNNLinearOp(const aops::LinearOpOptions& options) : LinearOp(options) {}

void QNNLinearOp::load(const ParameterFile::ptr_t& ploader) {
  switch (ploader->version()) {
    case ModelFileVersion::kV1: {
      weight_ = ploader->pull(getName() + ".weight");
      // using FullyConnected in QNN, reshape weight to 2D [m, n] where m=out_channels, n=in_channels
      weight_ = weight_.view({options_.out_channels, options_.in_channels});
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
      // using FullyConnected in QNN, reshape weight to 2D [m, n] where m=out_channels, n=in_channels
      weight_ = weight_.view({options_.out_channels, options_.in_channels});
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
  if (options_.bias) { setQuantScale(bias_, biasScale_.at<float>({0})); }

  // Convert bias to int32 for FullyConnected operator
  if (options_.bias) { bias_ = bias_.to(kInt32); }
}

void QNNLinearOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& i = inputs[0];
  auto i_shape = i.shape();

  MLLM_RT_ASSERT_EQ(i_shape[i_shape.size() - 1], options_.in_channels);

  DataTypes o_dtype = i.dtype();

  // flatten to 2D for QNN FullyConnected operator
  int batch_count = 1;
  for (size_t i = 0; i < i_shape.size() - 1; ++i) { batch_count *= i_shape[i]; }
  auto o_shape = std::vector<int>{batch_count, options_.out_channels};

  outputs.emplace_back(Tensor::empty(o_shape, o_dtype, kQNN));

  // attach quant scale to output tensor
  switch (options_.qnn_impl_type) {
    case aops::LinearImplTypes::kQNN_tensor_symm_w8a8: {
      setQuantScale(outputs[0], outputScale_.at<float>({0}) / (std::pow(2, 7) - 1));
      break;
    }
    case aops::LinearImplTypes::kQNN_tensor_symm_w8a16: {
      setQuantScale(outputs[0], outputScale_.at<float>({0}) / (std::pow(2, 15) - 1));
      break;
    }
    default: MLLM_ERROR_EXIT(1, "QNN not support other linear impl");
  }
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

  // only support false in HTP
  auto keep_dims = QNNParamScalarWrapper::create("keep_dims", false);

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
  // The difference between int8 and int16 is handled in reshape
  float outputScale = mllm::qnn::getQuantScale(outputs[0]->tensor_);

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

  qnnBackend->graphAddNode(graphName, qnnLinearOp->getName(), "FullyConnected", inputTensorNames, {outputs[0]->name()}, {},
                           {keep_dims});

  return true;
}

}  // namespace mllm::qnn
