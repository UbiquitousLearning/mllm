// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/qnn/op/QNNCastTypeOp.hpp"
#include "mllm/backends/qnn/QNNBackend.hpp"
#include "mllm/backends/qnn/QNNUtils.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/utils/Log.hpp"
#include "QnnTypes.h"
#include <cmath>

namespace mllm::qnn {

QNNCastTypeOp::QNNCastTypeOp(const aops::CastTypeOpOptions& options) : aops::CastTypeOp(options) {}

void QNNCastTypeOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // CastType operation maintains the same shape
  assert(inputs.size() == 1);
  const auto& input = inputs[0];

  outputs.emplace_back(Tensor::empty(input.shape(), options_.dtype, input.device()));

  if (options_.dtype == kInt8 || options_.dtype == kInt16) {
    // IMPORTANT: propagate quantization scale to output tensor for afterward ops
    // we assume user attached original quant scale in the frontend
    // FIXME: historically issues
    auto t = inputs[0];  // shadow copy for get scale
    if (options_.dtype == kInt8) {
      setQuantScale(outputs[0], getQuantScale(t) / (pow(2, 7) - 1));
    } else {
      setQuantScale(outputs[0], getQuantScale(t) / (pow(2, 15) - 1));
    }
  }
}

bool QNNCastTypePattern::addNode(const std::string& graphName, const ir::op_ptr_t& op,
                                 const std::vector<ir::tensor::TensorValue::ptr_t>& inputs,
                                 const std::vector<ir::tensor::TensorValue::ptr_t>& outputs) {
  // Get the CastTypeOp from the IR
  auto* linalgOp = dynamic_cast<ir::linalg::CastTypeOp*>(op.get());
  if (!linalgOp) {
    MLLM_ERROR("Failed to cast to linalg::CastTypeOp");
    return false;
  }

  auto* qnnCastTypeOp = dynamic_cast<QNNCastTypeOp*>(linalgOp->getAOp());
  if (!qnnCastTypeOp) {
    MLLM_ERROR("Failed to cast to QNNCastTypeOp");
    return false;
  }

  auto qnnBackend = std::dynamic_pointer_cast<QNNBackend>(Context::instance().getBackend(kQNN));
  if (!qnnBackend) {
    MLLM_ERROR("Failed to get QNN backend from context");
    return false;
  }

  // Determine operation type based on input/output dtypes
  const auto& inputDtype = inputs[0]->tensor_.dtype();
  const auto& outputDtype = outputs[0]->tensor_.dtype();

  bool isQuantize = (inputDtype == kFloat32 || inputDtype == kFloat16) && (outputDtype == kInt8 || outputDtype == kInt16);
  bool isDequantize = (inputDtype == kInt8 || inputDtype == kInt16) && (outputDtype == kFloat32 || outputDtype == kFloat16);

  if (isQuantize) {
    return addQuantizeNode(graphName, qnnCastTypeOp, inputs, outputs, qnnBackend);
  } else if (isDequantize) {
    return addDequantizeNode(graphName, qnnCastTypeOp, inputs, outputs, qnnBackend);
  } else {
    MLLM_ERROR("Unsupported CastType conversion from {} to {}", (int)inputDtype, (int)outputDtype);
    return false;
  }
}

bool QNNCastTypePattern::addQuantizeNode(const std::string& graphName, QNNCastTypeOp* qnnCastTypeOp,
                                         const std::vector<ir::tensor::TensorValue::ptr_t>& inputs,
                                         const std::vector<ir::tensor::TensorValue::ptr_t>& outputs,
                                         const std::shared_ptr<QNNBackend>& qnnBackend) {
  MLLM_RT_ASSERT(inputs[0]->tensor_.rank() == 4);  // FIXME: custom op only supports 4D tensor for now
  const auto& outputDtype = outputs[0]->tensor_.dtype();

  // get quantization scale, it is propagated from input to output in reshape step
  float quantScale = getQuantScale(outputs[0]->tensor_);

  // Create scale parameter tensor
  auto scaleParamName = qnnCastTypeOp->getName() + ".quantize_scale";
  auto scaleParam = QNNParamTensorWrapper::create("scale", scaleParamName, QNN_DATATYPE_FLOAT_32, std::vector<uint32_t>{1});
  float* scaleData = static_cast<float*>(scaleParam->alloc());
  scaleData[0] = quantScale;

  // Add output tensor with quantization parameters
  Qnn_QuantizeParams_t outputQuantizeParams = {QNN_DEFINITION_DEFINED,
                                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                               {.scaleOffsetEncoding = {.scale = quantScale, .offset = 0}}};

  auto qnn_output_tensor_type = mllm::qnn::getQnnOutputTensorType(outputs[0]);

  if (!qnnBackend->addTensor(graphName, outputs[0]->name(), qnn_output_tensor_type, outputs[0]->tensor_,
                             outputQuantizeParams)) {
    MLLM_ERROR("Failed to add output tensor {} to graph {}", outputs[0]->name(), graphName);
    return false;
  }

  // Add quantize node to the graph
  qnnBackend->graphAddNode(graphName, qnnCastTypeOp->getName(), "LLaMAQuantize", {inputs[0]->name()}, {outputs[0]->name()},
                           {scaleParam}, {}, QNN_Custom_Op_Package);

  return true;
}

bool QNNCastTypePattern::addDequantizeNode(const std::string& graphName, QNNCastTypeOp* qnnCastTypeOp,
                                           const std::vector<ir::tensor::TensorValue::ptr_t>& inputs,
                                           const std::vector<ir::tensor::TensorValue::ptr_t>& outputs,
                                           const std::shared_ptr<QNNBackend>& qnnBackend) {
  MLLM_RT_ASSERT(inputs[0]->tensor_.rank() == 4);  // FIXME: custom op only supports 4D tensor for now
  const auto& inputDtype = inputs[0]->tensor_.dtype();

  // get quantization scale, it is propagated to input in previous ops
  // NOTE: different from quantize
  float dequantScale = getQuantScale(inputs[0]->tensor_);

  // Create scale parameter tensor
  auto scaleParamName = qnnCastTypeOp->getName() + ".dequantize_scale";
  auto scaleParam = QNNParamTensorWrapper::create("scale", scaleParamName, QNN_DATATYPE_FLOAT_32, std::vector<uint32_t>{1});
  float* scaleData = static_cast<float*>(scaleParam->alloc());
  scaleData[0] = dequantScale;

  // Add output tensor (float type, no quantization)
  auto qnn_output_tensor_type = mllm::qnn::getQnnOutputTensorType(outputs[0]);

  if (!qnnBackend->addTensor(graphName, outputs[0]->name(), qnn_output_tensor_type, outputs[0]->tensor_)) {
    MLLM_ERROR("Failed to add output tensor {} to graph {}", outputs[0]->name(), graphName);
    return false;
  }

  // Add dequantize node to the graph
  qnnBackend->graphAddNode(graphName, qnnCastTypeOp->getName(), "LLaMADequantize", {inputs[0]->name()}, {outputs[0]->name()},
                           {scaleParam}, {}, QNN_Custom_Op_Package);

  return true;
}

}  // namespace mllm::qnn
