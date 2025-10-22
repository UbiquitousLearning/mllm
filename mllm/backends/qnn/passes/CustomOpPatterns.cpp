#include "CustomOpPatterns.hpp"
#include "mllm/backends/qnn/QNNBackend.hpp"
#include "mllm/backends/qnn/QNNUtils.hpp"
#include "mllm/backends/qnn/CustomLayers.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/utils/Log.hpp"
#include "QnnTypes.h"

namespace mllm::qnn {

bool QNNDequantizeAddPattern::addNode(const std::string& graphName, const ir::op_ptr_t& op,
                                      const std::vector<ir::tensor::TensorValue::ptr_t>& inputs,
                                      const std::vector<ir::tensor::TensorValue::ptr_t>& outputs) {
  // Get the CustomizedOp from the IR
  auto* linalgOp = dynamic_cast<ir::linalg::CustomizedOp*>(op.get());
  if (!linalgOp) {
    MLLM_ERROR("Failed to cast to linalg::CustomizedOp");
    return false;
  }

  // Get the DequantizeAddOp from the customized op
  auto* dequantizeAddOp = dynamic_cast<DequantizeAddOp*>(linalgOp->getAOp());
  if (!dequantizeAddOp) {
    MLLM_ERROR("Failed to cast to DequantizeAddOp");
    return false;
  }

  auto qnnBackend = std::dynamic_pointer_cast<QNNBackend>(Context::instance().getBackend(kQNN));
  if (!qnnBackend) {
    MLLM_ERROR("Failed to get QNN backend from context");
    return false;
  }

  // Check tensor constraints - DequantizeAdd only supports 4D tensors for now
  if (inputs[0]->tensor_.rank() != 4) {
    MLLM_ERROR("DequantizeAdd custom op only supports 4D tensor, got {}D", inputs[0]->tensor_.rank());
    return false;
  }

  // Get input and output data types
  const auto& inputDtype = inputs[0]->tensor_.dtype();
  const auto& outputDtype = outputs[0]->tensor_.dtype();

  // Verify this is a valid dequantization operation (quantized input to float output)
  if (!(inputDtype == kInt8 || inputDtype == kInt16) || !(outputDtype == kFloat32 || outputDtype == kFloat16)) {
    MLLM_ERROR("DequantizeAdd expects quantized input (Int8/Int16) and float output (Float32/Float16)");
    return false;
  }

  // Calculate dequantization scale based on input type
  float dequantScale = getQuantScale(inputs[0]->tensor_);

  // Create scale parameter tensor
  auto scaleParamName = dequantizeAddOp->getName() + ".dequantize_add_scale";
  auto scaleParam = QNNParamTensorWrapper::create("scale", scaleParamName, QNN_DATATYPE_FLOAT_32, std::vector<uint32_t>{1});
  float* scaleData = static_cast<float*>(scaleParam->alloc());
  scaleData[0] = dequantScale;

  // Get bias tensor (assumes it's available as a member of dequantizeAddOp)
  // Note: This assumes the bias tensor is loaded in the DequantizeAddOp::load method
  // and stored as weight_ member (based on the CustomLayers.cpp implementation)
  auto biasName = dequantizeAddOp->getName() + ".bias";

  // Add bias tensor as static tensor to the graph
  // The bias tensor dimensions should be [1, 1, 1, out_channels]
  auto bias_tensor = dequantizeAddOp->getWeightTensor();  // Use accessor method to get bias data
  if (!qnnBackend->addStaticTensor(graphName, biasName, bias_tensor)) {
    MLLM_ERROR("Failed to add bias tensor {} to graph {}", biasName, graphName);
    return false;
  }

  // Add output tensor with quantization parameters
  Qnn_QuantizeParams_t outputQuantizeParams = {QNN_DEFINITION_DEFINED,
                                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                               {.scaleOffsetEncoding = {.scale = dequantScale, .offset = 0}}};

  auto qnn_output_tensor_type = mllm::qnn::getQnnOutputTensorType(outputs[0]);

  if (!qnnBackend->addTensor(graphName, outputs[0]->name(), qnn_output_tensor_type, outputs[0]->tensor_,
                             outputQuantizeParams)) {
    MLLM_ERROR("Failed to add output tensor {} to graph {}", outputs[0]->name(), graphName);
    return false;
  }

  // Add dequantize add node to the graph
  // Input tensors: [input_tensor, bias_tensor]
  // Output tensors: [output_tensor]
  // Parameters: [scale_param]
  qnnBackend->graphAddNode(graphName, dequantizeAddOp->getName(), "LLaMADequantizeAdd", {inputs[0]->name(), biasName},
                           {outputs[0]->name()}, {scaleParam}, {}, QNN_Custom_Op_Package);

  return true;
}

}  // namespace mllm::qnn