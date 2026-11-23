#include "CustomOpPatterns.hpp"

#include <cstring>

#include "mllm/backends/qnn/custom-op-package/LLaMAPackage/src/ops/ShadowCorrectionPacketLayout.hpp"
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

bool QNNShadowCorrectionPacketPattern::addNode(const std::string& graph_name, const ir::op_ptr_t& op,
                                                const std::vector<ir::tensor::TensorValue::ptr_t>& inputs,
                                                const std::vector<ir::tensor::TensorValue::ptr_t>& outputs) {
  return addNodeWithMask(graph_name, op, inputs, outputs, {});
}

bool QNNShadowCorrectionPacketPattern::addNodeWithMask(const std::string& graph_name, const ir::op_ptr_t& op,
                                                        const std::vector<ir::tensor::TensorValue::ptr_t>& inputs,
                                                        const std::vector<ir::tensor::TensorValue::ptr_t>& outputs,
                                                        const std::string& mask_name) {
  const auto* ir_op = dynamic_cast<ir::linalg::CustomizedOp*>(op.get());
  const auto* packet_op = ir_op ? dynamic_cast<ShadowCorrectionPacketOp*>(ir_op->getAOp()) : nullptr;
  if (!packet_op || (inputs.size() != 2 && inputs.size() != 3) || outputs.size() != 1) { return false; }
  const bool masked = !mask_name.empty();
  const bool guarded = inputs.size() == 3 && !masked;
  const auto& shape = inputs[0]->tensor_.shape();
  if (shape[1] * shape[2] != ::shadow_correction_packet::tile_rows) {
    MLLM_ERROR("ShadowCorrectionPacket requires an M256 QNN graph, got {}", graph_name);
    return false;
  }
  auto backend = std::dynamic_pointer_cast<QNNBackend>(Context::instance().getBackend(kQNN));
  if (!backend || !backend->addTensor(graph_name, outputs[0]->name(), getQnnOutputTensorType(outputs[0]), outputs[0]->tensor_)) {
    return false;
  }

  auto controls = QNNParamTensorWrapper::create("controls", packet_op->getName() + ".controls", QNN_DATATYPE_INT_32,
                                                std::vector<uint32_t>{1, 1, 1, 3});
  const auto& words = packet_op->controls();
  std::memcpy(controls->alloc(), words.data(), words.size() * sizeof(words[0]));
  std::vector<std::string> input_names{inputs[0]->name(), inputs[1]->name()};
  if (masked) { input_names.push_back(mask_name); }
  if (guarded) { input_names.push_back(inputs[2]->name()); }
  const char* type = masked ? "ShadowCorrectionPacketMasked" : (guarded ? "ShadowCorrectionPacketGuarded" : "ShadowCorrectionPacket");
  backend->graphAddNode(graph_name, packet_op->getName(), type, input_names, {outputs[0]->name()}, {controls}, {},
                        QNN_Custom_Op_Package);
  if (guarded) { MLLM_INFO("QNN_SHADOW_CORRECTION_ACTIVATION_CODES graph={}", graph_name); }
  MLLM_INFO("QNN_SHADOW_CORRECTION_PACKET graph={} positive={} negative={} flags={}", graph_name, words[0], words[1], words[2]);
  return true;
}

}  // namespace mllm::qnn
