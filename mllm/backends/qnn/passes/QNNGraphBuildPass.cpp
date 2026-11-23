// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/qnn/passes/QNNGraphBuildPass.hpp"
#include "QnnTypes.h"
#include "mllm/backends/qnn/QNNBackend.hpp"
#include "mllm/backends/qnn/QNNUtils.hpp"
#include "mllm/backends/qnn/CustomLayers.hpp"
#include "mllm/backends/qnn/ShadowCorrectionGraph.hpp"
#include "mllm/backends/qnn/custom-op-package/LLaMAPackage/src/ops/ShadowCorrectionPacketLayout.hpp"
#include "mllm/backends/qnn/op/QNNCastTypeOp.hpp"
#include "mllm/backends/qnn/op/QNNElewiseOp.hpp"
#include "mllm/backends/qnn/op/QNNRMSNormOp.hpp"
#include "mllm/backends/qnn/op/QNNSiLUOp.hpp"
#include "mllm/backends/qnn/op/QNNTransposeOp.hpp"
#include "mllm/backends/qnn/op/QNNViewOp.hpp"
#include "mllm/backends/qnn/op/QNNX2XOp.hpp"
#include "mllm/backends/qnn/passes/CustomOpPatterns.hpp"
#include "mllm/compile/ir/builtin/Op.hpp"
#include "mllm/compile/ir/cf/Op.hpp"
#include "mllm/compile/ir/graph/Op.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/tensor/Value.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/OpTypes.hpp"
#include "mllm/engine/Context.hpp"
#include "mllm/utils/Log.hpp"

#include "mllm/backends/qnn/op/QNNLinearOp.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string_view>

namespace mllm::qnn {

namespace {

using TensorValue = ir::tensor::TensorValue;

TensorValue::ptr_t shadowRawGraphOutput(const ir::graph::SubGraphOp::ptr_t& graph) {
  namespace packet = ::shadow_correction_packet;
  auto region = graph->getTopRegion();
  if (!region) { return {}; }

  TensorValue::ptr_t actual_packet;
  for (const auto& op : region->ops()) {
    const auto custom = std::dynamic_pointer_cast<ir::linalg::CustomizedOp>(op);
    if (custom && dynamic_cast<ShadowCorrectionPacketOp*>(custom->getAOp()) && custom->outputs().size() == 1) {
      actual_packet = custom->outputs().front()->cast_<TensorValue>();
      break;
    }
  }
  if (!actual_packet) { return {}; }

  for (const auto& op : region->ops()) {
    if (!op->isa_<ir::cf::ReturnOp>() || op->inputs().size() != 4) { continue; }
    const auto raw = op->inputs().front()->cast_<TensorValue>();
    const auto returned_packet = op->inputs().back()->cast_<TensorValue>();
    const auto& raw_shape = raw->tensor_.shape();
    if (returned_packet == actual_packet && raw->getAttr("is_graph_output") && raw_shape.size() == 4
        && raw_shape[0] == 1 && raw_shape[1] * raw_shape[2] == packet::tile_rows
        && raw_shape[3] == packet::in_width && raw->tensor_.dtype() == kFloat32) {
      return raw;
    }
  }
  return {};
}

bool isQwenMlpGraph(std::string_view name) {
  constexpr std::string_view prefix = "model.layers.";
  if (!name.starts_with(prefix) || (!name.ends_with("_1") && !name.ends_with("_2"))) { return false; }
  name.remove_prefix(prefix.size());
  name.remove_suffix(2);
  return !name.empty() && name.find_first_not_of("0123456789") == std::string_view::npos;
}

bool flagEnabled(const char* variable) {
  const char* value = std::getenv(variable);
  return value && std::string_view(value) == "1";
}

struct ShadowMaskLowering {
  ir::op_ptr_t multiply;
  ir::op_ptr_t packet;
  std::string name;
};

ShadowMaskLowering findShadowMaskLowering(const std::string& graph_name, const ir::graph::SubGraphOp::ptr_t& graph) {
  namespace packet = ::shadow_correction_packet;
  // The replay companion retains the original multiply and raw activation.
  if (!flagEnabled("MLLM_QNN_SHADOW_PRODUCER_MASK") || !isQwenMlpGraph(graph_name)) { return {}; }

  const std::string multiply_name = graph_name + ".Mul.0";
  ir::op_ptr_t multiply;
  for (const auto& op : graph->getTopRegion()->ops()) {
    const auto* mul = dynamic_cast<const ir::linalg::MulOp*>(op.get());
    if (mul && dynamic_cast<QNNMulOp*>(mul->getAOp()) && mul->getAOp()->getName() == multiply_name
        && op->inputs().size() == 2 && op->outputs().size() == 1) {
      const auto product = op->outputs().front()->cast_<TensorValue>();
      const auto& shape = product->tensor_.shape();
      const auto is_fp32_tile = [&shape](const TensorValue::ptr_t& value) {
        return value->tensor_.dtype() == kFloat32 && value->tensor_.shape() == shape;
      };
      if (!product->getAttr("is_graph_output") && shape.size() == 4 && shape[0] == 1
          && shape[1] * shape[2] == packet::tile_rows && shape[3] == packet::in_width
          && is_fp32_tile(product) && is_fp32_tile(op->inputs().front()->cast_<TensorValue>())
          && is_fp32_tile(op->inputs().back()->cast_<TensorValue>())) {
        multiply = op;
      }
    }

    const auto* custom = dynamic_cast<const ir::linalg::CustomizedOp*>(op.get());
    if (multiply && custom && dynamic_cast<ShadowCorrectionPacketOp*>(custom->getAOp())
        && !op->inputs().empty()
        && op->inputs().front()->cast_<TensorValue>()->name() == multiply->outputs().front()->cast_<TensorValue>()->name()) {
      return {multiply, op, multiply_name + ".activation_mask"};
    }
  }
  return {};
}

bool addShadowMaskedMultiply(const std::string& graph_name, const std::shared_ptr<QNNModel>& model,
                             const ShadowMaskLowering& lowering) {
  namespace packet = ::shadow_correction_packet;
  const auto product = lowering.multiply->outputs().front()->cast_<TensorValue>();
  const auto gate = lowering.multiply->inputs().front()->cast_<TensorValue>();
  const auto up = lowering.multiply->inputs().back()->cast_<TensorValue>();
  const auto* packet_op = static_cast<const ShadowCorrectionPacketOp*>(
      lowering.packet->cast_<ir::linalg::CustomizedOp>()->getAOp());
  auto mask_shape = product->tensor_.shape();
  mask_shape.back() = packet::activation_mask_words_per_row;
  const auto mask = Tensor::empty(mask_shape, kInt32, kQNN);
  if (model->addTensor(product->name(), getQnnOutputTensorType(product), product->tensor_) != MODEL_NO_ERROR
      || model->addTensor(lowering.name, QNN_TENSOR_TYPE_NATIVE, mask) != MODEL_NO_ERROR) { return false; }

  auto controls = QNNParamTensorWrapper::create("controls", lowering.name + ".controls", QNN_DATATYPE_INT_32,
                                                std::vector<uint32_t>{1, 1, 1, 3});
  const auto& words = packet_op->controls();
  std::memcpy(controls->alloc(), words.data(), words.size() * sizeof(words[0]));
  const auto err = model->addNode(QNN_OPCONFIG_VERSION_1, lowering.name + ".producer", QNN_Custom_Op_Package,
                                  "ShadowMulActivationMask", {controls}, {}, {gate->name(), up->name()},
                                  {product->name(), lowering.name});
  if (err != MODEL_NO_ERROR) { return false; }
  MLLM_INFO("QNN_SHADOW_PRODUCER_MASK graph={} tensor={}", graph_name, lowering.name);
  return true;
}

ir::op_ptr_t findMlpQuantizeConsumer(const std::string& graph_name, const ir::graph::SubGraphOp::ptr_t& graph,
                                    const ir::op_ptr_t& mul) {
  const auto* mul_op = dynamic_cast<const ir::linalg::MulOp*>(mul.get());
  if (!mul_op || mul_op->getAOp()->getName() != std::string(shadowMainGraphName(graph_name)) + ".Mul.0") { return {}; }
  auto reject = [&](const char* reason) -> ir::op_ptr_t {
    MLLM_INFO("QNN_MLP_HVX_MUL_QUANTIZE16_REJECT graph={} reason={}", graph_name, reason);
    return {};
  };
  if (!dynamic_cast<QNNMulOp*>(mul_op->getAOp()) || mul->inputs().size() != 2 || mul->outputs().size() != 1) {
    return reject("multiply_signature");
  }
  const auto* gate = dynamic_cast<const TensorValue*>(mul->inputs().front().get_weak());
  const auto* up = dynamic_cast<const TensorValue*>(mul->inputs().back().get_weak());
  auto* product = dynamic_cast<TensorValue*>(mul->outputs().front().get_weak());
  if (!gate || !up || !product || product->name().empty()) { return reject("multiply_tensors"); }
  if (getQnnOutputTensorType(product->cast_<TensorValue>()) != QNN_TENSOR_TYPE_NATIVE) {
    return reject("exported_product");
  }
  const auto& shape = product->tensor_.shape();
  if (shape != Tensor::shape_t{1, 16, 16, 8960} && shape != Tensor::shape_t{1, 256, 1, 8960}) {
    return reject("multiply_shape");
  }
  if (product->tensor_.dtype() != kFloat32 || gate->tensor_.dtype() != kFloat32 || up->tensor_.dtype() != kFloat32
      || gate->tensor_.shape() != shape || up->tensor_.shape() != shape) {
    return reject("multiply_operand_layout");
  }

  // QNN resolves tensor edges by their names. Follow that edge and reject
  // any other consumer, including returns and downstream reshapes.
  ir::op_ptr_t consumer;
  bool seen_producer = false;
  for (const auto& op : graph->getTopRegion()->ops()) {
    for (const auto& input : op->inputs()) {
      auto* tensor = dynamic_cast<TensorValue*>(input.get_weak());
      if (tensor && tensor->name() == product->name()) {
        if (!seen_producer || consumer) { return reject("nonprivate_product"); }
        consumer = op;
      }
    }
    if (op == mul) { seen_producer = true; }
  }
  if (!consumer) { return reject("no_consumer"); }
  auto* cast = dynamic_cast<ir::linalg::CastTypeOp*>(consumer.get());
  if (!cast || !dynamic_cast<QNNCastTypeOp*>(cast->getAOp())
      || cast->inputs().size() != 1 || cast->outputs().size() != 1) {
    const auto* linalg = dynamic_cast<const ir::linalg::LinalgIROp*>(consumer.get());
    MLLM_INFO("QNN_MLP_HVX_MUL_QUANTIZE16_CONSUMER graph={} op={}", graph_name,
              linalg ? linalg->getAOp()->getName() : "<non-linalg>");
    return reject("consumer_signature");
  }
  const auto* cast_input = dynamic_cast<const TensorValue*>(cast->inputs().front().get_weak());
  const auto* quantized = dynamic_cast<const TensorValue*>(cast->outputs().front().get_weak());
  if (!cast_input || !quantized || cast_input->tensor_.dtype() != kFloat32
      || quantized->tensor_.dtype() != kInt16 || cast_input->tensor_.shape() != shape || quantized->tensor_.shape() != shape) {
    return reject("cast_layout");
  }
  return consumer;
}

bool addMlpMulQuantize(const std::string& graph_name, const std::shared_ptr<QNNModel>& model, const ir::op_ptr_t& mul,
                       const ir::op_ptr_t& cast) {
  const auto gate = mul->inputs().front()->cast_<TensorValue>();
  const auto up = mul->inputs().back()->cast_<TensorValue>();
  const auto output = cast->outputs().front()->cast_<TensorValue>();
  const auto& cast_name = cast->cast_<ir::linalg::CastTypeOp>()->getAOp()->getName();
  const auto quant_params = createQuantizeParams(output->tensor_);
  const float scale = quant_params.scaleOffsetEncoding.scale;
  if (!(scale > 0.0F && std::isfinite(scale))) {
    MLLM_ERROR("Invalid INT16 quantization scale for graph '{}'", graph_name);
    return false;
  }
  auto scale_param =
      QNNParamTensorWrapper::create("scale", cast_name + ".quantize_scale", QNN_DATATYPE_FLOAT_32, std::vector<uint32_t>{1});
  *static_cast<float*>(scale_param->alloc()) = scale;
  if (model->addTensor(output->name(), getQnnOutputTensorType(output), output->tensor_, quant_params) != MODEL_NO_ERROR) {
    MLLM_ERROR("Failed to add fused INT16 output {} to graph '{}'", output->name(), graph_name);
    return false;
  }
  const auto err = model->addNode(QNN_OPCONFIG_VERSION_1, cast_name + ".llama_hvx_mul_quantize16", QNN_Custom_Op_Package,
                                  "LLaMAMulQuantize16", {scale_param}, {}, {gate->name(), up->name()}, {output->name()});
  if (err != MODEL_NO_ERROR) {
    MLLM_ERROR("Failed to add fused multiply and quantize to graph '{}': {}", graph_name, static_cast<int>(err));
    return false;
  }
  MLLM_INFO("QNN_MLP_HVX_MUL_QUANTIZE16_LOWERING graph={} node={}", graph_name, cast_name);
  return true;
}

}  // namespace

QNNGraphBuildPass::QNNGraphBuildPass() {
  regPattern<QNNAddPattern, QNNMulPattern, QNNLinearPattern, QNNRMSNormPattern, QNNViewPattern, QNNTransposePattern,
             QNNX2XPattern, QNNCastTypePattern, QNNSiLUPattern>();

  // register custom op patterns
  patterns_.emplace((mllm::OpTypes)mllm::Context::instance().lookupCustomizedOpId(mllm::kQNN, "DequantizeAdd"),
                    std::make_shared<QNNDequantizeAddPattern>());
  patterns_.emplace(static_cast<mllm::OpTypes>(mllm::Context::instance().lookupCustomizedOpId(mllm::kQNN, "ShadowCorrectionPacket")),
                    std::make_shared<QNNShadowCorrectionPacketPattern>());
}

uint8_t QNNGraphBuildPass::run(const ir::node_ptr_t& op) {
  // The top op should be ModuleOp
  if (!op->isa_<ir::ModuleOp>()) {
    MLLM_ERROR("QNNGraphBuildPass expects ModuleOp as top level op");
    return ir::PASS_RET_FAILURE;
  }

  auto module_op = op->cast_<ir::ModuleOp>();
  auto top_region = module_op->getTopRegion();

  if (!top_region) {
    MLLM_ERROR("ModuleOp has no top region");
    return ir::PASS_RET_FAILURE;
  }

  // Find the subgraphs we need to process
  std::vector<ir::graph::SubGraphOp::ptr_t> graphs_ir_to_be_compiled;

  for (auto& region_op : top_region->ops()) {
    if (auto sub_graph_op = std::dynamic_pointer_cast<ir::graph::SubGraphOp>(region_op)) {
      auto symbol_attr = sub_graph_op->getSymbolAttr();
      if (symbol_attr) {
        auto sub_graph_name = symbol_attr->str();

        if (sub_graph_op->getDevice() == DeviceTypes::kQNN) { graphs_ir_to_be_compiled.push_back(sub_graph_op); }
      }
    }
  }

  // Verify all subgraph has no CallGraphOp
  for (auto& graph_ir : graphs_ir_to_be_compiled) {
    auto graph_region = graph_ir->getTopRegion();
    if (!graph_region) continue;

    for (auto& region_op : graph_region->ops()) {
      if (region_op->isa_<ir::graph::CallGraphOp>()) {
        MLLM_ERROR("Found CallGraphOp in SubGraph. You should call GraphInlinePass before QNNGraphBuildPass. "
                   "For now, manually inline CallGraphOp is highly recommended.");
        return ir::PASS_RET_FAILURE;
      }
    }
  }

  const char* choice = std::getenv("MLLM_QNN_SHADOW_COMPACT_CORRECTION");
  const bool compact_shadow = choice && std::string_view(choice) == "1";
  for (auto& graph_ir : graphs_ir_to_be_compiled) {
    const std::string name = graph_ir->getSymbolAttr()->str();
    const auto raw = compact_shadow ? shadowRawGraphOutput(graph_ir) : nullptr;
    if (!raw) {
      if (!buildQnnGraph(graph_ir, name)) { return ir::PASS_RET_FAILURE; }
      continue;
    }

    const std::string replay_name = name + std::string(shadow_raw_graph_suffix);
    if (!buildQnnGraph(graph_ir, replay_name)) { return ir::PASS_RET_FAILURE; }
    const auto original_output = raw->getAttr("is_graph_output");
    raw->removeAttr("is_graph_output");
    bool built = false;
    try {
      built = buildQnnGraph(graph_ir, name);
    } catch (...) {
      raw->setAttr("is_graph_output", original_output);
      throw;
    }
    raw->setAttr("is_graph_output", original_output);
    if (!built) { return ir::PASS_RET_FAILURE; }
    MLLM_INFO("QNN_SHADOW_COMPACT_GRAPH graph={} replay={}", name, replay_name);
  }

  return ir::PASS_RET_SUCCESS;
}

bool QNNGraphBuildPass::buildQnnGraph(const ir::graph::SubGraphOp::ptr_t& sub_graph_op,
                                      const std::string& graph_name) {
  auto& mllm_ctx = mllm::Context::instance();
  auto backend = mllm_ctx.getBackend(kQNN);

  if (!backend) {
    MLLM_ERROR("QNN backend not found in context");
    return false;
  }

  // Cast to QNN backend to access QNN-specific methods
  auto qnn_backend = std::dynamic_pointer_cast<QNNBackend>(backend);
  if (!qnn_backend) {
    MLLM_ERROR("Failed to cast backend to QNN backend");
    return false;
  }

  auto graph_region = sub_graph_op->getTopRegion();
  if (!graph_region) {
    MLLM_ERROR("SubGraphOp has no top region");
    return false;
  }

  // Create QNN model using the backend
  auto qnn_model = qnn_backend->createQnnGraph(graph_name);
  if (!qnn_model) {
    MLLM_ERROR("Failed to create QNN model for graph '{}'", graph_name);
    return false;
  }

  // A loaded context already owns finalized graph tensors, nodes, and weights.
  // Re-adding trace inputs would report duplicate tensors and, worse, can hide
  // a partial rebuild behind a nominally successful pass. The trace is still
  // needed to bind mllm execution ops to these graph names, but QNN graph
  // construction must be skipped.
  if (qnn_model->isGraphFinalized()) { return true; }

  // Add graph inputs first
  for (auto& input : sub_graph_op->inputs()) {
    auto input_tensor = input->cast_<ir::tensor::TensorValue>();

    auto quantize_param = DEFAULT_QUANTIZE_PARAMS;
    if (input_tensor->tensor_.dtype() == kInt8 || input_tensor->tensor_.dtype() == kInt16) {
      // the scale is set in Op::load and should be scaled to int range
      auto scale = getQuantScale(input_tensor->tensor_);
      quantize_param = Qnn_QuantizeParams_t{QNN_DEFINITION_DEFINED,
                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                            {.scaleOffsetEncoding = {.scale = scale, .offset = 0}}};
    }
    ModelError_t err =
        qnn_model->addTensor(input_tensor->name(), QNN_TENSOR_TYPE_APP_WRITE, input_tensor->tensor_, quantize_param);
    if (err != MODEL_NO_ERROR) {
      MLLM_ERROR("Failed to add input tensor {} to graph '{}'", input_tensor->name(), graph_name);
      return false;
    }
  }

  // Process each operation in the subgraph
  const char* fuse_request = std::getenv("MLLM_QNN_MLP_HVX_MUL_QUANTIZE16");
  const bool fuse_mul_quantize = fuse_request && std::string_view(fuse_request) == "1" && isQwenMlpGraph(shadowMainGraphName(graph_name));
  const auto& ops = graph_region->ops();
  const auto masked = findShadowMaskLowering(graph_name, sub_graph_op);
  ir::op_ptr_t fused_consumer;
  if (fuse_mul_quantize) { MLLM_INFO("QNN_MLP_HVX_MUL_QUANTIZE16_SCAN graph={}", graph_name); }
  for (const auto& region_op : ops) {
    if (region_op == fused_consumer) { continue; }
    if (region_op == masked.multiply) {
      if (!addShadowMaskedMultiply(graph_name, qnn_model, masked)) { return false; }
      continue;
    }
    if (fuse_mul_quantize && !fused_consumer) {
      if (auto consumer = findMlpQuantizeConsumer(graph_name, sub_graph_op, region_op)) {
        if (!addMlpMulQuantize(graph_name, qnn_model, region_op, consumer)) { return false; }
        fused_consumer = std::move(consumer);
        continue;
      }
    }
    // Handle linalg operations
    if (auto linalg_op = std::dynamic_pointer_cast<ir::linalg::LinalgIROp>(region_op)) {
      auto op_types = linalg_op->getAOpTypes();
      std::vector<ir::tensor::TensorValue::ptr_t> op_inputs;
      std::vector<ir::tensor::TensorValue::ptr_t> op_outputs;

      // Collect op inputs
      for (auto& input_val : linalg_op->inputs()) { op_inputs.emplace_back(input_val->cast_<ir::tensor::TensorValue>()); }
      // Collect op outputs
      for (auto& output_val : linalg_op->outputs()) { op_outputs.emplace_back(output_val->cast_<ir::tensor::TensorValue>()); }

      if (region_op == masked.packet) {
        if (!QNNShadowCorrectionPacketPattern::addNodeWithMask(graph_name, region_op, op_inputs, op_outputs, masked.name)) {
          return false;
        }
        continue;
      }
      if (patterns_.contains(op_types)) {
        auto pattern = patterns_[op_types];
        if (!pattern->addNode(graph_name, linalg_op, op_inputs, op_outputs)) {
          MLLM_ERROR("Failed to add node for op type: {} in graph '{}'", optype2Str(op_types), graph_name);
          return false;
        }
      } else {
        MLLM_WARN("No pattern registered for op type: {}", optype2Str(op_types));
      }
    } else if (auto ret_op = std::dynamic_pointer_cast<ir::cf::ReturnOp>(region_op)) {
      continue;
    } else {
      MLLM_WARN("Unsupported op type in QNN subgraph: {}", (int)region_op->getKind());
    }
  }

  // Finalize the QNN graph
  if (!qnn_backend->graphFinalize(graph_name)) {
    MLLM_ERROR("Failed to finalize QNN graph '{}'", graph_name);
    return false;
  }
  return true;
}

}  // namespace mllm::qnn
