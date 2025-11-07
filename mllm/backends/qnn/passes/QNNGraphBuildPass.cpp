// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/qnn/passes/QNNGraphBuildPass.hpp"
#include "QnnTypes.h"
#include "mllm/backends/qnn/QNNBackend.hpp"
#include "mllm/backends/qnn/QNNUtils.hpp"
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

#include <memory>

namespace mllm::qnn {

QNNGraphBuildPass::QNNGraphBuildPass() {
  regPattern<QNNAddPattern, QNNMulPattern, QNNLinearPattern, QNNRMSNormPattern, QNNViewPattern, QNNTransposePattern,
             QNNX2XPattern, QNNCastTypePattern, QNNSiLUPattern>();

  // register custom op patterns
  patterns_.emplace((mllm::OpTypes)mllm::Context::instance().lookupCustomizedOpId(mllm::kQNN, "DequantizeAdd"),
                    std::make_shared<QNNDequantizeAddPattern>());
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

  // QNN Graph build and compile
  for (auto& graph_ir : graphs_ir_to_be_compiled) { buildQnnGraph(graph_ir); }

  return ir::PASS_RET_SUCCESS;
}

void QNNGraphBuildPass::buildQnnGraph(const ir::graph::SubGraphOp::ptr_t& sub_graph_op) {
  auto& mllm_ctx = mllm::Context::instance();
  auto backend = mllm_ctx.getBackend(kQNN);

  if (!backend) {
    MLLM_ERROR("QNN backend not found in context");
    return;
  }

  // Cast to QNN backend to access QNN-specific methods
  auto qnn_backend = std::dynamic_pointer_cast<QNNBackend>(backend);
  if (!qnn_backend) {
    MLLM_ERROR("Failed to cast backend to QNN backend");
    return;
  }

  auto graph_region = sub_graph_op->getTopRegion();
  if (!graph_region) {
    MLLM_ERROR("SubGraphOp has no top region");
    return;
  }

  std::string graph_name = sub_graph_op->getSymbolAttr()->str();

  // Create QNN model using the backend
  auto qnn_model = qnn_backend->createQnnGraph(graph_name);
  if (!qnn_model) {
    MLLM_ERROR("Failed to create QNN model for graph '{}'", graph_name);
    return;
  }

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
    ModelError_t err = qnn_model->addTensor(input_tensor->name(), QNN_TENSOR_TYPE_APP_WRITE, input_tensor->tensor_, quantize_param);
    if (err != MODEL_NO_ERROR) {
      MLLM_ERROR("Failed to add input tensor {} to graph '{}'", input_tensor->name(), graph_name);
      return;
    }
  }

  // Record MLLM expected output order from ReturnOp
  std::vector<std::string> expectedOutputOrder;
  ir::cf::ReturnOp::ptr_t return_op = nullptr;

  // Process each operation in the subgraph
  for (auto& region_op : graph_region->ops()) {
    // Handle linalg operations
    if (auto linalg_op = std::dynamic_pointer_cast<ir::linalg::LinalgIROp>(region_op)) {
      auto op_types = linalg_op->getAOpTypes();
      std::vector<ir::tensor::TensorValue::ptr_t> op_inputs;
      std::vector<ir::tensor::TensorValue::ptr_t> op_outputs;

      // Collect op inputs
      for (auto& input_val : linalg_op->inputs()) { op_inputs.emplace_back(input_val->cast_<ir::tensor::TensorValue>()); }
      // Collect op outputs
      for (auto& output_val : linalg_op->outputs()) { op_outputs.emplace_back(output_val->cast_<ir::tensor::TensorValue>()); }

      if (patterns_.contains(op_types)) {
        auto pattern = patterns_[op_types];
        if (!pattern->addNode(graph_name, linalg_op, op_inputs, op_outputs)) {
          MLLM_ERROR("Failed to add node for op type: {} in graph '{}'", optype2Str(op_types), graph_name);
          return;
        }
      } else {
        MLLM_WARN("No pattern registered for op type: {}", optype2Str(op_types));
      }
    } else if (auto ret_op = std::dynamic_pointer_cast<ir::cf::ReturnOp>(region_op)) {
      // Record ReturnOp to extract expected output order
      return_op = ret_op;
    } else {
      MLLM_WARN("Unsupported op type in QNN subgraph: {}", (int)region_op->getKind());
    }
  }

  // Extract MLLM expected output order from ReturnOp inputs
  if (return_op) {
    for (auto& input : return_op->inputs()) {
      auto output_tensor = input->cast_<ir::tensor::TensorValue>();
      if (output_tensor) {
        expectedOutputOrder.push_back(output_tensor->name());
      }
    }
    // Set expected output order in QNN model
    qnn_model->setExpectedOutputOrder(expectedOutputOrder);
    // MLLM_INFO("QNNGraphBuildPass: Recorded MLLM expected output order for graph '{}' with {} outputs", graph_name,
    //           expectedOutputOrder.size());
  } else {
    MLLM_WARN("QNNGraphBuildPass: No ReturnOp found in graph '{}', cannot determine expected output order", graph_name);
  }

  // Finalize the QNN graph
  if (!qnn_backend->graphFinalize(graph_name)) {
    MLLM_ERROR("Failed to finalize QNN graph '{}'", graph_name);
    return;
  }
}

}  // namespace mllm::qnn
