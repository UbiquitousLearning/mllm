// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/qnn/passes/QNNGraphIOTensorPass.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/compile/ir/builtin/Op.hpp"
#include "mllm/compile/ir/graph/Op.hpp"
#include "mllm/compile/ir/cf/Op.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::qnn {

namespace {

/**
 * @brief Visit and process a call graph operation
 * @param ir_ctx IR context for creating attributes
 * @param call_op Call graph operation to process
 */
void visitCallGraph(const std::shared_ptr<ir::IRContext>& ir_ctx, const ir::graph::CallGraphOp::ptr_t& call_op);

/**
 * @brief Visit and process a subgraph operation
 * @param ir_ctx IR context for creating attributes
 * @param subgraph_op Subgraph operation to process
 */
void visitSubGraph(const std::shared_ptr<ir::IRContext>& ir_ctx, const ir::graph::SubGraphOp::ptr_t& subgraph_op) {
  // Mark all inputs of the subgraph as graph inputs
  for (auto& input : subgraph_op->inputs()) { input->setAttr("is_graph_input", ir_ctx->create<ir::BoolAttr>(true)); }

  // Get the top region of the subgraph
  auto region = subgraph_op->getTopRegion();

  // Traverse all operations in the region
  for (auto& op : region->ops()) {
    // Handle call graph operations recursively
    if (op->isa_<ir::graph::CallGraphOp>()) {
      visitCallGraph(ir_ctx, op->cast_<ir::graph::CallGraphOp>());
    }
    // Handle return operations - their inputs are subgraph outputs
    else if (op->isa_<ir::cf::ReturnOp>()) {
      // The input values of this cf::ReturnOp are the outputs of the subgraph
      for (auto& input : op->inputs()) { input->setAttr("is_graph_output", ir_ctx->create<ir::BoolAttr>(true)); }
    }
  }
}

void visitCallGraph(const std::shared_ptr<ir::IRContext>& ir_ctx, const ir::graph::CallGraphOp::ptr_t& call_op) {
  // Get the symbol referenced by this call graph operation
  auto symbol_attr = call_op->getSymbolAttr();
  MLLM_RT_ASSERT(symbol_attr != nullptr);

  // Lookup the subgraph in the symbol table and visit it
  auto subgraph_op = ir_ctx->lookupSymbolTable(symbol_attr->str())->cast_<ir::graph::SubGraphOp>();
  MLLM_RT_ASSERT(subgraph_op != nullptr);

  visitSubGraph(ir_ctx, subgraph_op);
}

}  // anonymous namespace

uint8_t QNNGraphIOTensorPass::run(const ir::node_ptr_t& op) {
  // The top operation should be ModuleOp
  MLLM_RT_ASSERT(op->isa_<ir::ModuleOp>());

  auto module_op = op->cast_<ir::ModuleOp>();
  auto writer = ir::IRWriter(getCtx(), module_op->getTopRegion());

  // Find the main call graph operation
  ir::graph::CallGraphOp::ptr_t main_call_graph_op = nullptr;

  writer.walk<ir::graph::CallGraphOp>(
      [&](ir::IRWriter& /*writer*/, const ir::graph::CallGraphOp::ptr_t& call_op) -> ir::IRWriter::WalkResult {
        // Ensure there is only one main call graph operation in the module
        MLLM_RT_ASSERT_EQ(main_call_graph_op, nullptr);

        main_call_graph_op = call_op;
        return ir::IRWriter::WalkResult::WALK_CONTINUE;
      });

  // Visit and process the main call graph
  if (main_call_graph_op != nullptr) { visitCallGraph(getCtx(), main_call_graph_op); }

  return ir::PASS_RET_SUCCESS;
}

ir::Pass::ptr_t createQNNGraphIOTensorPass() { return std::make_shared<QNNGraphIOTensorPass>(); }

}  // namespace mllm::qnn