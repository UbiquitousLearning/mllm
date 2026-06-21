// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/qnn/passes/QNNOpNamingPass.hpp"
#include "mllm/compile/ir/builtin/Op.hpp"
#include "mllm/compile/ir/graph/Op.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/tensor/Value.hpp"
#include "mllm/compile/ir/Node.hpp"
#include "mllm/core/OpTypes.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/utils/Log.hpp"
#include <unordered_map>

namespace mllm::qnn {

namespace {

/**
 * @brief Visit and process a call graph operation
 * @param ir_ctx IR context
 * @param call_op Call graph operation to process
 */
void visitCallGraph(const std::shared_ptr<ir::IRContext>& ir_ctx, const ir::graph::CallGraphOp::ptr_t& call_op);

/**
 * @brief Visit and process a subgraph operation, assigning names to unnamed operations
 * @param ir_ctx IR context
 * @param subgraph_op Subgraph operation to process
 */
void visitSubGraph(const std::shared_ptr<ir::IRContext>& ir_ctx, const ir::graph::SubGraphOp::ptr_t& subgraph_op) {
  auto region = subgraph_op->getTopRegion();

  std::unordered_map<OpTypes, int32_t> intra_optype_cnt;

  for (auto& op : region->ops()) {
    // If has call graph op
    if (op->isa_<ir::graph::CallGraphOp>()) {
      visitCallGraph(ir_ctx, op->cast_<ir::graph::CallGraphOp>());
    } else if (op->isa_<ir::linalg::LinalgIROp>()) {
      auto mllm_op = op->cast_<ir::linalg::LinalgIROp>()->getAOp();
      auto mllm_op_type = op->cast_<ir::linalg::LinalgIROp>()->getAOpTypes();

      // If this op has no name, assign a unique name
      if (mllm_op->getName().empty()) {
        if (!intra_optype_cnt.count(mllm_op_type)) { intra_optype_cnt.insert({mllm_op_type, -1}); }
        intra_optype_cnt[mllm_op_type] += 1;

        // Create unique name: module_name.op_type.index
        std::string unique_name = subgraph_op->getSymbolAttr()->str() + "." + optype2Str(mllm_op_type) + "."
                                  + std::to_string(intra_optype_cnt[mllm_op_type]);

        mllm_op->setName(unique_name);
      }
    }
    // For other operation types, we can add handling if needed
    // Currently just skip them as they might not be relevant for QNN
  }
}

void visitCallGraph(const std::shared_ptr<ir::IRContext>& ir_ctx, const ir::graph::CallGraphOp::ptr_t& call_op) {
  // Panic if input of call graph has no name (for validation)
  auto& inputs = call_op->inputs();
  for (auto& input : inputs) {
    MLLM_RT_ASSERT(input->isa_<ir::tensor::TensorValue>() && !input->cast_<ir::tensor::TensorValue>()->name().empty());
  }

  // Lookup the subgraph and visit it
  auto subgraph_op = ir_ctx->lookupSymbolTable(call_op->getSymbolAttr()->str())->cast_<ir::graph::SubGraphOp>();
  MLLM_RT_ASSERT(subgraph_op != nullptr);

  visitSubGraph(ir_ctx, subgraph_op);
}

}  // anonymous namespace

uint8_t QNNOpNamingPass::run(const ir::node_ptr_t& op) {
  // The top op should be ModuleOp
  MLLM_RT_ASSERT(op->isa_<ir::ModuleOp>());

  auto module_op = op->cast_<ir::ModuleOp>();
  auto writer = ir::IRWriter(getCtx(), module_op->getTopRegion());

  // Find the main CallGraphOp
  ir::graph::CallGraphOp::ptr_t call_main_graph_op = nullptr;
  writer.walk<ir::graph::CallGraphOp>(
      [&](ir::IRWriter& /*writer*/, const ir::graph::CallGraphOp::ptr_t& call_op) -> ir::IRWriter::WalkResult {
        // Make sure there is only one call graph op in the ModuleOp
        MLLM_RT_ASSERT_EQ(call_main_graph_op, nullptr);

        call_main_graph_op = call_op;
        return ir::IRWriter::WalkResult::WALK_CONTINUE;
      });

  // Visit all graphs and assign names to unnamed operations
  if (call_main_graph_op != nullptr) { visitCallGraph(getCtx(), call_main_graph_op); }

  return ir::PASS_RET_SUCCESS;
}

ir::Pass::ptr_t createQNNOpNamingPass() { return std::make_shared<QNNOpNamingPass>(); }

}  // namespace mllm::qnn