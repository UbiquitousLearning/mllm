// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/qnn/aot/passes/MergeLLMHeadIntoMainGraphPass.hpp"
#include "mllm/compile/ir/builtin/Op.hpp"
#include "mllm/compile/ir/graph/Op.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/tensor/Value.hpp"
#include "mllm/compile/ir/cf/Op.hpp"
#include "mllm/compile/ir/Node.hpp"
#include "mllm/core/OpTypes.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::qnn::aot {

uint8_t MergeLLMHeadIntoMainGraphPass::run(const ir::node_ptr_t& op) {
  // The top op should be ModuleOp
  MLLM_RT_ASSERT(op->isa_<ir::ModuleOp>());

  auto module_op = op->cast_<ir::ModuleOp>();
  auto writer = ir::IRWriter(getCtx(), module_op->getTopRegion());

  // Find the main CallGraphOp
  ir::graph::CallGraphOp::ptr_t call_main_graph_op = nullptr;
  writer.walk<ir::graph::CallGraphOp>(
      [&](ir::IRWriter& /*writer*/, const ir::graph::CallGraphOp::ptr_t& call_op) -> ir::IRWriter::WalkResult {
        MLLM_RT_ASSERT_EQ(call_main_graph_op, nullptr);
        call_main_graph_op = call_op;
        return ir::IRWriter::WalkResult::WALK_CONTINUE;
      });

  if (call_main_graph_op == nullptr) { return ir::PASS_RET_SUCCESS; }

  // Get the main graph
  auto main_graph = getCtx()->lookupSymbolTable(call_main_graph_op->getSymbolAttr()->str())->cast_<ir::graph::SubGraphOp>();
  MLLM_RT_ASSERT(main_graph != nullptr);

  // Find the LLM head linear op in the main graph
  // The requirement says: "lm_head" is a linear linalg op in TopModule
  ir::linalg::LinalgIROp::ptr_t llm_head_op = nullptr;

  writer.walk<ir::linalg::LinalgIROp>(
      [&](ir::IRWriter& /*writer*/, const ir::linalg::LinalgIROp::ptr_t& linear_op) -> ir::IRWriter::WalkResult {
        auto name = linear_op->getAOp()->getName();
        if (name == "lm_head" || name == "lm_head_out") { llm_head_op = linear_op; }
        return ir::IRWriter::WalkResult::WALK_CONTINUE;
      });

  if (llm_head_op == nullptr) {
    // No LLM head found, nothing to merge
    return ir::PASS_RET_SUCCESS;
  }

  // Find the ReturnOp in the main graph
  auto main_region = main_graph->getTopRegion();
  ir::cf::ReturnOp::ptr_t return_op = nullptr;
  for (auto& op : main_region->ops()) {
    if (op->isa_<ir::cf::ReturnOp>()) {
      return_op = op->cast_<ir::cf::ReturnOp>();
      break;
    }
  }

  if (return_op == nullptr) {
    // No return op found, nothing to merge
    return ir::PASS_RET_SUCCESS;
  }

  // Remove op in top module
  {
    auto rw = ir::IRWriter(getCtx(), module_op->getTopRegion());
    rw.removeOp(llm_head_op);
  }

  // Entering in main graph and move linalg op into this graph
  {
    auto rw = ir::IRWriter(getCtx(), main_graph->getTopRegion());
    rw.insertOpAtPos(return_op, ir::IRWriter::Position::BEFORE, llm_head_op);
  }

  // Make sure the output is right
  {
    MLLM_RT_ASSERT_EQ(llm_head_op->inputs().size(), 1);
    MLLM_RT_ASSERT_EQ(llm_head_op->outputs().size(), 1);
    auto llm_head_input = llm_head_op->inputs().front();
    auto llm_head_output = llm_head_op->outputs().front();
    auto& graph_region_outs = main_graph->outputs();

    // Replace llm_head_input with llm_head_output in graph outputs
    std::replace(graph_region_outs.begin(), graph_region_outs.end(), llm_head_input, llm_head_output);
    std::replace(return_op->inputs().begin(), return_op->inputs().end(), llm_head_input, llm_head_output);
    std::replace(main_graph->getTopRegion()->outputs().begin(), main_graph->getTopRegion()->outputs().end(), llm_head_input,
                 llm_head_output);
  }

  // Make sure the call_graph_op's output is right
  {
    MLLM_RT_ASSERT_EQ(llm_head_op->inputs().size(), 1);
    MLLM_RT_ASSERT_EQ(llm_head_op->outputs().size(), 1);
    auto llm_head_input = llm_head_op->inputs().front();
    auto llm_head_output = llm_head_op->outputs().front();
    auto& graph_region_outs = call_main_graph_op->outputs();

    // Replace llm_head_input with llm_head_output in call graph outputs
    std::replace(graph_region_outs.begin(), graph_region_outs.end(), llm_head_input, llm_head_output);
  }

  return ir::PASS_RET_SUCCESS;
}

ir::Pass::ptr_t createMergeLLMHeadIntoMainGraphPass() { return std::make_shared<MergeLLMHeadIntoMainGraphPass>(); }

}  // namespace mllm::qnn::aot
