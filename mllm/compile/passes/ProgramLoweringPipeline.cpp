/**
 * @file ProgramLoweringPipeline.cpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-30
 *
 */
#include "mllm/compile/ir/Node.hpp"
#include "mllm/compile/ir/builtin/Op.hpp"
#include "mllm/compile/ir/graph/Op.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/program/Op.hpp"
#include "mllm/compile/ir/tensor/Value.hpp"
#include "mllm/compile/passes/Pattern.hpp"
#include "mllm/compile/passes/ProgramLoweringPipeline.hpp"

namespace mllm::ir {

bool LinalgIR2ProgramPattern::isMatch(const op_ptr_t& node) { return node->isa_<ir::linalg::LinalgIROp>(); }

bool LinalgIR2ProgramPattern::rewrite(IRWriter& writer, const op_ptr_t& node) {
  std::vector<ir::tensor::TensorValue::ptr_t> ins;
  std::vector<ir::tensor::TensorValue::ptr_t> ous;

  for (auto& input : node->inputs()) {
    if (input->isa_<ir::tensor::TensorValue>()) { ins.push_back(input->cast_<ir::tensor::TensorValue>()); }
  }

  for (auto& output : node->outputs()) {
    if (output->isa_<ir::tensor::TensorValue>()) { ous.push_back(output->cast_<ir::tensor::TensorValue>()); }
  }

  auto mllm_op = node->cast_<ir::linalg::LinalgIROp>()->getAOp();

  auto new_op = writer.createAndReplaceOp<ir::program::InstructionOp>(node, ins, ous);
  new_op->pushMllmOp(mllm_op);

  return true;
}

LinalgIR2ProgramPattern::ptr_t LinalgIR2ProgramPattern::create() { return std::make_shared<LinalgIR2ProgramPattern>(); }

Linalg2ProgramPass::Linalg2ProgramPass() { regPattern<LinalgIR2ProgramPattern>(); }

uint8_t Linalg2ProgramPass::run(const node_ptr_t& op) {
  // The top op should be ModuleOp
  MLLM_RT_ASSERT(op->isa_<ir::ModuleOp>());

  // Find the subgraph we need to process
  auto r = ir::IRWriter(getCtx(), op->cast_<ir::ModuleOp>()->getTopRegion());

  // Graphs need to be write.
  std::vector<ir::graph::SubGraphOp::ptr_t> graphs_need_to_transform;
  r.walk<ir::graph::SubGraphOp>([&](ir::IRWriter& reader, const ir::graph::SubGraphOp::ptr_t& op) -> ir::IRWriter::WalkResult {
    graphs_need_to_transform.emplace_back(op);
    return ir::IRWriter::WalkResult::WALK_CONTINUE;
  });

  // Rewrite
  for (auto& g : graphs_need_to_transform) {
    auto g_r = ir::IRWriter(getCtx(), g->getTopRegion());
    g_r.walk<ir::linalg::LinalgIROp>([&](ir::IRWriter& rw, const ir::Op::ptr_t& op) -> ir::IRWriter::WalkResult {
      for (auto& p : patterns_) {
        if (p->isMatch(op)) { p->rewrite(g_r, op); }
      }
      return ir::IRWriter::WalkResult::WALK_CONTINUE;
    });
  }

  return ir::PASS_RET_SUCCESS;
}

Pass::ptr_t createLinalg2ProgramPass() { return std::make_shared<Linalg2ProgramPass>(); }

bool GraphSubGraph2ProgramPattern::isMatch(const op_ptr_t& node) { return node->isa_<ir::graph::SubGraphOp>(); }

bool GraphSubGraph2ProgramPattern::rewrite(IRWriter& writer, const op_ptr_t& node) {
  auto subgraph_op = node->cast_<ir::graph::SubGraphOp>();
  auto old_region = subgraph_op->getTopRegion();
  auto old_symbol = subgraph_op->getSymbolAttr();

  auto type = program::FragmentType::kCode;

  // Set fragment type
  if (old_symbol->str() == "init") { type = program::FragmentType::kData; }

  // Remove global symbol
  writer.getContext()->removeFromSymbolTable(old_symbol->str());
  auto fragment = writer.createAndReplaceOp<ir::program::FragmentOp>(node, type, old_symbol);

  // Region reset to fragment
  auto fragment_region = fragment->getTopRegion();
  fragment_region->inputs() = old_region->inputs();
  fragment_region->outputs() = old_region->outputs();
  fragment_region->ops() = old_region->ops();

  return true;
}

GraphSubGraph2ProgramPattern::ptr_t GraphSubGraph2ProgramPattern::create() {
  return std::make_shared<GraphSubGraph2ProgramPattern>();
}

bool GraphCallGraph2ProgramPattern::isMatch(const op_ptr_t& node) { return node->isa_<ir::graph::CallGraphOp>(); }

bool GraphCallGraph2ProgramPattern::rewrite(IRWriter& writer, const op_ptr_t& node) {
  auto call_graph_op = node->cast_<ir::graph::CallGraphOp>();

  // We found the entry point !
  if (call_graph_op->belongsTo()->isa_<ModuleOp>()) {
    std::vector<val_weak_ptr_t> inputs;
    std::vector<val_weak_ptr_t> outputs;
    for (auto& i : call_graph_op->inputs()) { inputs.emplace_back(i->cast_<Val>()); }
    for (auto& o : call_graph_op->outputs()) { outputs.emplace_back(o->cast_<Val>()); }

    writer.createAndReplaceOp<ir::program::EntryPointOp>(node, call_graph_op->getSymbolAttr(), inputs, outputs);
    return true;
  }

  // If not entry point.
  // TODO

  return true;
}

GraphCallGraph2ProgramPattern::ptr_t GraphCallGraph2ProgramPattern::create() {
  return std::make_shared<GraphCallGraph2ProgramPattern>();
}

uint8_t Graph2ProgramPass::run(const node_ptr_t& op) {
  // The top op should be ModuleOp
  MLLM_RT_ASSERT(op->isa_<ir::ModuleOp>());

  // Find the subgraph we need to process
  auto r = ir::IRWriter(getCtx(), op->cast_<ir::ModuleOp>()->getTopRegion());

  // Rewrite
  r.walk<ir::graph::GraphIROp>([&](ir::IRWriter& reader, const ir::graph::GraphIROp::ptr_t& op) -> ir::IRWriter::WalkResult {
    for (auto& p : patterns_) {
      if (p->isMatch(op)) { p->rewrite(r, op); }
    }
    return ir::IRWriter::WalkResult::WALK_CONTINUE;
  });

  return ir::PASS_RET_SUCCESS;
}

Graph2ProgramPass::Graph2ProgramPass() { regPattern<GraphSubGraph2ProgramPattern, GraphCallGraph2ProgramPattern>(); }

Pass::ptr_t createGraph2ProgramPass() { return std::make_shared<Graph2ProgramPass>(); }

uint8_t CF2ProgramPass::run(const node_ptr_t& op) { return ir::PASS_RET_SUCCESS; }

Pass::ptr_t createCF2ProgramPass() { return std::make_shared<CF2ProgramPass>(); }

std::vector<Pass::ptr_t> createProgramLoweringPipeline() {
  std::vector<Pass::ptr_t> ret;

  ret.push_back(createLinalg2ProgramPass());
  ret.push_back(createGraph2ProgramPass());
  ret.push_back(createCF2ProgramPass());

  return ret;
}

}  // namespace mllm::ir
