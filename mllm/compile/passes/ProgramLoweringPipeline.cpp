// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <nlohmann/json.hpp>

#include "mllm/compile/ir/cf/Op.hpp"
#include "mllm/compile/ir/tensor/Op.hpp"
#include "mllm/core/OpTypes.hpp"
#include "mllm/compile/ir/Node.hpp"
#include "mllm/compile/ir/builtin/Op.hpp"
#include "mllm/compile/ir/graph/Op.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/program/Op.hpp"
#include "mllm/compile/ir/tensor/Value.hpp"
#include "mllm/compile/passes/Pattern.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/compile/passes/ProgramModeConfigPass.hpp"
#include "mllm/compile/passes/ProgramLoweringPipeline.hpp"
#include "mllm/compile/jit/binary/LinalgIRSerialization.hpp"
#include "mllm/compile/passes/ExtractConstantTensorPass.hpp"
#include "mllm/compile/passes/ProgramIntrinsicIdIndexPass.hpp"
#include "mllm/compile/passes/FlattenTensorAndLinalgSymbol2ProgramSymbolPass.hpp"
#include "mllm/utils/Enumerate.hpp"

namespace mllm::ir {

bool LinalgIR2ProgramPattern::isMatch(const op_ptr_t& node) {
  return node->isa_<ir::linalg::LinalgIROp>() && (!node->isa_<ir::linalg::RegisterOp>());
}

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
  auto mllm_op_name = optype2Str(mllm_op->getOpType());
  auto mllm_op_options = jit::binary::dumpLinalgIROptions(node->cast_<ir::linalg::LinalgIROp>()).dump();
  auto kernel_op = writer.createAndReplaceOp<ir::program::KernelLaunchOp>(node, ins, ous, mllm_op_name, mllm_op_options);

  if (!mllm_op->getName().empty()) {
    kernel_op->setAttr("symbol_name", writer.getContext()->create<ir::StrAttr>(mllm_op->getName()));
  }

  return true;
}

LinalgIR2ProgramPattern::ptr_t LinalgIR2ProgramPattern::create() { return std::make_shared<LinalgIR2ProgramPattern>(); }

Linalg2ProgramPass::Linalg2ProgramPass() { regPattern<LinalgIR2ProgramPattern>(); }

uint8_t Linalg2ProgramPass::run(const node_ptr_t& this_top_op) {
  // The top op should be ModuleOp
  MLLM_RT_ASSERT(this_top_op->isa_<ir::ModuleOp>());

  // Find the subgraph we need to process
  auto r = ir::IRWriter(getCtx(), this_top_op->cast_<ir::ModuleOp>()->getTopRegion());

  // Rewrite linalg OP in the top module
  r.walk<ir::linalg::LinalgIROp>([&](ir::IRWriter& rw, const ir::Op::ptr_t& op) -> ir::IRWriter::WalkResult {
    for (auto& p : patterns_) {
      if (p->isMatch(op)) { p->rewrite(r, op); }
    }
    return ir::IRWriter::WalkResult::WALK_CONTINUE;
  });

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
  if (old_symbol->str() == "deinit") { type = program::FragmentType::kData; }
  if (old_symbol->str() == "op_symbol_table") { type = program::FragmentType::kTable; }

  // Remove global symbol
  writer.getContext()->removeFromSymbolTable(old_symbol->str());
  auto fragment = writer.createAndReplaceOp<ir::program::FragmentOp>(node, type, old_symbol);

  // Region reset to fragment
  auto fragment_region = fragment->getTopRegion();
  fragment_region->inputs() = old_region->inputs();
  fragment_region->outputs() = old_region->outputs();
  fragment_region->ops() = old_region->ops();

  ir::IRWriter w(writer.getContext(), fragment_region);
  w.createAtPos<ir::program::LabelOp>(fragment_region->ops().front(), IRWriter::Position::BEFORE,
                                      w.getContext()->create<SymbolAttr>(old_symbol->str() + ".__entry"));

  return true;
}

GraphSubGraph2ProgramPattern::ptr_t GraphSubGraph2ProgramPattern::create() {
  return std::make_shared<GraphSubGraph2ProgramPattern>();
}

bool GraphCallGraph2ProgramPattern::isMatch(const op_ptr_t& node) { return node->isa_<ir::graph::CallGraphOp>(); }

bool GraphCallGraph2ProgramPattern::rewrite(IRWriter& writer, const op_ptr_t& node) {
  auto call_graph_op = node->cast_<ir::graph::CallGraphOp>();

  // If not entry point. Need to jump.
  writer.createAndReplaceOp<ir::program::JumpOp>(node, call_graph_op->getSymbolAttr()->str() + ".__entry");

  return true;
}

GraphCallGraph2ProgramPattern::ptr_t GraphCallGraph2ProgramPattern::create() {
  return std::make_shared<GraphCallGraph2ProgramPattern>();
}

uint8_t Graph2ProgramPass::run(const node_ptr_t& this_top_op) {
  // The top op should be ModuleOp
  MLLM_RT_ASSERT(this_top_op->isa_<ir::ModuleOp>());

  // Find the subgraph we need to process
  auto r = ir::IRWriter(getCtx(), this_top_op->cast_<ir::ModuleOp>()->getTopRegion());

  // Find the callGraphOp, there only can be one callGraphOp in the top Module Region right now.
  std::vector<ir::graph::CallGraphOp::ptr_t> call_graph_ops;
  r.walk<ir::graph::CallGraphOp>([&](ir::IRWriter& reader, const ir::graph::CallGraphOp::ptr_t& op) {
    call_graph_ops.emplace_back(op);
    return ir::IRWriter::WalkResult::WALK_CONTINUE;
  });
  MLLM_RT_ASSERT_EQ(call_graph_ops.size(), 1);
  std::vector<ir::val_ptr_t> bind_input;
  std::vector<ir::val_ptr_t> bind_output;
  for (auto& i : call_graph_ops[0]->inputs()) { bind_input.emplace_back(i->cast_<ir::Val>()); }
  for (auto& o : call_graph_ops[0]->outputs()) { bind_output.emplace_back(o->cast_<ir::Val>()); }

  // IMPORTANT: Insert input bind
  for (auto [_id, _i_bind] : enumerate(bind_input)) {
    r.createAtPos<ir::program::BindOp>(call_graph_ops[0], ir::IRWriter::BEFORE, _id,
                                       _i_bind->cast_<ir::tensor::TensorValue>()->tensor_.uuid(), ir::program::BindOp::kInput);
  }
  // IMPORTANT: Insert Exit
  r.createAtPos<ir::program::ExitOp>(call_graph_ops[0], ir::IRWriter::AFTER);
  // IMPORTANT: Insert output bind
  for (auto [_id, _i_bind] : enumerate(bind_output)) {
    r.createAtPos<ir::program::BindOp>(call_graph_ops[0], ir::IRWriter::AFTER, _id,
                                       _i_bind->cast_<ir::tensor::TensorValue>()->tensor_.uuid(), ir::program::BindOp::kOutput);
  }

  // Rewrite the outmost GraphIROp
  r.walk<ir::graph::GraphIROp>([&](ir::IRWriter& reader, const ir::graph::GraphIROp::ptr_t& op) -> ir::IRWriter::WalkResult {
    for (auto& p : patterns_) {
      if (p->isMatch(op)) { p->rewrite(r, op); }
    }
    return ir::IRWriter::WalkResult::WALK_CONTINUE;
  });

  // Rewrite the callGraphOp in the Module Region and Program region.
  r.walk<ir::graph::CallGraphOp>([&](ir::IRWriter& reader, const ir::graph::GraphIROp::ptr_t& op) -> ir::IRWriter::WalkResult {
    for (auto& p : patterns_) {
      if (p->isMatch(op)) { p->rewrite(r, op); }
    }
    return ir::IRWriter::WalkResult::WALK_CONTINUE;
  });
  r.walk<ir::program::FragmentOp>(
      [&](ir::IRWriter& reader, const ir::program::FragmentOp::ptr_t& op) -> ir::IRWriter::WalkResult {
        auto rr = ir::IRWriter(getCtx(), op->getTopRegion());
        rr.walk<ir::graph::CallGraphOp>(
            [&](ir::IRWriter& inner_reader, const ir::graph::CallGraphOp::ptr_t& callop) -> ir::IRWriter::WalkResult {
              for (auto& p : patterns_) {
                if (p->isMatch(callop)) { p->rewrite(inner_reader, callop); }
              }
              return ir::IRWriter::WalkResult::WALK_CONTINUE;
            });
        return ir::IRWriter::WalkResult::WALK_CONTINUE;
      });

  // Flatten all program.fragment into a single program.fragment <code>
  auto flatten_code_segment = r.create<ir::program::FragmentOp>(
      program::FragmentType::kCode, getCtx()->create<ir::SymbolAttr>("__MLLM_JIT_PACKAGE_CODE_SEGMENT"));
  auto flatten_code_segment_rw = ir::IRWriter(getCtx(), flatten_code_segment->getTopRegion());

  // Remember we are now in the top module
  r.walk<ir::program::ProgramIROp>(
      [&](ir::IRWriter& reader, const ir::program::ProgramIROp::ptr_t& op) -> ir::IRWriter::WalkResult {
        if (!op->isa_<ir::program::FragmentOp>()) {
          flatten_code_segment_rw.insertOpAtLast(op);
          r.removeOpWithoutEdgeCut(op);
        } else {
          auto fragment_op_name = op->cast_<ir::program::FragmentOp>()->getSymbolAttr()->str();
          if (fragment_op_name == "__MLLM_JIT_PACKAGE_CODE_SEGMENT" || fragment_op_name == "init"
              || fragment_op_name == "deinit" || fragment_op_name == "op_symbol_table") {
            return ir::IRWriter::WalkResult::WALK_CONTINUE;
          }
          auto f_op = op->cast_<ir::program::FragmentOp>();
          auto f_op_rw = ir::IRWriter(getCtx(), f_op->getTopRegion());
          f_op_rw.walk<ir::Op>([&](ir::IRWriter& f_op_reader, const ir::Op::ptr_t& f_op_sub_op) -> ir::IRWriter::WalkResult {
            flatten_code_segment_rw.insertOpAtLast(f_op_sub_op);
            f_op_reader.removeOpWithoutEdgeCut(f_op_sub_op);
            return ir::IRWriter::WalkResult::WALK_CONTINUE;
          });
          r.removeOp(op);
          r.getContext()->removeFromSymbolTable(op->cast_<ir::program::FragmentOp>()->getSymbolAttr()->str());
        }
        return ir::IRWriter::WalkResult::WALK_CONTINUE;
      });

  return ir::PASS_RET_SUCCESS;
}

Graph2ProgramPass::Graph2ProgramPass() { regPattern<GraphSubGraph2ProgramPattern, GraphCallGraph2ProgramPattern>(); }

Pass::ptr_t createGraph2ProgramPass() { return std::make_shared<Graph2ProgramPass>(); }

bool CFRet2ProgramPattern::isMatch(const op_ptr_t& node) { return node->isa_<ir::cf::ReturnOp>(); }

bool CFRet2ProgramPattern::rewrite(IRWriter& writer, const op_ptr_t& node) {
  writer.createAndReplaceOp<ir::program::RetOp>(node);
  return true;
}

CFRet2ProgramPattern::ptr_t CFRet2ProgramPattern::create() { return std::make_shared<CFRet2ProgramPattern>(); }

CF2ProgramPass::CF2ProgramPass() { regPattern<CFRet2ProgramPattern>(); }

uint8_t CF2ProgramPass::run(const node_ptr_t& this_top_op) {  // The top op should be ModuleOp
  MLLM_RT_ASSERT(this_top_op->isa_<ir::ModuleOp>());

  // Find the subgraph we need to process
  auto r = ir::IRWriter(getCtx(), this_top_op->cast_<ir::ModuleOp>()->getTopRegion());

  // Graphs need to be write.
  std::vector<ir::graph::SubGraphOp::ptr_t> graphs_need_to_transform;
  r.walk<ir::graph::SubGraphOp>([&](ir::IRWriter& reader, const ir::graph::SubGraphOp::ptr_t& op) -> ir::IRWriter::WalkResult {
    graphs_need_to_transform.emplace_back(op);
    return ir::IRWriter::WalkResult::WALK_CONTINUE;
  });

  // Rewrite
  for (auto& g : graphs_need_to_transform) {
    auto g_r = ir::IRWriter(getCtx(), g->getTopRegion());
    g_r.walk<ir::cf::ControlFlowIROp>([&](ir::IRWriter& rw, const ir::Op::ptr_t& op) -> ir::IRWriter::WalkResult {
      for (auto& p : patterns_) {
        if (p->isMatch(op)) { p->rewrite(g_r, op); }
      }
      return ir::IRWriter::WalkResult::WALK_CONTINUE;
    });
  }

  return ir::PASS_RET_SUCCESS;
}

Pass::ptr_t createCF2ProgramPass() { return std::make_shared<CF2ProgramPass>(); }

bool TensorFreeOp2ProgramPattern::isMatch(const op_ptr_t& node) { return node->isa_<ir::tensor::FreeOp>(); }

bool TensorFreeOp2ProgramPattern::rewrite(IRWriter& writer, const op_ptr_t& node) {
  writer.createAndReplaceOp<ir::program::FreeOp>(node, node->cast_<ir::tensor::FreeOp>()->getFreedTensor());
  return true;
}

TensorFreeOp2ProgramPattern::ptr_t TensorFreeOp2ProgramPattern::create() {
  return std::make_shared<TensorFreeOp2ProgramPattern>();
}

Tensor2ProgramPass::Tensor2ProgramPass() { regPattern<TensorFreeOp2ProgramPattern>(); }

uint8_t Tensor2ProgramPass::run(const node_ptr_t& this_top_op) {
  // The top op should be ModuleOp
  MLLM_RT_ASSERT(this_top_op->isa_<ir::ModuleOp>());

  // Find the subgraph we need to process
  auto r = ir::IRWriter(getCtx(), this_top_op->cast_<ir::ModuleOp>()->getTopRegion());

  // Rewrite linalg OP in the top module
  r.walk<ir::tensor::TensorIROp>([&](ir::IRWriter& rw, const ir::Op::ptr_t& op) -> ir::IRWriter::WalkResult {
    for (auto& p : patterns_) {
      if (p->isMatch(op)) { p->rewrite(r, op); }
    }
    return ir::IRWriter::WalkResult::WALK_CONTINUE;
  });

  // Graphs need to be write.
  std::vector<ir::graph::SubGraphOp::ptr_t> graphs_need_to_transform;
  r.walk<ir::graph::SubGraphOp>([&](ir::IRWriter& reader, const ir::graph::SubGraphOp::ptr_t& op) -> ir::IRWriter::WalkResult {
    graphs_need_to_transform.emplace_back(op);
    return ir::IRWriter::WalkResult::WALK_CONTINUE;
  });

  // Rewrite
  for (auto& g : graphs_need_to_transform) {
    auto g_r = ir::IRWriter(getCtx(), g->getTopRegion());
    g_r.walk<ir::tensor::TensorIROp>([&](ir::IRWriter& rw, const ir::Op::ptr_t& op) -> ir::IRWriter::WalkResult {
      for (auto& p : patterns_) {
        if (p->isMatch(op)) { p->rewrite(g_r, op); }
      }
      return ir::IRWriter::WalkResult::WALK_CONTINUE;
    });
  }

  return ir::PASS_RET_SUCCESS;
}

Pass::ptr_t createTensor2ProgramPass() { return std::make_shared<Tensor2ProgramPass>(); }

std::vector<Pass::ptr_t> createProgramLoweringPipeline(const ProgramLoweringPipelineOptions& options) {
  std::vector<Pass::ptr_t> ret;

  // Transform Linalg op first
  ret.push_back(createExtractConstantTensorPass());
  ret.push_back(createLinalg2ProgramPass());
  ret.push_back(createTensor2ProgramPass());

  // Other transformation passes

  ret.push_back(createCF2ProgramPass());
  ret.push_back(createGraph2ProgramPass());

  // Program intrinsic Id Indexing.
  ret.push_back(createFlattenTensorAndLinalgSymbol2ProgramSymbolPass());
  if (options.enable_eager_flag) {
    ret.push_back(createProgramModeConfigPass({.mode = (int32_t)ir::program::ModeConfigFlag::kEager}));
  }
  ret.push_back(createProgramIntrinsicIdIndexPass());

  return ret;
}

}  // namespace mllm::ir
