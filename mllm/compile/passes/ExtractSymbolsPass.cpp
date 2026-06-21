// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/compile/ir/Node.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/graph/Op.hpp"
#include "mllm/compile/ir/builtin/Op.hpp"
#include "mllm/compile/passes/ExtractSymbolsPass.hpp"

namespace mllm::ir {

namespace MLLM_ANONYMOUS_NAMESPACE {
void recursiveRegisterLinalgSymbolOps(const IRContext::ptr_t& ctx, const ir::graph::SubGraphOp::ptr_t& this_graph_op,
                                      const ir::graph::SubGraphOp::ptr_t& symbol_table,
                                      std::unordered_map<std::string, ir::node_weak_ptr_t>& symbol_registered) {
  auto r = ir::IRWriter(ctx, this_graph_op->getTopRegion());
  r.walk<ir::graph::SubGraphOp>([&](ir::IRWriter& reader, const ir::graph::SubGraphOp::ptr_t& op) -> ir::IRWriter::WalkResult {
    recursiveRegisterLinalgSymbolOps(ctx, op, symbol_table, symbol_registered);
    return ir::IRWriter::WalkResult::WALK_CONTINUE;
  });

  r.walk<ir::linalg::LinalgIROp>(
      [&](ir::IRWriter& reader, const ir::linalg::LinalgIROp::ptr_t& op) -> ir::IRWriter::WalkResult {
        if (!op->getAOp()->getName().empty()) {
          if (!symbol_registered.count(op->getAOp()->getName())) {
            symbol_registered[op->getAOp()->getName()] = op;
            auto rrr = ir::IRWriter(ctx, symbol_table->getTopRegion());
            rrr.create<ir::linalg::RegisterOp>(op->getAOp(), op->getAOp()->getName());
          }
        }
        return ir::IRWriter::WalkResult::WALK_CONTINUE;
      });
}
}  // namespace MLLM_ANONYMOUS_NAMESPACE

uint8_t ExtractSymbolsPass::run(const node_ptr_t& op) {
  // The top op should be ModuleOp
  MLLM_RT_ASSERT(op->isa_<ir::ModuleOp>());

  // Find the subgraph we need to process
  auto r = ir::IRWriter(getCtx(), op->cast_<ir::ModuleOp>()->getTopRegion());

  // Insert a Symbol Op Table if not exists.
  ir::graph::SubGraphOp::ptr_t op_symbol_table = nullptr;
  if (!getCtx()->lookupSymbolTable("op_symbol_table")) {
    op_symbol_table = getCtx()->create<ir::graph::SubGraphOp>(getCtx()->create<SymbolAttr>("op_symbol_table"));
  } else {
    op_symbol_table = getCtx()->lookupSymbolTable("op_symbol_table")->cast_<ir::graph::SubGraphOp>();
  }

  std::unordered_map<std::string, ir::node_weak_ptr_t> linalg_symbols;

  r.walk<ir::graph::SubGraphOp>([&](ir::IRWriter& reader, const ir::graph::SubGraphOp::ptr_t& op) -> ir::IRWriter::WalkResult {
    if (op->getSymbolAttr()->str() == "op_symbol_table" || op->getSymbolAttr()->str() == "init"
        || op->getSymbolAttr()->str() == "deinit") {
      return ir::IRWriter::WalkResult::WALK_CONTINUE;
    }
    recursiveRegisterLinalgSymbolOps(getCtx(), op, op_symbol_table, linalg_symbols);
    return ir::IRWriter::WalkResult::WALK_CONTINUE;
  });

  r.walk<ir::linalg::LinalgIROp>(
      [&](ir::IRWriter& reader, const ir::linalg::LinalgIROp::ptr_t& op) -> ir::IRWriter::WalkResult {
        if (!op->getAOp()->getName().empty()) {
          if (!linalg_symbols.count(op->getAOp()->getName())) {
            linalg_symbols[op->getAOp()->getName()] = op;
            auto rrr = ir::IRWriter(getCtx(), op_symbol_table->getTopRegion());
            rrr.create<ir::linalg::RegisterOp>(op->getAOp(), op->getAOp()->getName());
          }
        }
        return ir::IRWriter::WalkResult::WALK_CONTINUE;
      });

  return ir::PASS_RET_SUCCESS;
}

Pass::ptr_t createExtractSymbolsPass() { return std::make_shared<ExtractSymbolsPass>(); }

}  // namespace mllm::ir
