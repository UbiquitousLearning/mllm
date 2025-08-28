// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/dbg/Op.hpp"
#include "mllm/compile/ir/graph/Op.hpp"
#include "mllm/compile/ir/builtin/Op.hpp"
#include "mllm/compile/passes/EliminateDbgInfoPass.hpp"

namespace mllm::ir {

namespace MLLM_ANONYMOUS_NAMESPACE {
void recursiveRemoveDbgInfo(const IRContext::ptr_t& ctx, const ir::graph::SubGraphOp::ptr_t& this_graph_op) {
  auto r = ir::IRWriter(ctx, this_graph_op->getTopRegion());
  r.walk<ir::graph::SubGraphOp>([&](ir::IRWriter& reader, const ir::graph::SubGraphOp::ptr_t& op) -> ir::IRWriter::WalkResult {
    recursiveRemoveDbgInfo(ctx, op);
    return ir::IRWriter::WalkResult::WALK_CONTINUE;
  });

  r.walk<ir::dbg::DbgIROp>([&](ir::IRWriter& reader, const ir::dbg::DbgIROp::ptr_t& op) -> ir::IRWriter::WalkResult {
    r.removeOp(op);
    return ir::IRWriter::WalkResult::WALK_CONTINUE;
  });
}
}  // namespace MLLM_ANONYMOUS_NAMESPACE

uint8_t EliminateDbgInfoPass::run(const node_ptr_t& op) {
  // The top op should be ModuleOp
  MLLM_RT_ASSERT(op->isa_<ir::ModuleOp>());

  // Find the subgraph we need to process
  auto r = ir::IRWriter(getCtx(), op->cast_<ir::ModuleOp>()->getTopRegion());

  r.walk<ir::graph::SubGraphOp>([&](ir::IRWriter& reader, const ir::graph::SubGraphOp::ptr_t& op) -> ir::IRWriter::WalkResult {
    recursiveRemoveDbgInfo(getCtx(), op);
    return ir::IRWriter::WalkResult::WALK_CONTINUE;
  });

  r.walk<ir::dbg::DbgIROp>([&](ir::IRWriter& reader, const ir::dbg::DbgIROp::ptr_t& op) -> ir::IRWriter::WalkResult {
    r.removeOp(op);
    return ir::IRWriter::WalkResult::WALK_CONTINUE;
  });

  return ir::PASS_RET_SUCCESS;
}

Pass::ptr_t createEliminateDbgInfoPass() { return std::make_shared<EliminateDbgInfoPass>(); }

}  // namespace mllm::ir
