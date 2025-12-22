// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <memory>
#include <unordered_map>

#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/graph/Op.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/builtin/Op.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/backends/qnn/aot/passes/AOTCompileContext.hpp"
#include "mllm/backends/qnn/aot/passes/MarkQnnGraphPass.hpp"

namespace mllm::qnn::aot {

namespace details {

ir::BoolAttr::ptr_t createTrueBoolAttr(const ir::IRContext::ptr_t& ctx) {
  auto ret = ctx->create<ir::BoolAttr>(true);
  return ret;
}

void recursiveVisitGraph(const ir::IRContext::ptr_t& ctx, const ir::graph::GraphIROp::ptr_t& graph) {
  if (!graph->getAttr("using_qnn")) { graph->setAttr("using_qnn", details::createTrueBoolAttr(ctx)); }
  auto writer = ir::IRWriter(ctx, graph->getTopRegion());
  writer.walk<ir::graph::CallGraphOp>(
      [&](ir::IRWriter& /*writer*/, const ir::graph::CallGraphOp::ptr_t& call_g) -> ir::IRWriter::WalkResult {
        auto g_name = call_g->getSymbolAttr()->str();
        auto g = ctx->lookupSymbolTable(g_name);
        MLLM_RT_ASSERT(g != nullptr);
        recursiveVisitGraph(ctx, g->cast_<ir::graph::GraphIROp>());
        return ir::IRWriter::WalkResult::WALK_CONTINUE;
      });
};

}  // namespace details

uint8_t MarkQnnGraphPass::run(const ir::node_ptr_t& op) {
  auto& aot_compile_ctx = AOTCompileContext::getInstance();
  auto config = aot_compile_ctx.getConfig();

  for (std::string item : config["graph_on_qnn"]) {
    auto g = getCtx()->lookupSymbolTable(item);
    MLLM_RT_ASSERT(g != nullptr);
    details::recursiveVisitGraph(getCtx(), g->cast_<ir::graph::GraphIROp>());
  }

  std::unordered_map<std::string, bool> op_visited;
  for (std::string item : config["op_on_qnn"]) { op_visited.insert({item, false}); }

  // Loop on the top module and check if this op need to be marked as qnn op.
  auto module_op = op->cast_<ir::ModuleOp>();
  auto writer = ir::IRWriter(getCtx(), module_op->getTopRegion());
  writer.walk<ir::linalg::LinalgIROp>(
      [&](ir::IRWriter& /*writer*/, const ir::linalg::LinalgIROp::ptr_t& linalg_op) -> ir::IRWriter::WalkResult {
        auto name = linalg_op->getAOp()->getName();
        if (op_visited.count(name) && !op_visited[name]) {
          linalg_op->setAttr("using_qnn", details::createTrueBoolAttr(getCtx()));
        }
        return ir::IRWriter::WalkResult::WALK_CONTINUE;
      });

  return ir::PASS_RET_SUCCESS;
}

ir::Pass::ptr_t createMarkQnnGraphPass() { return std::make_shared<MarkQnnGraphPass>(); }

}  // namespace mllm::qnn::aot
