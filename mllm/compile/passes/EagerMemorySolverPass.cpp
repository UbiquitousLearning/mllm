// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <unordered_map>

#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/graph/Op.hpp"
#include "mllm/compile/ir/builtin/Op.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/tensor/Op.hpp"
#include "mllm/compile/ir/tensor/Value.hpp"
#include "mllm/compile/passes/EagerMemorySolverPass.hpp"

namespace mllm::ir {

namespace MLLM_ANONYMOUS_NAMESPACE {

void recursiveIndexTensorOp(std::unordered_map<uint32_t, int32_t>& count, const IRContext::ptr_t& ctx,
                            const ir::graph::SubGraphOp::ptr_t& subgraph) {
  auto g = ir::IRWriter(ctx, subgraph->getTopRegion());
  g.walk<ir::Op>([&](ir::IRWriter& reader, const ir::Op::ptr_t& sub_op) -> ir::IRWriter::WalkResult {
    if (sub_op->isa_<ir::graph::CallGraphOp>()) {
      auto sub_subgraph = ctx->lookupSymbolTable(sub_op->cast_<ir::graph::CallGraphOp>()->getSymbolAttr()->str())
                              ->cast_<ir::graph::SubGraphOp>();
      recursiveIndexTensorOp(count, ctx, sub_subgraph);
    } else if (sub_op->isa_<ir::linalg::LinalgIROp>()) {
      // We only take consider the inputs, outputs should be used at least once.
      auto inputs = sub_op->inputs();
      for (auto input : inputs) {
        if (count.count(input->cast_<ir::tensor::TensorValue>()->tensor_.uuid())) {
          count[input->cast_<ir::tensor::TensorValue>()->tensor_.uuid()]++;
        } else {
          count[input->cast_<ir::tensor::TensorValue>()->tensor_.uuid()] = 1;
        }
      }
    }
    return ir::IRWriter::WalkResult::WALK_CONTINUE;
  });
}

void recursiveInsertFreeTensorOp(std::unordered_map<uint32_t, int32_t>& count, const IRContext::ptr_t& ctx,
                                 const ir::graph::SubGraphOp::ptr_t& subgraph) {
  auto g = ir::IRWriter(ctx, subgraph->getTopRegion());
  g.walk<ir::Op>([&](ir::IRWriter& reader, const ir::Op::ptr_t& sub_op) -> ir::IRWriter::WalkResult {
    if (sub_op->isa_<ir::graph::CallGraphOp>()) {
      auto sub_subgraph = ctx->lookupSymbolTable(sub_op->cast_<ir::graph::CallGraphOp>()->getSymbolAttr()->str())
                              ->cast_<ir::graph::SubGraphOp>();
      recursiveInsertFreeTensorOp(count, ctx, sub_subgraph);
    } else if (sub_op->isa_<ir::linalg::LinalgIROp>()) {
      // We only take consider the inputs, outputs should be used at least once.
      auto inputs = sub_op->inputs();
      for (auto input : inputs) {
        if (count.count(input->cast_<ir::tensor::TensorValue>()->tensor_.uuid())) {
          count[input->cast_<ir::tensor::TensorValue>()->tensor_.uuid()]--;
          if (count[input->cast_<ir::tensor::TensorValue>()->tensor_.uuid()] == 0) {
            reader.createAtPos<ir::tensor::FreeOp>(sub_op, ir::IRWriter::Position::AFTER,
                                                   input->cast_<ir::tensor::TensorValue>());
          }
        } else {
          MLLM_WARN("[EagerMemorySolverPass] Solver Algorithm Error.");
        }
      }
    }
    return ir::IRWriter::WalkResult::WALK_CONTINUE;
  });
}

}  // namespace MLLM_ANONYMOUS_NAMESPACE

uint8_t EagerMemorySolverPass::run(const node_ptr_t& op) {
  // The top op should be ModuleOp
  MLLM_RT_ASSERT(op->isa_<ir::ModuleOp>());

  // Find the subgraph we need to process
  auto r = ir::IRWriter(getCtx(), op->cast_<ir::ModuleOp>()->getTopRegion());

  // Find the first call op
  ir::graph::CallGraphOp::ptr_t call_op = nullptr;
  r.walk<ir::graph::CallGraphOp>(
      [&](ir::IRWriter& reader, const ir::graph::CallGraphOp::ptr_t& sub_op) -> ir::IRWriter::WalkResult {
        call_op = sub_op;
        return ir::IRWriter::WalkResult::WALK_CONTINUE;
      });

  std::unordered_map<uint32_t, int32_t> tensor_usage_count;
  recursiveIndexTensorOp(tensor_usage_count, getCtx(),
                         getCtx()->lookupSymbolTable(call_op->getSymbolAttr()->str())->cast_<ir::graph::SubGraphOp>());

  recursiveInsertFreeTensorOp(tensor_usage_count, getCtx(),
                              getCtx()->lookupSymbolTable(call_op->getSymbolAttr()->str())->cast_<ir::graph::SubGraphOp>());

  return ir::PASS_RET_SUCCESS;
}

Pass::ptr_t createEagerMemorySolverPass() { return std::make_shared<EagerMemorySolverPass>(); }

}  // namespace mllm::ir
