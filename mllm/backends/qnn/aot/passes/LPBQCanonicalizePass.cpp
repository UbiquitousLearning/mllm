// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/qnn/aot/passes/LPBQCanonicalizePass.hpp"
#include "mllm/compile/ir/builtin/Op.hpp"
#include "mllm/compile/ir/graph/Op.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/linalg/Attribute.hpp"
#include "mllm/compile/ir/tensor/Value.hpp"
#include "mllm/compile/ir/Node.hpp"
#include "mllm/core/OpTypes.hpp"
#include "mllm/core/aops/ViewOp.hpp"
#include "mllm/engine/Context.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/engine/Context.hpp"

namespace mllm::qnn::aot {

namespace {

void visitSubGraph(const std::shared_ptr<ir::IRContext>& ir_ctx, const ir::graph::SubGraphOp::ptr_t& subgraph_op);

void visitCallGraph(const std::shared_ptr<ir::IRContext>& ir_ctx, const ir::graph::CallGraphOp::ptr_t& call_op) {
  // Get the subgraph referenced by the call graph op
  auto subgraph_op = ir_ctx->lookupSymbolTable(call_op->getSymbolAttr()->str())->cast_<ir::graph::SubGraphOp>();
  MLLM_RT_ASSERT(subgraph_op != nullptr);

  visitSubGraph(ir_ctx, subgraph_op);
}

void visitSubGraph(const std::shared_ptr<ir::IRContext>& ir_ctx, const ir::graph::SubGraphOp::ptr_t& subgraph_op) {
  auto region = subgraph_op->getTopRegion();
  auto writer = ir::IRWriter(ir_ctx, region);

  // Walk through all operations in the subgraph
  writer.walk<ir::linalg::LinalgIROp>(
      [&](ir::IRWriter& w, const ir::linalg::LinalgIROp::ptr_t& linalg_op) -> ir::IRWriter::WalkResult {
        auto mllm_op = linalg_op->getAOp();
        auto mllm_op_type = linalg_op->getAOpTypes();

        // Check if this operation has quantization annotation
        if (!linalg_op->getAttr("quant_recipe")) { return ir::IRWriter::WalkResult::WALK_CONTINUE; }

        auto quant_attr = linalg_op->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonAnnotationAttr>();

        // TODO do rewrite when meets LPBQ weight
        if (quant_attr->annotation_.weights["weight"]->type == ir::linalg::QuantizationSpecType::kLPBQ) {
          auto o = linalg_op->outputs().front()->cast_<ir::tensor::TensorValue>();
          // [B, H, S0, S1] -> [B * H, S0, S1] -> [B, H, S0, S1]
          if (o->tensor_.rank() == 4) {
            auto B = o->tensor_.size(0);
            auto H = o->tensor_.size(1);
            auto S0 = o->tensor_.size(2);
            auto S1 = o->tensor_.size(3);
            // This output should be [B * H, S0, S1]
            o->tensor_ = o->tensor_.view({B * H, S0, S1});

            // Create ViewOp
            auto new_o = w.getContext()->create<ir::tensor::TensorValue>(
                Tensor::empty({B, H, S0, S1}, o->tensor_.dtype(), o->tensor_.device()));
            auto view_op = w.createAtPos<ir::linalg::ViewOp>(linalg_op, ir::IRWriter::AFTER,
                                                             mllm::Context::instance()
                                                                 .getBackend(o->tensor_.device())
                                                                 ->createOp(mllm::OpTypes::kView, mllm::aops::ViewOpOptions{}),
                                                             std::vector<ir::tensor::TensorValue::ptr_t>{o},
                                                             std::vector<ir::tensor::TensorValue::ptr_t>{new_o});

            // Find all operators that eats original o, and set them with new_o
            auto consumer_ops = o->consumerOps();
            for (auto cc_help_me : consumer_ops) {
              MLLM_RT_ASSERT(cc_help_me->isa_<ir::Op>());
              auto& inputs = cc_help_me->inputs();
              auto& outputs = cc_help_me->outputs();
            }
          }
        }

        return ir::IRWriter::WalkResult::WALK_CONTINUE;
      });

  // Also recursively visit any nested call graphs
  writer.walk<ir::graph::CallGraphOp>(
      [&](ir::IRWriter& w, const ir::graph::CallGraphOp::ptr_t& call_op) -> ir::IRWriter::WalkResult {
        visitCallGraph(ir_ctx, call_op);
        return ir::IRWriter::WalkResult::WALK_CONTINUE;
      });
}

}  // anonymous namespace

uint8_t LPBQCanonicalizePass::run(const ir::node_ptr_t& op) {
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

  // Visit all graphs and canonicalize LPBQ quantization specs
  if (call_main_graph_op != nullptr) { visitCallGraph(getCtx(), call_main_graph_op); }

  return ir::PASS_RET_SUCCESS;
}

ir::Pass::ptr_t createLPBQCanonicalizePass() { return std::make_shared<LPBQCanonicalizePass>(); }

}  // namespace mllm::qnn::aot
