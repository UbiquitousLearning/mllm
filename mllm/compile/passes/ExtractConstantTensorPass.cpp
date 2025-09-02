// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include <set>

#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/graph/Op.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/tensor/Op.hpp"
#include "mllm/compile/ir/builtin/Op.hpp"
#include "mllm/compile/ir/tensor/Value.hpp"
#include "mllm/compile/passes/ExtractConstantTensorPass.hpp"

namespace mllm::ir {

namespace MLLM_ANONYMOUS_NAMESPACE {}  // namespace MLLM_ANONYMOUS_NAMESPACE

uint8_t ExtractConstantTensorPass::run(const node_ptr_t& op) {
  // The top op should be ModuleOp
  MLLM_RT_ASSERT(op->isa_<ir::ModuleOp>());

  // Find the subgraph we need to process
  auto r = ir::IRWriter(getCtx(), op->cast_<ir::ModuleOp>()->getTopRegion());

  auto constant_tensor_set = std::set<ir::tensor::TensorValue::ptr_t>{};
  auto value_symbol_region = getCtx()->lookupSymbolTable("init")->cast_<ir::graph::SubGraphOp>();
  auto value_symbol_rw = ir::IRWriter(getCtx(), value_symbol_region->getTopRegion());

  r.walk<ir::linalg::LinalgIROp>([&constant_tensor_set](ir::IRWriter& reader, const ir::linalg::LinalgIROp::ptr_t& sub_op) {
    auto inputs = sub_op->inputs();
    auto outputs = sub_op->outputs();

    for (auto input : inputs) {
      if (input->isa_<ir::tensor::TensorValue>() && input->getAttr("constant")) {
        constant_tensor_set.insert(input->cast_<ir::tensor::TensorValue>());
      }
    }

    for (auto output : outputs) {
      if (output->isa_<ir::tensor::TensorValue>() && output->getAttr("constant")) {
        constant_tensor_set.insert(output->cast_<ir::tensor::TensorValue>());
      }
    }

    return ir::IRWriter::WalkResult::WALK_CONTINUE;
  });

  std::vector<ir::graph::SubGraphOp::ptr_t> graphs;
  r.walk<ir::graph::SubGraphOp>([&graphs](ir::IRWriter& reader, const ir::graph::SubGraphOp::ptr_t& sub_op) {
    if (sub_op->getSymbolAttr()->str() != "init" && sub_op->getSymbolAttr()->str() != "deinit") { graphs.push_back(sub_op); }
    return ir::IRWriter::WalkResult::WALK_CONTINUE;
  });

  for (auto& g : graphs) {
    auto g_rw = ir::IRWriter(getCtx(), g->getTopRegion());
    g_rw.walk<ir::linalg::LinalgIROp>(
        [&constant_tensor_set](ir::IRWriter& reader, const ir::linalg::LinalgIROp::ptr_t& sub_op) {
          auto inputs = sub_op->inputs();
          auto outputs = sub_op->outputs();

          for (auto input : inputs) {
            if (input->isa_<ir::tensor::TensorValue>() && input->getAttr("constant")) {
              constant_tensor_set.insert(input->cast_<ir::tensor::TensorValue>());
            }
          }

          for (auto output : outputs) {
            if (output->isa_<ir::tensor::TensorValue>() && output->getAttr("constant")) {
              constant_tensor_set.insert(output->cast_<ir::tensor::TensorValue>());
            }
          }

          return ir::IRWriter::WalkResult::WALK_CONTINUE;
        });
  }

  for (auto& sub_tensor : constant_tensor_set) { value_symbol_rw.create<ir::tensor::RegisterOp>(sub_tensor); }

  return ir::PASS_RET_SUCCESS;
}

Pass::ptr_t createExtractConstantTensorPass() { return std::make_shared<ExtractConstantTensorPass>(); }

}  // namespace mllm::ir
