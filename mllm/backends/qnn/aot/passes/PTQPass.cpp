// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <memory>

#include "mllm/backends/qnn/aot/passes/PTQPass.hpp"
#include "mllm/backends/qnn/aot/passes/AOTCompileContext.hpp"
#include "mllm/compile/ir/builtin/Op.hpp"
#include "mllm/compile/ir/graph/Op.hpp"
#include "mllm/compile/ir/linalg/Attribute.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/tensor/Value.hpp"
#include "mllm/compile/ir/cf/Op.hpp"
#include "mllm/compile/ir/Node.hpp"
#include "mllm/core/OpTypes.hpp"
#include "mllm/core/ParameterFile.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::qnn::aot {

namespace {

template<typename T>
void checkTypeLimits(Tensor in, int quant_min, int quant_max) {  // NOLINT
  auto numel = in.numel();
  for (int i = 0; i < numel; ++i) {
    MLLM_RT_ASSERT(*(in.ptr<T>() + i) >= quant_min);
    MLLM_RT_ASSERT(*(in.ptr<T>() + i) <= quant_max);
  }
}

void solveLinearWeight(const ir::IRContext::ptr_t& ctx, const ParameterFile::ptr_t& pf,
                       const ir::linalg::LinalgIROp::ptr_t& op) {
  auto mllm_op = op->getAOp();
  MLLM_INFO("PTQPass working on Op: {}'s weight", mllm_op->getName());
  auto weight_spec =
      op->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonAnnotationAttr>()->annotation_.weights.at("weight");

  if (weight_spec->solved) return;

  switch (weight_spec->type) {
    case ir::linalg::QuantizationSpecType::kLPBQ: {
      auto this_spec = std::static_pointer_cast<ir::linalg::QuantizationSpecLPBQ>(weight_spec);
      auto scale1 = pf->pull(mllm_op->getName() + ".scale1");  // using uint8 to store uint4
      auto scale2 = pf->pull(mllm_op->getName() + ".scale2");
      auto weight = pf->pull(mllm_op->getName() + ".weight");

      // FIXME weight maybe error, Check qnn eats int8 or uint8. Here weight using int8 to store int4.
      checkTypeLimits<int8_t>(weight, -8, 7);   // Int4
      checkTypeLimits<uint8_t>(scale1, 0, 16);  // UInt4

      this_spec->scale_level_0_int = scale1;
      this_spec->scale_level_1_fp = scale2;

      weight_spec->solved = true;
      break;
    }
    default: {
      NYI("quant recipe type not support");
    }
  }
}

void solveRMSNormWeight(const ir::IRContext::ptr_t& ctx, const ParameterFile::ptr_t& pf,
                        const ir::linalg::LinalgIROp::ptr_t& op) {
  auto mllm_op = op->getAOp();
  MLLM_INFO("PTQPass working on Op: {}'s weight", mllm_op->getName());
  auto weight_spec =
      op->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonAnnotationAttr>()->annotation_.weights.at("weight");

  if (weight_spec->solved) return;

  switch (weight_spec->type) {
    case ir::linalg::QuantizationSpecType::kRaw: {
      weight_spec->solved = true;
      break;
    }
    case ir::linalg::QuantizationSpecType::kAsymPerTensor: {
      auto this_spec = std::static_pointer_cast<ir::linalg::QuantizationSpecAsymPerTensor>(weight_spec);
      auto scale = pf->pull(mllm_op->getName() + ".scale");
      auto zero_point = pf->pull(mllm_op->getName() + ".zero_point");
      this_spec->scale = scale;
      this_spec->zero_point = zero_point;
      checkTypeLimits<uint16_t>(pf->pull(mllm_op->getName() + ".weight"), this_spec->quant_min, this_spec->quant_max);
      MLLM_RT_ASSERT(scale.dtype() == kFloat32);
      MLLM_RT_ASSERT(scale.rank() == 1);
      MLLM_RT_ASSERT(scale.item<float>() > 0);
      MLLM_RT_ASSERT(zero_point.dtype() == kInt32);
      MLLM_RT_ASSERT(zero_point.rank() == 1);
      MLLM_RT_ASSERT(zero_point.item<int32_t>() >= 0);
      weight_spec->solved = true;
      break;
    }
    default: {
      NYI("quant recipe type not support");
    }
  }
}

void solveEmbeddingWeight(const ir::IRContext::ptr_t& ctx, const ParameterFile::ptr_t& pf,
                          const ir::linalg::LinalgIROp::ptr_t& op) {
  auto mllm_op = op->getAOp();
  MLLM_INFO("PTQPass working on Op: {}'s weight", mllm_op->getName());
  auto weight_spec =
      op->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonAnnotationAttr>()->annotation_.weights.at("weight");

  if (weight_spec->solved) return;

  switch (weight_spec->type) {
    case ir::linalg::QuantizationSpecType::kRaw: {
      weight_spec->solved = true;
      break;
    }
    default: {
      NYI("quant recipe type not support");
    }
  }
}

void recursiveSolveWeights(const std::shared_ptr<ir::IRContext>& ir_ctx, const ir::graph::SubGraphOp::ptr_t& call_op,
                           const ParameterFile::ptr_t& pf) {
  auto wow = ir::IRWriter(ir_ctx, call_op->getTopRegion());
  wow.walk<ir::Op>([&](ir::IRWriter& w, const ir::Op::ptr_t& op) -> ir::IRWriter::WalkResult {
    if (op->isa_<ir::linalg::LinearOp>()) { solveLinearWeight(w.getContext(), pf, op->cast_<ir::linalg::LinalgIROp>()); }
    if (op->isa_<ir::linalg::RMSNormOp>()) { solveRMSNormWeight(w.getContext(), pf, op->cast_<ir::linalg::LinalgIROp>()); }
    if (op->isa_<ir::linalg::EmbeddingOp>()) { solveEmbeddingWeight(w.getContext(), pf, op->cast_<ir::linalg::LinalgIROp>()); }
    if (op->isa_<ir::graph::CallGraphOp>()) {
      auto ns = op->cast_<ir::graph::CallGraphOp>()->getSymbolAttr()->str();
      recursiveSolveWeights(w.getContext(), w.getContext()->lookupSymbolTable(ns)->cast_<ir::graph::SubGraphOp>(), pf);
    }
    return ir::IRWriter::WALK_CONTINUE;
  });
}

void __recursiveSolveNormalImpl(const ir::Val::ptr_t& v) {
  MLLM_RT_ASSERT(v->isa_<ir::tensor::TensorValue>());
  auto tv = v->cast_<ir::tensor::TensorValue>();
  MLLM_RT_ASSERT(tv->getAttr("quant_recipe"));
  auto f_spec = tv->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>();

  if (f_spec->spec_->solved) { return; }

  switch (f_spec->spec_->type) {
    case ir::linalg::QuantizationSpecType::kAsymPerTensor: {
      if (!tv->tensor_.hasAttachedView("scale") || !tv->tensor_.hasAttachedView("zero_point")) { return; }
      auto scale = tv->tensor_.getExtraTensorViewInTensor("scale");
      auto zero_point = tv->tensor_.getExtraTensorViewInTensor("zero_point");
      auto this_spec = std::static_pointer_cast<ir::linalg::QuantizationSpecAsymPerTensor>(f_spec->spec_);
      this_spec->scale = scale;
      this_spec->zero_point = zero_point;
      this_spec->solved = true;
      break;
    }
    case ir::linalg::QuantizationSpecType::kSymPerTensor: {
      if (!tv->tensor_.hasAttachedView("scale")) { return; }
      auto scale = tv->tensor_.getExtraTensorViewInTensor("scale");
      auto this_spec = std::static_pointer_cast<ir::linalg::QuantizationSpecSymPerTensor>(f_spec->spec_);
      this_spec->scale = scale;
      this_spec->solved = true;
      break;
    }
    case ir::linalg::QuantizationSpecType::kRaw: {
      auto this_spec = std::static_pointer_cast<ir::linalg::QuantizationSpecRaw>(f_spec->spec_);
      this_spec->solved = true;
      break;
    }
    default: {
      NYI("quant recipe type not support on tensor: {}", v->name());
    }
  }
}

void recursiveSolveNormal(const std::shared_ptr<ir::IRContext>& ir_ctx, const ir::graph::SubGraphOp::ptr_t& call_op,
                          const ParameterFile::ptr_t& pf) {
  auto wow = ir::IRWriter(ir_ctx, call_op->getTopRegion());
  wow.walk<ir::Op>([&](ir::IRWriter& w, const ir::Op::ptr_t& op) -> ir::IRWriter::WalkResult {
    if (op->isa_<ir::linalg::LinalgIROp>()) {
      MLLM_INFO("PTQPass relax working on Op: {}'s tensors", op->cast_<ir::linalg::LinalgIROp>()->getAOp()->getName());

      auto inputs = op->inputs();
      auto outputs = op->outputs();

      for (auto iii : inputs) { __recursiveSolveNormalImpl(iii->cast_<ir::Val>()); }
      for (auto ooo : outputs) { __recursiveSolveNormalImpl(ooo->cast_<ir::Val>()); }
    }

    if (op->isa_<ir::graph::CallGraphOp>()) {
      auto ns = op->cast_<ir::graph::CallGraphOp>()->getSymbolAttr()->str();
      recursiveSolveNormal(w.getContext(), w.getContext()->lookupSymbolTable(ns)->cast_<ir::graph::SubGraphOp>(), pf);
    }
    return ir::IRWriter::WALK_CONTINUE;
  });
}

}  // namespace

uint8_t PTQPass::run(const ir::node_ptr_t& op) {
  auto pf = AOTCompileContext::getInstance().getParamFile();

  // The top op should be ModuleOp
  MLLM_RT_ASSERT(op->isa_<ir::ModuleOp>());

  auto module_op = op->cast_<ir::ModuleOp>();
  auto writer = ir::IRWriter(getCtx(), module_op->getTopRegion());

  ir::graph::CallGraphOp::ptr_t call_main_graph_op = nullptr;
  writer.walk<ir::graph::CallGraphOp>(
      [&](ir::IRWriter& /*writer*/, const ir::graph::CallGraphOp::ptr_t& call_op) -> ir::IRWriter::WalkResult {
        MLLM_RT_ASSERT_EQ(call_main_graph_op, nullptr);

        call_main_graph_op = call_op;
        return ir::IRWriter::WalkResult::WALK_CONTINUE;
      });

  // Solve all registered weight
  recursiveSolveWeights(writer.getContext(),
                        getCtx()->lookupSymbolTable(call_main_graph_op->getSymbolAttr()->str())->cast_<ir::graph::SubGraphOp>(),
                        pf);

  // Solve other normal tensors
  recursiveSolveNormal(writer.getContext(),
                       getCtx()->lookupSymbolTable(call_main_graph_op->getSymbolAttr()->str())->cast_<ir::graph::SubGraphOp>(),
                       pf);

  return ir::PASS_RET_SUCCESS;
}

ir::Pass::ptr_t createPTQPass() { return std::make_shared<PTQPass>(); }

}  // namespace mllm::qnn::aot
