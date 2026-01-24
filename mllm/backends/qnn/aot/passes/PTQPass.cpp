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

void recursiveSolveWeights(const std::shared_ptr<ir::IRContext>& ir_ctx, const ir::graph::SubGraphOp::ptr_t& call_op,
                           const ParameterFile::ptr_t& pf) {
  auto wow = ir::IRWriter(ir_ctx, call_op->getTopRegion());
  wow.walk<ir::Op>([&](ir::IRWriter& w, const ir::Op::ptr_t& op) -> ir::IRWriter::WalkResult {
    if (op->isa_<ir::linalg::LinearOp>()) { solveLinearWeight(w.getContext(), pf, op->cast_<ir::linalg::LinalgIROp>()); }
    if (op->isa_<ir::linalg::Conv2DOp>()) {
      // Conv2D's Check same with Linear
      solveLinearWeight(w.getContext(), pf, op->cast_<ir::linalg::LinalgIROp>());
    }
    if (op->isa_<ir::linalg::RMSNormOp>()) { solveRMSNormWeight(w.getContext(), pf, op->cast_<ir::linalg::LinalgIROp>()); }
    if (op->isa_<ir::linalg::EmbeddingOp>()) { solveEmbeddingWeight(w.getContext(), pf, op->cast_<ir::linalg::LinalgIROp>()); }
    if (op->isa_<ir::graph::CallGraphOp>()) {
      auto ns = op->cast_<ir::graph::CallGraphOp>()->getSymbolAttr()->str();
      recursiveSolveWeights(w.getContext(), w.getContext()->lookupSymbolTable(ns)->cast_<ir::graph::SubGraphOp>(), pf);
    }
    return ir::IRWriter::WALK_CONTINUE;
  });
}

void _recursiveSolveNormalImpl(const ir::IRContext::ptr_t& ctx, const ir::Val::ptr_t& v) {
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
      auto min_v = this_spec->quant_min;
      auto max_v = this_spec->quant_max;

      // Check if this tensor is constant tensor. Then we need to quantize it.
      double constant_v = 0;
      DataTypes constant_dtype = kFloat32;
      if (tv->getAttr("constant")) {
        auto constant_ir = tv->getAttr("constant");
        if (constant_ir->isa_<ir::VectorFP32Attr>()) {
          auto ci = constant_ir->cast_<ir::VectorFP32Attr>();
          MLLM_RT_ASSERT_EQ(ci->data().size(), 1);
          constant_v = ci->data()[0];
          constant_dtype = kFloat32;
        } else if (constant_ir->isa_<ir::VectorInt16Attr>()) {
          auto ci = constant_ir->cast_<ir::VectorInt16Attr>();
          MLLM_RT_ASSERT_EQ(ci->data().size(), 1);
          constant_v = ci->data()[0];
          constant_dtype = kInt16;
        } else {
          NYI("Not implement constant attribute type");
        }

        // Calculate constant scale after PTQ
        MLLM_RT_ASSERT_EQ(scale.numel(), 1);
        MLLM_RT_ASSERT_EQ(zero_point.numel(), 1);
        auto scale_fp = scale.item<mllm_fp32_t>();
        auto zero_point_int32 = zero_point.item<mllm_int32_t>();
        auto quant_value = std::round(constant_v / scale_fp) + zero_point_int32;
        auto clamped_value =
            std::clamp(static_cast<int32_t>(quant_value), static_cast<int32_t>(min_v), static_cast<int32_t>(max_v));
        auto ptq_constant_v = static_cast<mllm_int32_t>(clamped_value);

        if (constant_ir->isa_<ir::VectorFP32Attr>()) {
          auto ci = constant_ir->cast_<ir::VectorFP32Attr>();
          ci->data()[0] = ptq_constant_v;

          // FIXME: We hard code uint16 here.
          tv->tensor_ = Tensor::ones({1}, kUInt16, kCPU);
          tv->tensor_.at<mllm_uint16_t>({0}) = ptq_constant_v;
        } else if (constant_ir->isa_<ir::VectorInt16Attr>()) {
          auto ci = constant_ir->cast_<ir::VectorInt16Attr>();
          ci->data()[0] = ptq_constant_v;

          // FIXME: We hard code uint16 here.
          tv->tensor_ = Tensor::ones({1}, kUInt16, kCPU);
          tv->tensor_.at<mllm_uint16_t>({0}) = ptq_constant_v;
        }

        auto _attr = ctx->create<ir::VectorUInt16Attr>(std::vector<uint16_t>{(uint16_t)ptq_constant_v});
        tv->removeAttr("constant");
        tv->setAttr("constant", _attr);

        MLLM_INFO("Constant tensor '{}' quantized (AsymPerTensor): before={}, after={}", tv->name(), constant_v,
                  ptq_constant_v);
      }

      this_spec->solved = true;
      break;
    }
    case ir::linalg::QuantizationSpecType::kSymPerTensor: {
      if (!tv->tensor_.hasAttachedView("scale")) { return; }
      auto scale = tv->tensor_.getExtraTensorViewInTensor("scale");
      auto this_spec = std::static_pointer_cast<ir::linalg::QuantizationSpecSymPerTensor>(f_spec->spec_);
      this_spec->scale = scale;
      auto min_v = this_spec->quant_min;
      auto max_v = this_spec->quant_max;

      // Check if this tensor is constant tensor. Then we need to quantize it.
      double constant_v = 0;
      DataTypes constant_dtype = kFloat32;
      if (tv->getAttr("constant")) {
        auto constant_ir = tv->getAttr("constant");
        if (constant_ir->isa_<ir::VectorFP32Attr>()) {
          auto ci = constant_ir->cast_<ir::VectorFP32Attr>();
          MLLM_RT_ASSERT_EQ(ci->data().size(), 1);
          constant_v = ci->data()[0];
          constant_dtype = kFloat32;
        } else if (constant_ir->isa_<ir::VectorInt16Attr>()) {
          auto ci = constant_ir->cast_<ir::VectorInt16Attr>();
          MLLM_RT_ASSERT_EQ(ci->data().size(), 1);
          constant_v = ci->data()[0];
          constant_dtype = kInt16;
        } else {
          NYI("Not implement constant attribute type");
        }

        // Calculate constant scale after PTQ
        MLLM_RT_ASSERT_EQ(scale.numel(), 1);
        auto scale_fp = scale.item<mllm_fp32_t>();
        auto quant_value = std::round(constant_v / scale_fp);
        auto clamped_value =
            std::clamp(static_cast<int32_t>(quant_value), static_cast<int32_t>(min_v), static_cast<int32_t>(max_v));
        auto ptq_constant_v = static_cast<mllm_int32_t>(clamped_value);

        if (constant_ir->isa_<ir::VectorFP32Attr>()) {
          auto ci = constant_ir->cast_<ir::VectorFP32Attr>();
          ci->data()[0] = ptq_constant_v;

          // FIXME: We hard code uint16 here.
          tv->tensor_ = Tensor::ones({1}, kUInt16, kCPU);
          tv->tensor_.at<mllm_uint16_t>({0}) = ptq_constant_v;
        } else if (constant_ir->isa_<ir::VectorInt16Attr>()) {
          auto ci = constant_ir->cast_<ir::VectorInt16Attr>();
          ci->data()[0] = ptq_constant_v;

          // FIXME: We hard code uint16 here.
          tv->tensor_ = Tensor::ones({1}, kUInt16, kCPU);
          tv->tensor_.at<mllm_uint16_t>({0}) = ptq_constant_v;
        }

        auto _attr = ctx->create<ir::VectorUInt16Attr>(std::vector<uint16_t>{(uint16_t)ptq_constant_v});
        tv->removeAttr("constant");
        tv->setAttr("constant", _attr);

        MLLM_INFO("Constant tensor '{}' quantized (SymPerTensor): before={}, after={}", tv->name(), constant_v, ptq_constant_v);
      }

      this_spec->solved = true;
      break;
    }
    case ir::linalg::QuantizationSpecType::kRaw: {
      auto this_spec = std::static_pointer_cast<ir::linalg::QuantizationSpecRaw>(f_spec->spec_);
      this_spec->solved = true;
      break;
    }
    default: {
      NYI("Quant recipe type not support on tensor: {}", v->name());
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

      for (auto iii : inputs) { _recursiveSolveNormalImpl(ir_ctx, iii->cast_<ir::Val>()); }
      for (auto ooo : outputs) { _recursiveSolveNormalImpl(ir_ctx, ooo->cast_<ir::Val>()); }
    }

    if (op->isa_<ir::graph::CallGraphOp>()) {
      auto ns = op->cast_<ir::graph::CallGraphOp>()->getSymbolAttr()->str();
      recursiveSolveNormal(w.getContext(), w.getContext()->lookupSymbolTable(ns)->cast_<ir::graph::SubGraphOp>(), pf);
    }
    return ir::IRWriter::WALK_CONTINUE;
  });
}

void recursiveCheckUnsolved(const std::shared_ptr<ir::IRContext>& ir_ctx, const ir::graph::SubGraphOp::ptr_t& call_op) {
  auto wow = ir::IRWriter(ir_ctx, call_op->getTopRegion());
  wow.walk<ir::Op>([&](ir::IRWriter& w, const ir::Op::ptr_t& op) -> ir::IRWriter::WalkResult {
    if (op->isa_<ir::linalg::LinalgIROp>()) {
      auto linalg_op = op->cast_<ir::linalg::LinalgIROp>();
      std::string op_name = linalg_op->getAOp()->getName();

      auto inputs = op->inputs();
      auto outputs = op->outputs();

      for (auto iii : inputs) {
        if (!iii->isa_<ir::tensor::TensorValue>()) continue;
        auto tv = iii->cast_<ir::tensor::TensorValue>();
        if (!tv->getAttr("quant_recipe")) continue;
        auto f_spec = tv->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>();
        if (!f_spec->spec_->solved) {
          MLLM_ERROR_EXIT(ExitCode::kCoreError, "PTQPass: TensorValue '{}' is not solved, used by Op: '{}'", tv->name(),
                          op_name);
        }
      }

      for (auto ooo : outputs) {
        if (!ooo->isa_<ir::tensor::TensorValue>()) continue;
        auto tv = ooo->cast_<ir::tensor::TensorValue>();
        if (!tv->getAttr("quant_recipe")) continue;
        auto f_spec = tv->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>();
        if (!f_spec->spec_->solved) {
          MLLM_ERROR_EXIT(ExitCode::kCoreError, "PTQPass: TensorValue '{}' is not solved, produced by Op: '{}'", tv->name(),
                          op_name);
        }
      }
    }

    if (op->isa_<ir::graph::CallGraphOp>()) {
      auto ns = op->cast_<ir::graph::CallGraphOp>()->getSymbolAttr()->str();
      recursiveCheckUnsolved(w.getContext(), w.getContext()->lookupSymbolTable(ns)->cast_<ir::graph::SubGraphOp>());
    }
    return ir::IRWriter::WALK_CONTINUE;
  });
}

void recursiveCheckConcatInputs(const std::shared_ptr<ir::IRContext>& ir_ctx, const ir::graph::SubGraphOp::ptr_t& call_op) {
  auto wow = ir::IRWriter(ir_ctx, call_op->getTopRegion());
  wow.walk<ir::Op>([&](ir::IRWriter& w, const ir::Op::ptr_t& op) -> ir::IRWriter::WalkResult {
    if (op->isa_<ir::linalg::ConcatOp>()) {
      auto concat_op = op->cast_<ir::linalg::ConcatOp>();
      std::string op_name = concat_op->getAOp()->getName();

      auto inputs = op->inputs();
      if (inputs.empty()) { return ir::IRWriter::WALK_CONTINUE; }

      // Get first input's scale and zero_point as reference
      Tensor ref_scale;
      Tensor ref_zero_point;
      bool has_ref = false;
      std::string ref_input_name;

      for (auto iii : inputs) {
        if (!iii->isa_<ir::tensor::TensorValue>()) continue;
        auto tv = iii->cast_<ir::tensor::TensorValue>();
        if (!tv->getAttr("quant_recipe")) continue;
        auto f_spec = tv->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>();

        if (f_spec->spec_->type == ir::linalg::QuantizationSpecType::kAsymPerTensor) {
          auto this_spec = std::static_pointer_cast<ir::linalg::QuantizationSpecAsymPerTensor>(f_spec->spec_);
          if (!this_spec->solved) continue;

          if (!has_ref) {
            ref_scale = this_spec->scale;
            ref_zero_point = this_spec->zero_point;
            ref_input_name = tv->name();
            has_ref = true;
          } else {
            // Check if scale and zero_point match
            auto cur_scale = this_spec->scale;
            auto cur_zero_point = this_spec->zero_point;

            MLLM_RT_ASSERT_EQ(ref_scale.numel(), 1);
            MLLM_RT_ASSERT_EQ(cur_scale.numel(), 1);
            MLLM_RT_ASSERT_EQ(ref_zero_point.numel(), 1);
            MLLM_RT_ASSERT_EQ(cur_zero_point.numel(), 1);

            auto ref_scale_v = ref_scale.item<mllm_fp32_t>();
            auto cur_scale_v = cur_scale.item<mllm_fp32_t>();
            auto ref_zp_v = ref_zero_point.item<mllm_int32_t>();
            auto cur_zp_v = cur_zero_point.item<mllm_int32_t>();

            if (std::abs(ref_scale_v - cur_scale_v) > 1e-6 || ref_zp_v != cur_zp_v) {
              MLLM_ERROR("PTQPass: ConcatOp '{}' has mismatched scale/zp between inputs. "
                         "Input '{}': scale={}, zp={}, scale_name={}, zp_name={}; Input '{}': scale={}, zp={}, scale_name={}, "
                         "zp_name={}",
                         op_name, ref_input_name, ref_scale_v, ref_zp_v, ref_scale.name(), ref_zero_point.name(), tv->name(),
                         cur_scale_v, cur_zp_v, cur_scale.name(), cur_zero_point.name());
            }
          }
        } else if (f_spec->spec_->type == ir::linalg::QuantizationSpecType::kSymPerTensor) {
          auto this_spec = std::static_pointer_cast<ir::linalg::QuantizationSpecSymPerTensor>(f_spec->spec_);
          if (!this_spec->solved) continue;

          if (!has_ref) {
            ref_scale = this_spec->scale;
            ref_input_name = tv->name();
            has_ref = true;
          } else {
            // Check if scale matches
            auto cur_scale = this_spec->scale;

            MLLM_RT_ASSERT_EQ(ref_scale.numel(), 1);
            MLLM_RT_ASSERT_EQ(cur_scale.numel(), 1);

            auto ref_scale_v = ref_scale.item<mllm_fp32_t>();
            auto cur_scale_v = cur_scale.item<mllm_fp32_t>();

            if (std::abs(ref_scale_v - cur_scale_v) > 1e-6) {
              MLLM_ERROR("PTQPass: ConcatOp '{}' has mismatched scale between inputs. "
                         "Input '{}': scale={}; Input '{}': scale={}",
                         op_name, ref_input_name, ref_scale_v, tv->name(), cur_scale_v);
            }
          }
        }
      }
    }

    if (op->isa_<ir::graph::CallGraphOp>()) {
      auto ns = op->cast_<ir::graph::CallGraphOp>()->getSymbolAttr()->str();
      recursiveCheckConcatInputs(w.getContext(), w.getContext()->lookupSymbolTable(ns)->cast_<ir::graph::SubGraphOp>());
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

  // Check for unsolved tensorValues and warn
  recursiveCheckUnsolved(
      writer.getContext(),
      getCtx()->lookupSymbolTable(call_main_graph_op->getSymbolAttr()->str())->cast_<ir::graph::SubGraphOp>());

  // Check Concat inputs have consistent scale and zero_point
  recursiveCheckConcatInputs(
      writer.getContext(),
      getCtx()->lookupSymbolTable(call_main_graph_op->getSymbolAttr()->str())->cast_<ir::graph::SubGraphOp>());

  return ir::PASS_RET_SUCCESS;
}

ir::Pass::ptr_t createPTQPass() { return std::make_shared<PTQPass>(); }

}  // namespace mllm::qnn::aot
