// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <regex>

#include "mllm/utils/Common.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/compile/ir/Node.hpp"
#include "mllm/compile/ir/graph/Op.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/tensor/Op.hpp"
#include "mllm/compile/ir/builtin/Op.hpp"
#include "mllm/compile/ir/tensor/Value.hpp"
#include "mllm/compile/ir/linalg/Attribute.hpp"
#include "mllm/backends/qnn/aot/passes/AOTCompileContext.hpp"
#include "mllm/backends/qnn/aot/passes/LLMQuantRecipePass.hpp"

namespace mllm::qnn::aot {

namespace {

void recursiveVisitGraph(const ir::IRContext::ptr_t& ctx,
                         const std::vector<std::pair<int, ir::Pattern::ptr_t>>& patterns_w_priority_,
                         std::unordered_map<std::string, ir::Pattern::ptr_t>& _named_pattern,
                         const ir::graph::SubGraphOp::ptr_t& sub_graph_ir) {
  auto rw = ir::IRWriter(ctx, sub_graph_ir->getTopRegion());

  // If this graph's inputs has no QuantRecipe, we will assign it here.
  for (auto& input_node : sub_graph_ir->getTopRegion()->inputs()) {
    if (input_node->isa_<ir::tensor::TensorValue>() && !input_node->getAttr("quant_recipe")) {
      auto input_spec = genSimpleQuantizationSpecAttr(ctx, input_node->cast_<ir::tensor::TensorValue>());
      input_node->setAttr("quant_recipe", input_spec);
    }
  }

  rw.walk<ir::Op>([&](ir::IRWriter& iw, const ir::Op::ptr_t& some_op) -> ir::IRWriter::WalkResult {
    if (some_op->isa_<ir::linalg::LinalgIROp>()) {
      if (!some_op->getAttr("quant_recipe")) {
        for (auto& pattern : patterns_w_priority_) {
          if (pattern.second->isMatch(some_op)) {
            for (auto& _named_pattern_ : _named_pattern) {
              if (_named_pattern_.second == pattern.second) {
                MLLM_INFO("LLMQuantizationRecipePass Processing op: {} with pass: {}",
                          some_op->cast_<ir::linalg::LinalgIROp>()->getAOp()->getName(), _named_pattern_.first);
              }
            }

            if (!pattern.second->rewrite(iw, some_op)) {
              for (auto& _named_pattern_ : _named_pattern) {
                if (_named_pattern_.second == pattern.second) {
                  MLLM_ERROR_EXIT(ExitCode::kCoreError, "Failed at pass: {} on op(ptr): {}", _named_pattern_.first,
                                  some_op->cast_<ir::linalg::LinalgIROp>()->getAOp()->getName());
                }
              }
            }
            break;
          }
        }
      }
    } else if (some_op->isa_<ir::graph::CallGraphOp>()) {
      auto call_op = some_op->cast_<ir::graph::CallGraphOp>();
      auto next_g = ctx->lookupSymbolTable(call_op->getSymbolAttr()->str())->cast_<ir::graph::SubGraphOp>();
      recursiveVisitGraph(ctx, patterns_w_priority_, _named_pattern, next_g);
    }
    return ir::IRWriter::WalkResult::WALK_CONTINUE;
  });
}

}  // namespace

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//
ir::linalg::LinalgIRQuantizatonSpecAttr::ptr_t genSimpleQuantizationSpecAttr(const ir::IRContext::ptr_t& ctx,
                                                                             const ir::tensor::TensorValue::ptr_t& v) {
  ir::linalg::LinalgIRQuantizatonSpecAttr::ptr_t ret = nullptr;

  ir::linalg::QuantizationSpec::ptr_t spec = nullptr;

  switch (v->tensor_.dtype()) {
    case kInt8PerTensorSym: {
      spec = ir::linalg::QuantizationSpecSymPerTensor::create(-128, 127, kInt8, kFloat32, Tensor::nil());
      break;
    }
    case kUInt8PerTensorSym: {
      spec = ir::linalg::QuantizationSpecSymPerTensor::create(0, 255, kUInt8, kFloat32, Tensor::nil());
      break;
    }
    case kInt16PerTensorSym: {
      spec = ir::linalg::QuantizationSpecSymPerTensor::create(-32768, 32767, kInt16, kFloat32, Tensor::nil());
      break;
    }
    case kUInt16PerTensorSym: {
      spec = ir::linalg::QuantizationSpecSymPerTensor::create(0, 65535, kUInt16, kFloat32, Tensor::nil());
      break;
    }
    case kUInt16PerTensorAsy: {
      spec =
          ir::linalg::QuantizationSpecAsymPerTensor::create(0, 65535, kUInt16, kFloat32, kInt32, Tensor::nil(), Tensor::nil());
      break;
    }
    case kUInt8:
    case kUInt16:
    case kUInt32:
    case kInt8:
    case kInt16:
    case kInt32:
    case kUInt64:
    case kInt64:
    case kBFloat16:
    case kFloat16:
    case kFloat32: {
      spec = ir::linalg::QuantizationSpecRaw::create(v->tensor_.dtype());
      break;
    }
    default: {
      NYI("Only support [uint16, int16, uint8, int8] + [sym] and normal dtypes such as [float32, bfloat16, etc] for now.");
    }
  }

  ret = ctx->create<ir::linalg::LinalgIRQuantizatonSpecAttr>(spec);

  return ret;
}

bool shareQuantSpecSingleInputToSingleOutputAndSetOpQuantAnnoAttr(const ir::IRContext::ptr_t& ctx,
                                                                  const ir::linalg::LinalgIROp::ptr_t& op) {
  // OP has no quant_recipe
  MLLM_RETURN_FALSE_IF(op->getAttr("quant_recipe"));
  MLLM_RETURN_FALSE_IF_NOT(op->inputs().size() == 1);
  MLLM_RETURN_FALSE_IF_NOT(op->outputs().size() == 1);
  MLLM_RETURN_FALSE_IF_NOT(op->inputs().front()->getAttr("quant_recipe"));

  // Create annotation
  auto annotation_attr = ctx->create<ir::linalg::LinalgIRQuantizatonAnnotationAttr>();

  // Share
  auto quant_spec = op->inputs().front()->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>();
  annotation_attr->annotation_.inputs.emplace_back(quant_spec->spec_);
  annotation_attr->annotation_.outputs.emplace_back(quant_spec->spec_);
  op->outputs().front()->setAttr("quant_recipe", quant_spec);
  op->setAttr("quant_recipe", annotation_attr);

  return true;
}

bool noSharingSingleInAndSingleOutQuantAnnoAttr(const ir::IRContext::ptr_t& ctx, const ir::linalg::LinalgIROp::ptr_t& op) {
  // OP has no quant_recipe
  MLLM_RETURN_FALSE_IF(op->getAttr("quant_recipe"));
  MLLM_RETURN_FALSE_IF_NOT(op->inputs().size() == 1);
  MLLM_RETURN_FALSE_IF_NOT(op->outputs().size() == 1);

  // Create annotation
  auto annotation_attr = ctx->create<ir::linalg::LinalgIRQuantizatonAnnotationAttr>();

  // Not share
  ir::linalg::LinalgIRQuantizatonSpecAttr::ptr_t quant_spec_i0 = nullptr;
  ir::linalg::LinalgIRQuantizatonSpecAttr::ptr_t quant_spec_o0 = nullptr;

  if (!op->inputs().front()->getAttr("quant_recipe")) {
    quant_spec_i0 = genSimpleQuantizationSpecAttr(ctx, op->inputs().front()->cast_<ir::tensor::TensorValue>());
    op->inputs().front()->setAttr("quant_recipe", quant_spec_i0);
  }

  if (!op->outputs().front()->getAttr("quant_recipe")) {
    quant_spec_o0 = genSimpleQuantizationSpecAttr(ctx, op->outputs().front()->cast_<ir::tensor::TensorValue>());
    op->outputs().front()->setAttr("quant_recipe", quant_spec_o0);
  }

  quant_spec_i0 = op->inputs().front()->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>();
  quant_spec_o0 = op->outputs().front()->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>();

  annotation_attr->annotation_.inputs.emplace_back(quant_spec_i0->spec_);
  annotation_attr->annotation_.outputs.emplace_back(quant_spec_o0->spec_);
  op->setAttr("quant_recipe", annotation_attr);

  return true;
}

ir::linalg::LinalgIRQuantizatonSpecAttr::ptr_t cloneQuantizationSpecType(
    const ir::IRContext::ptr_t& ctx, const ir::linalg::LinalgIRQuantizatonSpecAttr::ptr_t& from) {
  // clone all type, but not clone scale or other self-contained data.
  if (!from || !from->spec_) { return nullptr; }

  auto from_spec = from->spec_;
  ir::linalg::QuantizationSpec::ptr_t cloned_spec = nullptr;

  switch (from_spec->type) {
    case ir::linalg::QuantizationSpecType::kRaw: {
      auto raw_spec = std::static_pointer_cast<ir::linalg::QuantizationSpecRaw>(from_spec);
      cloned_spec = ir::linalg::QuantizationSpecRaw::create(raw_spec->type_);
      break;
    }
    case ir::linalg::QuantizationSpecType::kSymPerTensor: {
      auto sym_spec = std::static_pointer_cast<ir::linalg::QuantizationSpecSymPerTensor>(from_spec);
      cloned_spec = ir::linalg::QuantizationSpecSymPerTensor::create(sym_spec->quant_min, sym_spec->quant_max,
                                                                     sym_spec->quant_to_type, sym_spec->scale_type,
                                                                     Tensor::nil());  // Not cloning scale
      break;
    }
    case ir::linalg::QuantizationSpecType::kSymPerChannel: {
      auto sym_spec = std::static_pointer_cast<ir::linalg::QuantizationSpecSymPerChannel>(from_spec);
      cloned_spec = ir::linalg::QuantizationSpecSymPerChannel::create(
          sym_spec->quant_min, sym_spec->quant_max, sym_spec->ch_axis, sym_spec->quant_to_type, sym_spec->scale_type,
          Tensor::nil());  // Not cloning scale
      break;
    }
    case ir::linalg::QuantizationSpecType::kSymPerBlock: {
      auto sym_spec = std::static_pointer_cast<ir::linalg::QuantizationSpecSymPerBlock>(from_spec);
      cloned_spec = ir::linalg::QuantizationSpecSymPerBlock::create(sym_spec->quant_min, sym_spec->quant_max,
                                                                    sym_spec->block_size, sym_spec->quant_to_type,
                                                                    sym_spec->scale_type, Tensor::nil());  // Not cloning scale
      break;
    }
    case ir::linalg::QuantizationSpecType::kAsymPerTensor: {
      auto asym_spec = std::static_pointer_cast<ir::linalg::QuantizationSpecAsymPerTensor>(from_spec);
      cloned_spec = ir::linalg::QuantizationSpecAsymPerTensor::create(
          asym_spec->quant_min, asym_spec->quant_max, asym_spec->quant_to_type, asym_spec->scale_type,
          asym_spec->zero_point_type, Tensor::nil(), Tensor::nil());  // Not cloning scale and zero_point
      break;
    }
    case ir::linalg::QuantizationSpecType::kAsymPerChannel: {
      auto asym_spec = std::static_pointer_cast<ir::linalg::QuantizationSpecAsymPerChannel>(from_spec);
      cloned_spec = ir::linalg::QuantizationSpecAsymPerChannel::create(
          asym_spec->quant_min, asym_spec->quant_max, asym_spec->ch_axis, asym_spec->quant_to_type, asym_spec->scale_type,
          asym_spec->zero_point_type, Tensor::nil(), Tensor::nil());  // Not cloning scale and zero_point
      break;
    }
    case ir::linalg::QuantizationSpecType::kAsymPerBlock: {
      auto asym_spec = std::static_pointer_cast<ir::linalg::QuantizationSpecAsymPerBlock>(from_spec);
      cloned_spec = ir::linalg::QuantizationSpecAsymPerBlock::create(
          asym_spec->quant_min, asym_spec->quant_max, asym_spec->block_size, asym_spec->quant_to_type, asym_spec->scale_type,
          asym_spec->zero_point_type, Tensor::nil(), Tensor::nil());  // Not cloning scale and zero_point
      break;
    }
    case ir::linalg::QuantizationSpecType::kLPBQ: {
      auto lpbq_spec = std::static_pointer_cast<ir::linalg::QuantizationSpecLPBQ>(from_spec);
      cloned_spec = ir::linalg::QuantizationSpecLPBQ::create(
          lpbq_spec->quant_min, lpbq_spec->quant_max, lpbq_spec->block_size, lpbq_spec->ch_axis,
          lpbq_spec->scale_level_0_bitwidth, lpbq_spec->quant_to_type, lpbq_spec->scale_1_type, Tensor::nil(),
          Tensor::nil());  // Not cloning scale_level_0_int and scale_level_1_fp
      break;
    }
    case ir::linalg::QuantizationSpecType::kNone:
    default: {
      cloned_spec = nullptr;
      break;
    }
  }

  if (!cloned_spec) { return nullptr; }

  return ctx->create<ir::linalg::LinalgIRQuantizatonSpecAttr>(cloned_spec);
}

//===----------------------------------------------------------------------===//
// Sigmoid Pattern
//===----------------------------------------------------------------------===//
bool LLMQuantRecipeConv2DPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  if (op->isa_<ir::linalg::Conv2DOp>()) { return true; }
  return false;
}

bool LLMQuantRecipeConv2DPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) {
  auto conv2d_ir = node->cast_<ir::linalg::Conv2DOp>();

  auto config = AOTCompileContext::getInstance().getConfig()["quant_recipe"]["builtin_llm_pass"]["linear"];
  auto use_config = config["fallback"];

  // Get this op name
  auto op_name = conv2d_ir->getAOp()->getName();

  // config's key is regex pattern list. try to fit each config. If no matched config, use default fallback config
  // Config e.g.:
  // "fallback": {
  //    "method": "LPBQ",
  //    "sym": true,
  //    "precision": "w4a16",
  //    "block_size": 32
  // },
  // "regex pattern": {
  //    "method": "LPBQ",
  //    "sym": true,
  //    "precision": "w4a16",
  //    "block_size": 64
  // },
  for (auto it = config.begin(); it != config.end(); ++it) {
    const std::string& key = it.key();
    if (key == "fallback") { continue; }
    try {
      std::regex pattern(key);
      if (std::regex_match(op_name, pattern)) {
        use_config = it.value();
        break;  // Found a match, stop searching
      }
    } catch (const std::regex_error& e) {
      // If the key is not a valid regex, skip it
      continue;
    }
  }

  // Apply configuration
  // Suppose the first input has quant_recipe
  MLLM_RETURN_FALSE_IF_NOT(conv2d_ir->inputs().front()->getAttr("quant_recipe"));
  {
    auto annotation_attr = writer.create<ir::linalg::LinalgIRQuantizatonAnnotationAttr>();
    auto input_spec = conv2d_ir->inputs().front()->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>();
    annotation_attr->annotation_.inputs.emplace_back(input_spec->spec_);

    if (use_config["method"] == "LPBQ") {
      // Unpack
      std::string precision = use_config["precision"];
      bool sym = use_config["sym"];
      int block_size = use_config["block_size"];
      MLLM_RETURN_FALSE_IF_NOT(sym);

      ir::linalg::QuantizationSpecLPBQ::ptr_t weight_quant_spec = nullptr;

      if (precision == "w4a16") {
        // HWIO
        weight_quant_spec =
            ir::linalg::QuantizationSpecLPBQ::create(-7, 7, block_size, 3, 4, kInt4, kFloat32, Tensor::nil(), Tensor::nil());

        // output sym int16
        auto out_quant_spec = ir::linalg::QuantizationSpecAsymPerTensor::create(0, 65536 - 1, kUInt16, kFloat32, kInt32,
                                                                                Tensor::nil(), Tensor::nil());
        conv2d_ir->outputs().front()->setAttr("quant_recipe",
                                              writer.create<ir::linalg::LinalgIRQuantizatonSpecAttr>(out_quant_spec));

        annotation_attr->annotation_.outputs.emplace_back(out_quant_spec);
        annotation_attr->annotation_.weights.insert({"weight", weight_quant_spec});
      }

      auto weight_name = conv2d_ir->getAOp()->getName() + ".weight";
      auto weight_reg_tensor_ir = writer.getContext()->lookupSymbolTable(weight_name);
      MLLM_RETURN_FALSE_IF_NOT(weight_reg_tensor_ir);
      MLLM_RETURN_FALSE_IF_NOT(weight_reg_tensor_ir->isa_<ir::tensor::RegisterOp>());
      MLLM_RETURN_FALSE_IF_NOT(weight_reg_tensor_ir->outputs().front()->isa_<ir::tensor::TensorValue>());
      auto t = weight_reg_tensor_ir->outputs().front()->cast_<ir::tensor::TensorValue>();
      t->setAttr("quant_recipe", writer.create<ir::linalg::LinalgIRQuantizatonSpecAttr>(weight_quant_spec));
    } else {
      std::string s = use_config["method"];
      MLLM_WARN("Currently not support method: {}", s);
    }

    conv2d_ir->setAttr("quant_recipe", annotation_attr);
  }

  return true;
}

//===----------------------------------------------------------------------===//
// Sigmoid Pattern
//===----------------------------------------------------------------------===//
bool LLMQuantRecipeSigmoidPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  if (op->isa_<ir::linalg::SigmoidOp>()) { return true; }
  return false;
}

bool LLMQuantRecipeSigmoidPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) {
  return noSharingSingleInAndSingleOutQuantAnnoAttr(writer.getContext(), node->cast_<ir::linalg::LinalgIROp>());
}

//===----------------------------------------------------------------------===//
// Negative Pattern
//===----------------------------------------------------------------------===//
bool LLMQuantRecipeNegPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  if (op->isa_<ir::linalg::NegOp>()) { return true; }
  return false;
}

bool LLMQuantRecipeNegPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) {
  return noSharingSingleInAndSingleOutQuantAnnoAttr(writer.getContext(), node->cast_<ir::linalg::LinalgIROp>());
}

//===----------------------------------------------------------------------===//
// ReduceMin Pattern
//===----------------------------------------------------------------------===//
bool LLMQuantRecipeReduceMinPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  if (op->isa_<ir::linalg::ReduceMinOp>()) { return true; }
  return false;
}

bool LLMQuantRecipeReduceMinPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) {
  return noSharingSingleInAndSingleOutQuantAnnoAttr(writer.getContext(), node->cast_<ir::linalg::LinalgIROp>());
}

//===----------------------------------------------------------------------===//
// RoPE Pattern
//===----------------------------------------------------------------------===//
bool LLMQuantRecipeRoPEPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  if (op->isa_<ir::linalg::RoPEOp>()) { return true; }
  return false;
}

bool LLMQuantRecipeRoPEPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) {
  auto rope_ir = node->cast_<ir::linalg::RoPEOp>();
  auto i_0 = *(node->inputs().begin());                // x
  auto i_1 = *(std::next(node->inputs().begin()));     // cos
  auto i_2 = *(std::next(node->inputs().begin(), 2));  // sin
  auto o_0 = *(node->outputs().begin());               // embedded

  MLLM_RETURN_FALSE_IF_NOT(i_0->getAttr("quant_recipe"));

  // Ensure i_1 and i_2 have quant_recipe, generate if missing
  if (!i_1->getAttr("quant_recipe")) {
    auto i_1_spec = genSimpleQuantizationSpecAttr(writer.getContext(), i_1->cast_<ir::tensor::TensorValue>());
    i_1->setAttr("quant_recipe", i_1_spec);
  }
  if (!i_2->getAttr("quant_recipe")) {
    auto i_2_spec = genSimpleQuantizationSpecAttr(writer.getContext(), i_2->cast_<ir::tensor::TensorValue>());
    i_2->setAttr("quant_recipe", i_2_spec);
  }

  // Output inherits quant_recipe from input i_0
  o_0->setAttr("quant_recipe", i_0->getAttr("quant_recipe"));

  // Create annotation
  auto annotation_attr = writer.create<ir::linalg::LinalgIRQuantizatonAnnotationAttr>();
  annotation_attr->annotation_.inputs.emplace_back(
      i_0->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>()->spec_);
  annotation_attr->annotation_.inputs.emplace_back(
      i_1->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>()->spec_);
  annotation_attr->annotation_.inputs.emplace_back(
      i_2->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>()->spec_);
  annotation_attr->annotation_.outputs.emplace_back(
      o_0->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>()->spec_);

  node->setAttr("quant_recipe", annotation_attr);

  return true;
}

//===----------------------------------------------------------------------===//
// CastType Pattern
//===----------------------------------------------------------------------===//
bool LLMQuantRecipeCastTypePattern::isMatch(const mllm::ir::op_ptr_t& op) {
  if (op->isa_<ir::linalg::CastTypeOp>()) { return true; }
  return false;
}

bool LLMQuantRecipeCastTypePattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) {
  auto cast_type_ir = node->cast_<ir::linalg::CastTypeOp>();
  auto i_0 = *(node->inputs().begin());   // cast from
  auto o_0 = *(node->outputs().begin());  // cast to

  MLLM_RETURN_FALSE_IF_NOT(i_0->getAttr("quant_recipe"));

  auto o_0_spec = genSimpleQuantizationSpecAttr(writer.getContext(), o_0->cast_<ir::tensor::TensorValue>());
  o_0->setAttr("quant_recipe", o_0_spec);

  auto annotation_attr = writer.create<ir::linalg::LinalgIRQuantizatonAnnotationAttr>();
  annotation_attr->annotation_.inputs.emplace_back(
      i_0->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>()->spec_);
  annotation_attr->annotation_.outputs.emplace_back(
      o_0->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>()->spec_);
  node->setAttr("quant_recipe", annotation_attr);

  return true;
}

//===----------------------------------------------------------------------===//
// RMSNorm Pattern
//===----------------------------------------------------------------------===//
bool LLMQuantRecipeRMSNormPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  if (op->isa_<ir::linalg::RMSNormOp>()) { return true; }
  return false;
}

bool LLMQuantRecipeRMSNormPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) {
  auto ret = noSharingSingleInAndSingleOutQuantAnnoAttr(writer.getContext(), node->cast_<ir::linalg::LinalgIROp>());

  if (!ret) return false;

  auto rms_norm_ir = node->cast_<ir::linalg::RMSNormOp>();

  // RMS Norm's weight quantization method same as inputs, but not share, just same type
  auto weight_name = rms_norm_ir->getAOp()->getName() + ".weight";
  auto weight_reg_tensor_ir = writer.getContext()->lookupSymbolTable(weight_name);
  MLLM_RETURN_FALSE_IF_NOT(weight_reg_tensor_ir);
  MLLM_RETURN_FALSE_IF_NOT(weight_reg_tensor_ir->isa_<ir::tensor::RegisterOp>());
  MLLM_RETURN_FALSE_IF_NOT(weight_reg_tensor_ir->outputs().front()->isa_<ir::tensor::TensorValue>());
  auto t = weight_reg_tensor_ir->outputs().front()->cast_<ir::tensor::TensorValue>();

  // RMSNorm weight dtype must be uint16, force set to kUInt16PerTensorAsy
  MLLM_RETURN_FALSE_IF_NOT(t->tensor_.dtype() == kUInt16 || t->tensor_.dtype() == kUInt16PerTensorAsy);
  t->tensor_ = t->tensor_.__unsafeSetDType(kUInt16PerTensorAsy);

  // FIXME: This dtype is hardcoded. We should make it right.
  auto weight_spec_attr = writer.create<ir::linalg::LinalgIRQuantizatonSpecAttr>(
      ir::linalg::QuantizationSpecAsymPerTensor::create(0, 65536 - 1, kUInt16, kFloat32, kInt32, Tensor::nil(), Tensor::nil()));
  weight_reg_tensor_ir->outputs().front()->setAttr("quant_recipe", weight_spec_attr);

  // Get self anno
  node->getAttr("quant_recipe")
      ->cast_<ir::linalg::LinalgIRQuantizatonAnnotationAttr>()
      ->annotation_.weights.insert({"weight", weight_spec_attr->spec_});

  return true;
}

//===----------------------------------------------------------------------===//
// SiLU Pattern
//===----------------------------------------------------------------------===//
bool LLMQuantRecipeSiLUPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  if (op->isa_<ir::linalg::SiLUOp>()) { return true; }
  return false;
}

bool LLMQuantRecipeSiLUPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) {
  return noSharingSingleInAndSingleOutQuantAnnoAttr(writer.getContext(), node->cast_<ir::linalg::LinalgIROp>());
}

//===----------------------------------------------------------------------===//
// Index Pattern
//===----------------------------------------------------------------------===//
bool LLMQuantRecipeIndexPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  if (op->isa_<ir::linalg::IndexOp>()) { return true; }
  return false;
}

bool LLMQuantRecipeIndexPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) {
  auto index_ir = node->cast_<ir::linalg::IndexOp>();
  auto i_0 = *(node->inputs().begin());  // Index what

  if (!i_0->getAttr("quant_recipe")) {
    auto i_0_spec = genSimpleQuantizationSpecAttr(writer.getContext(), i_0->cast_<ir::tensor::TensorValue>());
    i_0->setAttr("quant_recipe", i_0_spec);
  }

  return shareQuantSpecSingleInputToSingleOutputAndSetOpQuantAnnoAttr(writer.getContext(),
                                                                      node->cast_<ir::linalg::LinalgIROp>());
}

//===----------------------------------------------------------------------===//
// Gather Pattern
//===----------------------------------------------------------------------===//
bool LLMQuantRecipeGatherPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  if (op->isa_<ir::linalg::GatherOp>()) { return true; }
  return false;
}

bool LLMQuantRecipeGatherPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) {
  auto gather_ir = node->cast_<ir::linalg::GatherOp>();
  auto i_0 = *(node->inputs().begin());

  if (!i_0->getAttr("quant_recipe")) {
    auto i_0_spec = genSimpleQuantizationSpecAttr(writer.getContext(), i_0->cast_<ir::tensor::TensorValue>());
    i_0->setAttr("quant_recipe", i_0_spec);
  }

  auto annotation_attr = writer.getContext()->create<ir::linalg::LinalgIRQuantizatonAnnotationAttr>();
  auto op = node->cast_<ir::linalg::LinalgIROp>();

  // Share
  auto quant_spec = op->inputs().front()->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>();
  annotation_attr->annotation_.inputs.emplace_back(quant_spec->spec_);
  annotation_attr->annotation_.outputs.emplace_back(quant_spec->spec_);
  op->outputs().front()->setAttr("quant_recipe", quant_spec);
  op->setAttr("quant_recipe", annotation_attr);

  return true;
}

//===----------------------------------------------------------------------===//
// Slice Pattern
//===----------------------------------------------------------------------===//
bool LLMQuantRecipeSlicePattern::isMatch(const mllm::ir::op_ptr_t& op) {
  if (op->isa_<ir::linalg::SliceOp>()) { return true; }
  return false;
}

bool LLMQuantRecipeSlicePattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) {
  return shareQuantSpecSingleInputToSingleOutputAndSetOpQuantAnnoAttr(writer.getContext(),
                                                                      node->cast_<ir::linalg::LinalgIROp>());
}

//===----------------------------------------------------------------------===//
// Elementwise Pattern
//===----------------------------------------------------------------------===//
bool LLMQuantRecipeElementwisePattern::isMatch(const mllm::ir::op_ptr_t& op) {
  if (op->isa_<ir::linalg::AddOp>()) { return true; }
  if (op->isa_<ir::linalg::SubOp>()) { return true; }
  if (op->isa_<ir::linalg::MulOp>()) { return true; }
  if (op->isa_<ir::linalg::DivOp>()) { return true; }
  return false;
}

bool LLMQuantRecipeElementwisePattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) {
  auto i_0 = node->inputs().front();
  auto i_1 = *(std::next(node->inputs().begin()));
  auto o_0 = node->outputs().front();

  MLLM_RETURN_FALSE_IF_NOT(i_0->getAttr("quant_recipe"));

  // i_1 maybe a constant, we need to create quant recipe for it
  if (!i_1->getAttr("quant_recipe")) {
    if (i_1->getAttr("constant")) {
      i_1->setAttr("quant_recipe",
                   cloneQuantizationSpecType(writer.getContext(),
                                             i_0->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>()));

    } else {
      MLLM_WARN("LLMQuantRecipeEqualPattern Only support constant Value as second inputs right now. Pls send us a issue or PR "
                "if you want to compare two normal tensor(rather than static-tensor).");
      return false;
    }
  }

  // Create a NEW quant_recipe for output (don't share with input) so that PTQ pass can solve it independently
  o_0->setAttr("quant_recipe", genSimpleQuantizationSpecAttr(writer.getContext(), o_0->cast_<ir::tensor::TensorValue>()));

  auto annotation_attr = writer.create<ir::linalg::LinalgIRQuantizatonAnnotationAttr>();
  annotation_attr->annotation_.inputs.emplace_back(
      i_0->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>()->spec_);
  annotation_attr->annotation_.inputs.emplace_back(
      i_1->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>()->spec_);
  annotation_attr->annotation_.outputs.emplace_back(
      o_0->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>()->spec_);

  node->setAttr("quant_recipe", annotation_attr);

  return true;
}

//===----------------------------------------------------------------------===//
// Transpose Pattern
//===----------------------------------------------------------------------===//
bool LLMQuantRecipeTransposePattern::isMatch(const mllm::ir::op_ptr_t& op) {
  if (op->isa_<ir::linalg::TransposeOp>()) { return true; }
  return false;
}

bool LLMQuantRecipeTransposePattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) {
  return shareQuantSpecSingleInputToSingleOutputAndSetOpQuantAnnoAttr(writer.getContext(),
                                                                      node->cast_<ir::linalg::LinalgIROp>());
}

//===----------------------------------------------------------------------===//
// Concat Pattern
//===----------------------------------------------------------------------===//
bool LLMQuantRecipeConcatPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  if (op->isa_<ir::linalg::ConcatOp>()) { return true; }
  return false;
}

bool LLMQuantRecipeConcatPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) {
  // Current support concat two Tensor. Inherent first tensor's Quant Spec.
  auto concat_ir = node->cast_<ir::linalg::ConcatOp>();
  auto i_0 = *(node->inputs().begin());             // t1
  auto i_1 = *(std::next(node->inputs().begin()));  // t2
  auto o_0 = *(node->outputs().begin());            // to1

  if (concat_ir->inputs().size() != 2) {
    MLLM_WARN("Current support concat two Tensor. Inherent first tensor's setting.");
    return false;
  }

  // Create quant_recipe if not present
  if (!i_0->getAttr("quant_recipe")) {
    auto i_0_spec = genSimpleQuantizationSpecAttr(writer.getContext(), i_0->cast_<ir::tensor::TensorValue>());
    i_0->setAttr("quant_recipe", i_0_spec);
  }
  if (!i_1->getAttr("quant_recipe")) {
    auto i_1_spec = genSimpleQuantizationSpecAttr(writer.getContext(), i_1->cast_<ir::tensor::TensorValue>());
    i_1->setAttr("quant_recipe", i_1_spec);
  }

  o_0->setAttr("quant_recipe", i_0->getAttr("quant_recipe"));

  auto annotation_attr = writer.create<ir::linalg::LinalgIRQuantizatonAnnotationAttr>();
  annotation_attr->annotation_.inputs.emplace_back(
      i_0->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>()->spec_);
  annotation_attr->annotation_.inputs.emplace_back(
      i_1->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>()->spec_);
  annotation_attr->annotation_.outputs.emplace_back(
      o_0->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>()->spec_);

  node->setAttr("quant_recipe", annotation_attr);

  return true;
}

//===----------------------------------------------------------------------===//
// Repeat Pattern
//===----------------------------------------------------------------------===//
bool LLMQuantRecipeRepeatPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  if (op->isa_<ir::linalg::RepeatOp>()) { return true; }
  return false;
}

bool LLMQuantRecipeRepeatPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) {
  return shareQuantSpecSingleInputToSingleOutputAndSetOpQuantAnnoAttr(writer.getContext(),
                                                                      node->cast_<ir::linalg::LinalgIROp>());
}

//===----------------------------------------------------------------------===//
// MatMul Pattern
//===----------------------------------------------------------------------===//
bool LLMQuantRecipeMatMulPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  if (op->isa_<ir::linalg::MatMulOp>()) { return true; }
  return false;
}

bool LLMQuantRecipeMatMulPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) {
  auto matmul_ir = node->cast_<ir::linalg::MatMulOp>();
  auto i_0 = *(node->inputs().begin());             // x
  auto i_1 = *(std::next(node->inputs().begin()));  // equal to
  auto o_0 = *(node->outputs().begin());

  MLLM_RETURN_FALSE_IF_NOT(i_0->getAttr("quant_recipe"));
  MLLM_RETURN_FALSE_IF_NOT(i_1->getAttr("quant_recipe"));

  auto o_spec = genSimpleQuantizationSpecAttr(writer.getContext(), o_0->cast_<ir::tensor::TensorValue>());
  o_0->setAttr("quant_recipe", o_spec);

  auto annotation_attr = writer.create<ir::linalg::LinalgIRQuantizatonAnnotationAttr>();
  node->setAttr("quant_recipe", annotation_attr);

  annotation_attr->annotation_.inputs.emplace_back(
      i_0->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>()->spec_);
  annotation_attr->annotation_.inputs.emplace_back(
      i_1->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>()->spec_);
  annotation_attr->annotation_.outputs.emplace_back(
      o_0->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>()->spec_);

  return true;
}

//===----------------------------------------------------------------------===//
// Equal Pattern
//===----------------------------------------------------------------------===//
bool LLMQuantRecipeEqualPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  if (op->isa_<ir::linalg::EqualOp>()) { return true; }
  return false;
}

bool LLMQuantRecipeEqualPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) {
  auto equal_ir = node->cast_<ir::linalg::EqualOp>();
  auto i_0 = *(node->inputs().begin());             // x
  auto i_1 = *(std::next(node->inputs().begin()));  // equal to
  auto o_0 = *(node->outputs().begin());

  MLLM_RETURN_FALSE_IF_NOT(i_0->getAttr("quant_recipe"));

  // i_1 maybe a constant, we need to create quant recipe for it
  if (!i_1->getAttr("quant_recipe")) {
    if (i_1->getAttr("constant")) {
      auto i_1_tensor = i_1->cast_<ir::tensor::TensorValue>()->tensor_;
      switch (i_1_tensor.dtype()) {
        case kUInt16:
        case kInt16:
        case kFloat32: {
          // Force all i_1 to be uint16 per tensor asy
          i_1->setAttr("quant_recipe",
                       writer.create<ir::linalg::LinalgIRQuantizatonSpecAttr>(ir::linalg::QuantizationSpecAsymPerTensor::create(
                           0, 65535, kUInt16, kFloat32, kInt32, Tensor::nil(), Tensor::nil())));
          break;
        }
        default: {
          MLLM_ERROR_EXIT(ExitCode::kCoreError, "Only support [int16, f32] for now.");
        }
      }
    } else {
      MLLM_WARN("LLMQuantRecipeEqualPattern Only support constant Value as second inputs right now. Pls send us a issue or PR "
                "if you want to compare two normal tensor(rather than static-tensor).");
      return false;
    }
  }

  // Configure output. output is uint8
  o_0->setAttr("quant_recipe", writer.create<ir::linalg::LinalgIRQuantizatonSpecAttr>(ir::linalg::QuantizationSpecRaw::create(
                                   o_0->cast_<ir::tensor::TensorValue>()->tensor_.dtype())));

  // Configure this op
  auto annotation_attr = writer.create<ir::linalg::LinalgIRQuantizatonAnnotationAttr>();
  annotation_attr->annotation_.inputs.emplace_back(
      i_0->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>()->spec_);
  annotation_attr->annotation_.inputs.emplace_back(
      i_1->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>()->spec_);
  annotation_attr->annotation_.outputs.emplace_back(
      o_0->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>()->spec_);
  node->setAttr("quant_recipe", annotation_attr);

  return true;
}

//===----------------------------------------------------------------------===//
// Where Pattern
//===----------------------------------------------------------------------===//
bool LLMQuantRecipeWherePattern::isMatch(const mllm::ir::op_ptr_t& op) {
  if (op->isa_<ir::linalg::WhereOp>()) { return true; }
  return false;
}

bool LLMQuantRecipeWherePattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) {
  auto where_ir = node->cast_<ir::linalg::WhereOp>();
  auto i_0 = *(node->inputs().begin());                        // mask
  auto i_1 = *(std::next(node->inputs().begin()));             // set when mask is true
  auto i_2 = *(std::next(std::next(node->inputs().begin())));  // set when mask is false
  auto o_0 = *(node->outputs().begin());

  MLLM_RETURN_FALSE_IF_NOT(i_0->getAttr("quant_recipe"));
  MLLM_RETURN_FALSE_IF_NOT(i_1->getAttr("quant_recipe"));
  MLLM_RETURN_FALSE_IF_NOT(i_2->getAttr("quant_recipe"));

  auto o_0_spec = genSimpleQuantizationSpecAttr(writer.getContext(), o_0->cast_<ir::tensor::TensorValue>());
  o_0->setAttr("quant_recipe", o_0_spec);

  auto annotation_attr = writer.create<ir::linalg::LinalgIRQuantizatonAnnotationAttr>();
  annotation_attr->annotation_.inputs.emplace_back(
      i_0->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>()->spec_);
  annotation_attr->annotation_.inputs.emplace_back(
      i_1->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>()->spec_);
  annotation_attr->annotation_.inputs.emplace_back(
      i_2->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>()->spec_);
  annotation_attr->annotation_.outputs.emplace_back(
      o_0->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>()->spec_);

  node->setAttr("quant_recipe", annotation_attr);

  return true;
}

//===----------------------------------------------------------------------===//
// Softmax Pattern
//===----------------------------------------------------------------------===//
bool LLMQuantRecipeSoftmaxPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  if (op->isa_<ir::linalg::SoftmaxOp>()) { return true; }
  return false;
}

bool LLMQuantRecipeSoftmaxPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) {
  return noSharingSingleInAndSingleOutQuantAnnoAttr(writer.getContext(), node->cast_<ir::linalg::LinalgIROp>());
}

//===----------------------------------------------------------------------===//
// Linear Pattern
//===----------------------------------------------------------------------===//
bool LLMQuantRecipeLinearPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  if (op->isa_<ir::linalg::LinearOp>()) { return true; }
  return false;
}

bool LLMQuantRecipeLinearPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) {
  auto linear_ir = node->cast_<ir::linalg::LinearOp>();

  auto config = AOTCompileContext::getInstance().getConfig()["quant_recipe"]["builtin_llm_pass"]["linear"];
  auto use_config = config["fallback"];

  // Get this op name
  auto op_name = linear_ir->getAOp()->getName();

  // config's key is regex pattern list. try to fit each config. If no matched config, use default fallback config
  // Config e.g.:
  // "fallback": {
  //    "method": "LPBQ",
  //    "sym": true,
  //    "precision": "w4a16",
  //    "block_size": 32
  // },
  // "regex pattern": {
  //    "method": "LPBQ",
  //    "sym": true,
  //    "precision": "w4a16",
  //    "block_size": 64
  // },
  for (auto it = config.begin(); it != config.end(); ++it) {
    const std::string& key = it.key();
    if (key == "fallback") { continue; }
    try {
      std::regex pattern(key);
      if (std::regex_match(op_name, pattern)) {
        use_config = it.value();
        break;  // Found a match, stop searching
      }
    } catch (const std::regex_error& e) {
      // If the key is not a valid regex, skip it
      continue;
    }
  }

  // Apply configuration
  // Suppose the first input has quant_recipe
  MLLM_RETURN_FALSE_IF_NOT(linear_ir->inputs().front()->getAttr("quant_recipe"));
  {
    auto annotation_attr = writer.create<ir::linalg::LinalgIRQuantizatonAnnotationAttr>();
    auto input_spec = linear_ir->inputs().front()->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>();
    annotation_attr->annotation_.inputs.emplace_back(input_spec->spec_);

    if (use_config["method"] == "LPBQ") {
      // Unpack
      std::string precision = use_config["precision"];
      bool sym = use_config["sym"];
      int block_size = use_config["block_size"];
      MLLM_RETURN_FALSE_IF_NOT(sym);

      ir::linalg::QuantizationSpecLPBQ::ptr_t weight_quant_spec = nullptr;

      if (precision == "w4a16") {
        weight_quant_spec =
            ir::linalg::QuantizationSpecLPBQ::create(-8, 7, block_size, 0, 4, kUInt4, kFloat32, Tensor::nil(), Tensor::nil());

        // output sym int16
        auto out_quant_spec = ir::linalg::QuantizationSpecAsymPerTensor::create(0, 65536 - 1, kUInt16, kFloat32, kInt32,
                                                                                Tensor::nil(), Tensor::nil());
        linear_ir->outputs().front()->setAttr("quant_recipe",
                                              writer.create<ir::linalg::LinalgIRQuantizatonSpecAttr>(out_quant_spec));

        annotation_attr->annotation_.outputs.emplace_back(out_quant_spec);
        annotation_attr->annotation_.weights.insert({"weight", weight_quant_spec});
      }

      auto weight_name = linear_ir->getAOp()->getName() + ".weight";
      auto weight_reg_tensor_ir = writer.getContext()->lookupSymbolTable(weight_name);
      MLLM_RETURN_FALSE_IF_NOT(weight_reg_tensor_ir);
      MLLM_RETURN_FALSE_IF_NOT(weight_reg_tensor_ir->isa_<ir::tensor::RegisterOp>());
      MLLM_RETURN_FALSE_IF_NOT(weight_reg_tensor_ir->outputs().front()->isa_<ir::tensor::TensorValue>());
      auto t = weight_reg_tensor_ir->outputs().front()->cast_<ir::tensor::TensorValue>();
      t->setAttr("quant_recipe", writer.create<ir::linalg::LinalgIRQuantizatonSpecAttr>(weight_quant_spec));
    } else {
      std::string s = use_config["method"];
      MLLM_WARN("Currently not support method: {}", s);
    }

    linear_ir->setAttr("quant_recipe", annotation_attr);
  }

  return true;
}

//===----------------------------------------------------------------------===//
// View Pattern
//===----------------------------------------------------------------------===//
bool LLMQuantRecipeViewPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  MLLM_RETURN_FALSE_IF_NOT(op->isa_<ir::linalg::ViewOp>());
  return true;
}

bool LLMQuantRecipeViewPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) {
  // View op share Quant Spec

  auto view_op = node->cast_<ir::linalg::ViewOp>();
  auto annotation_attr = writer.create<ir::linalg::LinalgIRQuantizatonAnnotationAttr>();

  if (view_op->inputs().front()->getAttr("quant_recipe"))
  // Pass by
  {
    auto quant_spec = view_op->inputs().front()->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>();
    view_op->outputs().front()->setAttr("quant_recipe", quant_spec);
    annotation_attr->annotation_.inputs.emplace_back(quant_spec->spec_);
    annotation_attr->annotation_.outputs.emplace_back(quant_spec->spec_);
    view_op->setAttr("quant_recipe", annotation_attr);
  } else
  // Using Raw dtype, shared inputs and outputs
  {
    auto input = view_op->inputs().front()->cast_<ir::tensor::TensorValue>();
    auto quant_spec =
        writer.create<ir::linalg::LinalgIRQuantizatonSpecAttr>(ir::linalg::QuantizationSpecRaw::create(input->tensor_.dtype()));
    annotation_attr->annotation_.inputs.emplace_back(quant_spec->spec_);
    annotation_attr->annotation_.outputs.emplace_back(quant_spec->spec_);
    view_op->inputs().front()->setAttr("quant_recipe", quant_spec);
    view_op->outputs().front()->setAttr("quant_recipe", quant_spec);
    view_op->setAttr("quant_recipe", annotation_attr);
  }

  return true;
}

//===----------------------------------------------------------------------===//
// Embedding Pattern
//===----------------------------------------------------------------------===//
bool LLMQuantRecipeEmbeddingPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  // Pattern:
  //
  // embedding(op)
  MLLM_RETURN_FALSE_IF_NOT(op->isa_<ir::linalg::EmbeddingOp>());

  // Already marked.
  MLLM_RETURN_FALSE_IF(op->getAttr("quant_recipe"));

  return true;
}

bool LLMQuantRecipeEmbeddingPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) {
  auto embedding_op = node->cast_<ir::linalg::EmbeddingOp>();
  auto i_0 = *(node->inputs().begin());
  auto o_0 = *(node->outputs().begin());

  auto annotation_attr = writer.create<ir::linalg::LinalgIRQuantizatonAnnotationAttr>();

  // i_0 logic stays the same
  if (!i_0->getAttr("quant_recipe")) {
    auto i_0_spec = genSimpleQuantizationSpecAttr(writer.getContext(), i_0->cast_<ir::tensor::TensorValue>());
    i_0->setAttr("quant_recipe", i_0_spec);
    annotation_attr->annotation_.inputs.emplace_back(
        i_0->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>()->spec_);
  } else {
    annotation_attr->annotation_.inputs.emplace_back(
        i_0->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>()->spec_);
  }

  // Weights - must be uint16, force set to kUInt16PerTensorAsy
  auto weight_name = embedding_op->getAOp()->getName() + ".weight";
  auto weight_reg_tensor_ir = writer.getContext()->lookupSymbolTable(weight_name);
  MLLM_RETURN_FALSE_IF_NOT(weight_reg_tensor_ir);
  MLLM_RETURN_FALSE_IF_NOT(weight_reg_tensor_ir->isa_<ir::tensor::RegisterOp>());
  MLLM_RETURN_FALSE_IF_NOT(weight_reg_tensor_ir->outputs().front()->isa_<ir::tensor::TensorValue>());
  auto weight_tensor = weight_reg_tensor_ir->outputs().front()->cast_<ir::tensor::TensorValue>();

  // Embedding weight dtype must be uint16, force set to kUInt16PerTensorAsy
  MLLM_RETURN_FALSE_IF_NOT(weight_tensor->tensor_.dtype() == kUInt16 || weight_tensor->tensor_.dtype() == kUInt16PerTensorAsy);
  weight_tensor->tensor_ = weight_tensor->tensor_.__unsafeSetDType(kUInt16PerTensorAsy);

  // Create weight spec with kUInt16PerTensorAsy (AsymPerTensor)
  auto weight_spec =
      ir::linalg::QuantizationSpecAsymPerTensor::create(0, 65535, kUInt16, kFloat32, kInt32, Tensor::nil(), Tensor::nil());
  auto weight_spec_attr = writer.getContext()->create<ir::linalg::LinalgIRQuantizatonSpecAttr>(weight_spec);
  weight_reg_tensor_ir->outputs().front()->setAttr("quant_recipe", weight_spec_attr);
  annotation_attr->annotation_.weights.insert({"weight", weight_spec_attr->spec_});

  // o_0's quant recipe shares with weight
  o_0->setAttr("quant_recipe", weight_spec_attr);
  annotation_attr->annotation_.outputs.emplace_back(weight_spec_attr->spec_);

  // Attach to quantize node
  node->setAttr("quant_recipe", annotation_attr);

  return true;
}

//===----------------------------------------------------------------------===//
// Qwen3 Attention Pattern
//===----------------------------------------------------------------------===//
bool LLMQuantRecipeQwen3AttentionPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  auto linear_name_match = [](const ir::linalg::LinearOp::ptr_t& linear_op, const std::string& match) -> bool {
    auto name = linear_op->getAOp()->getName();
    return name.ends_with(match);
  };

  // Q, K, V Liner
  MLLM_RETURN_FALSE_IF_NOT(op->isa_<ir::linalg::LinearOp>());
  MLLM_RETURN_FALSE_IF_NOT(linear_name_match(op->cast_<ir::linalg::LinearOp>(), "q_proj"));
  auto cur_op = op->nextOp();
  MLLM_RETURN_FALSE_IF_NOT(cur_op->isa_<ir::linalg::LinearOp>());
  MLLM_RETURN_FALSE_IF_NOT(linear_name_match(cur_op->cast_<ir::linalg::LinearOp>(), "k_proj"));
  cur_op = cur_op->nextOp();
  MLLM_RETURN_FALSE_IF_NOT(cur_op->isa_<ir::linalg::LinearOp>());
  MLLM_RETURN_FALSE_IF_NOT(linear_name_match(cur_op->cast_<ir::linalg::LinearOp>(), "v_proj"));

  bool find_o_proj = false;
  do {
    cur_op = cur_op->nextOp();
    if (cur_op && cur_op->isa_<ir::linalg::LinearOp>() && linear_name_match(cur_op->cast_<ir::linalg::LinearOp>(), "o_proj")) {
      find_o_proj = true;
      break;
    }
  } while (cur_op);
  MLLM_RETURN_FALSE_IF_NOT(find_o_proj);

  return true;
}

bool LLMQuantRecipeQwen3AttentionPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) {
  // Find Q, K, V Linear and O Linear. O Linear is the end of this pattern.
  auto q_linear_ir = node->cast_<ir::linalg::LinearOp>();
  auto k_linear_ir = node->nextOp()->cast_<ir::linalg::LinearOp>();
  auto v_linear_ir = node->nextOp()->nextOp()->cast_<ir::linalg::LinearOp>();
  WeakOwner<ir::linalg::LinearOp> o_linear_ir = nullptr;
  {
    auto linear_name_match = [](const ir::linalg::LinearOp::ptr_t& linear_op, const std::string& match) -> bool {
      auto name = linear_op->getAOp()->getName();
      return name.ends_with(match);
    };
    auto cur_op = node->nextOp();
    bool find_o_proj = false;
    do {
      cur_op = cur_op->nextOp();
      if (cur_op && cur_op->isa_<ir::linalg::LinearOp>()
          && linear_name_match(cur_op->cast_<ir::linalg::LinearOp>(), "o_proj")) {
        find_o_proj = true;
        o_linear_ir = cur_op->cast_<ir::linalg::LinearOp>();
        break;
      }
    } while (cur_op);
    MLLM_RETURN_FALSE_IF_NOT(find_o_proj);
  }

  // TODO Maybe something need to be done here!

  return true;
}

//===----------------------------------------------------------------------===//
// LLMQuantRecipePass
//===----------------------------------------------------------------------===//
LLMQuantRecipePass::LLMQuantRecipePass() {
  auto config = AOTCompileContext::getInstance().getConfig();
  // Register all patterns
  addPattern(LLMQuantRecipeNegPattern::create(), "neg", 0);
  addPattern(LLMQuantRecipeConv2DPattern::create(), "conv2d", 0);
  addPattern(LLMQuantRecipeSlicePattern::create(), "slice", 0);
  addPattern(LLMQuantRecipeSigmoidPattern::create(), "sigmoid", 0);
  addPattern(LLMQuantRecipeReduceMinPattern::create(), "reduce_min", 0);
  addPattern(LLMQuantRecipeRoPEPattern::create(), "rope", 0);
  addPattern(LLMQuantRecipeCastTypePattern::create(), "cast_type", 0);
  addPattern(LLMQuantRecipeRMSNormPattern::create(), "rms_norm", 0);
  addPattern(LLMQuantRecipeSiLUPattern::create(), "silu", 0);
  addPattern(LLMQuantRecipeIndexPattern::create(), "index", 0);
  addPattern(LLMQuantRecipeElementwisePattern::create(), "elementwise", 0);
  addPattern(LLMQuantRecipeTransposePattern::create(), "transpose", 0);
  addPattern(LLMQuantRecipeConcatPattern::create(), "concat", 0);
  addPattern(LLMQuantRecipeRepeatPattern::create(), "repeat", 0);
  addPattern(LLMQuantRecipeMatMulPattern::create(), "matmul", 0);
  addPattern(LLMQuantRecipeEqualPattern::create(), "equal", 0);
  addPattern(LLMQuantRecipeWherePattern::create(), "where", 0);
  addPattern(LLMQuantRecipeSoftmaxPattern::create(), "softmax", 0);
  addPattern(LLMQuantRecipeLinearPattern::create(), "linear", 0);
  addPattern(LLMQuantRecipeEmbeddingPattern::create(), "embedding", 0);
  addPattern(LLMQuantRecipeViewPattern::create(), "view", 0);
  addPattern(LLMQuantRecipeGatherPattern::create(), "gather", 0);
}

uint8_t LLMQuantRecipePass::run(const ir::node_ptr_t& op) {
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

  if (call_main_graph_op == nullptr) { return ir::PASS_RET_SUCCESS; }

  auto main_graph = getCtx()->lookupSymbolTable(call_main_graph_op->getSymbolAttr()->str())->cast_<ir::graph::SubGraphOp>();
  MLLM_RT_ASSERT(main_graph != nullptr);

  // Sort patterns by priority in descending order
  auto sorted_patterns = pattern_with_priority_;
  std::sort(sorted_patterns.begin(), sorted_patterns.end(),
            [](const std::pair<int, ir::Pattern::ptr_t>& a, const std::pair<int, ir::Pattern::ptr_t>& b) {
              return a.first > b.first;
            });

  // Visit all graphs at tail. Handling elementwise, transpose, rms_norm, tile, rope op, etc.
  recursiveVisitGraph(getCtx(), sorted_patterns, patterns_, main_graph);

  return 0;
}

void LLMQuantRecipePass::addPattern(const ir::Pattern::ptr_t& p, const std::string& name, int priority) {
  patterns_.insert({name, p});
  pattern_with_priority_.emplace_back(priority, p);
}

ir::Pass::ptr_t createLLMQuantRecipePass() { return std::make_shared<LLMQuantRecipePass>(); }

}  // namespace mllm::qnn::aot
