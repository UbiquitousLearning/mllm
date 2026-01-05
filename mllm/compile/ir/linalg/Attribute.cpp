// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include <memory>
#include <sstream>
#include <utility>

#include "mllm/core/DataTypes.hpp"
#include "mllm/compile/ir/linalg/Attribute.hpp"

namespace mllm::ir::linalg {

LinalgIRAttr::~LinalgIRAttr() = default;

LinalgIRAttr::LinalgIRAttr() : Attr(RK_Attr_LinalgIRAttr) {}

LinalgIRAttr::LinalgIRAttr(const NodeKind& kind) : Attr(kind) {}

LinalgIRQuantizatonAnnotationAttr::~LinalgIRQuantizatonAnnotationAttr() = default;

LinalgIRQuantizatonAnnotationAttr::LinalgIRQuantizatonAnnotationAttr()
    : LinalgIRAttr(RK_Attr_LinalgIRAttr_QuantizationAnnotation) {}

LinalgIRQuantizatonAnnotationAttr::LinalgIRQuantizatonAnnotationAttr(const NodeKind& kind) : LinalgIRAttr(kind) {}

void LinalgIRQuantizatonAnnotationAttr::dump(IRPrinter& p) {
  auto gen_quant_spec_str = [](const QuantizationSpec::ptr_t& q) -> std::string {
    std::stringstream ss;
    ss << "QuantSpec(";
    switch (q->type) {
      case QuantizationSpecType::kSymPerTensor: {
        auto _q = std::static_pointer_cast<QuantizationSpecSymPerTensor>(q);
        ss << "SymPerTensor(";
        ss << "quant_min: " << _q->quant_min;
        ss << ", quant_max: " << _q->quant_max;
        ss << ", quant_to_type: " << nameOfType(_q->quant_to_type);
        ss << ", scale_type: " << nameOfType(_q->scale_type);
        ss << ")";
        break;
      }
      case QuantizationSpecType::kSymPerChannel: {
        auto _q = std::static_pointer_cast<QuantizationSpecSymPerChannel>(q);
        ss << "SymPerChannel(";
        ss << "quant_min: " << _q->quant_min;
        ss << ", quant_max: " << _q->quant_max;
        ss << ", ch_axis: " << _q->ch_axis;
        ss << ", quant_to_type: " << nameOfType(_q->quant_to_type);
        ss << ", scale_type: " << nameOfType(_q->scale_type);
        ss << ")";
        break;
      }
      case QuantizationSpecType::kSymPerBlock: {
        auto _q = std::static_pointer_cast<QuantizationSpecSymPerBlock>(q);
        ss << "SymPerBlock(";
        ss << "quant_min: " << _q->quant_min;
        ss << ", quant_max: " << _q->quant_max;
        ss << ", block_size: " << _q->block_size;
        ss << ", quant_to_type: " << nameOfType(_q->quant_to_type);
        ss << ", scale_type: " << nameOfType(_q->scale_type);
        ss << ")";
        break;
      }
      case QuantizationSpecType::kAsymPerTensor: {
        auto _q = std::static_pointer_cast<QuantizationSpecAsymPerTensor>(q);
        ss << "AsymPerTensor(";
        ss << "quant_min: " << _q->quant_min;
        ss << ", quant_max: " << _q->quant_max;
        ss << ", quant_to_type: " << nameOfType(_q->quant_to_type);
        ss << ", scale_type: " << nameOfType(_q->scale_type);
        ss << ", zero_point_type: " << nameOfType(_q->zero_point_type);
        ss << ")";
        break;
      }
      case QuantizationSpecType::kAsymPerChannel: {
        auto _q = std::static_pointer_cast<QuantizationSpecAsymPerChannel>(q);
        ss << "AsymPerChannel(";
        ss << "quant_min: " << _q->quant_min;
        ss << ", quant_max: " << _q->quant_max;
        ss << ", ch_axis: " << _q->ch_axis;
        ss << ", quant_to_type: " << nameOfType(_q->quant_to_type);
        ss << ", scale_type: " << nameOfType(_q->scale_type);
        ss << ", zero_point_type: " << nameOfType(_q->zero_point_type);
        ss << ")";
        break;
      }
      case QuantizationSpecType::kAsymPerBlock: {
        auto _q = std::static_pointer_cast<QuantizationSpecAsymPerBlock>(q);
        ss << "AsymPerBlock(";
        ss << "quant_min: " << _q->quant_min;
        ss << ", quant_max: " << _q->quant_max;
        ss << ", block_size: " << _q->block_size;
        ss << ", quant_to_type: " << nameOfType(_q->quant_to_type);
        ss << ", scale_type: " << nameOfType(_q->scale_type);
        ss << ", zero_point_type: " << nameOfType(_q->zero_point_type);
        ss << ")";
        break;
      }
      case QuantizationSpecType::kLPBQ: {
        auto _q = std::static_pointer_cast<QuantizationSpecLPBQ>(q);
        ss << "LPBQ(";
        ss << "quant_min: " << _q->quant_min;
        ss << ", quant_max: " << _q->quant_max;
        ss << ", block_size: " << _q->block_size;
        ss << ", ch_axis: " << _q->ch_axis;
        ss << ", scale_level_0_bitwidth: " << _q->scale_level_0_bitwidth;
        ss << ", quant_to_type: " << nameOfType(_q->quant_to_type);
        ss << ", scale_1_type: " << nameOfType(_q->scale_1_type);
        ss << ")";
        break;
      }
      case QuantizationSpecType::kNone: {
        ss << "None()";
        break;
      }
      case QuantizationSpecType::kRaw: {
        auto _q = std::static_pointer_cast<QuantizationSpecRaw>(q);
        ss << "Raw(";
        ss << "type: " << nameOfType(_q->type_);
        ss << ")";
        break;
      }
    }
    ss << ", ";
    ss << "uuid=" << q->uuid;
    ss << ", ";
    ss << "solved=" << q->solved;
    ss << ")";
    return ss.str();
  };

  p.print("QuantAnnotation");
  p.lparentheses();
  for (int i = 0; i < annotation_.inputs.size(); ++i) {
    p.print("inputs_" + std::to_string(i));
    p.colon();
    p.print("{}", gen_quant_spec_str(annotation_.inputs[i]));
    if (i < annotation_.inputs.size() - 1) { p.comma(); }
  }
  p.comma();
  for (int i = 0; i < annotation_.outputs.size(); ++i) {
    p.print("outputs_" + std::to_string(i));
    p.colon();
    p.print("{}", gen_quant_spec_str(annotation_.outputs[i]));
    if (i < annotation_.outputs.size() - 1) { p.comma(); }
  }
  p.comma();
  int weight_idx = 0;
  for (const auto& [name, spec] : annotation_.weights) {
    p.print("weight_" + name);
    p.colon();
    p.print("{}", gen_quant_spec_str(spec));
    if (weight_idx < annotation_.weights.size() - 1) { p.comma(); }
    ++weight_idx;
  }
  p.rparentheses();
}

LinalgIRQuantizatonAnnotationAttr::ptr_t LinalgIRQuantizatonAnnotationAttr::build(IRContext* ctx) {
  auto ret = std::make_shared<LinalgIRQuantizatonAnnotationAttr>();
  return ret;
}

LinalgIRQuantizatonSpecAttr::~LinalgIRQuantizatonSpecAttr() = default;

LinalgIRQuantizatonSpecAttr::LinalgIRQuantizatonSpecAttr() : LinalgIRAttr(RK_Attr_LinalgIRAttr_QuantizationSpec) {}

LinalgIRQuantizatonSpecAttr::LinalgIRQuantizatonSpecAttr(const NodeKind& kind) : LinalgIRAttr(kind) {}

void LinalgIRQuantizatonSpecAttr::dump(IRPrinter& p) {
  auto gen_quant_spec_str = [](const QuantizationSpec::ptr_t& q) -> std::string {
    std::stringstream ss;
    ss << "QuantSpec(";
    switch (q->type) {
      case QuantizationSpecType::kSymPerTensor: {
        auto _q = std::static_pointer_cast<QuantizationSpecSymPerTensor>(q);
        ss << "SymPerTensor(";
        ss << "quant_min: " << _q->quant_min;
        ss << ", quant_max: " << _q->quant_max;
        ss << ", quant_to_type: " << nameOfType(_q->quant_to_type);
        ss << ", scale_type: " << nameOfType(_q->scale_type);
        ss << ")";
        break;
      }
      case QuantizationSpecType::kSymPerChannel: {
        auto _q = std::static_pointer_cast<QuantizationSpecSymPerChannel>(q);
        ss << "SymPerChannel(";
        ss << "quant_min: " << _q->quant_min;
        ss << ", quant_max: " << _q->quant_max;
        ss << ", ch_axis: " << _q->ch_axis;
        ss << ", quant_to_type: " << nameOfType(_q->quant_to_type);
        ss << ", scale_type: " << nameOfType(_q->scale_type);
        ss << ")";
        break;
      }
      case QuantizationSpecType::kSymPerBlock: {
        auto _q = std::static_pointer_cast<QuantizationSpecSymPerBlock>(q);
        ss << "SymPerBlock(";
        ss << "quant_min: " << _q->quant_min;
        ss << ", quant_max: " << _q->quant_max;
        ss << ", block_size: " << _q->block_size;
        ss << ", quant_to_type: " << nameOfType(_q->quant_to_type);
        ss << ", scale_type: " << nameOfType(_q->scale_type);
        ss << ")";
        break;
      }
      case QuantizationSpecType::kAsymPerTensor: {
        auto _q = std::static_pointer_cast<QuantizationSpecAsymPerTensor>(q);
        ss << "AsymPerTensor(";
        ss << "quant_min: " << _q->quant_min;
        ss << ", quant_max: " << _q->quant_max;
        ss << ", quant_to_type: " << nameOfType(_q->quant_to_type);
        ss << ", scale_type: " << nameOfType(_q->scale_type);
        ss << ", zero_point_type: " << nameOfType(_q->zero_point_type);
        ss << ")";
        break;
      }
      case QuantizationSpecType::kAsymPerChannel: {
        auto _q = std::static_pointer_cast<QuantizationSpecAsymPerChannel>(q);
        ss << "AsymPerChannel(";
        ss << "quant_min: " << _q->quant_min;
        ss << ", quant_max: " << _q->quant_max;
        ss << ", ch_axis: " << _q->ch_axis;
        ss << ", quant_to_type: " << nameOfType(_q->quant_to_type);
        ss << ", scale_type: " << nameOfType(_q->scale_type);
        ss << ", zero_point_type: " << nameOfType(_q->zero_point_type);
        ss << ")";
        break;
      }
      case QuantizationSpecType::kAsymPerBlock: {
        auto _q = std::static_pointer_cast<QuantizationSpecAsymPerBlock>(q);
        ss << "AsymPerBlock(";
        ss << "quant_min: " << _q->quant_min;
        ss << ", quant_max: " << _q->quant_max;
        ss << ", block_size: " << _q->block_size;
        ss << ", quant_to_type: " << nameOfType(_q->quant_to_type);
        ss << ", scale_type: " << nameOfType(_q->scale_type);
        ss << ", zero_point_type: " << nameOfType(_q->zero_point_type);
        ss << ")";
        break;
      }
      case QuantizationSpecType::kLPBQ: {
        auto _q = std::static_pointer_cast<QuantizationSpecLPBQ>(q);
        ss << "LPBQ(";
        ss << "quant_min: " << _q->quant_min;
        ss << ", quant_max: " << _q->quant_max;
        ss << ", block_size: " << _q->block_size;
        ss << ", ch_axis: " << _q->ch_axis;
        ss << ", scale_level_0_bitwidth: " << _q->scale_level_0_bitwidth;
        ss << ", quant_to_type: " << nameOfType(_q->quant_to_type);
        ss << ", scale_1_type: " << nameOfType(_q->scale_1_type);
        ss << ")";
        break;
      }
      case QuantizationSpecType::kNone: {
        ss << "None(";
        break;
      }
      case QuantizationSpecType::kRaw: {
        auto _q = std::static_pointer_cast<QuantizationSpecRaw>(q);
        ss << "Raw(";
        ss << "type: " << nameOfType(_q->type_);
        ss << ")";
        break;
      }
    }
    ss << ", ";
    ss << "uuid=" << q->uuid;
    ss << ", ";
    ss << "solved=" << q->solved;
    ss << ")";
    return ss.str();
  };

  p.print("{}", gen_quant_spec_str(spec_));
}

LinalgIRQuantizatonSpecAttr::ptr_t LinalgIRQuantizatonSpecAttr::build(IRContext* ctx) {
  auto ret = std::make_shared<LinalgIRQuantizatonSpecAttr>();
  return ret;
}

LinalgIRQuantizatonSpecAttr::ptr_t LinalgIRQuantizatonSpecAttr::build(IRContext* ctx, const QuantizationSpec::ptr_t& spec) {
  auto ret = std::make_shared<LinalgIRQuantizatonSpecAttr>();
  ret->spec_ = spec;
  return ret;
}

}  // namespace mllm::ir::linalg
