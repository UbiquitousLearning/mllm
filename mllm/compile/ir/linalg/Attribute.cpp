// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include <sstream>

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
      case QuantizationSpecType::kSymPerTensor: ss << "SymPerTensor"; break;
      case QuantizationSpecType::kSymPerChannel: ss << "SymPerChannel"; break;
      case QuantizationSpecType::kSymPerBlock: ss << "SymPerBlock"; break;
      case QuantizationSpecType::kAsymPerTensor: ss << "AsymPerTensor"; break;
      case QuantizationSpecType::kAsymPerChannel: ss << "AsymPerChannel"; break;
      case QuantizationSpecType::kAsymPerBlock: ss << "AsymPerBlock"; break;
      case QuantizationSpecType::kLPBQ: ss << "LPBQ"; break;
      case QuantizationSpecType::kNone: ss << "None"; break;
    }
    ss << ", ";
    ss << "uuid=" << q->uuid;
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
    p.print("{}", gen_quant_spec_str(annotation_.inputs[i]));
    if (i < annotation_.inputs.size() - 1) { p.comma(); }
  }
  p.comma();
  for (int i = 0; i < annotation_.weights.size(); ++i) {
    p.print("weight_" + std::to_string(i));
    p.colon();
    p.print("{}", gen_quant_spec_str(annotation_.inputs[i]));
    if (i < annotation_.inputs.size() - 1) { p.comma(); }
  }
  p.rparentheses();
}

}  // namespace mllm::ir::linalg
