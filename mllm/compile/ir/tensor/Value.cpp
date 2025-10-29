// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <string>

#include "mllm/compile/ir/tensor/Value.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/core/DataTypes.hpp"

namespace mllm::ir::tensor {

TensorIRValue::~TensorIRValue() = default;

TensorIRValue::TensorIRValue() : Val(RK_Val_TensorIRVal) {}

TensorIRValue::TensorIRValue(NodeKind kind) : Val(kind) {}

TensorValue::~TensorValue() = default;

TensorValue::TensorValue() : TensorIRValue(RK_Val_TensorIRVal_TensorVal) {}

TensorValue::ptr_t TensorValue::build(IRContext* ctx, const Tensor& tensor) {
  auto ret = std::make_shared<TensorValue>();
  ret->tensor_ = tensor;
  ret->name() = std::to_string(tensor.uuid());
  ret->setDevice(tensor.device());

  // If this tensor is parameter tensor or global tensor which has name
  switch (tensor.memType()) {
    case kGlobal:
    case kParamsNormal:
    case kParamsMMAP: ret->setSymbolAttr(SymbolAttr::build(ctx, tensor.name())); break;
    default: break;
  }

  return ret;
}

void TensorValue::dump(IRPrinter& p) {
  Val::dump(p);

  p.print("tensor");
  IRPrinter::langle();

  // shape
  {
    IRPrinter::lsbracket();
    auto size = tensor_.shape().size();
    for (int i = 0; i < size; ++i) {
      p.print("{}", tensor_.shape()[i]);
      if (i < size - 1) IRPrinter::comma();
    }
    IRPrinter::rsbracket();
  }

  IRPrinter::comma();

  // dtype
  p.print("{}", nameOfType(tensor_.dtype()));

  IRPrinter::comma();

  // device type
  p.print("{}", deviceTypes2Str(tensor_.device()));

  IRPrinter::rangle();

  if (hasSymbolAttr()) {
    IRPrinter::lsbracket();
    p.print("@{}", getSymbolAttr()->str());
    IRPrinter::rsbracket();
  }

  if (getAttr("constant")) {
    IRPrinter::lsbracket();
    p.print("constant: {}", getAttr("constant")->cast_<ir::VectorFP32Attr>()->data());
    IRPrinter::rsbracket();
  }

  dumpAttributes(p);
}

}  // namespace mllm::ir::tensor
