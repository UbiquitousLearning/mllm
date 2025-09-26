// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/compile/ir/tensor/Op.hpp"
#include <cstdint>
#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/mllm.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::ir::tensor {
TensorIROp::~TensorIROp() = default;

TensorIROp::TensorIROp() : Op(RK_Op_TensorIROp) { MLLM_EMPTY_SCOPE; }

TensorIROp::TensorIROp(NodeKind kind) : Op(kind) { MLLM_EMPTY_SCOPE; }

RegisterOp::~RegisterOp() = default;

RegisterOp::RegisterOp() : TensorIROp(RK_Op_TensorIROp_RegisterOp) { MLLM_EMPTY_SCOPE; }

void RegisterOp::dump(IRPrinter& p) {
  p.print("tensor.{}.register ", deviceTypes2Str(getDevice()));
  Op::dump(p);
  dumpAttributes(p);
}

RegisterOp::ptr_t RegisterOp::build(IRContext* ctx, const TensorValue::ptr_t& tensor_v) {
  auto ret = std::make_shared<RegisterOp>();

  // This op generate the tensor value.
  // The registerted tensor is marked as a produced value.
  (*ret)-- > tensor_v;

  // The symbol is registered
  MLLM_RT_ASSERT((tensor_v->tensor_.memType() <= TensorMemTypes::kParams_End
                  && tensor_v->tensor_.memType() >= TensorMemTypes::kParams_Start)
                 || (tensor_v->tensor_.memType() == TensorMemTypes::kGlobal) || tensor_v->getAttr("constant"));

  if (tensor_v->hasSymbolAttr()) {
    // Symbol
    auto symbol_attr = ctx->create<SymbolAttr>(tensor_v->getSymbolAttr()->str());
    ret->setSymbolAttr(symbol_attr);

    ctx->addToSymbolTable(ret, symbol_attr->str());
  } else {
    // Constant
    auto symbol_attr = ctx->create<SymbolAttr>("constant_" + std::to_string(tensor_v->tensor_.uuid()));
    ret->setSymbolAttr(symbol_attr);

    ctx->addToSymbolTable(ret, symbol_attr->str());
  }

  return ret;
}

TensorValue::ptr_t RegisterOp::getRegisteredTensor() { return outputs().front()->cast_<::mllm::ir::tensor::TensorValue>(); }

AllocOp::~AllocOp() = default;

AllocOp::AllocOp() : TensorIROp(RK_Op_TensorIROp_AllocOp) { MLLM_EMPTY_SCOPE; }

void AllocOp::dump(IRPrinter& p) {
  p.print("tensor.{}.alloc ", deviceTypes2Str(getDevice()));
  Op::dump(p);
  dumpAttributes(p);
}

AllocOp::ptr_t AllocOp::build(IRContext* ctx, const TensorValue::ptr_t& tensor_v) {
  auto ret = std::make_shared<AllocOp>();

  (*ret)-- > tensor_v;

  return ret;
}

TensorValue::ptr_t AllocOp::getAllocatedTensor() { return outputs().front()->cast_<::mllm::ir::tensor::TensorValue>(); }

FreeOp::~FreeOp() = default;

FreeOp::FreeOp() : TensorIROp(RK_Op_TensorIROp_FreeOp) { MLLM_EMPTY_SCOPE; }

void FreeOp::dump(IRPrinter& p) {
  p.print("tensor.{}.free ", deviceTypes2Str(getDevice()));
  Op::dump(p);
  dumpAttributes(p);
}

FreeOp::ptr_t FreeOp::build(IRContext* ctx, const TensorValue::ptr_t& tensor_v) {
  auto ret = std::make_shared<FreeOp>();

  (*tensor_v)-- > ret;

  return ret;
}

TensorValue::ptr_t FreeOp::getFreedTensor() { return inputs().front()->cast_<::mllm::ir::tensor::TensorValue>(); }

}  // namespace mllm::ir::tensor
