/**
 * @file Op.cpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-29
 *
 */

#include "mllm/compile/ir/program/Op.hpp"
#include "mllm/compile/ir/GeneratedRTTIKind.hpp"

namespace mllm::ir::program {

ProgramIROp::ProgramIROp() : Op(RK_Op_ProgramIROp) {}

ProgramIROp::ProgramIROp(const NodeKind& kind) : Op(kind) {}

InstructionOp::InstructionOp() : ProgramIROp(RK_Op_ProgramIROp_InstructionOp) {}

void InstructionOp::dump(IRPrinter& p) {
  // Not intrinsic !!!
  p.print("prog.{}.inst.", deviceTypes2Str(getDevice()));
  auto size = mllm_concrete_ops_.size();
  int cnt = 0;
  for (auto op : mllm_concrete_ops_) {
    p.print(optype2Str(op->getOpType()));
    if (cnt < size - 1) { p.print("+"); }
    cnt++;
  }
  Op::dump(p);
}

InstructionOp::ptr_t InstructionOp::build(IRContext* ctx, const std::vector<tensor::TensorValue::ptr_t>& inputs,
                                          const std::vector<tensor::TensorValue::ptr_t>& outputs) {
  auto inst = std::make_shared<InstructionOp>();
  for (auto& input : inputs) { (*input)-- > inst; }
  for (auto& output : outputs) { (*inst)-- > output; }
  return inst;
}

void InstructionOp::pushMllmOp(BaseOp* op) { mllm_concrete_ops_.push_back(op); }

FragmentOp::FragmentOp() : ProgramIROp(RK_Op_ProgramIROp_FragmentOp) {}

void FragmentOp::dump(IRPrinter& p) {
  auto fragment_type_2_str = [](FragmentType fragment_type) -> std::string {
    switch (fragment_type) {
      case FragmentType::kCode: return "code";
      case FragmentType::kData: return "data";
      case FragmentType::kText: return "text";
      default: return "unknown";
    }
  };

  p.print("prog.fragment @{} ", getSymbolAttr()->str());
  p.langle();
  p.print("{}", deviceTypes2Str(getDevice()));
  p.rangle();
  p.blank();
  p.langle();
  p.print("{}", fragment_type_2_str(type_));
  p.rangle();
  p.blank();
  p.lbrace();

  getTopRegion()->dump(p);

  p.rbrace();
}

FragmentOp::ptr_t FragmentOp::build(IRContext* ctx, FragmentType type, const SymbolAttr::ptr_t& symbol_attr) {
  auto ret = std::make_shared<FragmentOp>();
  ret->setSymbolAttr(symbol_attr);
  ret->createRegionAtTop();
  ret->type_ = type;
  ctx->addToSymbolTable(ret, symbol_attr->str());
  return ret;
}

}  // namespace mllm::ir::program
