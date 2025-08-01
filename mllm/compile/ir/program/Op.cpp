// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/compile/ir/program/Op.hpp"
#include "mllm/compile/ir/GeneratedRTTIKind.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"

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

std::vector<BaseOp*> InstructionOp::getMllmOps() const { return mllm_concrete_ops_; }

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

JumpOp::JumpOp() : ProgramIROp(RK_Op_ProgramIROp_JumpOp) {}

void JumpOp::dump(IRPrinter& p) {
  p.print("prog.jump");
  p.blank();
  p.print("{}", label_name_);
}

JumpOp::ptr_t JumpOp::build(IRContext* ctx, const std::string& label_name) {
  auto ret = std::make_shared<JumpOp>();
  ret->label_name_ = label_name;
  return ret;
}

LabelOp::LabelOp() : ProgramIROp(RK_Op_ProgramIROp_LabelOp) {}

void LabelOp::dump(IRPrinter& p) {
  p.print("prog.label");
  dumpAttributes(p);
}

LabelOp::ptr_t LabelOp::build(IRContext* ctx, const SymbolAttr::ptr_t& symbol_attr) {
  auto ret = std::make_shared<LabelOp>();
  ret->setSymbolAttr(symbol_attr);
  return ret;
}

EntryPointOp::EntryPointOp() : ProgramIROp(RK_Op_ProgramIROp_EntryPointOp) {}

void EntryPointOp::dump(IRPrinter& p) {
  p.print("prog.entry_point");
  Op::dump(p);
  dumpAttributes(p);
}

EntryPointOp::ptr_t EntryPointOp::build(IRContext* ctx, const SymbolAttr::ptr_t& symbol_attr,
                                        const std::vector<val_weak_ptr_t>& inputs, const std::vector<val_weak_ptr_t>& outputs) {
  auto ret = std::make_shared<EntryPointOp>();
  ret->setSymbolAttr(symbol_attr);

  for (auto& i : inputs) {
    i.get_weak()->outputs().emplace_back(ret);
    ret->inputs().emplace_back(i);
  }

  for (auto& o : outputs) {
    ret->outputs().emplace_back(o);
    o.get_weak()->inputs().emplace_back(ret);
  }

  return ret;
}

}  // namespace mllm::ir::program
