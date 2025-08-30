// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/compile/ir/program/Op.hpp"
#include "mllm/compile/ir/GeneratedRTTIKind.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"

namespace mllm::ir::program {

ProgramIROp::ProgramIROp() : Op(RK_Op_ProgramIROp) {}

ProgramIROp::ProgramIROp(const NodeKind& kind) : Op(kind) {}

FragmentOp::FragmentOp() : ProgramIROp(RK_Op_ProgramIROp_FragmentOp) {}

void FragmentOp::dump(IRPrinter& p) {
  auto fragment_type_2_str = [](FragmentType fragment_type) -> std::string {
    switch (fragment_type) {
      case FragmentType::kCode: return "code";
      case FragmentType::kData: return "data";
      case FragmentType::kTable: return "table";
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

  {
    auto& ops = getTopRegion()->ops();
    size_t cnt = 0;
    auto size = ops.size();
    for (auto& op : ops) {
      if (op->isa_<ProgramIROp>()) { p.print("addr:{:#016x} ", op->cast_<ProgramIROp>()->getProgramIntrinsicId()); }
      op->dump(p);
      if (cnt < size - 1) p.newline();
      cnt++;
    }
  }

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

KernelLaunchOp::KernelLaunchOp() : ProgramIROp(RK_Op_ProgramIROp_KernelLaunchOp) {}

void KernelLaunchOp::dump(IRPrinter& p) {
  p.print("prog.kernel_launch");
  Op::dump(p);
  dumpAttributes(p);
}

KernelLaunchOp::ptr_t KernelLaunchOp::build(IRContext* ctx, const std::vector<tensor::TensorValue::ptr_t>& inputs,
                                            const std::vector<tensor::TensorValue::ptr_t>& outputs, const std::string& op_type,
                                            const std::string& op_options) {
  auto ret = std::make_shared<KernelLaunchOp>();
  ret->setAttr("op_options", ctx->create<StrAttr>(op_options));
  ret->setAttr("op_type", ctx->create<StrAttr>(op_type));
  for (auto& input : inputs) { (*input)-- > ret; }
  for (auto& output : outputs) { (*ret)-- > output; }
  return ret;
}

JumpOp::JumpOp() : ProgramIROp(RK_Op_ProgramIROp_JumpOp) {}

void JumpOp::dump(IRPrinter& p) {
  p.print("prog.jump");
  p.blank();
  p.print("{}", label_name_);
  dumpAttributes(p);
}

const std::string& JumpOp::labelName() const { return label_name_; }

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

ExitOp::ExitOp() : ProgramIROp(RK_Op_ProgramIROp_ExitOp) {}

void ExitOp::dump(IRPrinter& p) { p.print("prog.exit"); }

ExitOp::ptr_t ExitOp::build(IRContext* ctx) { return std::make_shared<ExitOp>(); }

RetOp::RetOp() : ProgramIROp(RK_Op_ProgramIROp_RetOp) {}

void RetOp::dump(IRPrinter& p) { p.print("prog.ret"); }

RetOp::ptr_t RetOp::build(IRContext* ctx) { return std::make_shared<RetOp>(); }

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
