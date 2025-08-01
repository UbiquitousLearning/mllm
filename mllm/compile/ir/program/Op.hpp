// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <vector>

#include "mllm/core/BaseOp.hpp"
#include "mllm/compile/ir/Node.hpp"
#include "mllm/compile/ir/tensor/Value.hpp"
#include "mllm/compile/ir/builtin/Interface.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"

namespace mllm::ir::program {

class ProgramIROp : public Op {
 public:
  DEFINE_SPECIFIC_IR_CLASS(ProgramIROp);

  ~ProgramIROp() override = default;

  ProgramIROp();

  explicit ProgramIROp(const NodeKind& kind);

  static inline bool classof(const Node* node) { RTTI_RK_OP_PROGRAMIROP_IMPL(node); }
};

class InstructionOp final : public ProgramIROp {
 public:
  DEFINE_SPECIFIC_IR_CLASS(InstructionOp);

  ~InstructionOp() override = default;

  InstructionOp();

  void dump(IRPrinter& p) override;

  static ptr_t build(IRContext* ctx, const std::vector<tensor::TensorValue::ptr_t>& inputs,
                     const std::vector<tensor::TensorValue::ptr_t>& outputs);

  void pushMllmOp(BaseOp* op);

  std::vector<BaseOp*> getMllmOps() const;

  static inline bool classof(const Node* node) { RTTI_RK_OP_PROGRAMIROP_INSTRUCTIONOP_IMPL(node); }

 private:
  // If has multiple ops. It supposed that those ops can be fused when generating code.
  std::vector<BaseOp*> mllm_concrete_ops_;
};

enum class FragmentType : int32_t {
  kCode,
  kData,
  kText,
};

class FragmentOp final : public ProgramIROp, public SymbolInterface<FragmentOp> {
 public:
  DEFINE_SPECIFIC_IR_CLASS(FragmentOp);

  ~FragmentOp() override = default;

  FragmentOp();

  void dump(IRPrinter& p) override;

  static ptr_t build(IRContext* ctx, FragmentType type, const SymbolAttr::ptr_t& symbol_attr);

  static inline bool classof(const Node* node) { RTTI_RK_OP_PROGRAMIROP_FRAGMENTOP_IMPL(node); }

 private:
  FragmentType type_ = FragmentType::kCode;
};

class JumpOp final : public ProgramIROp {
 public:
  DEFINE_SPECIFIC_IR_CLASS(JumpOp);

  ~JumpOp() override = default;

  JumpOp();

  void dump(IRPrinter& p) override;

  static ptr_t build(IRContext* ctx, const std::string& label_name);

  static inline bool classof(const Node* node) { RTTI_RK_OP_PROGRAMIROP_JUMPOP_IMPL(node); }

 private:
  std::string label_name_;
};

class LabelOp final : public ProgramIROp, public SymbolInterface<LabelOp> {
 public:
  DEFINE_SPECIFIC_IR_CLASS(LabelOp);

  ~LabelOp() override = default;

  LabelOp();

  void dump(IRPrinter& p) override;

  static ptr_t build(IRContext* ctx, const SymbolAttr::ptr_t& symbol_attr);

  static inline bool classof(const Node* node) { RTTI_RK_OP_PROGRAMIROP_LABELOP_IMPL(node); }
};

class EntryPointOp final : public ProgramIROp, public SymbolInterface<EntryPointOp> {
 public:
  DEFINE_SPECIFIC_IR_CLASS(EntryPointOp);

  ~EntryPointOp() override = default;

  EntryPointOp();

  void dump(IRPrinter& p) override;

  static ptr_t build(IRContext* ctx, const SymbolAttr::ptr_t& symbol_attr, const std::vector<val_weak_ptr_t>& inputs,
                     const std::vector<val_weak_ptr_t>& outputs);

  static inline bool classof(const Node* node) { RTTI_RK_OP_PROGRAMIROP_ENTRYPOINTOP_IMPL(node); }
};

}  // namespace mllm::ir::program
