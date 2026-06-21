// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <vector>

#include "mllm/compile/ir/Node.hpp"
#include "mllm/compile/ir/builtin/Interface.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/compile/ir/tensor/Value.hpp"

namespace mllm::ir::program {

class ProgramIROp : public Op {
 public:
  DEFINE_SPECIFIC_IR_CLASS(ProgramIROp);

  ~ProgramIROp() override = default;

  ProgramIROp();

  explicit ProgramIROp(const NodeKind& kind);

  static inline bool classof(const Node* node) { RTTI_RK_OP_PROGRAMIROP_IMPL(node); }

  void setProgramIntrinsicId(uint64_t program_intrinsic_id) { program_intrinsic_id_ = program_intrinsic_id; }

  uint64_t getProgramIntrinsicId() const { return program_intrinsic_id_; }

 private:
  uint64_t program_intrinsic_id_ = -1;
};

enum class FragmentType : int32_t {
  kCode,
  kData,
  kTable,
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

class KernelLaunchOp final : public ProgramIROp {
 public:
  DEFINE_SPECIFIC_IR_CLASS(KernelLaunchOp);

  ~KernelLaunchOp() override = default;

  KernelLaunchOp();

  void dump(IRPrinter& p) override;

  static ptr_t build(IRContext* ctx, const std::vector<tensor::TensorValue::ptr_t>& inputs,
                     const std::vector<tensor::TensorValue::ptr_t>& outputs, const std::string& op_type,
                     const std::string& op_options);

  static inline bool classof(const Node* node) { RTTI_RK_OP_PROGRAMIROP_KERNELLAUNCHOP_IMPL(node); }
};

class KernelSymbolOp final : public ProgramIROp, public SymbolInterface<KernelSymbolOp> {
 public:
  DEFINE_SPECIFIC_IR_CLASS(KernelSymbolOp);

  ~KernelSymbolOp() override = default;

  KernelSymbolOp();

  void dump(IRPrinter& p) override;

  static ptr_t build(IRContext* ctx, const SymbolAttr::ptr_t& symbol_attr, const std::string& op_type,
                     const std::string& op_options);

  static inline bool classof(const Node* node) { RTTI_RK_OP_PROGRAMIROP_KERNELSYMBOLOP_IMPL(node); }
};

class ValueSymbolOp final : public ProgramIROp, public SymbolInterface<ValueSymbolOp> {
 public:
  DEFINE_SPECIFIC_IR_CLASS(ValueSymbolOp);

  ~ValueSymbolOp() override = default;

  ValueSymbolOp();

  void dump(IRPrinter& p) override;

  static ptr_t build(IRContext* ctx, const val_ptr_t& value_ir, const SymbolAttr::ptr_t& symbol_attr);

  static inline bool classof(const Node* node) { RTTI_RK_OP_PROGRAMIROP_VALUESYMBOLOP_IMPL(node); }
};

class JumpOp final : public ProgramIROp {
 public:
  DEFINE_SPECIFIC_IR_CLASS(JumpOp);

  ~JumpOp() override = default;

  JumpOp();

  void dump(IRPrinter& p) override;

  const std::string& labelName() const;

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

class ExitOp final : public ProgramIROp {
 public:
  DEFINE_SPECIFIC_IR_CLASS(ExitOp);

  ~ExitOp() override = default;

  ExitOp();

  void dump(IRPrinter& p) override;

  static ptr_t build(IRContext* ctx);

  static inline bool classof(const Node* node) { RTTI_RK_OP_PROGRAMIROP_EXITOP_IMPL(node); }
};

class RetOp final : public ProgramIROp {
 public:
  DEFINE_SPECIFIC_IR_CLASS(RetOp);

  ~RetOp() override = default;

  RetOp();

  void dump(IRPrinter& p) override;

  static ptr_t build(IRContext* ctx);

  static inline bool classof(const Node* node) { RTTI_RK_OP_PROGRAMIROP_RETOP_IMPL(node); }
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

class AllocOp final : public ProgramIROp {
 public:
  DEFINE_SPECIFIC_IR_CLASS(AllocOp);

  ~AllocOp() override = default;

  AllocOp();

  void dump(IRPrinter& p) override;

  static ptr_t build(IRContext* ctx, const val_ptr_t& v_ir);

  static inline bool classof(const Node* node) { RTTI_RK_OP_PROGRAMIROP_ALLOCOP_IMPL(node); }
};

class FreeOp final : public ProgramIROp {
 public:
  DEFINE_SPECIFIC_IR_CLASS(FreeOp);

  ~FreeOp() override = default;

  FreeOp();

  void dump(IRPrinter& p) override;

  static ptr_t build(IRContext* ctx, const val_ptr_t& v_ir);

  static inline bool classof(const Node* node) { RTTI_RK_OP_PROGRAMIROP_FREEOP_IMPL(node); }
};

enum class ModeConfigFlag : uint32_t {
  kClear = 0x0,
  kEager = 0x1,
  kStaticPlanned = 0x2,
};

class ModeConfigOp final : public ProgramIROp {
 public:
  DEFINE_SPECIFIC_IR_CLASS(ModeConfigOp);

  ~ModeConfigOp() override = default;

  ModeConfigOp();

  void dump(IRPrinter& p) override;

  static ptr_t build(IRContext* ctx, ModeConfigFlag flag);

  static inline bool classof(const Node* node) { RTTI_RK_OP_PROGRAMIROP_MODECONFIGOP_IMPL(node); }
};

class BindOp final : public ProgramIROp {
 public:
  DEFINE_SPECIFIC_IR_CLASS(BindOp);

  enum BindType : int32_t {
    kInput = 0,
    kOutput = 1,
  };

  ~BindOp() override = default;

  BindOp();

  void dump(IRPrinter& p) override;

  static ptr_t build(IRContext* ctx, uint32_t input_pos, uint32_t program_uuid, BindType type);

  static inline bool classof(const Node* node) { RTTI_RK_OP_PROGRAMIROP_BINDOP_IMPL(node); }
};

}  // namespace mllm::ir::program
