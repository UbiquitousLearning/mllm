// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/compile/ir/Node.hpp"
#include "mllm/compile/ir/builtin/Interface.hpp"
#include "mllm/compile/ir/tensor/Value.hpp"

namespace mllm::ir::tensor {

class TensorIROp : public Op {
 public:
  DEFINE_SPECIFIC_IR_CLASS(TensorIROp);

  ~TensorIROp() override;

  TensorIROp();

  explicit TensorIROp(NodeKind kind);

  static inline bool classof(const Node* node) { RTTI_RK_OP_TENSORIROP_IMPL(node); }
};

class RegisterOp : public TensorIROp, public SymbolInterface<RegisterOp> {
 public:
  DEFINE_SPECIFIC_IR_CLASS(RegisterOp);

  ~RegisterOp() override;

  RegisterOp();

  void dump(IRPrinter& p) final;

  static ptr_t build(IRContext* ctx, const TensorValue::ptr_t& tensor_v);

  TensorValue::ptr_t getRegisteredTensor();

  static inline bool classof(const Node* node) { RTTI_RK_OP_TENSORIROP_REGISTEROP_IMPL(node); }
};

class AllocOp : public TensorIROp {
 public:
  DEFINE_SPECIFIC_IR_CLASS(AllocOp);

  ~AllocOp() override;

  AllocOp();

  void dump(IRPrinter& p) final;

  static ptr_t build(IRContext* ctx, const TensorValue::ptr_t& tensor_v);

  TensorValue::ptr_t getAllocatedTensor();

  static inline bool classof(const Node* node) { RTTI_RK_OP_TENSORIROP_ALLOCOP_IMPL(node); }
};

class FreeOp : public TensorIROp {
 public:
  DEFINE_SPECIFIC_IR_CLASS(FreeOp);

  ~FreeOp() override;

  FreeOp();

  void dump(IRPrinter& p) final;

  static ptr_t build(IRContext* ctx, const TensorValue::ptr_t& tensor_v);

  TensorValue::ptr_t getFreedTensor();

  static inline bool classof(const Node* node) { RTTI_RK_OP_TENSORIROP_FREEOP_IMPL(node); }
};

}  // namespace mllm::ir::tensor
