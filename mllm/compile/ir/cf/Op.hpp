// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <vector>

#include "mllm/compile/ir/GeneratedRTTIKind.hpp"
#include "mllm/compile/ir/Node.hpp"

namespace mllm::ir::cf {

class ControlFlowIROp : public Op {
 public:
  DEFINE_SPECIFIC_IR_CLASS(ControlFlowIROp);

  ControlFlowIROp();

  explicit ControlFlowIROp(NodeKind kind);

  ~ControlFlowIROp() override;

  static inline bool classof(const Node* node) { RTTI_RK_OP_CONTROLFLOWIROP_IMPL(node); }
};

class ReturnOp : public ControlFlowIROp {
 public:
  DEFINE_SPECIFIC_IR_CLASS(ReturnOp);

  ReturnOp();

  ~ReturnOp() override;

  static ptr_t build(IRContext* ctx, const val_ptr_t& val);

  static ptr_t build(IRContext* ctx, const std::vector<val_ptr_t>& vals);

  void dump(IRPrinter& p) override;

  static inline bool classof(const Node* node) { RTTI_RK_OP_CONTROLFLOWIROP_RETURNOP_IMPL(node); }
};

}  // namespace mllm::ir::cf