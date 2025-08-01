// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "mllm/compile/ir/builtin/Interface.hpp"
#include "mllm/compile/ir/Node.hpp"
#include "mllm/compile/ir/GeneratedRTTIKind.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/compile/ir/NodeRTTIClassOfImpl.hpp"
#include "mllm/compile/ir/IRPrinter.hpp"

namespace mllm::ir {

class BuiltinIROp : public Op {
 public:
  DEFINE_SPECIFIC_IR_CLASS(BuiltinIROp);

  ~BuiltinIROp() override;
  BuiltinIROp();
  explicit BuiltinIROp(NodeKind kind);

  static inline bool classof(const Node* node) { RTTI_RK_OP_BUILTINIROP_IMPL(node); }
};

class ModuleOp : public BuiltinIROp, public SymbolInterface<ModuleOp> {
 public:
  DEFINE_SPECIFIC_IR_CLASS(ModuleOp);

  ~ModuleOp() override;
  ModuleOp();

  void dump(IRPrinter& p) final;

  static ptr_t build(IRContext* ctx, const std::shared_ptr<SymbolAttr>& symbol_attr);

  static inline bool classof(const Node* node) { RTTI_RK_OP_BUILTINIROP_MODULEOP_IMPL(node); }
};

}  // namespace mllm::ir