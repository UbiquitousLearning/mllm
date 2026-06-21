// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/compile/ir/builtin/Interface.hpp"
#include "mllm/compile/ir/Node.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/nn/AbstractNnNode.hpp"

namespace mllm::ir::graph {

class GraphIROp : public Op {
 public:
  DEFINE_SPECIFIC_IR_CLASS(GraphIROp);

  ~GraphIROp() override;

  GraphIROp();

  explicit GraphIROp(const NodeKind& kind);

  static inline bool classof(const Node* node) { RTTI_RK_OP_GRAPHIROP_IMPL(node); }
};

class SubGraphOp : public GraphIROp, public SymbolInterface<SubGraphOp> {
 public:
  DEFINE_SPECIFIC_IR_CLASS(SubGraphOp);

  ~SubGraphOp() override;

  SubGraphOp();

  static ptr_t build(IRContext* ctx, const SymbolAttr::ptr_t& symbol_attr,
                     const ::mllm::nn::AbstractNnNode::ptr_t& hierarchy_base);

  static ptr_t build(IRContext* ctx, const SymbolAttr::ptr_t& symbol_attr);

  void dump(IRPrinter& p) override;

  ::mllm::nn::AbstractNnNode::ptr_t abstract_nn_node_ = nullptr;

  static inline bool classof(const Node* node) { RTTI_RK_OP_GRAPHIROP_SUBGRAPHOP_IMPL(node); }
};

class CallGraphOp : public GraphIROp, public SymbolInterface<CallGraphOp> {
 public:
  DEFINE_SPECIFIC_IR_CLASS(CallGraphOp);

  ~CallGraphOp() override;

  CallGraphOp();

  void dump(IRPrinter& p) override;

  static ptr_t build(IRContext* ctx, const SymbolAttr::ptr_t& symbol_attr);

  static inline bool classof(const Node* node) { RTTI_RK_OP_GRAPHIROP_CALLGRAPHOP_IMPL(node); }
};
}  // namespace mllm::ir::graph