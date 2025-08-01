// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/compile/ir/graph/Op.hpp"
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/compile/ir/GeneratedRTTIKind.hpp"

namespace mllm::ir::graph {

GraphIROp::~GraphIROp() = default;

GraphIROp::GraphIROp() : Op(RK_Op_GraphIROp) {}

GraphIROp::GraphIROp(const NodeKind& kind) : Op(kind) {}

SubGraphOp::~SubGraphOp() = default;

SubGraphOp::SubGraphOp() : GraphIROp(RK_Op_GraphIROp_SubGraphOp) {}

SubGraphOp::ptr_t SubGraphOp::build(IRContext* ctx, const SymbolAttr::ptr_t& symbol_attr,
                                    const ::mllm::nn::AbstractNnNode::ptr_t& abstract_nn_node) {
  auto ret = std::make_shared<SubGraphOp>();
  ret->setSymbolAttr(symbol_attr);
  ret->createRegionAtTop();
  ret->abstract_nn_node_ = abstract_nn_node;

  ctx->addToSymbolTable(ret, symbol_attr->str());

  return ret;
}

SubGraphOp::ptr_t SubGraphOp::build(IRContext* ctx, const SymbolAttr::ptr_t& symbol_attr) {
  auto ret = std::make_shared<SubGraphOp>();
  ret->setSymbolAttr(symbol_attr);
  ret->createRegionAtTop();
  ret->abstract_nn_node_ = nullptr;

  ctx->addToSymbolTable(ret, symbol_attr->str());
  return ret;
}

void SubGraphOp::dump(IRPrinter& p) {
  p.print("graph.SubGraphOp @{} ", getSymbolAttr()->str());
  p.langle();
  p.print("{}", abstract_nn_node_ == nullptr ? "notype" : deviceTypes2Str(abstract_nn_node_->getDevice()));
  p.rangle();
  p.blank();
  p.lbrace();

  getTopRegion()->dump(p);

  p.rbrace();
}

CallGraphOp::~CallGraphOp() = default;

CallGraphOp::CallGraphOp() : GraphIROp(RK_Op_GraphIROp_CallGraphOp) {}

void CallGraphOp::dump(IRPrinter& p) {
  p.print("graph.CallGraphOp @{} ", getSymbolAttr()->str());
  Op::dump(p);
}

CallGraphOp::ptr_t CallGraphOp::build(IRContext* ctx, const SymbolAttr::ptr_t& symbol_attr) {
  auto ret = std::make_shared<CallGraphOp>();
  ret->setSymbolAttr(symbol_attr);
  return ret;
}

}  // namespace mllm::ir::graph