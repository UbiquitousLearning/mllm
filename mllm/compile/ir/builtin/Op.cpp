/**
 * @file Op.cpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-27
 *
 */
#include <memory>

#include "mllm/compile/ir/builtin/Op.hpp"
#include "mllm/compile/ir/GeneratedRTTIKind.hpp"

namespace mllm::ir {

BuiltinIROp::~BuiltinIROp() = default;

BuiltinIROp::BuiltinIROp() : Op(RK_Op_BuiltinIROp) {}

BuiltinIROp::BuiltinIROp(NodeKind kind) : Op(kind) {}

ModuleOp::~ModuleOp() = default;

ModuleOp::ModuleOp() : BuiltinIROp(RK_Op_BuiltinIROp_ModuleOp) {}

ModuleOp::ptr_t ModuleOp::build(IRContext* ctx, const std::shared_ptr<SymbolAttr>& symbol_attr) {
  auto ret = std::make_shared<ModuleOp>();

  ret->setSymbolAttr(symbol_attr);
  ret->createRegionAtTop();

  return ret;
}

void ModuleOp::dump(IRPrinter& p) {
  p.print("@{} ", getSymbolAttr()->str());
  getTopRegion()->dump(p);
  p.newline();
}

}  // namespace mllm::ir