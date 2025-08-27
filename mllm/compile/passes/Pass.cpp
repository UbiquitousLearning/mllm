// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/compile/passes/Pass.hpp"

namespace mllm::ir {

uint8_t Pass::run(const node_ptr_t& op) { return PASS_RET_SUCCESS; }

void Pass::setCtx(const std::shared_ptr<IRContext>& ctx) { ctx_ = ctx; }

std::shared_ptr<IRContext> Pass::getCtx() { return ctx_; }

uint8_t PatternMatchPass::run(const node_ptr_t& op) { return PASS_RET_SUCCESS; }

void PatternMatchPass::setCtx(const std::shared_ptr<IRContext>& ctx) {
  for (auto& pattern : patterns_) { pattern->setIRContext(ctx.get()); }
  Pass::setCtx(ctx);
}

}  // namespace mllm::ir
