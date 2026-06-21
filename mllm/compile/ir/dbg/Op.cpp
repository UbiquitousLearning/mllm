// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/compile/ir/dbg/Op.hpp"
#include "mllm/compile/ir/GeneratedRTTIKind.hpp"

namespace mllm::ir::dbg {
DbgIROp::~DbgIROp() = default;

DbgIROp::DbgIROp() : Op(RK_Op_DbgIROp) {}

DbgIROp::DbgIROp(const NodeKind& kind) : Op(kind) {}

CommentOp::~CommentOp() = default;

CommentOp::CommentOp() : DbgIROp(RK_Op_DbgIROp_CommentOp) {}

CommentOp::ptr_t CommentOp::build(IRContext* ctx, const std::string& comment) {
  auto ret = std::make_shared<CommentOp>();
  ret->comments_ = comment;
  return ret;
}

void CommentOp::dump(IRPrinter& p) { p.print("{}", comments_); }
}  // namespace mllm::ir::dbg
