// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "mllm/compile/ir/Node.hpp"

namespace mllm::ir::dbg {

class DbgIROp : public Op {
 public:
  DEFINE_SPECIFIC_IR_CLASS(DbgIROp);

  ~DbgIROp() override;

  DbgIROp();

  explicit DbgIROp(const NodeKind& kind);

  static inline bool classof(const Node* node) { RTTI_RK_OP_DBGIROP_IMPL(node); }
};

class CommentOp : public DbgIROp {
 public:
  DEFINE_SPECIFIC_IR_CLASS(CommentOp);

  ~CommentOp() override;

  CommentOp();

  static ptr_t build(IRContext* ctx, const std::string& comment);

  void dump(IRPrinter& p) override;

  static inline bool classof(const Node* node) { RTTI_RK_OP_DBGIROP_COMMENTOP_IMPL(node); }

  std::string comments_;
};

}  // namespace mllm::ir::dbg
