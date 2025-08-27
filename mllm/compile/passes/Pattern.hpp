// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/compile/ir/Node.hpp"

namespace mllm::ir {

class Pattern {
 public:
  using ptr_t = std::shared_ptr<Pattern>;

  Pattern() = default;

  ~Pattern() = default;

  virtual bool isMatch(const op_ptr_t& node) = 0;

  virtual bool rewrite(IRWriter& writer, const op_ptr_t& node) = 0;

  void setIRContext(IRContext* ctx);

 private:
  IRContext* ir_ctx_;
};

}  // namespace mllm::ir
