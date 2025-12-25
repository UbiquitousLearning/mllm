// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/compile/passes/Pattern.hpp"
#include "mllm/compile/ir/Node.hpp"
#include "mllm/compile/ir/tensor/Value.hpp"
#include <vector>

namespace mllm::qnn::aot {

class QnnAOTBasePattern : public ir::Pattern {
 public:
  QnnAOTBasePattern() = default;

  bool isMatch(const mllm::ir::op_ptr_t& op) override { return false; }

  bool rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) override { return false; }

  virtual bool compile(ir::IRWriter& writer, const ir::op_ptr_t& op) = 0;
};

class QnnAOTQuantRecipeBasePattern : public ir::Pattern {
 public:
  QnnAOTQuantRecipeBasePattern() = default;

  bool isMatch(const mllm::ir::op_ptr_t& op) override { return false; }

  bool rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) override { return false; }
};

}  // namespace mllm::qnn::aot
