// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <nlohmann/json.hpp>
#include "mllm/compile/ir/Node.hpp"

namespace mllm::jit::binary {

class IRSerializer {
 public:
  IRSerializer() = default;

  // Will loop all ops in this Module
  nlohmann::json visit(const ir::IRContext::ptr_t& module);

  nlohmann::json visitLinalgOp();

  nlohmann::json visitCFOp();

  nlohmann::json visitTensorOp();

  nlohmann::json visitGraphOp();

  nlohmann::json visitDbgOp();

  nlohmann::json visitBuiltinOp();

 private:
  nlohmann::json code_;
};

}  // namespace mllm::jit::binary
