// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/compile/jit/binary/IRSerialization.hpp"
#include "mllm/compile/jit/JITUtils.hpp"

namespace mllm::jit::binary {

// Will loop all ops in this Module
nlohmann::json IRSerializer::visit(const ir::IRContext::ptr_t& module) {
  // Before serialize
  code_["compilation_time"] = 0;
  code_["little_endian"] = isLittleEndian();

  /// TODO
  // Start serialize
  //

  // After serialize
  code_["op_counts"] = 0;

  return code_;
}

nlohmann::json IRSerializer::visitLinalgOp() {}

nlohmann::json IRSerializer::visitCFOp() {}

nlohmann::json IRSerializer::visitTensorOp() {}

nlohmann::json IRSerializer::visitGraphOp() {}

nlohmann::json IRSerializer::visitDbgOp() {}

nlohmann::json IRSerializer::visitBuiltinOp() {}

}  // namespace mllm::jit::binary
