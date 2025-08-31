// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/compile/jit/interpreter/IRInterpreter.hpp"

namespace mllm::jit::interpreter {

void IRInterpreter::loadParam(const ParameterFile::ptr_t& parameter_file) {
  // TODO
}

void IRInterpreter::loadAndLinkPrograms(const std::string& source_code, const nlohmann::json& program) {
  // TODO
}

std::vector<Tensor> IRInterpreter::run(const std::vector<Tensor>& inputs) {
  // TODO
  return {};
}

}  // namespace mllm::jit::interpreter
