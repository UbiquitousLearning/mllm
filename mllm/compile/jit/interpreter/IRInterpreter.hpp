// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <string>
#include <vector>
#include <nlohmann/json_fwd.hpp>

#include "mllm/core/ParameterFile.hpp"
#include "mllm/core/Tensor.hpp"

namespace mllm::jit::interpreter {

// IRInterpreter Only handle program IR
class IRInterpreter {
 public:
  IRInterpreter() = default;

  ~IRInterpreter() = default;

  void load(const ParameterFile::ptr_t& parameter_file);

  void loadAndLinkPrograms(const std::string& source_code, const nlohmann::json& program);

  std::vector<Tensor> run(const std::vector<Tensor>& inputs);

 private:
  uint64_t program_counter = 0;
  std::string source_code_;
};

}  // namespace mllm::jit::interpreter
