// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <string>
#include <vector>
#include <variant>
#include <unordered_map>
#include <nlohmann/json_fwd.hpp>

#include "mllm/compile/ir/program/Op.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/compile/ir/GeneratedRTTIKind.hpp"

namespace mllm::jit::interpreter {

using program_id_t = uint64_t;

struct ProgramIRInstructionModeConfigPayload {
  ir::program::ModeConfigFlag flag;
};

struct ProgramIRInstructionKernelLaunchPayload {
  BaseOp::ptr_t mllm_op = nullptr;
};

struct ProgramIRInstructionJumpPayload {
  int32_t offset;
};

struct ProgramIRInstruction {
  ir::NodeKind program_kind;
  std::variant<ProgramIRInstructionModeConfigPayload, ProgramIRInstructionKernelLaunchPayload, ProgramIRInstructionJumpPayload>
      payload;
};

struct IRInterpreterSymbolTable {
  std::unordered_map<program_id_t, ProgramIRInstruction> program_id_2_op_table;
  std::unordered_map<std::string, program_id_t> symbol_name_2_program_id_table;
};

class IRInterpreterProgram {};

// IRInterpreter Only handle program IR
class IRInterpreter {
 public:
  IRInterpreter() = default;

  ~IRInterpreter() = default;

  void loadParam(const ParameterFile::ptr_t& parameter_file);

  void loadAndLinkPrograms(const std::string& source_code, const nlohmann::json& program);

  std::vector<Tensor> run(const std::vector<Tensor>& inputs);

 private:
  program_id_t program_counter = 0;
  std::string source_code_;
};

}  // namespace mllm::jit::interpreter
