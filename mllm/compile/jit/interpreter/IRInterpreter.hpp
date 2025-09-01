// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <stack>
#include <string>
#include <vector>
#include <variant>
#include <unordered_map>
#include <nlohmann/json.hpp>

#include "mllm/compile/ir/program/Op.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/DeviceTypes.hpp"
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
  std::vector<uint32_t> inputs_uuid;
  std::vector<uint32_t> outputs_uuid;
  DeviceTypes device_type;
};

struct ProgramIRInstructionKernelSymbolPayload {
  BaseOp::ptr_t mllm_op = nullptr;
  std::string name;
};

struct ProgramIRInstructionJumpPayload {
  int32_t offset;
};

struct ProgramIRInstructionFreePayload {
  std::vector<uint32_t> inputs_uuid;
};

struct ProgramIRInstructionExitPayload {
  std::vector<uint32_t> inputs_uuid;
};

struct ProgramIRInstructionLabelPayload {
  std::string label;
};

struct ProgramIRInstructionBindPayload {
  int32_t pos_id;
  int32_t in_program_uuid;
  ir::program::BindOp::BindType bind_type;
};

struct ProgramIRInstruction {
  ir::NodeKind program_kind;
  std::variant<std::nullopt_t, ProgramIRInstructionModeConfigPayload, ProgramIRInstructionKernelLaunchPayload,
               ProgramIRInstructionKernelSymbolPayload, ProgramIRInstructionJumpPayload, ProgramIRInstructionFreePayload,
               ProgramIRInstructionExitPayload, ProgramIRInstructionLabelPayload, ProgramIRInstructionBindPayload>
      payload;

  ProgramIRInstruction() : program_kind(ir::NodeKind::RK_Op_ProgramIROp_ExitOp), payload(std::nullopt) {}

  ProgramIRInstruction(
      ir::NodeKind kind,
      const std::variant<std::nullopt_t, ProgramIRInstructionModeConfigPayload, ProgramIRInstructionKernelLaunchPayload,
                         ProgramIRInstructionKernelSymbolPayload, ProgramIRInstructionJumpPayload,
                         ProgramIRInstructionFreePayload, ProgramIRInstructionExitPayload, ProgramIRInstructionLabelPayload,
                         ProgramIRInstructionBindPayload>& p)
      : program_kind(kind), payload(p) {}
};

struct IRInterpreterSymbolTable {
  std::unordered_map<program_id_t, ProgramIRInstruction> program_id_2_op_table;
  std::unordered_map<std::string, program_id_t> symbol_name_2_program_id_table;
};

// IRInterpreter Only handle program IR
class IRInterpreter {
  enum class InterpreterMode {
    kEager,
    kStatic,
  };

 public:
  IRInterpreter() = default;

  ~IRInterpreter() = default;

  void loadParam(const ParameterFile::ptr_t& parameter_file);

  void loadAndLinkPrograms(const std::string& source_code, const nlohmann::json& program);

  std::vector<Tensor> run(const std::vector<Tensor>& inputs);

 private:
  int32_t eager();

  int32_t static_();

  InterpreterMode mode_ = InterpreterMode::kStatic;

  std::string source_code_;
  std::vector<Tensor> global_inputs_;
  std::vector<Tensor> global_outputs_;
  nlohmann::json program_info_;
  program_id_t program_counter_ = 0;
  IRInterpreterSymbolTable symbol_table_;
  std::stack<program_id_t> program_stack_;
  std::unordered_map<uint32_t, Tensor> uuid_2_tensor_;
  std::unordered_map<uint32_t, uint32_t> in_program_2_uuid_;
  std::unordered_map<uint32_t, uint32_t> uuid_2_in_program_;
  std::unordered_map<program_id_t, ProgramIRInstruction> program_;
};

}  // namespace mllm::jit::interpreter
