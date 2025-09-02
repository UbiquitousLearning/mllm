// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include <nlohmann/json.hpp>
#include <optional>
#include <fmt/core.h>

#include "mllm/compile/jit/interpreter/IRInterpreter.hpp"
#include "mllm/compile/ir/GeneratedRTTIKind.hpp"
#include "mllm/compile/ir/program/Op.hpp"
#include "mllm/compile/jit/interpreter/AopsFromJson.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/engine/Context.hpp"
#include "mllm/utils/Enumerate.hpp"

namespace mllm::jit::interpreter {

void IRInterpreter::loadParam(const ParameterFile::ptr_t& parameter_file) {
  // Reset program counter
  program_counter_ = 0;

  // Load parameters
  do {
    switch (symbol_table_.program_id_2_op_table[program_counter_].program_kind) {
      case ir::NodeKind::RK_Op_ProgramIROp_KernelSymbolOp: {
        auto payload =
            std::get<ProgramIRInstructionKernelSymbolPayload>(symbol_table_.program_id_2_op_table[program_counter_].payload);
        payload.mllm_op->setName(payload.name);
        payload.mllm_op->load(parameter_file);
        program_counter_++;
        break;
      }
      case ir::NodeKind::RK_Op_ProgramIROp_ValueSymbolOp: {
        program_counter_++;
        break;
      }
      default: {
        MLLM_ERROR("Invalid instruction in ir interpreter. [pc={:#016x}]", program_counter_);
        program_counter_++;
      }
    }
  } while (program_counter_ < symbol_table_.program_id_2_op_table.size());
}

void IRInterpreter::loadAndLinkPrograms(const std::string& source_code, const nlohmann::json& program) {
  // 1. Load program head info.
  source_code_ = source_code;
  program_info_ = nlohmann::json();
  program_info_["compilation_time"] = program["compilation_time"];
  program_info_["little_endian"] = program["little_endian"];
  program_info_["platform"] = program["platform"];
  program_info_["cpu_architecture"] = program["cpu_architecture"];
  program_info_["cxx_compiler"] = program["cxx_compiler"];
  program_info_["cxx_compiler_version"] = program["cxx_compiler_version"];
  program_info_["cpp_standard"] = program["cpp_standard"];
  program_info_["op_counts"] = program["op_counts"];
  fmt::print("Program Info:\n");
  fmt::print("  compilation_time      : {}\n", program_info_["compilation_time"].dump());
  fmt::print("  little_endian         : {}\n", program_info_["little_endian"].dump());
  fmt::print("  platform              : {}\n", program_info_["platform"].dump());
  fmt::print("  cpu_architecture      : {}\n", program_info_["cpu_architecture"].dump());
  fmt::print("  cxx_compiler          : {}\n", program_info_["cxx_compiler"].dump());
  fmt::print("  cxx_compiler_version  : {}\n", program_info_["cxx_compiler_version"].dump());
  fmt::print("  cpp_standard          : {}\n", program_info_["cpp_standard"].dump());
  fmt::print("  op_counts             : {}\n", program_info_["op_counts"].dump());

  // Get ops in module
  nlohmann::json ops_in_module = program["code"]["module"];

  // 2. Load program symbol table
  {
    nlohmann::json symbol_table_segment;
    for (const auto& op : ops_in_module) {
      if (!op["op"].is_null() && op["op"] == "fragment" && op["symbol"] == "__MLLM_JIT_PACKAGE_SYMBOL_TABLE_SEGMENT") {
        symbol_table_segment = op;
        break;
      }
    }

    // Symbol instruction
    auto instructions = symbol_table_segment["instructions"];

    for (auto& instruction : instructions) {
      if (instruction.contains("dialect") && instruction["dialect"] == "program" && instruction["op"] == "kernel_symbol") {
        symbol_table_.program_id_2_op_table[instruction["program_id"]] = ProgramIRInstruction{
            ir::NodeKind::RK_Op_ProgramIROp_KernelSymbolOp,
            ProgramIRInstructionKernelSymbolPayload{.mllm_op = aopsFromJson(instruction), .name = instruction["symbol"]}};
        symbol_table_.symbol_name_2_program_id_table[instruction["symbol"]] = instruction["program_id"];
      } else if (instruction.contains("dialect") && instruction["dialect"] == "program"
                 && instruction["op"] == "value_symbol") {
        symbol_table_.program_id_2_op_table[instruction["program_id"]] =
            ProgramIRInstruction{ir::NodeKind::RK_Op_ProgramIROp_ValueSymbolOp, std::nullopt};

        // Register constant symbol.
        if (instruction.contains("constant")) {
          uuid_2_tensor_[instruction["outputs"][0]] = Tensor::fromVector<float>(
              instruction["constant"].get<std::vector<float>>(), instruction["shape"].get<Tensor::shape_t>());
        }
      }
    }
  }

  // 3. Load program and Link.
  {
    nlohmann::json code_segment;
    for (const auto& op : ops_in_module) {
      if (!op["op"].is_null() && op["op"] == "fragment" && op["symbol"] == "__MLLM_JIT_PACKAGE_CODE_SEGMENT") {
        code_segment = op;
        break;
      }
    }

    // code instruction
    auto instructions = code_segment["instructions"];
    for (auto& instruction : instructions) {
      // program.mode_config
      if (instruction.contains("dialect") && instruction["dialect"] == "program" && instruction["op"] == "mode_config") {
        program_[instruction["program_id"]] = ProgramIRInstruction{
            ir::NodeKind::RK_Op_ProgramIROp_ModeConfigOp,
            ProgramIRInstructionModeConfigPayload{.flag = static_cast<ir::program::ModeConfigFlag>(instruction["flag"])}};
      }
      // program.ret
      else if (instruction.contains("dialect") && instruction["dialect"] == "program" && instruction["op"] == "ret") {
        program_[instruction["program_id"]] = ProgramIRInstruction{
            ir::NodeKind::RK_Op_ProgramIROp_RetOp,
            std::nullopt,
        };
      }
      // program.kernel_launch
      else if (instruction.contains("dialect") && instruction["dialect"] == "program" && instruction["op"] == "kernel_launch") {
        BaseOp::ptr_t mllm_op = nullptr;
        mllm_op = instruction.contains("symbol_name")  // Link to Symbol Table if possible
                      ? std::get<ProgramIRInstructionKernelSymbolPayload>(
                            symbol_table_
                                .program_id_2_op_table[symbol_table_.symbol_name_2_program_id_table[instruction["symbol_name"]]]
                                .payload)
                            .mllm_op
                      : aopsFromJson(instruction);

        program_[instruction["program_id"]] = ProgramIRInstruction{
            ir::NodeKind::RK_Op_ProgramIROp_KernelLaunchOp,
            ProgramIRInstructionKernelLaunchPayload{
                .mllm_op = mllm_op,
            },
        };

        // Process inputs and outputs uuid
        ProgramIRInstructionKernelLaunchPayload& kernel_payload =
            std::get<ProgramIRInstructionKernelLaunchPayload>(program_[instruction["program_id"]].payload);
        for (auto& i_uuid : instruction["inputs"]) { kernel_payload.inputs_uuid.emplace_back(i_uuid); }
        for (auto& o_uuid : instruction["outputs"]) { kernel_payload.outputs_uuid.emplace_back(o_uuid); }
        kernel_payload.device_type = str2DeviceType(instruction["device"]);
      }
      // program.free
      else if (instruction.contains("dialect") && instruction["dialect"] == "program" && instruction["op"] == "free") {
        ProgramIRInstructionFreePayload payload;

        if (instruction.contains("inputs")) {
          for (const auto& input_uuid : instruction["inputs"]) { payload.inputs_uuid.push_back(input_uuid.get<uint32_t>()); }
        }

        program_[instruction["program_id"]] = ProgramIRInstruction{
            ir::NodeKind::RK_Op_ProgramIROp_FreeOp,
            payload,
        };
      }
      // program.jump
      else if (instruction.contains("dialect") && instruction["dialect"] == "program" && instruction["op"] == "jump") {
        ProgramIRInstructionJumpPayload payload;

        if (instruction.contains("offset")) { payload.offset = instruction["offset"].get<int32_t>(); }

        program_[instruction["program_id"]] = ProgramIRInstruction{
            ir::NodeKind::RK_Op_ProgramIROp_JumpOp,
            payload,
        };
      }
      // program.exit
      else if (instruction.contains("dialect") && instruction["dialect"] == "program" && instruction["op"] == "exit") {
        ProgramIRInstructionExitPayload payload;
        program_[instruction["program_id"]] = ProgramIRInstruction{
            ir::NodeKind::RK_Op_ProgramIROp_ExitOp,
            payload,
        };
      }
      // program.label
      else if (instruction.contains("dialect") && instruction["dialect"] == "program" && instruction["op"] == "label") {
        ProgramIRInstructionLabelPayload payload;

        if (instruction.contains("label")) { payload.label = instruction["label"].get<std::string>(); }

        program_[instruction["program_id"]] = ProgramIRInstruction{
            ir::NodeKind::RK_Op_ProgramIROp_LabelOp,
            payload,
        };
      }
      // program.bind
      else if (instruction.contains("dialect") && instruction["dialect"] == "program" && instruction["op"] == "bind") {
        ProgramIRInstructionBindPayload payload;
        if (instruction.contains("input_pos")) { payload.pos_id = instruction["input_pos"].get<int32_t>(); }
        if (instruction.contains("program_uuid")) { payload.in_program_uuid = instruction["program_uuid"].get<int32_t>(); }
        if (instruction.contains("type")) {
          std::string type_str = instruction["type"];
          payload.bind_type =
              (type_str == "input") ? ir::program::BindOp::BindType::kInput : ir::program::BindOp::BindType::kOutput;
        }
        program_[instruction["program_id"]] = ProgramIRInstruction{
            ir::NodeKind::RK_Op_ProgramIROp_BindOp,
            payload,
        };
      }
    }
  }
}

std::vector<Tensor> IRInterpreter::run(const std::vector<Tensor>& inputs) {
  // Set inputs
  global_inputs_ = inputs;

  // Reset program counter
  program_counter_ = 0;

  // Reset interpreter mode to static
  mode_ = InterpreterMode::kStatic;

  // Loop
  auto prog_size = program_.size();

  do {
    int32_t ret_val = 0;
    switch (mode_) {
      case InterpreterMode::kStatic: {
        ret_val = static_();
        break;
      }
      case InterpreterMode::kEager: {
        ret_val = eager();
        break;
      }
    }
    if (ret_val) { break; }
  } while (program_counter_ < prog_size);

  return global_outputs_;
}

int32_t IRInterpreter::eager() {
  auto& instruction = program_[program_counter_];
  switch (instruction.program_kind) {
    case ir::NodeKind::RK_Op_ProgramIROp_BindOp: {
      auto payload = std::get<ProgramIRInstructionBindPayload>(instruction.payload);
      switch (payload.bind_type) {
        case ir::program::BindOp::BindType::kInput: {
          // Register the bind op
          in_program_2_uuid_[payload.in_program_uuid] = global_inputs_[payload.pos_id].uuid();
          uuid_2_in_program_[global_inputs_[payload.pos_id].uuid()] = payload.in_program_uuid;
          uuid_2_tensor_[payload.in_program_uuid] = global_inputs_[payload.pos_id];
          break;
        }
        case ir::program::BindOp::BindType::kOutput: {
          if (global_outputs_.size() <= payload.pos_id) { global_outputs_.resize(payload.pos_id + 1); }
          global_outputs_[payload.pos_id] = uuid_2_tensor_[payload.in_program_uuid];
          break;
        }
      }
      program_counter_++;
      break;
    }
    case ir::NodeKind::RK_Op_ProgramIROp_KernelLaunchOp: {
      auto payload = std::get<ProgramIRInstructionKernelLaunchPayload>(instruction.payload);

      // Get inputs
      std::vector<Tensor> inputs;
      inputs.reserve(payload.inputs_uuid.size());
      for (auto uuid : payload.inputs_uuid) { inputs.push_back(uuid_2_tensor_[uuid]); }

      // Build task
      auto task = Task::createExecuteOpTask(payload.mllm_op, inputs, {});

      // Send task to backend
      Context::instance().dispatcherManager()->submit(static_cast<int32_t>(payload.device_type), task);

      // Get output and register it
      for (auto [_id, out] : enumerate(task->outputs)) {
        uuid_2_tensor_[payload.outputs_uuid[_id]] = out;
        in_program_2_uuid_[payload.outputs_uuid[_id]] = out.uuid();
        uuid_2_in_program_[out.uuid()] = payload.outputs_uuid[_id];
      }

      program_counter_++;
      break;
    }
    case ir::NodeKind::RK_Op_ProgramIROp_KernelSymbolOp: {
      MLLM_ERROR("Invalid instruction in ir interpreter.");
      program_counter_++;
      break;
    }
    case ir::NodeKind::RK_Op_ProgramIROp_JumpOp: {
      auto payload = std::get<ProgramIRInstructionJumpPayload>(instruction.payload);
      program_stack_.push(program_counter_);
      program_counter_ += payload.offset;
      break;
    }
    case ir::NodeKind::RK_Op_ProgramIROp_FreeOp: {
      auto payload = std::get<ProgramIRInstructionFreePayload>(instruction.payload);
      if (!payload.inputs_uuid.empty()) {
        for (const auto& uuid : payload.inputs_uuid) {
          auto it = uuid_2_tensor_.find(uuid);
          if (it != uuid_2_tensor_.end()) { uuid_2_tensor_.erase(it); }
        }
      }
      program_counter_++;
      break;
    }
    case ir::NodeKind::RK_Op_ProgramIROp_ExitOp: {
      return 1;
    }
    case ir::NodeKind::RK_Op_ProgramIROp_LabelOp: {
      program_counter_++;
      break;
    }
    case ir::NodeKind::RK_Op_ProgramIROp_RetOp: {
      program_counter_ = program_stack_.top();
      program_stack_.pop();
      program_counter_++;
      break;
    }
    case ir::NodeKind::RK_Op_ProgramIROp_ModeConfigOp: {
      auto payload = std::get<ProgramIRInstructionModeConfigPayload>(instruction.payload);
      if (payload.flag == ir::program::ModeConfigFlag::kEager) {
        mode_ = InterpreterMode::kEager;
      } else if (payload.flag == ir::program::ModeConfigFlag::kStaticPlanned) {
        mode_ = InterpreterMode::kStatic;
      }
      program_counter_++;
      break;
    }
    default: {
      MLLM_ERROR("Invalid instruction in ir interpreter.");
      program_counter_++;
      break;
    }
  }
  return 0;
}

int32_t IRInterpreter::static_() {
  // TODO
  // We need impl the static mode. Currently, we just bypass the static mode as an eager mode.
  auto& instruction = program_[program_counter_];
  switch (instruction.program_kind) {
    case ir::NodeKind::RK_Op_ProgramIROp_BindOp: {
      auto payload = std::get<ProgramIRInstructionBindPayload>(instruction.payload);
      switch (payload.bind_type) {
        case ir::program::BindOp::BindType::kInput: {
          // Register the bind op
          in_program_2_uuid_[payload.in_program_uuid] = global_inputs_[payload.pos_id].uuid();
          uuid_2_in_program_[global_inputs_[payload.pos_id].uuid()] = payload.in_program_uuid;
          uuid_2_tensor_[payload.in_program_uuid] = global_inputs_[payload.pos_id];
          break;
        }
        case ir::program::BindOp::BindType::kOutput: {
          if (global_outputs_.size() <= payload.pos_id) { global_outputs_.resize(payload.pos_id + 1); }
          global_outputs_[payload.pos_id] = uuid_2_tensor_[payload.in_program_uuid];
          break;
        }
      }
      program_counter_++;
      break;
    }
    case ir::NodeKind::RK_Op_ProgramIROp_KernelLaunchOp: {
      auto payload = std::get<ProgramIRInstructionKernelLaunchPayload>(instruction.payload);

      // Get inputs
      std::vector<Tensor> inputs;
      inputs.reserve(payload.inputs_uuid.size());
      for (auto uuid : payload.inputs_uuid) { inputs.push_back(uuid_2_tensor_[uuid]); }

      // Build task
      auto task = Task::createExecuteOpTask(payload.mllm_op, inputs, {});

      // Send task to backend
      Context::instance().dispatcherManager()->submit(static_cast<int32_t>(payload.device_type), task);

      // Get output and register it
      for (auto [_id, out] : enumerate(task->outputs)) {
        uuid_2_tensor_[payload.outputs_uuid[_id]] = out;
        in_program_2_uuid_[payload.outputs_uuid[_id]] = out.uuid();
        uuid_2_in_program_[out.uuid()] = payload.outputs_uuid[_id];
      }

      program_counter_++;
      break;
    }
    case ir::NodeKind::RK_Op_ProgramIROp_KernelSymbolOp: {
      MLLM_ERROR("Invalid instruction in ir interpreter.");
      program_counter_++;
      break;
    }
    case ir::NodeKind::RK_Op_ProgramIROp_JumpOp: {
      auto payload = std::get<ProgramIRInstructionJumpPayload>(instruction.payload);
      program_stack_.push(program_counter_);
      program_counter_ += payload.offset;
      break;
    }
    case ir::NodeKind::RK_Op_ProgramIROp_FreeOp: {
      auto payload = std::get<ProgramIRInstructionFreePayload>(instruction.payload);
      if (!payload.inputs_uuid.empty()) {
        for (const auto& uuid : payload.inputs_uuid) {
          auto it = uuid_2_tensor_.find(uuid);
          if (it != uuid_2_tensor_.end()) { uuid_2_tensor_.erase(it); }
        }
      }
      program_counter_++;
      break;
    }
    case ir::NodeKind::RK_Op_ProgramIROp_ExitOp: {
      return 1;
    }
    case ir::NodeKind::RK_Op_ProgramIROp_LabelOp: {
      program_counter_++;
      break;
    }
    case ir::NodeKind::RK_Op_ProgramIROp_RetOp: {
      program_counter_ = program_stack_.top();
      program_stack_.pop();
      program_counter_++;
      break;
    }
    case ir::NodeKind::RK_Op_ProgramIROp_ModeConfigOp: {
      auto payload = std::get<ProgramIRInstructionModeConfigPayload>(instruction.payload);
      if (payload.flag == ir::program::ModeConfigFlag::kEager) {
        mode_ = InterpreterMode::kEager;
      } else if (payload.flag == ir::program::ModeConfigFlag::kStaticPlanned) {
        mode_ = InterpreterMode::kStatic;
      }
      program_counter_++;
      break;
    }
    default: {
      MLLM_ERROR("Invalid instruction in ir interpreter.");
      program_counter_++;
      break;
    }
  }
  return 0;
}

}  // namespace mllm::jit::interpreter
