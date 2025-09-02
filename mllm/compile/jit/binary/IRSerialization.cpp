// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <chrono>
#include <iomanip>
#include <sstream>
#include <fstream>

#include "mllm/compile/jit/binary/IRSerialization.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/compile/ir/builtin/Op.hpp"
#include "mllm/compile/ir/program/Op.hpp"
#include "mllm/compile/ir/tensor/Value.hpp"
#include "mllm/compile/jit/JITUtils.hpp"
#include "mllm/utils/CPUArchHelper.hpp"
#include "mllm/utils/PlatformRTHelper.hpp"

namespace mllm::jit::binary {

// Will loop all ops in this Module
nlohmann::json IRSerializer::visit(const ir::IRContext::ptr_t& ctx) {
  // Before serialize
  // 1. Timestamp
  {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    code_["compilation_time"] = ss.str();
  }

  // Little endian or big endian
  code_["little_endian"] = isLittleEndian();

  // Platform
  code_["platform"] = MLLM_CURRENT_PLATFORM_STRING;

  // CPU Architecture
  code_["cpu_architecture"] = mllm::cpu::CURRENT_ARCH_STRING;

  // Compiler information
#if defined(__clang__)
  code_["cxx_compiler"] = "Clang";
  code_["cxx_compiler_version"] =
      std::to_string(__clang_major__) + "." + std::to_string(__clang_minor__) + "." + std::to_string(__clang_patchlevel__);
#elif defined(__GNUC__)
  code_["cxx_compiler"] = "GCC";
  code_["cxx_compiler_version"] =
      std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__) + "." + std::to_string(__GNUC_PATCHLEVEL__);
#elif defined(_MSC_VER)
  code_["cxx_compiler"] = "MSVC";
  code_["cxx_compiler_version"] = std::to_string(_MSC_VER);
#else
  code_["cxx_compiler"] = "unknown";
  code_["cxx_compiler_version"] = "unknown";
#endif

  code_["cpp_standard"] = std::to_string(__cplusplus);

  // Start serialize
  code_["code"] = visitBuiltinOp(ctx, ctx->topLevelOp()->cast_<ir::BuiltinIROp>());

  // After serialize
  // Count ops in __MLLM_JIT_PACKAGE_CODE_SEGMENT fragment
  size_t op_counts = 0;
  auto code_fragment = ctx->lookupSymbolTable("__MLLM_JIT_PACKAGE_CODE_SEGMENT");
  if (code_fragment && code_fragment->isa_<ir::program::FragmentOp>()) {
    auto fragment_op = code_fragment->cast_<ir::program::FragmentOp>();
    op_counts = fragment_op->getTopRegion()->ops().size();
  }
  code_["op_counts"] = op_counts;

  return code_;
}

nlohmann::json IRSerializer::visitLinalgOp(const ir::IRContext::ptr_t& ctx, const ir::linalg::LinalgIROp::ptr_t& op) {
  return {};
}

nlohmann::json IRSerializer::visitCFOp(const ir::IRContext::ptr_t& ctx, const ir::cf::ControlFlowIROp::ptr_t& op) { return {}; }

nlohmann::json IRSerializer::visitTensorOp(const ir::IRContext::ptr_t& ctx, const ir::tensor::TensorIROp::ptr_t& op) {
  return {};
}

nlohmann::json IRSerializer::visitGraphOp(const ir::IRContext::ptr_t& ctx, const ir::graph::GraphIROp::ptr_t& op) { return {}; }

nlohmann::json IRSerializer::visitDbgOp(const ir::IRContext::ptr_t& ctx, const ir::dbg::DbgIROp::ptr_t& op) { return {}; }

nlohmann::json IRSerializer::visitBuiltinOp(const ir::IRContext::ptr_t& ctx, const ir::BuiltinIROp::ptr_t& op) {
  if (op->isa_<ir::ModuleOp>()) {
    nlohmann::json j;
    j["module"] = nlohmann::json::array();
    auto r = ir::IRWriter(ctx, op->cast_<ir::ModuleOp>()->getTopRegion());
    r.walk<ir::Op>([&](ir::IRWriter& rw, const ir::Op::ptr_t& sub_op) {
      if (sub_op->isa_<ir::program::ProgramIROp>()) {
        auto _j = visitProgramOp(ctx, sub_op->cast_<ir::program::ProgramIROp>());
        j["module"].push_back(_j);
      }
      return ir::IRWriter::WalkResult::WALK_CONTINUE;
    });
    return j;
  }
  return {};
}

nlohmann::json IRSerializer::visitProgramOp(const ir::IRContext::ptr_t& ctx, const ir::program::ProgramIROp::ptr_t& op) {
  if (op->isa_<ir::program::FragmentOp>()) { return visitProgramFragmentOp(ctx, op->cast_<ir::program::FragmentOp>()); }
  return {};
}

nlohmann::json IRSerializer::visitProgramFragmentOp(const ir::IRContext::ptr_t& ctx, const ir::program::FragmentOp::ptr_t& op) {
  auto r = ir::IRWriter(ctx, op->cast_<ir::program::FragmentOp>()->getTopRegion());
  nlohmann::json j;
  j["dialect"] = "program";
  j["op"] = "fragment";
  j["symbol"] = op->getSymbolAttr()->str();
  j["instructions"] = nlohmann::json::array();
  r.walk<ir::Op>([&](ir::IRWriter& rw, const ir::Op::ptr_t& sub_op) {
    if (sub_op->isa_<ir::program::FreeOp>()) {
      auto _j = visitProgramFreeOp(ctx, sub_op->cast_<ir::program::FreeOp>());
      j["instructions"].push_back(_j);
    } else if (sub_op->isa_<ir::program::LabelOp>()) {
      auto _j = visitProgramLabelOp(ctx, sub_op->cast_<ir::program::LabelOp>());
      j["instructions"].push_back(_j);
    } else if (sub_op->isa_<ir::program::JumpOp>()) {
      auto _j = visitProgramJumpOp(ctx, sub_op->cast_<ir::program::JumpOp>());
      j["instructions"].push_back(_j);
    } else if (sub_op->isa_<ir::program::KernelLaunchOp>()) {
      auto _j = visitProgramKernelLaunchOp(ctx, sub_op->cast_<ir::program::KernelLaunchOp>());
      j["instructions"].push_back(_j);
    } else if (sub_op->isa_<ir::program::RetOp>()) {
      auto _j = visitProgramRetOp(ctx, sub_op->cast_<ir::program::RetOp>());
      j["instructions"].push_back(_j);
    } else if (sub_op->isa_<ir::program::KernelSymbolOp>()) {
      auto _j = visitProgramKernelSymbolOp(ctx, sub_op->cast_<ir::program::KernelSymbolOp>());
      j["instructions"].push_back(_j);
    } else if (sub_op->isa_<ir::program::ValueSymbolOp>()) {
      auto _j = visitProgramValueSymbolOp(ctx, sub_op->cast_<ir::program::ValueSymbolOp>());
      j["instructions"].push_back(_j);
    } else if (sub_op->isa_<ir::program::ModeConfigOp>()) {
      auto _j = visitProgramModeConfigOp(ctx, sub_op->cast_<ir::program::ModeConfigOp>());
      j["instructions"].push_back(_j);
    } else if (sub_op->isa_<ir::program::ExitOp>()) {
      auto _j = visitProgramExitOp(ctx, sub_op->cast_<ir::program::ExitOp>());
      j["instructions"].push_back(_j);
    } else if (sub_op->isa_<ir::program::BindOp>()) {
      auto _j = visitProgramBindOp(ctx, sub_op->cast_<ir::program::BindOp>());
      j["instructions"].push_back(_j);
    }
    j["instructions"].back()["device"] = deviceTypes2Str(sub_op->getDevice());
    return ir::IRWriter::WalkResult::WALK_CONTINUE;
  });

  return j;
}

nlohmann::json IRSerializer::visitProgramFreeOp(const ir::IRContext::ptr_t& ctx, const ir::program::FreeOp::ptr_t& op) {
  nlohmann::json j;
  j["dialect"] = "program";
  j["op"] = "free";
  j["inputs"] = nlohmann::json::array();
  j["inputs"].push_back(op->inputs().front()->cast_<ir::tensor::TensorValue>()->tensor_.uuid());
  j["outputs"] = nlohmann::json::array();
  j["program_id"] = op->getProgramIntrinsicId();
  return j;
}

nlohmann::json IRSerializer::visitProgramLabelOp(const ir::IRContext::ptr_t& ctx, const ir::program::LabelOp::ptr_t& op) {
  nlohmann::json j;
  j["dialect"] = "program";
  j["op"] = "label";
  j["symbol"] = op->getSymbolAttr()->str();
  j["program_id"] = op->getProgramIntrinsicId();
  return j;
}

nlohmann::json IRSerializer::visitProgramJumpOp(const ir::IRContext::ptr_t& ctx, const ir::program::JumpOp::ptr_t& op) {
  nlohmann::json j;
  j["dialect"] = "program";
  j["op"] = "jump";
  j["target"] = op->labelName();
  j["program_id"] = op->getProgramIntrinsicId();
  j["offset"] = op->getAttr("offset")->cast_<ir::IntAttr>()->data();
  return j;
}

nlohmann::json IRSerializer::visitProgramKernelLaunchOp(const ir::IRContext::ptr_t& ctx,
                                                        const ir::program::KernelLaunchOp::ptr_t& op) {
  nlohmann::json j;
  j["dialect"] = "program";
  j["op"] = "kernel_launch";
  j["program_id"] = op->getProgramIntrinsicId();
  j["inputs"] = nlohmann::json::array();
  for (const auto& input : op->inputs()) {
    if (auto value = input) { j["inputs"].push_back(value->cast_<ir::tensor::TensorValue>()->tensor_.uuid()); }
  }
  j["outputs"] = nlohmann::json::array();
  for (const auto& output : op->outputs()) {
    if (auto value = output) { j["outputs"].push_back(value->cast_<ir::tensor::TensorValue>()->tensor_.uuid()); }
  }
  if (auto op_type_attr = op->getAttr("op_type")) { j["op_type"] = op_type_attr->cast_<ir::StrAttr>()->data(); }
  if (auto op_options_attr = op->getAttr("op_options")) {
    j["op_options"] = nlohmann::json::parse(op_options_attr->cast_<ir::StrAttr>()->data());
  }
  if (op->getAttr("symbol_name")) { j["symbol_name"] = op->getAttr("symbol_name")->cast_<ir::StrAttr>()->data(); }
  return j;
}

nlohmann::json IRSerializer::visitProgramRetOp(const ir::IRContext::ptr_t& ctx, const ir::program::RetOp::ptr_t& op) {
  nlohmann::json j;
  j["dialect"] = "program";
  j["op"] = "ret";
  j["program_id"] = op->getProgramIntrinsicId();
  return j;
}

nlohmann::json IRSerializer::visitProgramKernelSymbolOp(const ir::IRContext::ptr_t& ctx,
                                                        const ir::program::KernelSymbolOp::ptr_t& op) {
  nlohmann::json j;
  j["dialect"] = "program";
  j["op"] = "kernel_symbol";
  j["symbol"] = op->getSymbolAttr()->str();
  j["program_id"] = op->getProgramIntrinsicId();

  if (auto op_type_attr = op->getAttr("op_type")) { j["op_type"] = op_type_attr->cast_<ir::StrAttr>()->data(); }
  if (auto op_options_attr = op->getAttr("op_options")) {
    j["op_options"] = nlohmann::json::parse(op_options_attr->cast_<ir::StrAttr>()->data());
  }

  return j;
}

nlohmann::json IRSerializer::visitProgramValueSymbolOp(const ir::IRContext::ptr_t& ctx,
                                                       const ir::program::ValueSymbolOp::ptr_t& op) {
  nlohmann::json j;
  j["dialect"] = "program";
  j["op"] = "value_symbol";
  j["symbol"] = op->getSymbolAttr()->str();
  j["program_id"] = op->getProgramIntrinsicId();
  j["outputs"] = nlohmann::json::array();
  for (const auto& output : op->outputs()) {
    if (auto value = output) { j["outputs"].push_back(value->cast_<ir::tensor::TensorValue>()->tensor_.uuid()); }
  }
  MLLM_RT_ASSERT_EQ(op->outputs().size(), 1);
  if (op->outputs().front()->cast_<ir::tensor::TensorValue>()->getAttr("constant")) {
    j["constant"] =
        op->outputs().front()->cast_<ir::tensor::TensorValue>()->getAttr("constant")->cast_<ir::VectorFP32Attr>()->data();
    j["shape"] = op->outputs().front()->cast_<ir::tensor::TensorValue>()->tensor_.shape();
  }
  return j;
}

nlohmann::json IRSerializer::visitProgramModeConfigOp(const ir::IRContext::ptr_t& ctx,
                                                      const ir::program::ModeConfigOp::ptr_t& op) {
  nlohmann::json j;
  j["dialect"] = "program";
  j["op"] = "mode_config";
  j["program_id"] = op->getProgramIntrinsicId();
  if (auto flag_attr = op->getAttr("flag")) { j["flag"] = flag_attr->cast_<ir::IntAttr>()->data(); }
  return j;
}

nlohmann::json IRSerializer::visitProgramExitOp(const ir::IRContext::ptr_t& ctx, const ir::program::ExitOp::ptr_t& op) {
  nlohmann::json j;
  j["dialect"] = "program";
  j["op"] = "exit";
  j["program_id"] = op->getProgramIntrinsicId();
  return j;
}

nlohmann::json IRSerializer::visitProgramBindOp(const ir::IRContext::ptr_t& ctx, const ir::program::BindOp::ptr_t& op) {
  nlohmann::json j;
  j["dialect"] = "program";
  j["op"] = "bind";
  j["program_id"] = op->getProgramIntrinsicId();

  if (auto input_pos_attr = op->getAttr("input_pos")) { j["input_pos"] = input_pos_attr->cast_<ir::IntAttr>()->data(); }
  if (auto program_uuid_attr = op->getAttr("program_uuid")) {
    j["program_uuid"] = program_uuid_attr->cast_<ir::IntAttr>()->data();
  }
  if (auto type_attr = op->getAttr("type")) {
    auto type = static_cast<ir::program::BindOp::BindType>(type_attr->cast_<ir::IntAttr>()->data());
    j["type"] = type == ir::program::BindOp::BindType::kInput ? "input" : "output";
  }
  return j;
}

nlohmann::json& IRSerializer::getCode() { return code_; }

void IRSerializer::save(const std::string& path) {
  std::ofstream file(path);
  file << code_;
}

}  // namespace mllm::jit::binary
