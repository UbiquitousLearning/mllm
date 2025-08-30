// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <chrono>
#include <iomanip>
#include <sstream>

#include "mllm/compile/jit/binary/IRSerialization.hpp"
#include "mllm/compile/jit/JITUtils.hpp"
#include "mllm/utils/CPUArchHelper.hpp"
#include "mllm/utils/PlatformRTHelper.hpp"

namespace mllm::jit::binary {

// Will loop all ops in this Module
nlohmann::json IRSerializer::visit(const ir::IRContext::ptr_t& module) {
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
