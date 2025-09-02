// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <nlohmann/json.hpp>
#include <string>

#include "mllm/compile/ir/Node.hpp"
#include "mllm/compile/ir/builtin/Op.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/dbg/Op.hpp"
#include "mllm/compile/ir/cf/Op.hpp"
#include "mllm/compile/ir/tensor/Op.hpp"
#include "mllm/compile/ir/program/Op.hpp"
#include "mllm/compile/ir/graph/Op.hpp"

namespace mllm::jit::binary {

class IRSerializer {
 public:
  IRSerializer() = default;

  // Will loop all ops in this Module
  nlohmann::json visit(const ir::IRContext::ptr_t& ctx);

  nlohmann::json visitLinalgOp(const ir::IRContext::ptr_t& ctx, const ir::linalg::LinalgIROp::ptr_t& op);

  nlohmann::json visitCFOp(const ir::IRContext::ptr_t& ctx, const ir::cf::ControlFlowIROp::ptr_t& op);

  nlohmann::json visitTensorOp(const ir::IRContext::ptr_t& ctx, const ir::tensor::TensorIROp::ptr_t& op);

  nlohmann::json visitGraphOp(const ir::IRContext::ptr_t& ctx, const ir::graph::GraphIROp::ptr_t& op);

  nlohmann::json visitDbgOp(const ir::IRContext::ptr_t& ctx, const ir::dbg::DbgIROp::ptr_t& op);

  nlohmann::json visitBuiltinOp(const ir::IRContext::ptr_t& ctx, const ir::BuiltinIROp::ptr_t& op);

  //===----------------------------------------------------------------------===//
  // ProgramOp visitor
  //===----------------------------------------------------------------------===//
  nlohmann::json visitProgramOp(const ir::IRContext::ptr_t& ctx, const ir::program::ProgramIROp::ptr_t& op);

  nlohmann::json visitProgramFragmentOp(const ir::IRContext::ptr_t& ctx, const ir::program::FragmentOp::ptr_t& op);

  nlohmann::json visitProgramFreeOp(const ir::IRContext::ptr_t& ctx, const ir::program::FreeOp::ptr_t& op);

  nlohmann::json visitProgramLabelOp(const ir::IRContext::ptr_t& ctx, const ir::program::LabelOp::ptr_t& op);

  nlohmann::json visitProgramJumpOp(const ir::IRContext::ptr_t& ctx, const ir::program::JumpOp::ptr_t& op);

  nlohmann::json visitProgramKernelLaunchOp(const ir::IRContext::ptr_t& ctx, const ir::program::KernelLaunchOp::ptr_t& op);

  nlohmann::json visitProgramRetOp(const ir::IRContext::ptr_t& ctx, const ir::program::RetOp::ptr_t& op);

  nlohmann::json visitProgramKernelSymbolOp(const ir::IRContext::ptr_t& ctx, const ir::program::KernelSymbolOp::ptr_t& op);

  nlohmann::json visitProgramValueSymbolOp(const ir::IRContext::ptr_t& ctx, const ir::program::ValueSymbolOp::ptr_t& op);

  nlohmann::json visitProgramModeConfigOp(const ir::IRContext::ptr_t& ctx, const ir::program::ModeConfigOp::ptr_t& op);

  nlohmann::json visitProgramExitOp(const ir::IRContext::ptr_t& ctx, const ir::program::ExitOp::ptr_t& op);

  nlohmann::json visitProgramBindOp(const ir::IRContext::ptr_t& ctx, const ir::program::BindOp::ptr_t& op);

  nlohmann::json& getCode();

  void save(const std::string& path);

 private:
  nlohmann::json code_;
};

}  // namespace mllm::jit::binary
