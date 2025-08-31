// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/program/Op.hpp"
#include "mllm/compile/ir/builtin/Op.hpp"
#include "mllm/compile/passes/ProgramModeConfigPass.hpp"

namespace mllm::ir {

ProgramModeConfigPass::ProgramModeConfigPass(const ProgramModeConfigPassOptions& options) : options_(options) {}

uint8_t ProgramModeConfigPass::run(const node_ptr_t& op) {
  // The top op should be ModuleOp
  MLLM_RT_ASSERT(op->isa_<ir::ModuleOp>());
  auto code_fragment = getCtx()->lookupSymbolTable("__MLLM_JIT_PACKAGE_CODE_SEGMENT")->cast_<ir::program::FragmentOp>();
  auto r = ir::IRWriter(getCtx(), code_fragment->getTopRegion());
  auto first_op = code_fragment->getTopRegion()->ops().front();
  r.createAtPos<ir::program::ModeConfigOp>(first_op, ir::IRWriter::BEFORE, (ir::program::ModeConfigFlag)options_.mode);
  return ir::PASS_RET_SUCCESS;
}

Pass::ptr_t createProgramModeConfigPass(const ProgramModeConfigPassOptions& options) {
  return std::make_shared<ProgramModeConfigPass>(options);
}

}  // namespace mllm::ir
