// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include <cstdint>
#include <unordered_map>

#include "mllm/compile/ir/Node.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/builtin/Op.hpp"
#include "mllm/compile/ir//program/Op.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/compile/passes/ProgramIntrinsicIdIndexPass.hpp"

namespace mllm::ir {

uint8_t ProgramIntrinsicIdIndexPass::run(const node_ptr_t& op) {
  // The top op should be ModuleOp
  MLLM_RT_ASSERT(op->isa_<ir::ModuleOp>());

  auto fragment = getCtx()->lookupSymbolTable("__MLLM_JIT_PACKAGE_CODE_SEGMENT");
  MLLM_RT_ASSERT(fragment != nullptr);

  uint64_t program_cnt = 0;
  for (auto& op : fragment->cast_<ir::program::FragmentOp>()->getTopRegion()->ops()) {
    if (op->isa_<ir::program::ProgramIROp>()) { op->cast_<ir::program::ProgramIROp>()->setProgramIntrinsicId(program_cnt++); }
  }

  // Find the subgraph we need to process
  auto r = ir::IRWriter(getCtx(), fragment->cast_<ir::program::FragmentOp>()->getTopRegion());
  std::unordered_map<std::string, uint64_t> program_name_to_id;

  r.walk<ir::program::LabelOp>([&](ir::IRWriter& rw, const ir::program::LabelOp::ptr_t& op) -> ir::IRWriter::WalkResult {
    MLLM_RT_ASSERT_EQ(program_name_to_id.count(op->getSymbolAttr()->str()), 0);
    program_name_to_id[op->getSymbolAttr()->str()] = op->getProgramIntrinsicId();
    return ir::IRWriter::WalkResult::WALK_CONTINUE;
  });

  r.walk<ir::program::JumpOp>([&](ir::IRWriter& rw, const ir::program::JumpOp::ptr_t& op) -> ir::IRWriter::WalkResult {
    MLLM_RT_ASSERT_EQ(program_name_to_id.count(op->labelName()), 1);
    auto label_id = program_name_to_id[op->labelName()];
    auto offset_id = label_id - op->getProgramIntrinsicId();
    op->setAttr("offset", getCtx()->create<IntAttr>(offset_id));
    return ir::IRWriter::WalkResult::WALK_CONTINUE;
  });

  // We process Symbol Index here.
  {
    auto symbol_fragment = getCtx()->lookupSymbolTable("__MLLM_JIT_PACKAGE_SYMBOL_TABLE_SEGMENT");
    MLLM_RT_ASSERT(symbol_fragment != nullptr);
    uint64_t symbol_cnt = 0;
    for (auto& op : symbol_fragment->cast_<ir::program::FragmentOp>()->getTopRegion()->ops()) {
      if (op->isa_<ir::program::ProgramIROp>()) { op->cast_<ir::program::ProgramIROp>()->setProgramIntrinsicId(symbol_cnt++); }
    }
  }

  return ir::PASS_RET_SUCCESS;
}

Pass::ptr_t createProgramIntrinsicIdIndexPass() { return std::make_shared<ProgramIntrinsicIdIndexPass>(); }

}  // namespace mllm::ir
