// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/Node.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/program/Op.hpp"
#include "mllm/compile/ir/tensor/Op.hpp"
#include "mllm/compile/ir/builtin/Op.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/compile/passes/FlattenTensorAndLinalgSymbol2ProgramSymbolPass.hpp"

namespace mllm::ir {

uint8_t FlattenTensorAndLinalgSymbol2ProgramSymbolPass::run(const node_ptr_t& this_top_op) {
  // The top op should be ModuleOp
  MLLM_RT_ASSERT(this_top_op->isa_<ir::ModuleOp>());

  auto r = ir::IRWriter(getCtx(), this_top_op->cast_<ir::ModuleOp>()->getTopRegion());

  auto init_fragment = getCtx()->lookupSymbolTable("init");
  MLLM_RT_ASSERT(init_fragment != nullptr);
  auto init_fragment_rw = ir::IRWriter(getCtx(), init_fragment->cast_<ir::program::FragmentOp>()->getTopRegion());

  MLLM_RT_ASSERT_EQ(getCtx()->lookupSymbolTable("__MLLM_JIT_PACKAGE_SYMBOL_TABLE_SEGMENT"), nullptr)
  auto flatten_table_segment = r.create<ir::program::FragmentOp>(
      program::FragmentType::kTable, getCtx()->create<ir::SymbolAttr>("__MLLM_JIT_PACKAGE_SYMBOL_TABLE_SEGMENT"));
  auto flatten_table_segment_rw = ir::IRWriter(getCtx(), flatten_table_segment->getTopRegion());

  // Loop init table, and transform Tensor Register IR to Value Symbol
  init_fragment_rw.walk<ir::tensor::RegisterOp>(
      [&](ir::IRWriter& inner_reader, const ir::tensor::RegisterOp::ptr_t& op) -> ir::IRWriter::WalkResult {
        auto program_symbol_op =
            flatten_table_segment_rw.create<ir::program::ValueSymbolOp>(op->getRegisteredTensor(), op->getSymbolAttr());
        getCtx()->removeFromSymbolTable(op->getSymbolAttr()->str());
        getCtx()->addToSymbolTable(program_symbol_op, op->getSymbolAttr()->str());
        inner_reader.removeOp(op);
        return ir::IRWriter::WalkResult::WALK_CONTINUE;
      });
  r.removeOp(init_fragment->cast_<ir::Op>());

  std::unordered_map<std::string, std::pair<std::string, std::string>> kernel_symbol_payload;
  auto code_segment = getCtx()->lookupSymbolTable("__MLLM_JIT_PACKAGE_CODE_SEGMENT");
  MLLM_RT_ASSERT(code_segment != nullptr);
  auto code_segment_rw = ir::IRWriter(getCtx(), code_segment->cast_<ir::program::FragmentOp>()->getTopRegion());
  code_segment_rw.walk<ir::program::KernelLaunchOp>(
      [&](ir::IRWriter& inner_reader, const ir::program::KernelLaunchOp::ptr_t& op) -> ir::IRWriter::WalkResult {
        if (op->getAttr("symbol_name")) {
          kernel_symbol_payload[op->getAttr("symbol_name")->cast_<ir::StrAttr>()->data()] = {
              op->getAttr("op_type")->cast_<ir::StrAttr>()->data(),
              op->getAttr("op_options")->cast_<ir::StrAttr>()->data(),
          };
        }
        return ir::IRWriter::WalkResult::WALK_CONTINUE;
      });

  // Loop op_symbol_table if it exists
  auto op_symbol_table_fragment = getCtx()->lookupSymbolTable("op_symbol_table");
  if (op_symbol_table_fragment) {
    auto op_symbol_table_fragment_rw =
        ir::IRWriter(getCtx(), op_symbol_table_fragment->cast_<ir::program::FragmentOp>()->getTopRegion());
    op_symbol_table_fragment_rw.walk<ir::linalg::RegisterOp>(
        [&](ir::IRWriter& inner_reader, const ir::linalg::RegisterOp::ptr_t& op) -> ir::IRWriter::WalkResult {
          auto _name = op->getSymbolAttr()->str();
          auto kernel_symbol_op = flatten_table_segment_rw.create<ir::program::KernelSymbolOp>(
              op->getSymbolAttr(), kernel_symbol_payload[_name].first, kernel_symbol_payload[_name].second);
          getCtx()->removeFromSymbolTable(op->getSymbolAttr()->str());
          getCtx()->addToSymbolTable(kernel_symbol_op, op->getSymbolAttr()->str());
          inner_reader.removeOp(op);
          return ir::IRWriter::WalkResult::WALK_CONTINUE;
        });
    r.removeOp(op_symbol_table_fragment->cast_<ir::Op>());
  }

  return ir::PASS_RET_SUCCESS;
}

Pass::ptr_t createFlattenTensorAndLinalgSymbol2ProgramSymbolPass() {
  return std::make_shared<FlattenTensorAndLinalgSymbol2ProgramSymbolPass>();
}

}  // namespace mllm::ir
