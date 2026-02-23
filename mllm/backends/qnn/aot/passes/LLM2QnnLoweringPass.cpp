// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <regex>

#include "mllm/backends/qnn/aot/passes/LLM2QnnLoweringPass.hpp"
#include "mllm/backends/qnn/aot/passes/AOTCompileContext.hpp"
#include "mllm/compile/ir/builtin/Op.hpp"
#include "mllm/compile/ir/graph/Op.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/Node.hpp"
#include "mllm/compile/passes/Pass.hpp"
#include "mllm/utils/Common.hpp"

#include "mllm/backends/qnn/aot/visitor/Conv2D.hpp"
#include "mllm/backends/qnn/aot/visitor/Elewise.hpp"
#include "mllm/backends/qnn/aot/visitor/Embedding.hpp"
#include "mllm/backends/qnn/aot/visitor/Gather.hpp"
#include "mllm/backends/qnn/aot/visitor/CastType.hpp"
#include "mllm/backends/qnn/aot/visitor/View.hpp"
#include "mllm/backends/qnn/aot/visitor/Index.hpp"
#include "mllm/backends/qnn/aot/visitor/RMSNorm.hpp"
#include "mllm/backends/qnn/aot/visitor/Linear.hpp"
#include "mllm/backends/qnn/aot/visitor/Concat.hpp"
#include "mllm/backends/qnn/aot/visitor/Slice.hpp"
#include "mllm/backends/qnn/aot/visitor/Transpose.hpp"
#include "mllm/backends/qnn/aot/visitor/Reduce.hpp"
#include "mllm/backends/qnn/aot/visitor/Equal.hpp"
#include "mllm/backends/qnn/aot/visitor/Sigmoid.hpp"
#include "mllm/backends/qnn/aot/visitor/Matmul.hpp"
#include "mllm/backends/qnn/aot/visitor/Repeat.hpp"
#include "mllm/backends/qnn/aot/visitor/Softmax.hpp"
#include "mllm/backends/qnn/aot/visitor/Where.hpp"

namespace mllm::qnn::aot {

LLM2QnnLoweringPass::LLM2QnnLoweringPass() {
  registerPatterns<QnnAOTEmbeddingPattern, QnnAOTCastTypePattern, QnnAOTAddPattern, QnnAOTMulPattern, QnnAOTNegPattern,
                   QnnAOTViewPattern, QnnAOTIndexPattern, QnnAOTGatherPattern, QnnAOTRMSNormPattern, QnnAOTLinearPattern,
                   QnnAOTTransposePattern, QnnAOTSlicePattern, QnnAOTConcatPattern, QnnAOTRepeatPattern, QnnAOTMatMulPattern,
                   QnnAOTReduceMaxPattern, QnnAOTReduceMinPattern, QnnAOTReduceMeanPattern, QnnAOTReduceSumPattern,
                   QnnAOTEqualPattern, QnnAOTWherePattern, QnnAOTSoftmaxPattern, QnnAOTSigmoidPattern, QnnAOTConv2DPattern>();
}

uint8_t LLM2QnnLoweringPass::run(const ir::node_ptr_t& op) {
  // The top op should be modelOp
  MLLM_RT_ASSERT(op->isa_<ir::ModuleOp>());

  auto model_op = op->cast_<ir::ModuleOp>();
  auto writer = ir::IRWriter(getCtx(), model_op->getTopRegion());

  // Check only has 1 call graph op in model_op
  ir::graph::CallGraphOp::ptr_t call_graph_op = nullptr;
  writer.walk<ir::graph::CallGraphOp>(
      [&](ir::IRWriter& /*writer*/, const ir::graph::CallGraphOp::ptr_t& call_op) -> ir::IRWriter::WalkResult {
        MLLM_RT_ASSERT(call_graph_op == nullptr);  // Should only have one CallGraphOp
        call_graph_op = call_op;
        return ir::IRWriter::WalkResult::WALK_CONTINUE;
      });

  if (call_graph_op == nullptr) {
    MLLM_ERROR("LLM2QnnLoweringPass: No CallGraphOp found in ModuleOp");
    return ir::PASS_RET_FAILURE;
  }

  // Check call graph op point to a subgraph named "model"
  auto symbol_attr = call_graph_op->getSymbolAttr();
  if (symbol_attr == nullptr || symbol_attr->str() != "model") {
    MLLM_ERROR("LLM2QnnLoweringPass: CallGraphOp should point to a subgraph named 'model'");
    return ir::PASS_RET_FAILURE;
  }

  // Get the "model" subgraph
  auto model_subgraph = getCtx()->lookupSymbolTable("model")->cast_<ir::graph::SubGraphOp>();
  if (model_subgraph == nullptr) {
    MLLM_ERROR("LLM2QnnLoweringPass: Cannot find 'model' subgraph in symbol table");
    return ir::PASS_RET_FAILURE;
  }

  // Collect all subgraphs from the modelOp's top region
  std::unordered_map<std::string, ir::graph::SubGraphOp::ptr_t> subgraphs;

  for (auto& region_op : model_op->getTopRegion()->ops()) {
    if (auto sub_graph_op = std::dynamic_pointer_cast<ir::graph::SubGraphOp>(region_op)) {
      auto symbol_attr = sub_graph_op->getSymbolAttr();
      if (symbol_attr && symbol_attr->str() != "init") { subgraphs[symbol_attr->str()] = sub_graph_op; }
    }
  }

  // Validate that we only have the expected subgraphs: model, model.0.s32, model.1.s16, etc.
  // Pattern: model.x.sN where x is a number and N can be 16, 32, 64, 96, etc.
  std::regex model_pattern(R"(^model(\.\d+\.s\d+)?$)");
  for (const auto& [name, _] : subgraphs) {
    if (!std::regex_match(name, model_pattern)) {
      MLLM_ERROR("LLM2QnnLoweringPass: Unexpected subgraph name {}, expected pattern: model or model.x.sx", name);
      return ir::PASS_RET_FAILURE;
    }
  }

  // Store subgraphs in the member variable
  subgraph_map_.clear();
  for (const auto& [name, subgraph] : subgraphs) {
    if (name != "model") { subgraph_map_[name] = subgraph; }
  }

  // Validate that at least one model.x.sN subgraph exists (required for the lowering)
  // We don't require specifically model.0.s32, but any model.x.sN pattern
  bool has_valid_subgraph = false;
  for (const auto& [name, _] : subgraph_map_) {
    if (std::regex_match(name, std::regex(R"(^model\.\d+\.s\d+$)"))) {
      has_valid_subgraph = true;
      break;
    }
  }

  if (!has_valid_subgraph) {
    MLLM_ERROR("LLM2QnnLoweringPass: No valid subgraph found (expected model.x.sN pattern)");
    return ir::PASS_RET_FAILURE;
  }

  // Sort subgraphs by name to ensure deterministic processing order
  std::vector<std::string> sorted_names;
  sorted_names.reserve(subgraph_map_.size());
  for (const auto& [name, _] : subgraph_map_) { sorted_names.push_back(name); }
  std::sort(sorted_names.begin(), sorted_names.end());

  // Get AOT Compile Context
  auto aot_cfg = AOTCompileContext::getInstance().getConfig();
  auto aot_env = AOTCompileContext::getInstance().getEnv();

  // FIXME: Only support one context right now.
  {
    int split_graph = aot_cfg["split_graph"];
    MLLM_RT_ASSERT_EQ(split_graph, 1);
    aot_env->createContext("context.0", true);
  }

  // Process each subgraph in order
  for (const auto& subgraph_name : sorted_names) {
    auto subgraph = subgraph_map_[subgraph_name];
    auto region = subgraph->getTopRegion();
    if (!region) continue;

    // Create IRWriter for this subgraph
    auto subgraph_writer = ir::IRWriter(getCtx(), region);

    auto aot_graph = aot_env->captureAOTGraph("context.0", subgraph_name);

    // Add sub-graph inputs
    for (auto& input : region->inputs()) {
      auto tensor_input = input->cast_<ir::tensor::TensorValue>();
      if (tensor_input) { aot_env->captureQnnAOTNodeTensor("context.0", subgraph_name, tensor_input); }
    }
    // Add sub-graph outputs
    for (auto& output : region->outputs()) {
      auto tensor_output = output->cast_<ir::tensor::TensorValue>();
      if (tensor_output) { aot_env->captureQnnAOTNodeTensor("context.0", subgraph_name, tensor_output); }
    }

    // Walk through all linalg operations in the subgraph
    subgraph_writer.walk<ir::linalg::LinalgIROp>(
        [&](ir::IRWriter& this_tough_writer, const ir::linalg::LinalgIROp::ptr_t& linalg_op) -> ir::IRWriter::WalkResult {
          if (!linalg_op->belongsTo()->getAttr("use_qnn")) {
            MLLM_WARN("Found none qnn op: {} in graph: {}", linalg_op->getAOp()->getName(), subgraph_name);
            return ir::IRWriter::WalkResult::WALK_BREAK;
          }
          bool processed = false;
          for (auto& [op_type, pass] : named_pattern_) {
            if (pass->isMatch(linalg_op)) {
              if (!pass->rewrite(this_tough_writer, linalg_op)) {
                MLLM_ERROR_EXIT(ExitCode::kCoreError, "Failed when processing op {} with pass {}",
                                linalg_op->getAOp()->getName(), optype2Str(op_type));
              } else {
                processed = true;
                break;
              }
            }
          }

          if (!processed) {
            MLLM_ERROR_EXIT(ExitCode::kCoreError, "Failed processing op {} on all passes", linalg_op->getAOp()->getName());
          }

          return ir::IRWriter::WalkResult::WALK_CONTINUE;
        });

    // Compile
    MLLM_RT_ASSERT(aot_graph->compile());
  }

  return ir::PASS_RET_SUCCESS;
}

ir::Pass::ptr_t createLLM2QnnLoweringPass() { return std::make_shared<LLM2QnnLoweringPass>(); }

}  // namespace mllm::qnn::aot
