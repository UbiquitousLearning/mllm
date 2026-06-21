// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/qnn/aot/passes/MarkTensorIO.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/compile/ir/builtin/Op.hpp"
#include "mllm/compile/ir/graph/Op.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/tensor/Value.hpp"
#include "mllm/compile/ir/cf/Op.hpp"
#include "mllm/compile/ir/Node.hpp"
#include "mllm/core/OpTypes.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/backends/qnn/aot/passes/AOTCompileContext.hpp"

namespace mllm::qnn::aot {

uint8_t MarkTensorIOPass::run(const ir::node_ptr_t& op) {
  auto& aot_compile_ctx = AOTCompileContext::getInstance();
  auto config = aot_compile_ctx.getConfig();

  if (!config.contains("split_graph") || config["split_graph"] != 1) {
    MLLM_ERROR_EXIT(
        ExitCode::kCoreError,
        "split_graph should be 1 in mark tensor IO pass. Pls send us a issue or give us a PR if you want split graph");
  }

  // The top op should be ModuleOp
  MLLM_RT_ASSERT(op->isa_<ir::ModuleOp>());

  auto module_op = op->cast_<ir::ModuleOp>();
  auto writer = ir::IRWriter(getCtx(), module_op->getTopRegion());

  // Find the main CallGraphOp
  ir::graph::CallGraphOp::ptr_t call_main_graph_op = nullptr;
  writer.walk<ir::graph::CallGraphOp>(
      [&](ir::IRWriter& /*writer*/, const ir::graph::CallGraphOp::ptr_t& call_op) -> ir::IRWriter::WalkResult {
        MLLM_RT_ASSERT_EQ(call_main_graph_op, nullptr);

        call_main_graph_op = call_op;
        return ir::IRWriter::WalkResult::WALK_CONTINUE;
      });

  // Visit all graphs and assign names to unnamed operations
  //   {
  //     "target_machine": {
  //         "htp_arch": "V81",
  //         "htp_chipset": "SM8850",
  //         "htp_try_best_performance": "HtpBurst",
  //         "htp_security_pd_session": "HtpSignedPd",
  //         "htp_vtcm_capability_in_mb": 8
  //     },
  //     "graph_on_qnn": [
  //         "model"
  //     ],
  //     "op_on_qnn": [
  //         "lm_head"
  //     ],
  //     "quant_recipe": {
  //         "llm_recipe": true,
  //         "builtin_qwen3_recipe": {
  //             "linear": "w4a16-lpbq",
  //             "kv_cache": {
  //                 "key": "int8-per-tensor",
  //                 "value": "int8-per-tensor"
  //             }
  //         }
  //     }
  // }
  //
  // "llm_recipe": true must be setted!
  if (call_main_graph_op != nullptr) {
    // Check if llm_recipe is set in config
    if (!config.contains("quant_recipe") || !config["quant_recipe"].contains("llm_recipe")
        || !config["quant_recipe"]["llm_recipe"].is_boolean() || !config["quant_recipe"]["llm_recipe"].get<bool>()) {
      MLLM_ERROR_EXIT(ExitCode::kAssert,
                      "MarkTensorIOPass: 'quant_recipe.llm_recipe' must be set to true in the AOT configuration. "
                      "Please ensure the config JSON contains: {{\"quant_recipe\": {{\"llm_recipe\": true, ...}}}}");
    }

    // Tag the inputs to call_graph to "qnn_graph_inputs"
    {
      auto inputs = call_main_graph_op->inputs();
      for (auto i : inputs) {
        if (i->isa_<ir::tensor::TensorValue>()) {
          if (i->prevOp()) { MLLM_ERROR_EXIT(ExitCode::kCoreError, "Expect inputs tensor value has no previous operators."); }
          i->setAttr("qnn_graph_inputs", getCtx()->create<ir::BoolAttr>(true));
        } else {
          MLLM_ERROR_EXIT(ExitCode::kCoreError, "Expect pure tensor value inputs to graph.CallGraphOp");
        }
      }
    }

    // Tag the outputs to call_graph to "qnn_graph_outputs"
    {
      auto outputs = call_main_graph_op->outputs();
      for (auto o : outputs) {
        if (o->isa_<ir::tensor::TensorValue>()) {
          // That means after this graph, we have a lm_head!
          // o 's outputs is cf.Return and lm_head(linear op)
          if (!o->outputs().empty() && o->outputs().size() == 2 && o->outputs().front()->isa_<ir::cf::ReturnOp>()
              && (*std::next(o->outputs().begin()))->isa_<ir::linalg::LinearOp>()) {
            auto lm_head_ir = (*std::next(o->outputs().begin()))->cast_<ir::linalg::LinearOp>();
            if (lm_head_ir->outputs().front()->outputs().empty()) {
              lm_head_ir->outputs().front()->setAttr("qnn_graph_outputs", getCtx()->create<ir::BoolAttr>(true));
            }
          } else {
            o->setAttr("qnn_graph_outputs", getCtx()->create<ir::BoolAttr>(true));
          }
        } else {
          MLLM_ERROR_EXIT(ExitCode::kCoreError, "Expect pure tensor value outputs to graph.CallGraphOp");
        }
      }
    }
  }

  return ir::PASS_RET_SUCCESS;
}

ir::Pass::ptr_t createMarkTensorIOPass() { return std::make_shared<MarkTensorIOPass>(); }

}  // namespace mllm::qnn::aot
