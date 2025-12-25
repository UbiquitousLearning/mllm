// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/utils/Common.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/compile/ir/graph/Op.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/builtin/Op.hpp"
#include "mllm/compile/ir/linalg/Attribute.hpp"
#include "mllm/backends/qnn/aot/passes/AOTCompileContext.hpp"
#include "mllm/backends/qnn/aot/passes/LLMQuantRecipePass.hpp"

namespace mllm::qnn::aot {
bool shareQuantSpecSingleInputToSingleOutputAndSetOpQuantAnnoAttr(const ir::IRContext::ptr_t& ctx,
                                                                  const ir::linalg::LinalgIROp::ptr_t& op) {
  // OP has no quant_recipe
  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("quant_recipe"));
  MLLM_RETURN_FALSE_IF_NOT(op->inputs().size() == 1);
  MLLM_RETURN_FALSE_IF_NOT(op->outputs().size() == 1);
  MLLM_RETURN_FALSE_IF_NOT(op->inputs().front()->getAttr("quant_recipe"));

  // Create annotation
  auto annotation_attr = ctx->create<ir::linalg::LinalgIRQuantizatonAnnotationAttr>();

  // Share
  auto quant_spec = op->inputs().front()->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>();
  annotation_attr->annotation_.inputs.emplace_back(quant_spec);
  annotation_attr->annotation_.outputs.emplace_back(quant_spec);
  op->outputs().front()->setAttr("quant_recipe", quant_spec);
  op->setAttr("quant_recipe", annotation_attr);

  return true;
}

//===----------------------------------------------------------------------===//
// Index Pattern
//===----------------------------------------------------------------------===//
bool LLMQuantRecipeIndexPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  if (op->isa_<ir::linalg::IndexOp>()) { return true; }
  return false;
}

bool LLMQuantRecipeIndexPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) {
  // TODO: Implement index pattern rewrite logic
  return true;
}

//===----------------------------------------------------------------------===//
// Elementwise Pattern
//===----------------------------------------------------------------------===//
bool LLMQuantRecipeElementwisePattern::isMatch(const mllm::ir::op_ptr_t& op) {
  if (op->isa_<ir::linalg::AddOp>()) { return true; }
  if (op->isa_<ir::linalg::SubOp>()) { return true; }
  if (op->isa_<ir::linalg::MulOp>()) { return true; }
  if (op->isa_<ir::linalg::DivOp>()) { return true; }
  return false;
}

bool LLMQuantRecipeElementwisePattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) {
  // TODO: Implement elementwise pattern rewrite logic
  return true;
}

//===----------------------------------------------------------------------===//
// Transpose Pattern
//===----------------------------------------------------------------------===//
bool LLMQuantRecipeTransposePattern::isMatch(const mllm::ir::op_ptr_t& op) {
  if (op->isa_<ir::linalg::TransposeOp>()) { return true; }
  return false;
}

bool LLMQuantRecipeTransposePattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) {
  return shareQuantSpecSingleInputToSingleOutputAndSetOpQuantAnnoAttr(writer.getContext(),
                                                                      node->cast_<ir::linalg::LinalgIROp>());
}

//===----------------------------------------------------------------------===//
// Concat Pattern
//===----------------------------------------------------------------------===//
bool LLMQuantRecipeConcatPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  if (op->isa_<ir::linalg::ConcatOp>()) { return true; }
  return false;
}

bool LLMQuantRecipeConcatPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) {
  // TODO: Implement concat pattern rewrite logic
  return true;
}

//===----------------------------------------------------------------------===//
// Repeat Pattern
//===----------------------------------------------------------------------===//
bool LLMQuantRecipeRepeatPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  if (op->isa_<ir::linalg::RepeatOp>()) { return true; }
  return false;
}

bool LLMQuantRecipeRepeatPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) {
  return shareQuantSpecSingleInputToSingleOutputAndSetOpQuantAnnoAttr(writer.getContext(),
                                                                      node->cast_<ir::linalg::LinalgIROp>());
}

//===----------------------------------------------------------------------===//
// MatMul Pattern
//===----------------------------------------------------------------------===//
bool LLMQuantRecipeMatMulPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  if (op->isa_<ir::linalg::MatMulOp>()) { return true; }
  return false;
}

bool LLMQuantRecipeMatMulPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) {
  // TODO: Implement matmul pattern rewrite logic
  return true;
}

//===----------------------------------------------------------------------===//
// Equal Pattern
//===----------------------------------------------------------------------===//
bool LLMQuantRecipeEqualPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  if (op->isa_<ir::linalg::EqualOp>()) { return true; }
  return false;
}

bool LLMQuantRecipeEqualPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) {
  // TODO: Implement equal pattern rewrite logic
  return true;
}

//===----------------------------------------------------------------------===//
// Where Pattern
//===----------------------------------------------------------------------===//
bool LLMQuantRecipeWherePattern::isMatch(const mllm::ir::op_ptr_t& op) {
  if (op->isa_<ir::linalg::WhereOp>()) { return true; }
  return false;
}

bool LLMQuantRecipeWherePattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) {
  // TODO: Implement where pattern rewrite logic
  return true;
}

//===----------------------------------------------------------------------===//
// Softmax Pattern
//===----------------------------------------------------------------------===//
bool LLMQuantRecipeSoftmaxPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  if (op->isa_<ir::linalg::SoftmaxOp>()) { return true; }
  return false;
}

bool LLMQuantRecipeSoftmaxPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) {
  return shareQuantSpecSingleInputToSingleOutputAndSetOpQuantAnnoAttr(writer.getContext(),
                                                                      node->cast_<ir::linalg::LinalgIROp>());
}

//===----------------------------------------------------------------------===//
// Linear Pattern
//===----------------------------------------------------------------------===//
bool LLMQuantRecipeLinearPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  if (op->isa_<ir::linalg::LinearOp>()) { return true; }
  return false;
}

bool LLMQuantRecipeLinearPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) {
  // TODO: Implement linear pattern rewrite logic
  return true;
}

//===----------------------------------------------------------------------===//
// View Pattern
//===----------------------------------------------------------------------===//
bool LLMQuantRecipeViewPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  MLLM_RETURN_FALSE_IF_NOT(op->isa_<ir::linalg::ViewOp>());
  return true;
}

bool LLMQuantRecipeViewPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) {
  // View op share Quant Spec

  auto view_op = node->cast_<ir::linalg::ViewOp>();
  auto annotation_attr = writer.create<ir::linalg::LinalgIRQuantizatonAnnotationAttr>();

  if (view_op->inputs().front()->getAttr("quant_recipe"))
  // Pass by
  {
    auto quant_spec = view_op->inputs().front()->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>();
    view_op->outputs().front()->setAttr("quant_recipe", quant_spec);
    annotation_attr->annotation_.inputs.emplace_back(quant_spec->spec_);
    annotation_attr->annotation_.outputs.emplace_back(quant_spec->spec_);
    view_op->setAttr("quant_recipe", annotation_attr);
  } else
  // Using Raw dtype, shared inputs and outputs
  {
    auto input = view_op->inputs().front()->cast_<ir::tensor::TensorValue>();
    auto quant_spec =
        writer.create<ir::linalg::LinalgIRQuantizatonSpecAttr>(ir::linalg::QuantizationSpecRaw::create(input->tensor_.dtype()));
    annotation_attr->annotation_.inputs.emplace_back(quant_spec->spec_);
    annotation_attr->annotation_.outputs.emplace_back(quant_spec->spec_);
    view_op->inputs().front()->setAttr("quant_recipe", quant_spec);
    view_op->outputs().front()->setAttr("quant_recipe", quant_spec);
    view_op->setAttr("quant_recipe", annotation_attr);
  }

  return true;
}

//===----------------------------------------------------------------------===//
// Embedding Pattern
//===----------------------------------------------------------------------===//
bool LLMQuantRecipeEmbeddingPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  // Pattern:
  //
  // embedding(op) -> quantize(op)
  MLLM_RETURN_FALSE_IF_NOT(op->isa_<ir::linalg::EmbeddingOp>());
  MLLM_RETURN_FALSE_IF_NOT(op->nextOp());
  MLLM_RETURN_FALSE_IF_NOT(op->nextOp()->isa_<ir::linalg::CastTypeOp>());

  // Already marked.
  MLLM_RETURN_FALSE_IF(op->getAttr("quant_recipe"));
  MLLM_RETURN_FALSE_IF(op->nextOp()->getAttr("quant_recipe"));

  return true;
}

bool LLMQuantRecipeEmbeddingPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) {
  auto embedding_op = node->cast_<ir::linalg::EmbeddingOp>();
  auto quantize_op = embedding_op->nextOp()->cast_<ir::linalg::CastTypeOp>();

  auto annotation_attr = writer.create<ir::linalg::LinalgIRQuantizatonAnnotationAttr>();

  // Inputs to this Quantization node must be raw type.
  {
    auto i_type = quantize_op->inputs().front()->cast_<ir::tensor::TensorValue>()->tensor_.dtype();
    MLLM_RT_ASSERT(i_type == kFloat32 || i_type == kFloat16);
    auto i_quant_spec = ir::linalg::QuantizationSpecRaw::create(i_type);
    annotation_attr->annotation_.inputs.emplace_back(i_quant_spec);
    quantize_op->inputs().front()->setAttr("quant_recipe",
                                           writer.create<ir::linalg::LinalgIRQuantizatonSpecAttr>(i_quant_spec));
  }

  // Outputs to this Quantization node must be int8 or int16
  {
    auto o_type = quantize_op->outputs().front()->cast_<ir::tensor::TensorValue>()->tensor_.dtype();
    ir::linalg::QuantizationSpec::ptr_t o_quant_spec = nullptr;
    switch (o_type) {
      case kInt8PerTensorSym: {
        o_quant_spec = ir::linalg::QuantizationSpecSymPerTensor::create(-128, 127, kInt8, kFloat32, Tensor::nil());
        break;
      }
      case kUInt8PerTensorSym: {
        o_quant_spec = ir::linalg::QuantizationSpecSymPerTensor::create(0, 255, kUInt8, kFloat32, Tensor::nil());
        break;
      }
      case kInt16PerTensorSym: {
        o_quant_spec = ir::linalg::QuantizationSpecSymPerTensor::create(-32768, 32767, kInt16, kFloat32, Tensor::nil());
        break;
      }
      case kUInt16PerTensorSym: {
        o_quant_spec = ir::linalg::QuantizationSpecSymPerTensor::create(0, 65535, kUInt16, kFloat32, Tensor::nil());
        break;
      }
      default: {
        NYI("Only support [uint16, int16, uint8, int8], [sym] for now.");
      }
    }

    annotation_attr->annotation_.outputs.emplace_back(o_quant_spec);
    quantize_op->outputs().front()->setAttr("quant_recipe",
                                            writer.create<ir::linalg::LinalgIRQuantizatonSpecAttr>(o_quant_spec));
  }

  // Attach to quantize node
  node->nextOp()->setAttr("quant_recipe", annotation_attr);

  return true;
}

//===----------------------------------------------------------------------===//
// Qwen3 Attention Pattern
//===----------------------------------------------------------------------===//
bool LLMQuantRecipeQwen3AttentionPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  auto linear_name_match = [](const ir::linalg::LinearOp::ptr_t& linear_op, const std::string& match) -> bool {
    auto name = linear_op->getAOp()->getName();
    return name.ends_with(match);
  };

  // Q, K, V Liner
  MLLM_RETURN_FALSE_IF_NOT(op->isa_<ir::linalg::LinearOp>());
  MLLM_RETURN_FALSE_IF_NOT(linear_name_match(op->cast_<ir::linalg::LinearOp>(), "q_proj"));
  auto cur_op = op->nextOp();
  MLLM_RETURN_FALSE_IF_NOT(cur_op->isa_<ir::linalg::LinearOp>());
  MLLM_RETURN_FALSE_IF_NOT(linear_name_match(cur_op->cast_<ir::linalg::LinearOp>(), "k_proj"));
  cur_op = cur_op->nextOp();
  MLLM_RETURN_FALSE_IF_NOT(cur_op->isa_<ir::linalg::LinearOp>());
  MLLM_RETURN_FALSE_IF_NOT(linear_name_match(cur_op->cast_<ir::linalg::LinearOp>(), "v_proj"));

  bool find_o_proj = false;
  do {
    cur_op = cur_op->nextOp();
    if (cur_op && cur_op->isa_<ir::linalg::LinearOp>() && linear_name_match(cur_op->cast_<ir::linalg::LinearOp>(), "o_proj")) {
      find_o_proj = true;
      break;
    }
  } while (cur_op);
  MLLM_RETURN_FALSE_IF_NOT(find_o_proj);

  return true;
}

bool LLMQuantRecipeQwen3AttentionPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& node) { return true; }

//===----------------------------------------------------------------------===//
// LLMQuantRecipePass
//===----------------------------------------------------------------------===//
LLMQuantRecipePass::LLMQuantRecipePass() {
  auto config = AOTCompileContext::getInstance().getConfig();
  // Register all patterns
  addPattern(LLMQuantRecipeIndexPattern::create(), "index", 0);
  addPattern(LLMQuantRecipeElementwisePattern::create(), "elementwise", 0);
  addPattern(LLMQuantRecipeTransposePattern::create(), "transpose", 0);
  addPattern(LLMQuantRecipeConcatPattern::create(), "concat", 0);
  addPattern(LLMQuantRecipeRepeatPattern::create(), "repeat", 0);
  addPattern(LLMQuantRecipeMatMulPattern::create(), "matmul", 0);
  addPattern(LLMQuantRecipeEqualPattern::create(), "equal", 0);
  addPattern(LLMQuantRecipeWherePattern::create(), "where", 0);
  addPattern(LLMQuantRecipeSoftmaxPattern::create(), "softmax", 0);
  addPattern(LLMQuantRecipeLinearPattern::create(), "linear", 0);
  addPattern(LLMQuantRecipeEmbeddingPattern::create(), "embedding", 0);
  addPattern(LLMQuantRecipeViewPattern::create(), "view", 0);
  if (config["quant_recipe"]["builtin_llm_pass"]["model"] == "qwen3") {
    addPattern(LLMQuantRecipeQwen3AttentionPattern::create(), "qwen3_attention", 100);
  }
}

uint8_t LLMQuantRecipePass::run(const ir::node_ptr_t& op) {
  // The top op should be ModuleOp
  MLLM_RT_ASSERT(op->isa_<ir::ModuleOp>());

  auto module_op = op->cast_<ir::ModuleOp>();
  auto writer = ir::IRWriter(getCtx(), module_op->getTopRegion());
  ir::graph::CallGraphOp::ptr_t call_main_graph_op = nullptr;
  writer.walk<ir::graph::CallGraphOp>(
      [&](ir::IRWriter& /*writer*/, const ir::graph::CallGraphOp::ptr_t& call_op) -> ir::IRWriter::WalkResult {
        MLLM_RT_ASSERT_EQ(call_main_graph_op, nullptr);
        call_main_graph_op = call_op;
        return ir::IRWriter::WalkResult::WALK_CONTINUE;
      });

  if (call_main_graph_op == nullptr) { return ir::PASS_RET_SUCCESS; }

  auto main_graph = getCtx()->lookupSymbolTable(call_main_graph_op->getSymbolAttr()->str())->cast_<ir::graph::SubGraphOp>();
  MLLM_RT_ASSERT(main_graph != nullptr);
  {
    auto rw = ir::IRWriter(getCtx(), main_graph->getTopRegion());

    // Sort patterns by priority in descending order
    auto sorted_patterns = pattern_with_priority_;
    std::sort(sorted_patterns.begin(), sorted_patterns.end(),
              [](const std::pair<int, ir::Pattern::ptr_t>& a, const std::pair<int, ir::Pattern::ptr_t>& b) {
                return a.first > b.first;
              });

    // TODO we need to make this loop recursive on nested graphs
    rw.walk<ir::linalg::LinalgIROp>(
        [&](ir::IRWriter& iw, const ir::linalg::LinalgIROp::ptr_t& linalg_op) -> ir::IRWriter::WalkResult {
          if (!linalg_op->getAttr("quant_recipe")) {
            for (auto& pattern : sorted_patterns) {
              if (pattern.second->isMatch(linalg_op)) {
                MLLM_RT_ASSERT(pattern.second->rewrite(iw, linalg_op));
                break;
              }
            }
          }
          return ir::IRWriter::WalkResult::WALK_CONTINUE;
        });
  }

  return 0;
}

void LLMQuantRecipePass::addPattern(const ir::Pattern::ptr_t& p, const std::string& name, int priority) {
  patterns_.insert({name, p});
  pattern_with_priority_.emplace_back(priority, p);
}

ir::Pass::ptr_t createLLMQuantRecipePass() { return std::make_shared<LLMQuantRecipePass>(); }

}  // namespace mllm::qnn::aot
