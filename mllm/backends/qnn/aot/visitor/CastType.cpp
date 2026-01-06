// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/backends/qnn/aot/QnnWrappersAPI.hpp"
#include "mllm/backends/qnn/aot/visitor/CastType.hpp"
#include "mllm/backends/qnn/aot/passes/AOTCompileContext.hpp"
#include "mllm/core/aops/CastTypeOp.hpp"
#include "mllm/core/DataTypes.hpp"

namespace mllm::qnn::aot {

static bool isFloat(DataTypes dtype) { return dtype == kFloat32 || dtype == kFloat16; }

static bool isInt(DataTypes dtype) {
  return dtype == kInt8 || dtype == kInt16 || dtype == kInt32 || dtype == kUInt8 || dtype == kUInt16 || dtype == kUInt32
         || dtype == kInt8PerTensorSym || dtype == kInt8PerChannelSym || dtype == kUInt8PerTensorSym
         || dtype == kUInt8PerChannelSym || dtype == kInt16PerTensorSym || dtype == kInt16PerChannelSym
         || dtype == kUInt16PerTensorSym || dtype == kUInt16PerChannelSym || dtype == kInt8PerTensorAsy
         || dtype == kInt8PerChannelAsy || dtype == kUInt8PerTensorAsy || dtype == kUInt8PerChannelAsy
         || dtype == kInt16PerTensorAsy || dtype == kInt16PerChannelAsy || dtype == kUInt16PerTensorAsy
         || dtype == kUInt16PerChannelAsy;
}

bool QnnAOTCastTypePattern::isMatch(const mllm::ir::op_ptr_t& op) {
  return op->isa_<mllm::ir::linalg::CastTypeOp>() && (op->getAttr("using_qnn") != nullptr);
}

bool QnnAOTCastTypePattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& op) {
  auto env = AOTCompileContext::getInstance().getEnv();

  auto cast_ir_op = op->cast_<mllm::ir::linalg::CastTypeOp>();
  if (!cast_ir_op) {
    MLLM_ERROR("Failed to cast to linalg::CastTypeOp");
    return false;
  }

  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_graph_name"));
  auto qnn_graph_name = op->getAttr("qnn_graph_name")->cast_<ir::StrAttr>()->data();
  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_context_name"));
  auto qnn_context_name = op->getAttr("qnn_context_name")->cast_<ir::StrAttr>()->data();

  auto base_op = cast_ir_op->getAOp();
  auto cast_op = dynamic_cast<mllm::aops::CastTypeOp*>(base_op);
  if (!cast_op) {
    MLLM_ERROR("Failed to cast to aops::CastTypeOp");
    return false;
  }

  auto target_dtype = cast_op->options().dtype;

  auto input_val = op->inputs().front()->cast_<ir::tensor::TensorValue>();
  auto output_val = op->outputs().front()->cast_<ir::tensor::TensorValue>();

  auto input_dtype = input_val->tensor_.dtype();

  std::string qnn_op_type;
  if (isFloat(input_dtype) && isInt(target_dtype)) {
    qnn_op_type = "Quantize";
  } else if (isInt(input_dtype) && isFloat(target_dtype)) {
    qnn_op_type = "Dequantize";
  } else if (input_dtype == kFloat32 && target_dtype == kFloat16) {
    qnn_op_type = "Cast";
  } else {
    MLLM_ERROR("Unsupported CastType for QNN: {} -> {}", (int)input_dtype, (int)target_dtype);
    return false;
  }

  auto qnn_op_node = QnnAOTNodeOperation::create(qnn_op_type);
  qnn_op_node->setPackageName("qti.aisw");

  qnn_op_node->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, input_val))
      ->emplaceOutput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, output_val))
      ->setName(base_op->getName());

  env->captureAOTNodeOp(qnn_context_name, qnn_graph_name, qnn_op_node);

  return true;
}

}  // namespace mllm::qnn::aot
