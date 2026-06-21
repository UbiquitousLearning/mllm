// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/qnn/QNNUtils.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/backends/qnn/aot/QnnWrappersAPI.hpp"
#include "mllm/backends/qnn/aot/visitor/Conv2D.hpp"
#include "mllm/backends/qnn/aot/passes/AOTCompileContext.hpp"
#include "mllm/core/aops/Conv2DOp.hpp"

namespace mllm::qnn::aot {

bool QnnAOTConv2DPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  return op->isa_<mllm::ir::linalg::Conv2DOp>() && (op->getAttr("using_qnn") != nullptr);
}

bool QnnAOTConv2DPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& op) {
  auto env = AOTCompileContext::getInstance().getEnv();

  auto linear_op = op->cast_<mllm::ir::linalg::Conv2DOp>();
  if (!linear_op) {
    MLLM_ERROR("Failed to cast to linalg::Conv2DOp");
    return false;
  }

  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_graph_name"));
  auto qnn_graph_name = op->getAttr("qnn_graph_name")->cast_<ir::StrAttr>()->data();
  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_context_name"));
  auto qnn_context_name = op->getAttr("qnn_context_name")->cast_<ir::StrAttr>()->data();

  auto input = op->inputs().front()->cast_<ir::tensor::TensorValue>();
  auto output = op->outputs().front()->cast_<ir::tensor::TensorValue>();

  auto base_op = linear_op->getAOp();
  auto real_linear_op = dynamic_cast<mllm::aops::Conv2DOp*>(base_op);
  if (!real_linear_op) {
    MLLM_ERROR("Failed to cast BaseOp to mllm::aops::Conv2DOp");
    return false;
  }

  // Retrieve weight from symbol table
  auto weight_val = writer.getContext()
                        ->lookupSymbolTable(base_op->getName() + ".weight")
                        ->outputs()
                        .front()
                        ->cast_<ir::tensor::TensorValue>();

  // Create QNN Conv2D Op
  auto qnn_op_node = QnnAOTNodeOperation::create("Conv2d");
  qnn_op_node->setPackageName("qti.aisw");

  qnn_op_node->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, input))
      ->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, weight_val, true));

  // Handle Bias
  if (real_linear_op->options().bias) {
    auto bias_val = writer.getContext()
                        ->lookupSymbolTable(base_op->getName() + ".bias")
                        ->outputs()
                        .front()
                        ->cast_<ir::tensor::TensorValue>();
    qnn_op_node->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, bias_val, true));
  }

  qnn_op_node->emplaceOutput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, output))
      ->setName(base_op->getName());

  // Add params: stride
  {
    auto stride_param =
        QNNParamTensorWrapper::create("stride", base_op->getName() + ".stride", QNN_DATATYPE_UINT_32, std::vector<uint32_t>{2});
    uint32_t* data = static_cast<uint32_t*>(stride_param->alloc());
    data[0] = 1;
    data[1] = 1;
    qnn_op_node->emplaceParamTensor(stride_param);
  }

  // Add params: pad amount
  {
    auto pad_amount_param =
        QNNParamTensorWrapper::create("pad_amount", base_op->getName() + ".pad_amount", QNN_DATATYPE_UINT_32,
                                      std::vector<uint32_t>{
                                          2,
                                          2,
                                      });
    // [[height_pad_before, height_pad_after], [width_pad_before, width_pad_after]]
    uint32_t* data = static_cast<uint32_t*>(pad_amount_param->alloc());
    data[0] = 0;
    data[1] = 0;
    data[2] = 0;
    data[3] = 0;
    qnn_op_node->emplaceParamTensor(pad_amount_param);
  }

  // Add params: dilation
  {
    auto dilation_param = QNNParamTensorWrapper::create("dilation", base_op->getName() + ".dilation", QNN_DATATYPE_UINT_32,
                                                        std::vector<uint32_t>{2});
    uint32_t* data = static_cast<uint32_t*>(dilation_param->alloc());
    data[0] = 1;
    data[1] = 1;
    qnn_op_node->emplaceParamTensor(dilation_param);
  }

  // Register this op node into one graph.
  env->captureAOTNodeOp(qnn_context_name, qnn_graph_name, qnn_op_node);

  return true;
}

}  // namespace mllm::qnn::aot
