// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/MatMulOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::aops {

MatMulOp::MatMulOp(const MatMulOpOptions& options) : BaseOp(OpTypes::kMatMul), options_(options) {}

void MatMulOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void MatMulOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::MatMulOp>(shared_from_this(), i_irs, o_irs);
}

void MatMulOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("MatMulOp::forward not implemented in aops base.");
}

void MatMulOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto shape_a = inputs[0].shape();
  auto shape_b = inputs[1].shape();

  std::vector<int32_t> shape_c;

  // check.
  auto size_a = shape_a.size();
  auto size_b = shape_b.size();
  shape_c.reserve(size_a - 2);
  for (int i = 0; i < size_a - 2; ++i) { shape_c.push_back(shape_a[i]); }

  // transform shape.
  // MxK, KxN
  if (!options_.transpose_a && !options_.transpose_b) {
    MLLM_RT_ASSERT_EQ(shape_a[size_a - 1], shape_b[size_b - 2]);
    shape_c.push_back(shape_a[size_a - 2]);
    shape_c.push_back(shape_b[size_b - 1]);
  }
  // MxK, NxK
  else if (!options_.transpose_a && options_.transpose_b) {
    MLLM_RT_ASSERT_EQ(shape_a[size_a - 1], shape_b[size_b - 1]);
    shape_c.push_back(shape_a[size_a - 2]);
    shape_c.push_back(shape_b[size_b - 2]);
  }
  // not supported
  else {
    NYI("MatMulOp::reshape with transpose_a={} and transpose_b={} is not supported yet", options_.transpose_a,
        options_.transpose_b);
  }

  // wrap to tensor
  auto o = Tensor::empty(shape_c, inputs[0].dtype(), inputs[0].device());
  outputs.emplace_back(o);
}

void MatMulOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { BaseOp::setup(inputs, outputs); }

}  // namespace mllm::aops