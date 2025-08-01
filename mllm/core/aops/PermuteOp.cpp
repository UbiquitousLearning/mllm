// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/PermuteOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::aops {

PermuteOp::PermuteOp(const PermuteOpOptions& options) : BaseOp(OpTypes::kPermute), options_(options) {}

void PermuteOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void PermuteOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::PermuteOp>(shared_from_this(), i_irs, o_irs);
}

void PermuteOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("PermuteOp::forward not implemented in aops base.");
}

void PermuteOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];
  auto& output = outputs[0];

  auto input_shape = input.shape();
  std::vector<int32_t> output_shape(input_shape.size());

  for (int i = 0; i < input_shape.size(); ++i) { output_shape[i] = input_shape[options_.axis[i]]; }

  outputs.emplace_back(Tensor::empty(output_shape, input.dtype(), input.device()));
}

void PermuteOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { BaseOp::setup(inputs, outputs); }

}  // namespace mllm::aops