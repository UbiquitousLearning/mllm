// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/MultimodalRoPEOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::aops {

MultimodalRoPEOp::MultimodalRoPEOp(const MultimodalRoPEOpOptions& options)
    : BaseOp(OpTypes::kMultimodalRoPE), options_(options) {}

void MultimodalRoPEOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void MultimodalRoPEOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::MultimodalRoPEOp>(shared_from_this(), i_irs, o_irs);
}

void MultimodalRoPEOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("MultimodalRoPEOp::forward not implemented in aops base.");
}

void MultimodalRoPEOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& i = inputs[0];
  outputs.emplace_back(Tensor::empty(i.shape(), i.dtype(), i.device()));
}

void MultimodalRoPEOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  BaseOp::setup(inputs, outputs);
}

}  // namespace mllm::aops
