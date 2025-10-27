// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/PadOp.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::aops {

PadOp::PadOp(const PadOpOptions& options) : BaseOp(OpTypes::kPad), options_(options) {}

void PadOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void PadOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::PadOp>(shared_from_this(), i_irs, o_irs);
}

void PadOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("PadOp::forward is not implemented in the base class");
}

void PadOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  if (inputs.empty()) { throw std::invalid_argument("PadOp::reshape: inputs empty"); }

  const auto& i = inputs[0];
  const int in_dims = i.shape().size();
  const auto& pad = options_.pad;  // [dim_last_low, dim_last_high, ...]

  std::vector<int32_t> out_shape = i.shape();
  for (int d = 0; d < in_dims; ++d) {
    int idx = (in_dims - 1 - d) * 2;
    int32_t low = (idx < pad.size()) ? pad[idx] : 0;
    int32_t high = (idx + 1 < pad.size()) ? pad[idx + 1] : 0;
    out_shape[d] += low + high;
  }

  outputs.emplace_back(Tensor::empty(out_shape, i.dtype(), i.device()));
}

void PadOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { BaseOp::setup(inputs, outputs); }

}  // namespace mllm::aops
