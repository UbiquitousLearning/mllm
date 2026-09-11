// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/SigmoidOp.hpp"
#include <stdexcept>
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::aops {

SigmoidOp::SigmoidOp(const SigmoidOpOptions& options) : BaseOp(OpTypes::kSigmoid), options_(options) {}

void SigmoidOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void SigmoidOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::SigmoidOp>(shared_from_this(), i_irs, o_irs);
}

void SigmoidOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("SigmoidOp::forward not implemented in aops base.");
}

void SigmoidOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  if (inputs.size() != 1) throw std::invalid_argument("Sigmoid expects one input");
  const auto& x = inputs[0];
  if (!options_.approximate && (x.dtype() != kFloat32 || x.device() != kCPU || !x.isContiguous()))
    throw std::invalid_argument("Accurate Sigmoid requires contiguous float32 CPU input");
  if (options_.isInplace()) {
    outputs.emplace_back(inputs[0]);
  } else {
    outputs.emplace_back(Tensor::empty(inputs[0].shape(), inputs[0].dtype(), inputs[0].device()));
  }
}

void SigmoidOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { BaseOp::setup(inputs, outputs); }

}  // namespace mllm::aops
