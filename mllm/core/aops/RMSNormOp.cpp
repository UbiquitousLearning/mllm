// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/RMSNormOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::aops {

RMSNormOp::RMSNormOp(const RMSNormOpOptions& options) : BaseOp(OpTypes::kRMSNorm), options_(options) {}

void RMSNormOp::load(const ParameterFile::ptr_t& ploader) {
  switch (ploader->version()) {
    case ModelFileVersion::kV1: {
      weight_ = ploader->pull(getName() + ".weight");
      // FIXME: need reshape.
      break;
    }
    case ModelFileVersion::kUserTemporary:
    case ModelFileVersion::kV2: {
      weight_ = ploader->pull(getName() + ".weight");
      break;
    }
    default: NYI("Unsupported model file version")
  }
}

void RMSNormOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::RMSNormOp>(shared_from_this(), i_irs, o_irs);
}

void RMSNormOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("RMSNormOp::forward not implemented in aops base.");
}

void RMSNormOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  if (options_.isInplace()) {
    outputs.emplace_back(inputs[0]);
  } else {
    const auto& i = inputs[0];
    outputs.emplace_back(Tensor::empty(i.shape(), i.dtype(), i.device()));
  }
}

void RMSNormOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  if (!options_.isInplace()) { BaseOp::setup(inputs, outputs); }
}

ParameterFile::ptr_t RMSNormOp::getParams() {
  auto p = ParameterFile::create();
  p->push(getName() + ".weight", weight_);
  return p;
}

}  // namespace mllm::aops
