// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/LayerNormOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/graph/Op.hpp"
#include "mllm/compile/ir/tensor/Op.hpp"

namespace mllm::aops {

LayerNormOp::LayerNormOp(const LayerNormOpOptions& options) : BaseOp(OpTypes::kLayerNorm), options_(options) {}

void LayerNormOp::load(const ParameterFile::ptr_t& ploader) {
  if (options_.elementwise_affine) {
    weight_ = ploader->pull(getName() + ".weight");
    weight_ = weight_.view(options_.normalized_shape);
  }
  if (options_.bias) {
    bias_ = ploader->pull(getName() + ".bias");
    bias_ = bias_.view(options_.normalized_shape);
  }
}

void LayerNormOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;

  // Register Params
  if (weight_ && !ir_ctx->lookupSymbolTable(getName() + ".weight")) {
    ir::IRWriterGuard guard(ir_ctx, ir_ctx->lookupSymbolTable("init")->cast_<ir::graph::SubGraphOp>()->getTopRegion());
    ir_ctx->create<ir::tensor::RegisterOp>(ir_ctx->create<ir::tensor::TensorValue>(weight_));
    if (options_.bias) { ir_ctx->create<ir::tensor::RegisterOp>(ir_ctx->create<ir::tensor::TensorValue>(bias_)); }
  }

  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::LayerNormOp>(shared_from_this(), i_irs, o_irs);
}

void LayerNormOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("LayerNormOp::forward not implemented in aops base.");
}

void LayerNormOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& i = inputs[0];
  outputs.emplace_back(Tensor::empty(i.shape(), i.dtype(), i.device()));
}

void LayerNormOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { BaseOp::setup(inputs, outputs); }

ParameterFile::ptr_t LayerNormOp::getParams() {
  auto p = ParameterFile::create();
  p->push(getName() + ".weight", weight_);
  if (options_.bias) { p->push(getName() + ".bias", bias_); }
  return p;
}

}  // namespace mllm::aops
