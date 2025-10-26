// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/LayerNorm2DOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/graph/Op.hpp"
#include "mllm/compile/ir/tensor/Op.hpp"

namespace mllm::aops {

LayerNorm2DOp::LayerNorm2DOp(const LayerNorm2DOpOptions& options) : BaseOp(OpTypes::kLayerNorm2D), options_(options) {}

void LayerNorm2DOp::load(const ParameterFile::ptr_t& ploader) {
  weight_ = ploader->pull(getName() + ".weight");
  weight_ = weight_.view({options_.num_channels});
  bias_ = ploader->pull(getName() + ".bias");
  bias_ = bias_.view({options_.num_channels});
}

void LayerNorm2DOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;

  // Register Params
  auto init_region = ir_ctx->lookupSymbolTable("init")->cast_<ir::graph::SubGraphOp>()->getTopRegion();
  if (weight_ && !ir_ctx->lookupSymbolTable(getName() + ".weight")) {
    ir::IRWriterGuard guard(ir_ctx, init_region);
    ir_ctx->create<ir::tensor::RegisterOp>(ir_ctx->create<ir::tensor::TensorValue>(weight_));
  }
  if (bias_ && !ir_ctx->lookupSymbolTable(getName() + ".bias")) {
    ir::IRWriterGuard guard(ir_ctx, init_region);
    ir_ctx->create<ir::tensor::RegisterOp>(ir_ctx->create<ir::tensor::TensorValue>(bias_));
  }

  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::LayerNorm2DOp>(shared_from_this(), i_irs, o_irs);
}

void LayerNorm2DOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("LayerNorm2DOp::forward not implemented in aops base.");
}

void LayerNorm2DOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& i = inputs[0];
  outputs.emplace_back(Tensor::empty(i.shape(), i.dtype(), i.device()));
}

void LayerNorm2DOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { BaseOp::setup(inputs, outputs); }

ParameterFile::ptr_t LayerNorm2DOp::getParams() {
  auto p = ParameterFile::create();
  p->push(getName() + ".weight", weight_);
  p->push(getName() + ".bias", bias_);
  return p;
}

}  // namespace mllm::aops
