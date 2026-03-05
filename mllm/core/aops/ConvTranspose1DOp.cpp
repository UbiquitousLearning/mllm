// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/ConvTranspose1DOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/graph/Op.hpp"
#include "mllm/compile/ir/tensor/Op.hpp"

namespace mllm::aops {

ConvTranspose1DOp::ConvTranspose1DOp(const ConvTranspose1DOpOptions& options)
    : BaseOp(OpTypes::kConvTranspose1D), options_(options) {}

void ConvTranspose1DOp::load(const ParameterFile::ptr_t& ploader) {
  switch (ploader->version()) {
    case ModelFileVersion::kV1: {
      weight_ = ploader->pull(getName() + ".weight");
      if (options_.bias) { bias_ = ploader->pull(getName() + ".bias"); }
      weight_ = weight_.view({options_.in_channels, options_.out_channels / options_.groups, options_.kernel_size});
      if (options_.bias) { bias_ = bias_.view({options_.out_channels}); }
      break;
    }
    case ModelFileVersion::kUserTemporary:
    case ModelFileVersion::kV2: {
      weight_ = ploader->pull(getName() + ".weight");
      if (options_.bias) { bias_ = ploader->pull(getName() + ".bias"); }
      break;
    }
    default: NYI("Unsupported model file version")
  }
}

void ConvTranspose1DOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;

  if (weight_ && !ir_ctx->lookupSymbolTable(getName() + ".weight")) {
    ir::IRWriterGuard guard(ir_ctx, ir_ctx->lookupSymbolTable("init")->cast_<ir::graph::SubGraphOp>()->getTopRegion());
    ir_ctx->create<ir::tensor::RegisterOp>(ir_ctx->create<ir::tensor::TensorValue>(weight_));
    if (options_.bias) { ir_ctx->create<ir::tensor::RegisterOp>(ir_ctx->create<ir::tensor::TensorValue>(bias_)); }
  }

  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::ConvTranspose1DOp>(shared_from_this(), i_irs, o_irs);
}

void ConvTranspose1DOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("ConvTranspose1DOp::forward not implemented in aops base.");
}

void ConvTranspose1DOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& i = inputs[0];
  const auto& ishape = i.shape();

  if (ishape.size() != 3) {
    MLLM_ERROR_EXIT(ExitCode::kCoreError, "ConvTranspose1DOp expects 3D input, got {} D", ishape.size());
    outputs.emplace_back(Tensor::empty(i.shape(), i.dtype(), i.device()));
    return;
  }

  const int batch = ishape[0];
  const int in_channels = ishape[1];
  const int sequence = ishape[2];

  MLLM_RT_ASSERT_EQ(in_channels, options_.in_channels);
  MLLM_RT_ASSERT_EQ(in_channels % options_.groups, 0);
  MLLM_RT_ASSERT_EQ(options_.out_channels % options_.groups, 0);

  const int kernel_size = options_.kernel_size;
  const int stride = options_.stride;
  const int dilation = options_.dilation;
  const int padding = options_.padding;
  const int output_padding = options_.output_padding;

  const int seq_out = (sequence - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;

  auto new_shape = std::vector<int32_t>{batch, options_.out_channels, seq_out};
  outputs.emplace_back(Tensor::empty(new_shape, i.dtype(), i.device()));
}

void ConvTranspose1DOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  BaseOp::setup(inputs, outputs);
}

ParameterFile::ptr_t ConvTranspose1DOp::getParams() {
  auto p = ParameterFile::create();
  p->push(getName() + ".weight", weight_);
  if (options_.bias) { p->push(getName() + ".bias", bias_); }
  return p;
}

}  // namespace mllm::aops
