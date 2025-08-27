// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/Conv3DOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/graph/Op.hpp"
#include "mllm/compile/ir/tensor/Op.hpp"

namespace mllm::aops {

Conv3DOp::Conv3DOp(const Conv3DOpOptions& options) : BaseOp(OpTypes::kConv3D), options_(options) {}

void Conv3DOp::load(const ParameterFile::ptr_t& ploader) {
  switch (ploader->version()) {
    case ModelFileVersion::kV1: {
      weight_ = ploader->pull(getName() + ".weight");
      if (options_.bias) { bias_ = ploader->pull(getName() + ".bias"); }
      weight_ = weight_.view({
          options_.out_channels,
          options_.in_channels,
          options_.kernel_size[0],
          options_.kernel_size[1],
          options_.kernel_size[2],
      });
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

void Conv3DOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;

  // Register Params
  if (weight_ && !ir_ctx->lookupSymbolTable(getName() + ".weight")) {
    ir::IRWriterGuard guard(ir_ctx, ir_ctx->lookupSymbolTable("init")->cast_<ir::graph::SubGraphOp>()->getTopRegion());
    ir_ctx->create<ir::tensor::RegisterOp>(ir_ctx->create<ir::tensor::TensorValue>(weight_));
    if (options_.bias) { ir_ctx->create<ir::tensor::RegisterOp>(ir_ctx->create<ir::tensor::TensorValue>(bias_)); }
  }

  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  auto _op = ir_ctx->create<ir::linalg::Conv3DOp>(shared_from_this(), i_irs, o_irs);
}

void Conv3DOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("Conv3DOp::forward not implemented in aops base.");
}

void Conv3DOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& i = inputs[0];
  const auto& ishape = i.shape();

  // Input must be 5D: [batch, channels, depth, height, width]
  if (ishape.size() != 5) {
    MLLM_ERROR_EXIT(ExitCode::kCoreError, "Conv3DOp expects 5D input, got {} D", ishape.size());
    outputs.emplace_back(Tensor::empty(i.shape(), i.dtype(), i.device()));
    return;
  }

  // QNN May not support batch. You can write passes to eliminate batch dim.
  const int batch = ishape[0];
  const int in_channels = ishape[1];  // channel axis in VLM
  const int in_depth = ishape[2];     // time axis in VLM
  const int in_height = ishape[3];    // height axis in VLM
  const int in_width = ishape[4];     // width axis in VLM

  MLLM_RT_ASSERT_EQ(in_channels, options_.in_channels);

  // Retrieve convolution parameters from options_
  const auto& kernel = options_.kernel_size;  // [kd, kh, kw]
  const auto& stride = options_.stride;       // [sd, sh, sw]
  const int out_channels = options_.out_channels;

  // FIXME we not consider padding, dilation and group right now.
  // padding is always 0,
  // dilation is always 1,

  auto out_shape = [](int dim_size, int kernel_size, int stride_size, int padding_size, int dilation_size) -> int32_t {
    // FIXME use floor.
    return ((dim_size + 2 * padding_size - dilation_size * (kernel_size - 1) - 1) / stride_size) + 1;
  };

  auto d_out = out_shape(in_depth, kernel[0], stride[0], 0, 1);
  auto h_out = out_shape(in_height, kernel[1], stride[1], 0, 1);
  auto w_out = out_shape(in_width, kernel[2], stride[2], 0, 1);

  auto new_shape = std::vector<int32_t>{batch, out_channels, d_out, h_out, w_out};

  outputs.emplace_back(Tensor::empty(new_shape, i.dtype(), i.device()));
}

void Conv3DOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { BaseOp::setup(inputs, outputs); }

ParameterFile::ptr_t Conv3DOp::getParams() {
  auto p = ParameterFile::create();
  p->push(getName() + ".weight", weight_);
  if (options_.bias) { p->push(getName() + ".bias", bias_); }
  return p;
}

}  // namespace mllm::aops
