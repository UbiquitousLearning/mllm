// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/Conv2DOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/graph/Op.hpp"
#include "mllm/compile/ir/tensor/Op.hpp"

namespace mllm::aops {

Conv2DOp::Conv2DOp(const Conv2DOpOptions& options) : BaseOp(OpTypes::kConv2D), options_(options) {}

void Conv2DOp::load(const ParameterFile::ptr_t& ploader) {
  switch (ploader->version()) {
    case ModelFileVersion::kV1: {
      weight_ = ploader->pull(getName() + ".weight");
      if (options_.bias) { bias_ = ploader->pull(getName() + ".bias"); }
      weight_ = weight_.view({
          options_.out_channels,
          options_.in_channels,
          options_.kernel_size[0],
          options_.kernel_size[1],
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
    default: NYI("Unsupported model file version");
  }
}

void Conv2DOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;

  // Register Params
  auto init_region = ir_ctx->lookupSymbolTable("init")->cast_<ir::graph::SubGraphOp>()->getTopRegion();
  if (weight_ && !ir_ctx->lookupSymbolTable(getName() + ".weight")) {
    ir::IRWriterGuard guard(ir_ctx, init_region);
    ir_ctx->create<ir::tensor::RegisterOp>(ir_ctx->create<ir::tensor::TensorValue>(weight_));
  }
  if (options_.bias && bias_ && !ir_ctx->lookupSymbolTable(getName() + ".bias")) {
    ir::IRWriterGuard guard(ir_ctx, init_region);
    ir_ctx->create<ir::tensor::RegisterOp>(ir_ctx->create<ir::tensor::TensorValue>(bias_));
  }

  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  auto _op = ir_ctx->create<ir::linalg::Conv2DOp>(shared_from_this(), i_irs, o_irs);
}

void Conv2DOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("Conv2DOp::forward not implemented in aops base.");
}

void Conv2DOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& i = inputs[0];
  const auto& ishape = i.shape();

  // Input must be 4D: [batch, channels, height, width]
  if (ishape.size() != 4) {
    MLLM_ERROR_EXIT(ExitCode::kCoreError, "Conv2DOp expects 4D input, got {} D", ishape.size());
    outputs.emplace_back(Tensor::empty(i.shape(), i.dtype(), i.device()));
    return;
  }

  const int batch = ishape[0];
  const int in_channels = ishape[1];  // channel axis
  const int in_height = ishape[2];    // height axis
  const int in_width = ishape[3];     // width axis

  MLLM_RT_ASSERT_EQ(in_channels, options_.in_channels);

  // Retrieve convolution parameters from options_
  // For Conv2D, kernel_size should be [kh, kw]
  const auto& kernel = options_.kernel_size;
  const auto& stride = options_.stride;      // [sh, sw]
  const auto& padding = options_.padding;    // [ph, pw] if available
  const auto& dilation = options_.dilation;  // [dh, dw] if available
  const int out_channels = options_.out_channels;
  MLLM_RT_ASSERT_EQ(kernel.size(), 2);
  MLLM_RT_ASSERT_EQ(stride.size(), 2);
  MLLM_RT_ASSERT_EQ(padding.size(), 2);
  MLLM_RT_ASSERT_EQ(dilation.size(), 2);

  // Output shape calculation for Conv2D
  auto out_shape = [](int dim_size, int kernel_size, int stride_size, int padding_size, int dilation_size) -> int32_t {
    const int dilated_kernel_size = dilation_size * (kernel_size - 1) + 1;
    return ((dim_size + 2 * padding_size - dilated_kernel_size) / stride_size) + 1;
  };

  // Calculate output height and width
  auto h_out = out_shape(in_height, kernel[0], stride[0], padding[0], dilation[0]);
  auto w_out = out_shape(in_width, kernel[1], stride[1], padding[1], dilation[1]);

  // Output shape: [batch, out_channels, h_out, w_out]
  auto new_shape = std::vector<int32_t>{batch, out_channels, h_out, w_out};

  outputs.emplace_back(Tensor::empty(new_shape, i.dtype(), i.device()));
}

void Conv2DOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { BaseOp::setup(inputs, outputs); }

ParameterFile::ptr_t Conv2DOp::getParams() {
  auto p = ParameterFile::create();
  p->push(getName() + ".weight", weight_);
  if (options_.bias) { p->push(getName() + ".bias", bias_); }
  return p;
}

}  // namespace mllm::aops
