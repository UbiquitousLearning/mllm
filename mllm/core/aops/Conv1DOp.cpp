// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/Conv1DOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/graph/Op.hpp"
#include "mllm/compile/ir/tensor/Op.hpp"

namespace mllm::aops {

Conv1DOp::Conv1DOp(const Conv1DOpOptions& options) : BaseOp(OpTypes::kConv1D), options_(options) {}

void Conv1DOp::load(const ParameterFile::ptr_t& ploader) {
  switch (ploader->version()) {
    case ModelFileVersion::kV1: {
      weight_ = ploader->pull(getName() + ".weight");
      if (options_.bias) { bias_ = ploader->pull(getName() + ".bias"); }
      weight_ = weight_.view({options_.out_channels, options_.in_channels / options_.groups, options_.kernel_size});
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

void Conv1DOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;

  // Register Params
  if (weight_ && !ir_ctx->lookupSymbolTable(getName() + ".weight")) {
    ir::IRWriterGuard guard(ir_ctx, ir_ctx->lookupSymbolTable("init")->cast_<ir::graph::SubGraphOp>()->getTopRegion());
    ir_ctx->create<ir::tensor::RegisterOp>(ir_ctx->create<ir::tensor::TensorValue>(weight_));
    if (options_.bias) { ir_ctx->create<ir::tensor::RegisterOp>(ir_ctx->create<ir::tensor::TensorValue>(bias_)); }
  }

  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::Conv1DOp>(shared_from_this(), i_irs, o_irs);
}

void Conv1DOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("Conv1DOp::forward not implemented in aops base.");
}

void Conv1DOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& i = inputs[0];
  const auto& ishape = i.shape();

  // Input must be 3D: [batch, channels, sequence]
  if (ishape.size() != 3) {
    MLLM_ERROR_EXIT(ExitCode::kCoreError, "Conv1DOp expects 3D input, got {} D", ishape.size());
    outputs.emplace_back(Tensor::empty(i.shape(), i.dtype(), i.device()));
    return;
  }

  // QNN May not support batch. You can write passes to eliminate batch dim.
  const int batch = ishape[0];
  const int in_channels = ishape[1];  // channel axis
  const int sequence = ishape[2];     // sequence axis

  MLLM_RT_ASSERT_EQ(in_channels, options_.in_channels);

  // Check groups parameter
  MLLM_RT_ASSERT_EQ(in_channels % options_.groups, 0);
  MLLM_RT_ASSERT_EQ(options_.out_channels % options_.groups, 0);

  // Retrieve convolution parameters from options_
  const int kernel_size = options_.kernel_size;
  const int stride = options_.stride;
  const int out_channels = options_.out_channels;
  const int dilation = options_.dilation;

  // Calculate output shape considering dilation
  auto out_shape = [](int dim_size, int kernel_size, int stride_size, int padding_size, int dilation_size) -> int32_t {
    // Standard convolution output size formula with dilation
    return ((dim_size + 2 * padding_size - dilation_size * (kernel_size - 1) - 1) / stride_size) + 1;
  };

  auto seq_out = out_shape(sequence, kernel_size, stride, options_.padding, dilation);

  auto new_shape = std::vector<int32_t>{batch, out_channels, seq_out};

  outputs.emplace_back(Tensor::empty(new_shape, i.dtype(), i.device()));
}

void Conv1DOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { BaseOp::setup(inputs, outputs); }

ParameterFile::ptr_t Conv1DOp::getParams() {
  auto p = ParameterFile::create();
  p->push(getName() + ".weight", weight_);
  if (options_.bias) { p->push(getName() + ".bias", bias_); }
  return p;
}

}  // namespace mllm::aops