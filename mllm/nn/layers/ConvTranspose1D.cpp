// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/nn/layers/ConvTranspose1D.hpp"

namespace mllm::nn {

ConvTranspose1D::ConvTranspose1D() : Layer(OpTypes::kConvTranspose1D, aops::ConvTranspose1DOpOptions{}) {}

ConvTranspose1D::ConvTranspose1D(int32_t in_channels, int32_t out_channels, int32_t kernel_size, int32_t stride_size,
                                 int32_t padding, int32_t output_padding, int32_t dilation, int32_t groups, bool bias)
    : Layer(OpTypes::kConvTranspose1D, aops::ConvTranspose1DOpOptions{.in_channels = in_channels,
                                                                      .out_channels = out_channels,
                                                                      .kernel_size = kernel_size,
                                                                      .stride = stride_size,
                                                                      .padding = padding,
                                                                      .output_padding = output_padding,
                                                                      .dilation = dilation,
                                                                      .groups = groups,
                                                                      .bias = bias}) {}

ConvTranspose1D::ConvTranspose1D(const aops::ConvTranspose1DOpOptions& options) : Layer(OpTypes::kConvTranspose1D, options) {}

Tensor ConvTranspose1D::weight() const {
  return std::static_pointer_cast<aops::ConvTranspose1DOp>(impl()->getInstancedOp())->weight();
}

Tensor ConvTranspose1D::bias() const {
  return std::static_pointer_cast<aops::ConvTranspose1DOp>(impl()->getInstancedOp())->bias();
}

}  // namespace mllm::nn
