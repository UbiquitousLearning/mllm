#include "mllm/nn/layers/Conv1D.hpp"
#include "mllm/core/aops/Conv1DOp.hpp"
#include "mllm/nn/Layer.hpp"

namespace mllm::nn {

Conv1D::Conv1D() : Layer(OpTypes::kConv1D, aops::Conv1DOpOptions{}) {}

Conv1D::Conv1D(int32_t in_channels, int32_t out_channels, int32_t kernel_size, int32_t stride_size, int32_t padding,
               int32_t dilation, int32_t groups, bool bias)
    : Layer(OpTypes::kConv1D, aops::Conv1DOpOptions{.in_channels = in_channels,
                                                    .out_channels = out_channels,
                                                    .kernel_size = kernel_size,
                                                    .stride = stride_size,
                                                    .bias = bias,
                                                    .padding = padding,
                                                    .groups = groups,
                                                    .dilation = dilation}) {}

Conv1D::Conv1D(const aops::Conv1DOpOptions& options) : Layer(OpTypes::kConv1D, options) {}

Tensor Conv1D::weight() const { return std::static_pointer_cast<aops::Conv1DOp>(impl()->getInstancedOp())->weight(); }

Tensor Conv1D::bias() const { return std::static_pointer_cast<aops::Conv1DOp>(impl()->getInstancedOp())->bias(); }

}  // namespace mllm::nn