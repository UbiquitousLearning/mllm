#include "mllm/nn/layers/Conv2D.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/Conv2DOp.hpp"

namespace mllm::nn {

Conv2D::Conv2D() : Layer(OpTypes::kConv2D, aops::Conv2DOpOptions{}) {}

Conv2D::Conv2D(int32_t in_channels, int32_t out_channels, const std::vector<int32_t>& kernel_size,
               const std::vector<int32_t>& stride_size, const std::vector<int32_t>& padding_size,
               const std::vector<int32_t>& dilation_size, bool bias, aops::Conv2DOpImplType impl_type)
    : Layer(OpTypes::kConv2D, aops::Conv2DOpOptions{
                                  .in_channels = in_channels,
                                  .out_channels = out_channels,
                                  .kernel_size = kernel_size,
                                  .stride = stride_size,
                                  .padding = padding_size,
                                  .dilation = dilation_size,
                                  .bias = bias,
                                  .impl_type = impl_type,
                              }) {}

Conv2D::Conv2D(const aops::Conv2DOpOptions& options) : Layer(OpTypes::kConv2D, options) {}

Tensor Conv2D::weight() const { return std::static_pointer_cast<aops::Conv2DOp>(impl()->getInstancedOp())->weight(); }

Tensor Conv2D::bias() const { return std::static_pointer_cast<aops::Conv2DOp>(impl()->getInstancedOp())->bias(); }

}  // namespace mllm::nn
