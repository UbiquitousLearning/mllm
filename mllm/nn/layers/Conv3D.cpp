#include "mllm/nn/layers/Conv3D.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/Conv3DOp.hpp"

namespace mllm::nn {

Conv3D::Conv3D() : Layer(OpTypes::kConv3D, aops::Conv3DOpOptions{}) {}

Conv3D::Conv3D(int32_t in_channels, int32_t out_channels, const std::vector<int32_t>& kernel_size,
               const std::vector<int32_t>& stride_size, bool bias, aops::Conv3DOpImplType impl_type)
    : Layer(OpTypes::kConv3D, aops::Conv3DOpOptions{
                                  .in_channels = in_channels,
                                  .out_channels = out_channels,
                                  .kernel_size = kernel_size,
                                  .stride = stride_size,
                                  .bias = bias,
                                  .impl_type = impl_type,
                              }) {}

Conv3D::Conv3D(const aops::Conv3DOpOptions& options) : Layer(OpTypes::kConv3D, options) {}

Tensor Conv3D::weight() const { return std::static_pointer_cast<aops::Conv3DOp>(impl()->getInstancedOp())->weight(); }

Tensor Conv3D::bias() const { return std::static_pointer_cast<aops::Conv3DOp>(impl()->getInstancedOp())->bias(); }

}  // namespace mllm::nn
