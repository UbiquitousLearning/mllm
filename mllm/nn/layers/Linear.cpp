/**
 * @file Linear.cpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-24
 *
 */
#include "mllm/nn/layers/Linear.hpp"

namespace mllm::nn {

Linear::Linear() : Layer(OpTypes::kLinear, aops::LinearOptions{}) {}

Linear::Linear(int32_t in_channels, int32_t out_channels, bool bias, aops::LinearImplTypes impl_type)
    : Layer(
          OpTypes::kLinear,
          aops::LinearOptions{.in_channels = in_channels, .out_channels = out_channels, .bias = bias, .impl_type = impl_type}) {
}

Linear::Linear(const aops::LinearOptions& options) : Layer(OpTypes::kLinear, options) {}

Tensor Linear::weight() const { return Tensor(impl()->refParams()->pull(impl()->getAbsoluteName() + ".weight")); }

Tensor Linear::bias() const {
  auto bias_name = impl()->getAbsoluteName() + ".bias";
  if (!impl()->refParams()->has(bias_name)) {
    MLLM_ERROR("There is no bias in the linear layer: {}", impl()->getAbsoluteName());
    return Tensor::nil();
  }
  return Tensor(impl()->refParams()->pull(bias_name));
}

}  // namespace mllm::nn
