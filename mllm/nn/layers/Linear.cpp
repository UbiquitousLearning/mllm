// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <memory>
#include "mllm/core/aops/LinearOp.hpp"
#include "mllm/nn/layers/Linear.hpp"

namespace mllm::nn {

Linear::Linear() : Layer(OpTypes::kLinear, aops::LinearOpOptions{}) {}

Linear::Linear(int32_t in_channels, int32_t out_channels, bool bias, aops::LinearImplTypes impl_type)
    : Layer(OpTypes::kLinear,
            aops::LinearOpOptions{
                .in_channels = in_channels, .out_channels = out_channels, .bias = bias, .impl_type = impl_type}) {}

Linear::Linear(const aops::LinearOpOptions& options) : Layer(OpTypes::kLinear, options) {}

Tensor Linear::weight() const { return std::static_pointer_cast<aops::LinearOp>(impl()->getInstancedOp())->weight(); }

Tensor Linear::bias() const {
  auto iop = std::static_pointer_cast<aops::LinearOp>(impl()->getInstancedOp());
  return iop->bias();
}

}  // namespace mllm::nn
