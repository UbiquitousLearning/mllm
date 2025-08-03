// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <memory>

#include "mllm/core/aops/RMSNormOp.hpp"
#include "mllm/nn/layers/RMSNorm.hpp"

namespace mllm::nn {

RMSNorm::RMSNorm()
    : Layer(OpTypes::kRMSNorm, aops::RMSNormOpOptions{
                                   .epsilon = 1e-5,
                                   .add_unit_offset = false,
                               }) {}

RMSNorm::RMSNorm(float epsilon, bool add_unit_offset)
    : Layer(OpTypes::kRMSNorm, aops::RMSNormOpOptions{.epsilon = epsilon, .add_unit_offset = add_unit_offset}) {}

RMSNorm::RMSNorm(const aops::RMSNormOpOptions& options) : Layer(OpTypes::kRMSNorm, options) {}

Tensor RMSNorm::weight() const { return std::static_pointer_cast<aops::RMSNormOp>(impl()->getInstancedOp())->weight(); }

}  // namespace mllm::nn
