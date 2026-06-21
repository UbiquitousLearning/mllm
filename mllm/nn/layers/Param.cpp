// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/ParamOp.hpp"
#include "mllm/nn/layers/Param.hpp"

namespace mllm::nn {

Param::Param() : Layer(OpTypes::kParam, aops::ParamOpOptions{}) {}

Param::Param(const aops::ParamOpOptions& options) : Layer(OpTypes::kParam, options) {}

Param::Param(const std::string& name, const Tensor::shape_t& shape)
    : Layer(OpTypes::kParam, aops::ParamOpOptions{
                                 .name = name,
                                 .shape = shape,
                             }) {}

[[nodiscard]] Tensor Param::weight() const {
  return std::static_pointer_cast<aops::ParamOp>(impl()->getInstancedOp())->weight();
}

}  // namespace mllm::nn
