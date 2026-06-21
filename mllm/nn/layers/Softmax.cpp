// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/SoftmaxOp.hpp"
#include "mllm/nn/layers/Softmax.hpp"

namespace mllm::nn {

Softmax::Softmax() : Layer(OpTypes::kSoftmax, aops::SoftmaxOpOptions{}) {}

Softmax::Softmax(const aops::SoftmaxOpOptions& options) : Layer(OpTypes::kSoftmax, options) {}

Softmax::Softmax(int32_t dim) : Layer(OpTypes::kSoftmax, aops::SoftmaxOpOptions{.axis = dim}) {}

}  // namespace mllm::nn
