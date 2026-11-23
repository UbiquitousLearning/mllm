// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/nn/layers/SparseSoftmaxValue.hpp"

namespace mllm::nn {

SparseSoftmaxValue::SparseSoftmaxValue() : SparseSoftmaxValue(aops::SparseSoftmaxValueOpOptions{}) {}

SparseSoftmaxValue::SparseSoftmaxValue(const aops::SparseSoftmaxValueOpOptions& options)
    : Layer(OpTypes::kSparseSoftmaxValue, options) {}

}  // namespace mllm::nn
