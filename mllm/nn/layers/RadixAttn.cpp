// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/RadixAttnOp.hpp"
#include "mllm/nn/layers/RadixAttn.hpp"

namespace mllm::nn {

RadixAttn::RadixAttn() : Layer(OpTypes::kRadixAttn, aops::RadixAttnOpOptions{}) {}

RadixAttn::RadixAttn(const aops::RadixAttnOpOptions& options) : Layer(OpTypes::kRadixAttn, options) {}

RadixAttn::RadixAttn(int32_t H_Q, int32_t H_KV)
    : Layer(OpTypes::kRadixAttn, aops::RadixAttnOpOptions{.H_Q = H_Q, .H_KV = H_KV}) {}

}  // namespace mllm::nn
