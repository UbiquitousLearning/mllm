// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/Tensor.hpp"
#include "mllm/backends/cpu/ops/RadixAttnOp.hpp"
#include "mllm/backends/cpu/kernels/common/radix_attn/fwd_bshd.hpp"

namespace mllm::cpu {

CPURadixAttnOp::CPURadixAttnOp(const aops::RadixAttnOpOptions& options) : aops::RadixAttnOp(options) {}

void CPURadixAttnOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {}

}  // namespace mllm::cpu
