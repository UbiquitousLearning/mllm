// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include "mllm/core/DataTypes.hpp"
#include "mllm/utils/CPUArchHelper.hpp"

#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)

namespace mllm::cpu::x86 {

// Safe sofmax for fp32. Not optimized for stride!=1 situation. When stride is set to 1, this
// function will utilize vexp1_fast_fp32 method to accelerate exp computation. This function not
// required (len % K == 0), any length is acceptable.
void softmax_v1_fp32(const mllm_fp32_t* __restrict X, mllm_fp32_t* __restrict Y, int len, int stride, int thread_count);

}  // namespace mllm::cpu::x86

#endif
