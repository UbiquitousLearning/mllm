// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include "mllm/core/DataTypes.hpp"
#include "mllm/utils/CPUArchHelper.hpp"

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)

#include <arm_neon.h>

namespace mllm::cpu::arm {

// Safe sofmax for fp32. Not optimized for stride!=1 situation. When stride is set to 1, this
// function will utilize vexp1_fast_fp32 method to accelerate exp computation. This function not
// required (len % K == 0), any length is acceptable.
void softmax_v1_fp32(const mllm_fp32_t* __restrict X, mllm_fp32_t* __restrict Y, int len, int stride, int thread_count);

#if !defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) || !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#error This file must be compiled for AArch64, FEAT_FP16. Set -DMLLM_ARM_BACKEND_COMPILE_OPTIONS=\"-march=armv8.2-a+fp16\" in tasks yaml.
#else

void softmax_v1_fp16(const mllm_fp16_t* __restrict X, mllm_fp16_t* __restrict Y, int len, int stride, int thread_count);

#endif  // fp16

}  // namespace mllm::cpu::arm

#endif