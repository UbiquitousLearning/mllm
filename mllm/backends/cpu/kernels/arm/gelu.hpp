// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/utils/CPUArchHelper.hpp"

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)

#include <arm_neon.h>
#include <cstdint>

namespace mllm::cpu::arm {

void gelu_fp32(float* __restrict__ Z, const float* __restrict__ X, int32_t N, int thread_cnt);

void quick_gelu_fp32(float* __restrict__ Z, const float* __restrict__ X, int32_t N, int thread_cnt);

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
void gelu_fp16(float16_t* __restrict__ Z, const float16_t* __restrict__ X, int32_t N, int thread_cnt);

void quick_gelu_fp16(float16_t* __restrict__ Z, const float16_t* __restrict__ X, int32_t N, int thread_cnt);
#endif

}  // namespace mllm::cpu::arm

#endif