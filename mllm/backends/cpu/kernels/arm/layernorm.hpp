// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include "mllm/core/DataTypes.hpp"
#include "mllm/utils/CPUArchHelper.hpp"

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)

#include <arm_neon.h>

namespace mllm::cpu::arm {

void layernorm_N_fp32(mllm_fp32_t* __restrict__ Z, const mllm_fp32_t* __restrict__ X, const mllm_fp32_t* __restrict__ gamma,
                      const mllm_fp32_t* __restrict__ beta, size_t N, mllm_fp32_t eps, int32_t thread_count);

void layernorm_N_fp16(mllm_fp16_t* __restrict__ Z, const mllm_fp16_t* __restrict__ X, const mllm_fp16_t* __restrict__ gamma,
                      const mllm_fp16_t* __restrict__ beta, size_t N, mllm_fp32_t eps, int32_t thread_count);

}  // namespace mllm::cpu::arm

#endif