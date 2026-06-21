// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include "mllm/core/DataTypes.hpp"
#include "mllm/utils/CPUArchHelper.hpp"

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)

#include <arm_neon.h>

namespace mllm::cpu::arm {

void silu_fp32(const mllm_fp32_t* __restrict X, mllm_fp32_t* __restrict Y, int len, int thread_count);

void silu_fp16(const mllm_fp16_t* __restrict X, mllm_fp16_t* __restrict Y, int len, int thread_count);

}  // namespace mllm::cpu::arm

#endif