// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/utils/CPUArchHelper.hpp"

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)

#include <arm_neon.h>

namespace mllm::cpu::arm {

// For NCHW
void layernorm2d_fp32(const float* x, const float* weight, const float* bias, float* y, int N, int C, int H, int W, float eps);

}  // namespace mllm::cpu::arm
#endif
