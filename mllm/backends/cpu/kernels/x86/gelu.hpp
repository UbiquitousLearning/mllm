// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/utils/CPUArchHelper.hpp"

#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)

#include <cstdint>

namespace mllm::cpu::x86 {

void gelu_fp32(float* __restrict__ Z, const float* __restrict__ X, int32_t N, int thread_cnt);

void quick_gelu_fp32(float* __restrict__ Z, const float* __restrict__ X, int32_t N, int thread_cnt);

}  // namespace mllm::cpu::x86

#endif
