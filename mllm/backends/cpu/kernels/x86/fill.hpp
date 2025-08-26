// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/DataTypes.hpp"
#include "mllm/utils/CPUArchHelper.hpp"

#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)

namespace mllm::cpu::x86 {

void fill_zeros(mllm_fp32_t* __restrict dst, size_t size, int thread_count);

void fill_ones(mllm_fp32_t* __restrict dst, size_t size, int thread_count);

void fill_specific_value(mllm_fp32_t* __restrict dst, size_t size, float value, int thread_count);

void fill_arange(mllm_fp32_t* __restrict dst, size_t size, float start, float end, float step, int thread_count);

void fill_random(mllm_fp32_t* __restrict dst, size_t size, float start, float end, uint64_t seed, int thread_count);

}  // namespace mllm::cpu::x86

#endif
