/**
 * @file fill.hpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-31
 *
 */
#pragma once

#include "mllm/core/DataTypes.hpp"
#include "mllm/utils/CPUArchHelper.hpp"

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)

namespace mllm::cpu::arm {
void fill_zeros(mllm_fp32_t* __restrict dst, size_t size, int thread_count);

void fill_ones(mllm_fp32_t* __restrict dst, size_t size, int thread_count);

void fill_specific_value(mllm_fp32_t* __restrict dst, size_t size, float value, int thread_count);

void fill_arange(mllm_fp32_t* __restrict dst, size_t size, float start, float end, float step, int thread_count);

void fill_random(mllm_fp32_t* __restrict dst, size_t size, float start, float end, uint64_t seed, int thread_count);

void fill_zeros_fp16(mllm_fp16_t* __restrict dst, size_t size, int thread_count);

void fill_ones_fp16(mllm_fp16_t* __restrict dst, size_t size, int thread_count);

void fill_specific_value_fp16(mllm_fp16_t* __restrict dst, size_t size, float value, int thread_count);

void fill_arange_fp16(mllm_fp16_t* __restrict dst, size_t size, float start, float end, float step, int thread_count);

void fill_random_fp16(mllm_fp16_t* __restrict dst, size_t size, float start, float end, uint64_t seed, int thread_count);

}  // namespace mllm::cpu::arm

#endif