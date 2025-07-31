/**
 * @file gelu.hpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-29
 *
 */
#pragma once

#include <arm_neon.h>
#include <cstdint>

namespace mllm::cpu::arm {

void gelu_fp32(float* __restrict__ Z, const float* __restrict__ X, int32_t N);

void quick_gelu_fp32(float* __restrict__ Z, const float* __restrict__ X, int32_t N);

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
void gelu_fp16(float16_t* __restrict__ Z, const float16_t* __restrict__ X, int32_t N);

void quick_gelu_fp16(float16_t* __restrict__ Z, const float16_t* __restrict__ X, int32_t N);
#endif

}  // namespace mllm::cpu::arm