// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/DataTypes.hpp"
#include "mllm/utils/CPUArchHelper.hpp"

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)

#include "mllm/backends/cpu/kernels/arm/macro.hpp"
#include "mllm/backends/cpu/kernels/arm/primitives.hpp"

namespace mllm::cpu::arm {

template<typename __ST, typename __VT, typename __ScalarOp, typename __VectorOp, typename __VectorReduceOp>
struct Reduce {
  static MLLM_CPU_ARM_FORCE_INLINE __ST run(const __ST* __restrict__ src, size_t size, int thread_count) {
    if (size == 0) return __ST(0);

    size_t vector_lanes = __VectorOp::lanes();
    size_t vec_size = (size / vector_lanes) * vector_lanes;
    size_t remaining = size % vector_lanes;

    __ST scalar_result;

    if (vec_size > 0) {
      __VT vec_result0 = __VectorLoad<__VT>::load(src, 0);
      __VT vec_result1 = __VectorLoad<__VT>::load(src, vector_lanes);
      __VT vec_result2 = __VectorLoad<__VT>::load(src, 2 * vector_lanes);
      __VT vec_result3 = __VectorLoad<__VT>::load(src, 3 * vector_lanes);

      size_t i = 4 * vector_lanes;
      for (; i + 3 * vector_lanes < vec_size; i += 4 * vector_lanes) {
        vec_result0 = __VectorOp::cal(vec_result0, __VectorLoad<__VT>::load(src, i));
        vec_result1 = __VectorOp::cal(vec_result1, __VectorLoad<__VT>::load(src, i + vector_lanes));
        vec_result2 = __VectorOp::cal(vec_result2, __VectorLoad<__VT>::load(src, i + 2 * vector_lanes));
        vec_result3 = __VectorOp::cal(vec_result3, __VectorLoad<__VT>::load(src, i + 3 * vector_lanes));
      }

      vec_result0 = __VectorOp::cal(vec_result0, vec_result1);
      vec_result2 = __VectorOp::cal(vec_result2, vec_result3);
      vec_result0 = __VectorOp::cal(vec_result0, vec_result2);

      for (; i < vec_size; i += vector_lanes) {
        __VT v = __VectorLoad<__VT>::load(src, i);
        vec_result0 = __VectorOp::cal(vec_result0, v);
      }

      scalar_result = __VectorReduceOp::cal(vec_result0);
    } else {
      scalar_result = src[0];
    }

    for (size_t i = vec_size; i < size; ++i) { scalar_result = __ScalarOp::cal(scalar_result, src[i]); }

    return scalar_result;
  }
};

void reduce_sum_fp32(mllm_fp32_t* dst, const mllm_fp32_t* src, size_t size, int32_t thread_count);

void reduce_mul_fp32(mllm_fp32_t* dst, const mllm_fp32_t* src, size_t size, int32_t thread_count);

void reduce_max_fp32(mllm_fp32_t* dst, const mllm_fp32_t* src, size_t size, int32_t thread_count);

void reduce_min_fp32(mllm_fp32_t* dst, const mllm_fp32_t* src, size_t size, int32_t thread_count);

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
void reduce_sum_fp16(mllm_fp16_t* dst, const mllm_fp16_t* src, size_t size, int32_t thread_count);

void reduce_mul_fp16(mllm_fp16_t* dst, const mllm_fp16_t* src, size_t size, int32_t thread_count);

void reduce_max_fp16(mllm_fp16_t* dst, const mllm_fp16_t* src, size_t size, int32_t thread_count);

void reduce_min_fp16(mllm_fp16_t* dst, const mllm_fp16_t* src, size_t size, int32_t thread_count);
#endif

void reduce_sum_int8(mllm_int8_t* dst, const mllm_int8_t* src, size_t size, int32_t thread_count);

void reduce_mul_int8(mllm_int8_t* dst, const mllm_int8_t* src, size_t size, int32_t thread_count);

void reduce_max_int8(mllm_int8_t* dst, const mllm_int8_t* src, size_t size, int32_t thread_count);

void reduce_min_int8(mllm_int8_t* dst, const mllm_int8_t* src, size_t size, int32_t thread_count);

void reduce_sum_int16(mllm_int16_t* dst, const mllm_int16_t* src, size_t size, int32_t thread_count);

void reduce_mul_int16(mllm_int16_t* dst, const mllm_int16_t* src, size_t size, int32_t thread_count);

void reduce_max_int16(mllm_int16_t* dst, const mllm_int16_t* src, size_t size, int32_t thread_count);

void reduce_min_int16(mllm_int16_t* dst, const mllm_int16_t* src, size_t size, int32_t thread_count);

void reduce_sum_int32(mllm_int32_t* dst, const mllm_int32_t* src, size_t size, int32_t thread_count);

void reduce_mul_int32(mllm_int32_t* dst, const mllm_int32_t* src, size_t size, int32_t thread_count);

void reduce_max_int32(mllm_int32_t* dst, const mllm_int32_t* src, size_t size, int32_t thread_count);

void reduce_min_int32(mllm_int32_t* dst, const mllm_int32_t* src, size_t size, int32_t thread_count);
}  // namespace mllm::cpu::arm

#endif