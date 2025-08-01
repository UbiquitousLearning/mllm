// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/DataTypes.hpp"
#include "mllm/utils/CPUArchHelper.hpp"

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)

#include <arm_neon.h>

#include "mllm/backends/cpu/kernels/arm/primitives.hpp"

namespace mllm::cpu::arm {

//===----------------------------------------------------------------------===//
// Loops
//===----------------------------------------------------------------------===//
// Element wise loop
template<typename __ST, typename __VT, typename __ScalarOp, typename __VectorOp, size_t loop_unroll_size = 4>
struct __ew_loop {
  static inline void run(__ST* __restrict dst, const __ST* __restrict src0, const __ST* __restrict src1, size_t size) {
    size_t constexpr lanes = __VectorOp::lanes();
    size_t vec_size = size & ~(lanes - 1);
    size_t i = 0;
    for (; i + lanes * loop_unroll_size <= vec_size; i += lanes * loop_unroll_size) {
#pragma unroll
      for (size_t u = 0; u < loop_unroll_size; ++u) {
        __VT va = __VectorLoad<__VT>::load(src0, i + u * lanes);
        __VT vb = __VectorLoad<__VT>::load(src1, i + u * lanes);
        __VT result = __VectorOp::cal(va, vb);
        __VectorStore<__VT>::store(dst, i + u * lanes, result);
      }
    }
    for (; i < vec_size; i += lanes) {
      __VT va = __VectorLoad<__VT>::load(src0, i);
      __VT vb = __VectorLoad<__VT>::load(src1, i);
      __VT result = __VectorOp::cal(va, vb);
      __VectorStore<__VT>::store(dst, i, result);
    }
    for (; i < size; ++i) {
      __ST sa = __ScalarLoad<__ST>::load(src0, i);
      __ST sb = __ScalarLoad<__ST>::load(src1, i);
      __ST result = __ScalarOp::cal(sa, sb);
      __ScalarStore<__ST>::store(dst, i, result);
    }
  }
};

template<typename __ST, typename __VT, typename __ScalarOp, typename __VectorOp, size_t loop_unroll_size = 4>
struct ParallelElementwiseLoop {
  static inline void run(__ST* __restrict dst, const __ST* __restrict src0, const __ST* __restrict src1, size_t size,
                         size_t thread_count) {
    if (thread_count > 1) {
      size_t constexpr lanes = __VectorOp::lanes();
      size_t vec_size = size & ~(lanes - 1);
      size_t chunk_size = (vec_size + thread_count - 1) / thread_count;
      chunk_size = (chunk_size + lanes - 1) & ~(lanes - 1);

#pragma omp parallel for if (thread_count > 1) num_threads(thread_count)
      for (size_t start = 0; start < vec_size; start += chunk_size) {
        size_t end = std::min(start + chunk_size, vec_size);
        size_t local_size = end - start;
        __ew_loop<__ST, __VT, __ScalarOp, __VectorOp, loop_unroll_size>::run(dst + start, src0 + start, src1 + start,
                                                                             local_size);
      }
      if (vec_size < size) {
        size_t remaining_size = size - vec_size;
        __ew_loop<__ST, __VT, __ScalarOp, __VectorOp, loop_unroll_size>::run(dst + vec_size, src0 + vec_size, src1 + vec_size,
                                                                             remaining_size);
      }
    } else {
      __ew_loop<__ST, __VT, __ScalarOp, __VectorOp, loop_unroll_size>::run(dst, src0, src1, size);
    }
  }
};

template<typename __ST, typename __VT, typename __ScalarOp, typename __VectorOp, size_t loop_unroll_size = 4>
struct __ew_loop_array_scalar {
  static inline void run(__ST* __restrict dst, const __ST* __restrict src0, const __ST src1, size_t size) {
    size_t constexpr lanes = __VectorOp::lanes();
    size_t vec_size = size & ~(lanes - 1);
    size_t i = 0;

    __VT vb_scalar = __VectorLoad<__VT>::load_scalar(src1);

    for (; i + lanes * loop_unroll_size <= vec_size; i += lanes * loop_unroll_size) {
#pragma unroll
      for (size_t u = 0; u < loop_unroll_size; ++u) {
        __VT va = __VectorLoad<__VT>::load(src0, i + u * lanes);
        __VT result = __VectorOp::cal(va, vb_scalar);
        __VectorStore<__VT>::store(dst, i + u * lanes, result);
      }
    }
    for (; i < vec_size; i += lanes) {
      __VT va = __VectorLoad<__VT>::load(src0, i);
      __VT result = __VectorOp::cal(va, vb_scalar);
      __VectorStore<__VT>::store(dst, i, result);
    }
    for (; i < size; ++i) {
      __ST sa = __ScalarLoad<__ST>::load(src0, i);
      __ST result = __ScalarOp::cal(sa, src1);
      __ScalarStore<__ST>::store(dst, i, result);
    }
  }
};

template<typename __ST, typename __VT, typename __ScalarOp, typename __VectorOp, size_t loop_unroll_size = 4>
struct ParallelElementwiseLoopArrayScalar {
  static inline void run(__ST* __restrict dst, const __ST* __restrict src0, const __ST src1, size_t size, size_t thread_count) {
    if (thread_count > 1) {
      size_t constexpr lanes = __VectorOp::lanes();
      size_t vec_size = size & ~(lanes - 1);
      size_t chunk_size = (vec_size + thread_count - 1) / thread_count;
      chunk_size = (chunk_size + lanes - 1) & ~(lanes - 1);

#pragma omp parallel for if (thread_count > 1) num_threads(thread_count)
      for (size_t start = 0; start < vec_size; start += chunk_size) {
        size_t end = std::min(start + chunk_size, vec_size);
        size_t local_size = end - start;
        __ew_loop_array_scalar<__ST, __VT, __ScalarOp, __VectorOp, loop_unroll_size>::run(dst + start, src0 + start, src1,
                                                                                          local_size);
      }
      if (vec_size < size) {
        size_t remaining_size = size - vec_size;
        __ew_loop_array_scalar<__ST, __VT, __ScalarOp, __VectorOp, loop_unroll_size>::run(dst + vec_size, src0 + vec_size, src1,
                                                                                          remaining_size);
      }
    } else {
      __ew_loop_array_scalar<__ST, __VT, __ScalarOp, __VectorOp, loop_unroll_size>::run(dst, src0, src1, size);
    }
  }
};

//===----------------------------------------------------------------------===//
// INSTANCE
//===----------------------------------------------------------------------===//
void ew_add_fp32(mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ src0, const mllm_fp32_t* __restrict__ src1,
                 size_t size, int thread_count);

void ew_sub_fp32(mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ src0, const mllm_fp32_t* __restrict__ src1,
                 size_t size, int thread_count);

void ew_mul_fp32(mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ src0, const mllm_fp32_t* __restrict__ src1,
                 size_t size, int thread_count);

void ew_div_fp32(mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ src0, const mllm_fp32_t* __restrict__ src1,
                 size_t size, int thread_count);

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
void ew_add_fp16(mllm_fp16_t* __restrict__ dst, const mllm_fp16_t* __restrict__ src0, const mllm_fp16_t* __restrict__ src1,
                 size_t size, int thread_count);

void ew_sub_fp16(mllm_fp16_t* __restrict__ dst, const mllm_fp16_t* __restrict__ src0, const mllm_fp16_t* __restrict__ src1,
                 size_t size, int thread_count);

void ew_mul_fp16(mllm_fp16_t* __restrict__ dst, const mllm_fp16_t* __restrict__ src0, const mllm_fp16_t* __restrict__ src1,
                 size_t size, int thread_count);

void ew_div_fp16(mllm_fp16_t* __restrict__ dst, const mllm_fp16_t* __restrict__ src0, const mllm_fp16_t* __restrict__ src1,
                 size_t size, int thread_count);
#endif

void ew_add_int8(mllm_int8_t* __restrict__ dst, const mllm_int8_t* __restrict__ src0, const mllm_int8_t* __restrict__ src1,
                 size_t size, int thread_count);

void ew_sub_int8(mllm_int8_t* __restrict__ dst, const mllm_int8_t* __restrict__ src0, const mllm_int8_t* __restrict__ src1,
                 size_t size, int thread_count);

void ew_mul_int8(mllm_int8_t* __restrict__ dst, const mllm_int8_t* __restrict__ src0, const mllm_int8_t* __restrict__ src1,
                 size_t size, int thread_count);

void ew_div_int8(mllm_int8_t* __restrict__ dst, const mllm_int8_t* __restrict__ src0, const mllm_int8_t* __restrict__ src1,
                 size_t size, int thread_count);

void ew_add_int16(mllm_int16_t* __restrict__ dst, const mllm_int16_t* __restrict__ src0, const mllm_int16_t* __restrict__ src1,
                  size_t size, int thread_count);

void ew_sub_int16(mllm_int16_t* __restrict__ dst, const mllm_int16_t* __restrict__ src0, const mllm_int16_t* __restrict__ src1,
                  size_t size, int thread_count);

void ew_mul_int16(mllm_int16_t* __restrict__ dst, const mllm_int16_t* __restrict__ src0, const mllm_int16_t* __restrict__ src1,
                  size_t size, int thread_count);

void ew_div_int16(mllm_int16_t* __restrict__ dst, const mllm_int16_t* __restrict__ src0, const mllm_int16_t* __restrict__ src1,
                  size_t size, int thread_count);

void ew_add_int32(mllm_int32_t* __restrict__ dst, const mllm_int32_t* __restrict__ src0, const mllm_int32_t* __restrict__ src1,
                  size_t size, int thread_count);

void ew_sub_int32(mllm_int32_t* __restrict__ dst, const mllm_int32_t* __restrict__ src0, const mllm_int32_t* __restrict__ src1,
                  size_t size, int thread_count);

void ew_mul_int32(mllm_int32_t* __restrict__ dst, const mllm_int32_t* __restrict__ src0, const mllm_int32_t* __restrict__ src1,
                  size_t size, int thread_count);

void ew_div_int32(mllm_int32_t* __restrict__ dst, const mllm_int32_t* __restrict__ src0, const mllm_int32_t* __restrict__ src1,
                  size_t size, int thread_count);

void ew_add_fp32_scalar(mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ src0, const mllm_fp32_t src1,
                        size_t size, int thread_count);

void ew_sub_fp32_scalar(mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ src0, const mllm_fp32_t src1,
                        size_t size, int thread_count);

void ew_mul_fp32_scalar(mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ src0, const mllm_fp32_t src1,
                        size_t size, int thread_count);

void ew_div_fp32_scalar(mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ src0, const mllm_fp32_t src1,
                        size_t size, int thread_count);

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
void ew_add_fp16_scalar(mllm_fp16_t* __restrict__ dst, const mllm_fp16_t* __restrict__ src0, const mllm_fp16_t src1,
                        size_t size, int thread_count);

void ew_sub_fp16_scalar(mllm_fp16_t* __restrict__ dst, const mllm_fp16_t* __restrict__ src0, const mllm_fp16_t src1,
                        size_t size, int thread_count);

void ew_mul_fp16_scalar(mllm_fp16_t* __restrict__ dst, const mllm_fp16_t* __restrict__ src0, const mllm_fp16_t src1,
                        size_t size, int thread_count);

void ew_div_fp16_scalar(mllm_fp16_t* __restrict__ dst, const mllm_fp16_t* __restrict__ src0, const mllm_fp16_t src1,
                        size_t size, int thread_count);
#endif

void ew_add_int8_scalar(mllm_int8_t* __restrict__ dst, const mllm_int8_t* __restrict__ src0, const mllm_int8_t src1,
                        size_t size, int thread_count);

void ew_sub_int8_scalar(mllm_int8_t* __restrict__ dst, const mllm_int8_t* __restrict__ src0, const mllm_int8_t src1,
                        size_t size, int thread_count);

void ew_mul_int8_scalar(mllm_int8_t* __restrict__ dst, const mllm_int8_t* __restrict__ src0, const mllm_int8_t src1,
                        size_t size, int thread_count);

void ew_div_int8_scalar(mllm_int8_t* __restrict__ dst, const mllm_int8_t* __restrict__ src0, const mllm_int8_t src1,
                        size_t size, int thread_count);

void ew_add_int16_scalar(mllm_int16_t* __restrict__ dst, const mllm_int16_t* __restrict__ src0, const mllm_int16_t src1,
                         size_t size, int thread_count);

void ew_sub_int16_scalar(mllm_int16_t* __restrict__ dst, const mllm_int16_t* __restrict__ src0, const mllm_int16_t src1,
                         size_t size, int thread_count);

void ew_mul_int16_scalar(mllm_int16_t* __restrict__ dst, const mllm_int16_t* __restrict__ src0, const mllm_int16_t src1,
                         size_t size, int thread_count);

void ew_div_int16_scalar(mllm_int16_t* __restrict__ dst, const mllm_int16_t* __restrict__ src0, const mllm_int16_t src1,
                         size_t size, int thread_count);

void ew_add_int32_scalar(mllm_int32_t* __restrict__ dst, const mllm_int32_t* __restrict__ src0, const mllm_int32_t src1,
                         size_t size, int thread_count);

void ew_sub_int32_scalar(mllm_int32_t* __restrict__ dst, const mllm_int32_t* __restrict__ src0, const mllm_int32_t src1,
                         size_t size, int thread_count);

void ew_mul_int32_scalar(mllm_int32_t* __restrict__ dst, const mllm_int32_t* __restrict__ src0, const mllm_int32_t src1,
                         size_t size, int thread_count);

void ew_div_int32_scalar(mllm_int32_t* __restrict__ dst, const mllm_int32_t* __restrict__ src0, const mllm_int32_t src1,
                         size_t size, int thread_count);
}  // namespace mllm::cpu::arm

#endif
