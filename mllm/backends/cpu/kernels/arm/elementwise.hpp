// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/DataTypes.hpp"
#include "mllm/utils/CPUArchHelper.hpp"
#include "mllm/core/Parallel.hpp"

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)

#include <arm_neon.h>

#include "mllm/backends/cpu/kernels/arm/primitives.hpp"
#include "mllm/backends/cpu/kernels/arm/math.hpp"

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

      MLLM_AUTO_PARALLEL_FOR_BEGIN(start, 0, vec_size, chunk_size) {
        size_t end = std::min((size_t)(start + chunk_size), vec_size);
        size_t local_size = end - start;
        __ew_loop<__ST, __VT, __ScalarOp, __VectorOp, loop_unroll_size>::run(dst + start, src0 + start, src1 + start,
                                                                             local_size);
      }
      MLLM_AUTO_PARALLEL_FOR_END()

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

      MLLM_AUTO_PARALLEL_FOR_BEGIN(start, 0, vec_size, chunk_size) {
        size_t end = std::min((size_t)(start + chunk_size), vec_size);
        size_t local_size = end - start;
        __ew_loop_array_scalar<__ST, __VT, __ScalarOp, __VectorOp, loop_unroll_size>::run(dst + start, src0 + start, src1,
                                                                                          local_size);
      }
      MLLM_AUTO_PARALLEL_FOR_END()

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

// loop unary

template<typename __ST, typename __VT, typename __ScalarOp, typename __VectorOp, size_t loop_unroll_size = 4>
struct __ew_loop_unary {
  static inline void run(__ST* __restrict dst, const __ST* __restrict src0, size_t size) {
    size_t constexpr lanes = __VectorOp::lanes();
    size_t vec_size = size & ~(lanes - 1);
    size_t i = 0;
    for (; i + lanes * loop_unroll_size <= vec_size; i += lanes * loop_unroll_size) {
#pragma unroll
      for (size_t u = 0; u < loop_unroll_size; ++u) {
        __VT va = __VectorLoad<__VT>::load(src0, i + u * lanes);
        __VT result = __VectorOp::cal(va);
        __VectorStore<__VT>::store(dst, i + u * lanes, result);
      }
    }
    for (; i < vec_size; i += lanes) {
      __VT va = __VectorLoad<__VT>::load(src0, i);
      __VT result = __VectorOp::cal(va);
      __VectorStore<__VT>::store(dst, i, result);
    }
    for (; i < size; ++i) {
      __ST sa = __ScalarLoad<__ST>::load(src0, i);
      __ST result = __ScalarOp::cal(sa);
      __ScalarStore<__ST>::store(dst, i, result);
    }
  }
};

template<typename __ST, typename __VT, typename __ScalarOp, typename __VectorOp, size_t loop_unroll_size = 4>
struct ParallelElementwiseLoopUnary {
  static inline void run(__ST* __restrict dst, const __ST* __restrict src0, size_t size, size_t thread_count) {
    if (thread_count > 1) {
      size_t constexpr lanes = __VectorOp::lanes();
      size_t vec_size = size & ~(lanes - 1);
      size_t chunk_size = (vec_size + thread_count - 1) / thread_count;
      chunk_size = (chunk_size + lanes - 1) & ~(lanes - 1);

      MLLM_AUTO_PARALLEL_FOR_BEGIN(start, 0, vec_size, chunk_size) {
        size_t end = std::min((size_t)(start + chunk_size), vec_size);
        size_t local_size = end - start;
        __ew_loop_unary<__ST, __VT, __ScalarOp, __VectorOp, loop_unroll_size>::run(dst + start, src0 + start, local_size);
      }
      MLLM_AUTO_PARALLEL_FOR_END()

      if (vec_size < size) {
        size_t remaining_size = size - vec_size;
        __ew_loop_unary<__ST, __VT, __ScalarOp, __VectorOp, loop_unroll_size>::run(dst + vec_size, src0 + vec_size,
                                                                                   remaining_size);
      }
    } else {
      __ew_loop_unary<__ST, __VT, __ScalarOp, __VectorOp, loop_unroll_size>::run(dst, src0, size);
    }
  }
};

// Loop for clip operation with both min and max
template<typename __ST, typename __VT, typename __ScalarOp, typename __VectorOp, size_t loop_unroll_size = 4>
struct __clip_loop {
  static inline void run(__ST* __restrict dst, const __ST* __restrict src, __ST min_val, __ST max_val, size_t size) {
    __VectorOp vector_op_instance(min_val, max_val);
    size_t constexpr lanes = vector_op_instance.lanes();
    size_t vec_size = size & ~(lanes - 1);
    size_t i = 0;

    __ScalarOp scalar_op(min_val, max_val);
    __VectorOp vector_op(min_val, max_val);

    for (; i + lanes * loop_unroll_size <= vec_size; i += lanes * loop_unroll_size) {
#pragma unroll
      for (size_t u = 0; u < loop_unroll_size; ++u) {
        __VT va = __VectorLoad<__VT>::load(src, i + u * lanes);
        __VT result = vector_op.cal(va);
        __VectorStore<__VT>::store(dst, i + u * lanes, result);
      }
    }
    for (; i < vec_size; i += lanes) {
      __VT va = __VectorLoad<__VT>::load(src, i);
      __VT result = vector_op.cal(va);
      __VectorStore<__VT>::store(dst, i, result);
    }
    for (; i < size; ++i) {
      __ST sa = __ScalarLoad<__ST>::load(src, i);
      __ST result = scalar_op.cal(sa);
      __ScalarStore<__ST>::store(dst, i, result);
    }
  }
};

template<typename __ST, typename __VT, typename __ScalarOp, typename __VectorOp, size_t loop_unroll_size = 4>
struct ParallelClipLoop {
  static inline void run(__ST* __restrict dst, const __ST* __restrict src, __ST min_val, __ST max_val, size_t size,
                         size_t thread_count) {
    if (thread_count > 1) {
      __VectorOp vector_op_instance(min_val, max_val);
      size_t constexpr lanes = vector_op_instance.lanes();
      size_t vec_size = size & ~(lanes - 1);
      size_t chunk_size = (vec_size + thread_count - 1) / thread_count;
      chunk_size = (chunk_size + lanes - 1) & ~(lanes - 1);

      MLLM_AUTO_PARALLEL_FOR_BEGIN(start, 0, vec_size, chunk_size) {
        size_t end = std::min((size_t)(start + chunk_size), vec_size);
        size_t local_size = end - start;
        __clip_loop<__ST, __VT, __ScalarOp, __VectorOp, loop_unroll_size>::run(dst + start, src + start, min_val, max_val,
                                                                               local_size);
      }
      MLLM_AUTO_PARALLEL_FOR_END()

      if (vec_size < size) {
        size_t remaining_size = size - vec_size;
        __clip_loop<__ST, __VT, __ScalarOp, __VectorOp, loop_unroll_size>::run(dst + vec_size, src + vec_size, min_val, max_val,
                                                                               remaining_size);
      }
    } else {
      __clip_loop<__ST, __VT, __ScalarOp, __VectorOp, loop_unroll_size>::run(dst, src, min_val, max_val, size);
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

// ------------ complex input type

#define DEFINE_EW_FP32_COMPLEX_OP(NAME, OP)                                                                         \
  void ew_##NAME##_fp32_complex(mllm_complex_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ src0,        \
                                const mllm_complex_fp32_t* __restrict__ src1, size_t size, int thread_count);       \
  void ew_##NAME##_fp32_complex_scalar(mllm_complex_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ src0, \
                                       const mllm_complex_fp32_t src1, size_t size, int thread_count);
DEFINE_EW_FP32_COMPLEX_OP(add, +)
DEFINE_EW_FP32_COMPLEX_OP(sub, -)
DEFINE_EW_FP32_COMPLEX_OP(mul, *)
DEFINE_EW_FP32_COMPLEX_OP(div, /)

#undef EW_FP32_COMPLEX_OP

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

//===----------------------------------------------------------------------===//
// Abs operations
//===----------------------------------------------------------------------===//
void ew_abs_fp32(mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ src0, size_t size, int thread_count);

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
void ew_abs_fp16(mllm_fp16_t* __restrict__ dst, const mllm_fp16_t* __restrict__ src0, size_t size, int thread_count);
#endif

void ew_abs_int8(mllm_int8_t* __restrict__ dst, const mllm_int8_t* __restrict__ src0, size_t size, int thread_count);

void ew_abs_int16(mllm_int16_t* __restrict__ dst, const mllm_int16_t* __restrict__ src0, size_t size, int thread_count);

void ew_abs_int32(mllm_int32_t* __restrict__ dst, const mllm_int32_t* __restrict__ src0, size_t size, int thread_count);

void ew_abs_complex_fp32(mllm_fp32_t* __restrict__ dst, const mllm_complex_fp32_t* __restrict__ src0, size_t size,
                         int thread_count);

void ew_abs_complex_fp64(mllm_fp32_t* __restrict__ dst, const mllm_complex_fp64_t* __restrict__ src0, size_t size,
                         int thread_count);

//===----------------------------------------------------------------------===//
// Log operations
//===----------------------------------------------------------------------===//
void ew_log_fp32(mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ src0, size_t size, int thread_count);

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
void ew_log_fp16(mllm_fp16_t* __restrict__ dst, const mllm_fp16_t* __restrict__ src0, size_t size, int thread_count);
#endif

//===----------------------------------------------------------------------===//
// Exp operations
//===----------------------------------------------------------------------===//
void ew_exp_fp32(mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ src0, size_t size, int thread_count);

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
void ew_exp_fp16(mllm_fp16_t* __restrict__ dst, const mllm_fp16_t* __restrict__ src0, size_t size, int thread_count);
#endif

//===----------------------------------------------------------------------===//
// Sin operations
//===----------------------------------------------------------------------===//
void ew_sin_fp32(mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ src0, size_t size, int thread_count);

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
void ew_sin_fp16(mllm_fp16_t* __restrict__ dst, const mllm_fp16_t* __restrict__ src0, size_t size, int thread_count);
#endif

//===----------------------------------------------------------------------===//
// Cos operations
//===----------------------------------------------------------------------===//
void ew_cos_fp32(mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ src0, size_t size, int thread_count);

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
void ew_cos_fp16(mllm_fp16_t* __restrict__ dst, const mllm_fp16_t* __restrict__ src0, size_t size, int thread_count);
#endif

//===----------------------------------------------------------------------===//
// Clip Kernel API
//===----------------------------------------------------------------------===//
void clip_fp32(mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ src, mllm_fp32_t min_val, mllm_fp32_t max_val,
               size_t size, int thread_count);

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
void clip_fp16(mllm_fp16_t* __restrict__ dst, const mllm_fp16_t* __restrict__ src, mllm_fp16_t min_val, mllm_fp16_t max_val,
               size_t size, int thread_count);
#endif

void clip_int8(mllm_int8_t* __restrict__ dst, const mllm_int8_t* __restrict__ src, mllm_int8_t min_val, mllm_int8_t max_val,
               size_t size, int thread_count);

void clip_int16(mllm_int16_t* __restrict__ dst, const mllm_int16_t* __restrict__ src, mllm_int16_t min_val,
                mllm_int16_t max_val, size_t size, int thread_count);

void clip_int32(mllm_int32_t* __restrict__ dst, const mllm_int32_t* __restrict__ src, mllm_int32_t min_val,
                mllm_int32_t max_val, size_t size, int thread_count);

}  // namespace mllm::cpu::arm

#endif
