// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/DataTypes.hpp"
#include "mllm/core/Parallel.hpp"
#include "mllm/utils/CPUArchHelper.hpp"
#include "mllm/utils/UnsafeMacros.hpp"
#include "mllm/backends/cpu/kernels/arm/elementwise.hpp"

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)

namespace mllm::cpu::arm {

void ew_add_fp32(mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ src0, const mllm_fp32_t* __restrict__ src1,
                 size_t size, int thread_count) {
  ParallelElementwiseLoop<mllm_fp32_t, float32x4_t, __ScalarAdd<mllm_fp32_t>, __VectorAdd<float32x4_t>>::run(
      dst, src0, src1, size, thread_count);
}

void ew_sub_fp32(mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ src0, const mllm_fp32_t* __restrict__ src1,
                 size_t size, int thread_count) {
  ParallelElementwiseLoop<mllm_fp32_t, float32x4_t, __ScalarSub<mllm_fp32_t>, __VectorSub<float32x4_t>>::run(
      dst, src0, src1, size, thread_count);
}

void ew_mul_fp32(mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ src0, const mllm_fp32_t* __restrict__ src1,
                 size_t size, int thread_count) {
  ParallelElementwiseLoop<mllm_fp32_t, float32x4_t, __ScalarMul<mllm_fp32_t>, __VectorMul<float32x4_t>>::run(
      dst, src0, src1, size, thread_count);
}

void ew_div_fp32(mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ src0, const mllm_fp32_t* __restrict__ src1,
                 size_t size, int thread_count) {
  ParallelElementwiseLoop<mllm_fp32_t, float32x4_t, __ScalarDiv<mllm_fp32_t>, __VectorDiv<float32x4_t>>::run(
      dst, src0, src1, size, thread_count);
}

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
void ew_add_fp16(mllm_fp16_t* __restrict__ dst, const mllm_fp16_t* __restrict__ src0, const mllm_fp16_t* __restrict__ src1,
                 size_t size, int thread_count) {
  ParallelElementwiseLoop<mllm_fp16_t, float16x8_t, __ScalarAdd<mllm_fp16_t>, __VectorAdd<float16x8_t>>::run(
      dst, src0, src1, size, thread_count);
}

void ew_sub_fp16(mllm_fp16_t* __restrict__ dst, const mllm_fp16_t* __restrict__ src0, const mllm_fp16_t* __restrict__ src1,
                 size_t size, int thread_count) {
  ParallelElementwiseLoop<mllm_fp16_t, float16x8_t, __ScalarSub<mllm_fp16_t>, __VectorSub<float16x8_t>>::run(
      dst, src0, src1, size, thread_count);
}

void ew_mul_fp16(mllm_fp16_t* __restrict__ dst, const mllm_fp16_t* __restrict__ src0, const mllm_fp16_t* __restrict__ src1,
                 size_t size, int thread_count) {
  ParallelElementwiseLoop<mllm_fp16_t, float16x8_t, __ScalarMul<mllm_fp16_t>, __VectorMul<float16x8_t>>::run(
      dst, src0, src1, size, thread_count);
}

void ew_div_fp16(mllm_fp16_t* __restrict__ dst, const mllm_fp16_t* __restrict__ src0, const mllm_fp16_t* __restrict__ src1,
                 size_t size, int thread_count) {
  ParallelElementwiseLoop<mllm_fp16_t, float16x8_t, __ScalarDiv<mllm_fp16_t>, __VectorDiv<float16x8_t>>::run(
      dst, src0, src1, size, thread_count);
}
#endif

void ew_add_int8(mllm_int8_t* __restrict__ dst, const mllm_int8_t* __restrict__ src0, const mllm_int8_t* __restrict__ src1,
                 size_t size, int thread_count) {
  ParallelElementwiseLoop<mllm_int8_t, int8x16_t, __ScalarAdd<mllm_int8_t>, __VectorAdd<int8x16_t>>::run(dst, src0, src1, size,
                                                                                                         thread_count);
}

void ew_sub_int8(mllm_int8_t* __restrict__ dst, const mllm_int8_t* __restrict__ src0, const mllm_int8_t* __restrict__ src1,
                 size_t size, int thread_count) {
  ParallelElementwiseLoop<mllm_int8_t, int8x16_t, __ScalarSub<mllm_int8_t>, __VectorSub<int8x16_t>>::run(dst, src0, src1, size,
                                                                                                         thread_count);
}

void ew_mul_int8(mllm_int8_t* __restrict__ dst, const mllm_int8_t* __restrict__ src0, const mllm_int8_t* __restrict__ src1,
                 size_t size, int thread_count) {
  ParallelElementwiseLoop<mllm_int8_t, int8x16_t, __ScalarMul<mllm_int8_t>, __VectorMul<int8x16_t>>::run(dst, src0, src1, size,
                                                                                                         thread_count);
}

void ew_div_int8(mllm_int8_t* __restrict__ dst, const mllm_int8_t* __restrict__ src0, const mllm_int8_t* __restrict__ src1,
                 size_t size, int thread_count) {
  ParallelElementwiseLoop<mllm_int8_t, int8x16_t, __ScalarDiv<mllm_int8_t>, __VectorDiv<int8x16_t>>::run(dst, src0, src1, size,
                                                                                                         thread_count);
}

void ew_add_int16(mllm_int16_t* __restrict__ dst, const mllm_int16_t* __restrict__ src0, const mllm_int16_t* __restrict__ src1,
                  size_t size, int thread_count) {
  ParallelElementwiseLoop<mllm_int16_t, int16x8_t, __ScalarAdd<mllm_int16_t>, __VectorAdd<int16x8_t>>::run(dst, src0, src1,
                                                                                                           size, thread_count);
}

void ew_sub_int16(mllm_int16_t* __restrict__ dst, const mllm_int16_t* __restrict__ src0, const mllm_int16_t* __restrict__ src1,
                  size_t size, int thread_count) {
  ParallelElementwiseLoop<mllm_int16_t, int16x8_t, __ScalarSub<mllm_int16_t>, __VectorSub<int16x8_t>>::run(dst, src0, src1,
                                                                                                           size, thread_count);
}

void ew_mul_int16(mllm_int16_t* __restrict__ dst, const mllm_int16_t* __restrict__ src0, const mllm_int16_t* __restrict__ src1,
                  size_t size, int thread_count) {
  ParallelElementwiseLoop<mllm_int16_t, int16x8_t, __ScalarMul<mllm_int16_t>, __VectorMul<int16x8_t>>::run(dst, src0, src1,
                                                                                                           size, thread_count);
}

void ew_div_int16(mllm_int16_t* __restrict__ dst, const mllm_int16_t* __restrict__ src0, const mllm_int16_t* __restrict__ src1,
                  size_t size, int thread_count) {
  ParallelElementwiseLoop<mllm_int16_t, int16x8_t, __ScalarDiv<mllm_int16_t>, __VectorDiv<int16x8_t>>::run(dst, src0, src1,
                                                                                                           size, thread_count);
}

void ew_add_int32(mllm_int32_t* __restrict__ dst, const mllm_int32_t* __restrict__ src0, const mllm_int32_t* __restrict__ src1,
                  size_t size, int thread_count) {
  ParallelElementwiseLoop<mllm_int32_t, int32x4_t, __ScalarAdd<mllm_int32_t>, __VectorAdd<int32x4_t>>::run(dst, src0, src1,
                                                                                                           size, thread_count);
}

void ew_sub_int32(mllm_int32_t* __restrict__ dst, const mllm_int32_t* __restrict__ src0, const mllm_int32_t* __restrict__ src1,
                  size_t size, int thread_count) {
  ParallelElementwiseLoop<mllm_int32_t, int32x4_t, __ScalarSub<mllm_int32_t>, __VectorSub<int32x4_t>>::run(dst, src0, src1,
                                                                                                           size, thread_count);
}

void ew_mul_int32(mllm_int32_t* __restrict__ dst, const mllm_int32_t* __restrict__ src0, const mllm_int32_t* __restrict__ src1,
                  size_t size, int thread_count) {
  ParallelElementwiseLoop<mllm_int32_t, int32x4_t, __ScalarMul<mllm_int32_t>, __VectorMul<int32x4_t>>::run(dst, src0, src1,
                                                                                                           size, thread_count);
}

void ew_div_int32(mllm_int32_t* __restrict__ dst, const mllm_int32_t* __restrict__ src0, const mllm_int32_t* __restrict__ src1,
                  size_t size, int thread_count) {
  ParallelElementwiseLoop<mllm_int32_t, int32x4_t, __ScalarDiv<mllm_int32_t>, __VectorDiv<int32x4_t>>::run(dst, src0, src1,
                                                                                                           size, thread_count);
}

void ew_add_fp32_scalar(mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ src0, const mllm_fp32_t src1,
                        size_t size, int thread_count) {
  ParallelElementwiseLoopArrayScalar<mllm_fp32_t, float32x4_t, __ScalarAdd<mllm_fp32_t>, __VectorAdd<float32x4_t>>::run(
      dst, src0, src1, size, thread_count);
}

void ew_sub_fp32_scalar(mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ src0, const mllm_fp32_t src1,
                        size_t size, int thread_count) {
  ParallelElementwiseLoopArrayScalar<mllm_fp32_t, float32x4_t, __ScalarSub<mllm_fp32_t>, __VectorSub<float32x4_t>>::run(
      dst, src0, src1, size, thread_count);
}

void ew_mul_fp32_scalar(mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ src0, const mllm_fp32_t src1,
                        size_t size, int thread_count) {
  ParallelElementwiseLoopArrayScalar<mllm_fp32_t, float32x4_t, __ScalarMul<mllm_fp32_t>, __VectorMul<float32x4_t>>::run(
      dst, src0, src1, size, thread_count);
}

void ew_div_fp32_scalar(mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ src0, const mllm_fp32_t src1,
                        size_t size, int thread_count) {
  ParallelElementwiseLoopArrayScalar<mllm_fp32_t, float32x4_t, __ScalarDiv<mllm_fp32_t>, __VectorDiv<float32x4_t>>::run(
      dst, src0, src1, size, thread_count);
}

// ------------ complex input type
#define EW_FP32_COMPLEX_OP(NAME, OP)                                                                                \
                                                                                                                    \
  void ew_##NAME##_fp32_complex(mllm_complex_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ src0,        \
                                const mllm_complex_fp32_t* __restrict__ src1, size_t size, int thread_count) {      \
    MLLM_CONDITIONAL_PARALLEL_FOR(thread_count > 1, thread_count, i, 0, size, 1, { dst[i] = src0[i] OP src1[i]; }); \
  }                                                                                                                 \
                                                                                                                    \
  void ew_##NAME##_fp32_complex_scalar(mllm_complex_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ src0, \
                                       const mllm_complex_fp32_t src1, size_t size, int thread_count) {             \
    MLLM_CONDITIONAL_PARALLEL_FOR(thread_count > 1, thread_count, i, 0, size, 1, { dst[i] = src0[i] OP src1; });    \
  }

EW_FP32_COMPLEX_OP(add, +)
EW_FP32_COMPLEX_OP(sub, -)
EW_FP32_COMPLEX_OP(mul, *)
EW_FP32_COMPLEX_OP(div, /)

#undef EW_FP32_COMPLEX_OP

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
void ew_add_fp16_scalar(mllm_fp16_t* __restrict__ dst, const mllm_fp16_t* __restrict__ src0, const mllm_fp16_t src1,
                        size_t size, int thread_count) {
  ParallelElementwiseLoopArrayScalar<mllm_fp16_t, float16x8_t, __ScalarAdd<mllm_fp16_t>, __VectorAdd<float16x8_t>>::run(
      dst, src0, src1, size, thread_count);
}

void ew_sub_fp16_scalar(mllm_fp16_t* __restrict__ dst, const mllm_fp16_t* __restrict__ src0, const mllm_fp16_t src1,
                        size_t size, int thread_count) {
  ParallelElementwiseLoopArrayScalar<mllm_fp16_t, float16x8_t, __ScalarSub<mllm_fp16_t>, __VectorSub<float16x8_t>>::run(
      dst, src0, src1, size, thread_count);
}

void ew_mul_fp16_scalar(mllm_fp16_t* __restrict__ dst, const mllm_fp16_t* __restrict__ src0, const mllm_fp16_t src1,
                        size_t size, int thread_count) {
  ParallelElementwiseLoopArrayScalar<mllm_fp16_t, float16x8_t, __ScalarMul<mllm_fp16_t>, __VectorMul<float16x8_t>>::run(
      dst, src0, src1, size, thread_count);
}

void ew_div_fp16_scalar(mllm_fp16_t* __restrict__ dst, const mllm_fp16_t* __restrict__ src0, const mllm_fp16_t src1,
                        size_t size, int thread_count) {
  ParallelElementwiseLoopArrayScalar<mllm_fp16_t, float16x8_t, __ScalarDiv<mllm_fp16_t>, __VectorDiv<float16x8_t>>::run(
      dst, src0, src1, size, thread_count);
}
#endif

void ew_add_int8_scalar(mllm_int8_t* __restrict__ dst, const mllm_int8_t* __restrict__ src0, const mllm_int8_t src1,
                        size_t size, int thread_count) {
  ParallelElementwiseLoopArrayScalar<mllm_int8_t, int8x16_t, __ScalarAdd<mllm_int8_t>, __VectorAdd<int8x16_t>>::run(
      dst, src0, src1, size, thread_count);
}

void ew_sub_int8_scalar(mllm_int8_t* __restrict__ dst, const mllm_int8_t* __restrict__ src0, const mllm_int8_t src1,
                        size_t size, int thread_count) {
  ParallelElementwiseLoopArrayScalar<mllm_int8_t, int8x16_t, __ScalarSub<mllm_int8_t>, __VectorSub<int8x16_t>>::run(
      dst, src0, src1, size, thread_count);
}

void ew_mul_int8_scalar(mllm_int8_t* __restrict__ dst, const mllm_int8_t* __restrict__ src0, const mllm_int8_t src1,
                        size_t size, int thread_count) {
  ParallelElementwiseLoopArrayScalar<mllm_int8_t, int8x16_t, __ScalarMul<mllm_int8_t>, __VectorMul<int8x16_t>>::run(
      dst, src0, src1, size, thread_count);
}

void ew_div_int8_scalar(mllm_int8_t* __restrict__ dst, const mllm_int8_t* __restrict__ src0, const mllm_int8_t src1,
                        size_t size, int thread_count) {
  ParallelElementwiseLoopArrayScalar<mllm_int8_t, int8x16_t, __ScalarDiv<mllm_int8_t>, __VectorDiv<int8x16_t>>::run(
      dst, src0, src1, size, thread_count);
}

void ew_add_int16_scalar(mllm_int16_t* __restrict__ dst, const mllm_int16_t* __restrict__ src0, const mllm_int16_t src1,
                         size_t size, int thread_count) {
  ParallelElementwiseLoopArrayScalar<mllm_int16_t, int16x8_t, __ScalarAdd<mllm_int16_t>, __VectorAdd<int16x8_t>>::run(
      dst, src0, src1, size, thread_count);
}

void ew_sub_int16_scalar(mllm_int16_t* __restrict__ dst, const mllm_int16_t* __restrict__ src0, const mllm_int16_t src1,
                         size_t size, int thread_count) {
  ParallelElementwiseLoopArrayScalar<mllm_int16_t, int16x8_t, __ScalarSub<mllm_int16_t>, __VectorSub<int16x8_t>>::run(
      dst, src0, src1, size, thread_count);
}

void ew_mul_int16_scalar(mllm_int16_t* __restrict__ dst, const mllm_int16_t* __restrict__ src0, const mllm_int16_t src1,
                         size_t size, int thread_count) {
  ParallelElementwiseLoopArrayScalar<mllm_int16_t, int16x8_t, __ScalarMul<mllm_int16_t>, __VectorMul<int16x8_t>>::run(
      dst, src0, src1, size, thread_count);
}

void ew_div_int16_scalar(mllm_int16_t* __restrict__ dst, const mllm_int16_t* __restrict__ src0, const mllm_int16_t src1,
                         size_t size, int thread_count) {
  ParallelElementwiseLoopArrayScalar<mllm_int16_t, int16x8_t, __ScalarDiv<mllm_int16_t>, __VectorDiv<int16x8_t>>::run(
      dst, src0, src1, size, thread_count);
}

void ew_add_int32_scalar(mllm_int32_t* __restrict__ dst, const mllm_int32_t* __restrict__ src0, const mllm_int32_t src1,
                         size_t size, int thread_count) {
  ParallelElementwiseLoopArrayScalar<mllm_int32_t, int32x4_t, __ScalarAdd<mllm_int32_t>, __VectorAdd<int32x4_t>>::run(
      dst, src0, src1, size, thread_count);
}

void ew_sub_int32_scalar(mllm_int32_t* __restrict__ dst, const mllm_int32_t* __restrict__ src0, const mllm_int32_t src1,
                         size_t size, int thread_count) {
  ParallelElementwiseLoopArrayScalar<mllm_int32_t, int32x4_t, __ScalarSub<mllm_int32_t>, __VectorSub<int32x4_t>>::run(
      dst, src0, src1, size, thread_count);
}

void ew_mul_int32_scalar(mllm_int32_t* __restrict__ dst, const mllm_int32_t* __restrict__ src0, const mllm_int32_t src1,
                         size_t size, int thread_count) {
  ParallelElementwiseLoopArrayScalar<mllm_int32_t, int32x4_t, __ScalarMul<mllm_int32_t>, __VectorMul<int32x4_t>>::run(
      dst, src0, src1, size, thread_count);
}

void ew_div_int32_scalar(mllm_int32_t* __restrict__ dst, const mllm_int32_t* __restrict__ src0, const mllm_int32_t src1,
                         size_t size, int thread_count) {
  ParallelElementwiseLoopArrayScalar<mllm_int32_t, int32x4_t, __ScalarDiv<mllm_int32_t>, __VectorDiv<int32x4_t>>::run(
      dst, src0, src1, size, thread_count);
}

void ew_abs_fp32(mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ src0, size_t size, int thread_count) {
  ParallelElementwiseLoopUnary<mllm_fp32_t, float32x4_t, __ScalarAbs<mllm_fp32_t>, __VectorAbs<float32x4_t>>::run(
      dst, src0, size, thread_count);
}

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
void ew_abs_fp16(mllm_fp16_t* __restrict__ dst, const mllm_fp16_t* __restrict__ src0, size_t size, int thread_count) {
  ParallelElementwiseLoopUnary<mllm_fp16_t, float16x8_t, __ScalarAbs<mllm_fp16_t>, __VectorAbs<float16x8_t>>::run(
      dst, src0, size, thread_count);
}
#endif

void ew_abs_int8(mllm_int8_t* __restrict__ dst, const mllm_int8_t* __restrict__ src0, size_t size, int thread_count) {
  ParallelElementwiseLoopUnary<mllm_int8_t, int8x16_t, __ScalarAbs<mllm_int8_t>, __VectorAbs<int8x16_t>>::run(dst, src0, size,
                                                                                                              thread_count);
}

void ew_abs_int16(mllm_int16_t* __restrict__ dst, const mllm_int16_t* __restrict__ src0, size_t size, int thread_count) {
  ParallelElementwiseLoopUnary<mllm_int16_t, int16x8_t, __ScalarAbs<mllm_int16_t>, __VectorAbs<int16x8_t>>::run(dst, src0, size,
                                                                                                                thread_count);
}

void ew_abs_int32(mllm_int32_t* __restrict__ dst, const mllm_int32_t* __restrict__ src0, size_t size, int thread_count) {
  ParallelElementwiseLoopUnary<mllm_int32_t, int32x4_t, __ScalarAbs<mllm_int32_t>, __VectorAbs<int32x4_t>>::run(dst, src0, size,
                                                                                                                thread_count);
}

__MLLM_UNSAFE_OPT_BEGIN_FAST_MATH
void ew_abs_complex_fp32(mllm_fp32_t* __restrict__ dst, const mllm_complex_fp32_t* __restrict__ src0, size_t size,
                         int thread_count) {
  MLLM_CONDITIONAL_PARALLEL_FOR(thread_count > 1, thread_count, i, 0, size, 1, { dst[i] = std::abs(src0[i]); });
}
__MLLM_UNSAFE_OPT_END

__MLLM_UNSAFE_OPT_BEGIN_FAST_MATH
void ew_abs_complex_fp64(mllm_fp32_t* __restrict__ dst, const mllm_complex_fp64_t* __restrict__ src0, size_t size,
                         int thread_count) {
  MLLM_CONDITIONAL_PARALLEL_FOR(thread_count > 1, thread_count, i, 0, size, 1, { dst[i] = (mllm_fp32_t)std::abs(src0[i]); });
}
__MLLM_UNSAFE_OPT_END

// log operation
void ew_log_fp32(mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ src0, size_t size, int thread_count) {
  //   std::log implementation
  //   MLLM_CONDITIONAL_PARALLEL_FOR(thread_count > 1, thread_count, i, 0, size, 1, { dst[i] = std::log(src0[i]); });
  ParallelElementwiseLoopUnary<mllm_fp32_t, float32x4_t, __ScalarLog<mllm_fp32_t>, __VectorLog<float32x4_t>>::run(
      dst, src0, size, thread_count);
}

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
void ew_log_fp16(mllm_fp16_t* __restrict__ dst, const mllm_fp16_t* __restrict__ src0, size_t size, int thread_count) {
  //   std::log implementation
  //   MLLM_CONDITIONAL_PARALLEL_FOR(thread_count > 1, thread_count, i, 0, size, 1, { dst[i] = std::log(src0[i]); });
  ParallelElementwiseLoopUnary<mllm_fp16_t, float16x8_t, __ScalarLog<mllm_fp16_t>, __VectorLog<float16x8_t>>::run(
      dst, src0, size, thread_count);
}
#endif

// exp operation
void ew_exp_fp32(mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ src0, size_t size, int thread_count) {
  ParallelElementwiseLoopUnary<mllm_fp32_t, float32x4_t, __ScalarExp<mllm_fp32_t>, __VectorExp<float32x4_t>>::run(
      dst, src0, size, thread_count);
}

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
void ew_exp_fp16(mllm_fp16_t* __restrict__ dst, const mllm_fp16_t* __restrict__ src0, size_t size, int thread_count) {
  ParallelElementwiseLoopUnary<mllm_fp16_t, float16x8_t, __ScalarExp<mllm_fp16_t>, __VectorExp<float16x8_t>>::run(
      dst, src0, size, thread_count);
}
#endif

// sin operation
void ew_sin_fp32(mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ src0, size_t size, int thread_count) {
  MLLM_CONDITIONAL_PARALLEL_FOR(thread_count > 1, thread_count, i, 0, size, 1, { dst[i] = std::sin(src0[i]); });
}

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
void ew_sin_fp16(mllm_fp16_t* __restrict__ dst, const mllm_fp16_t* __restrict__ src0, size_t size, int thread_count) {
  MLLM_CONDITIONAL_PARALLEL_FOR(thread_count > 1, thread_count, i, 0, size, 1, { dst[i] = std::sin(src0[i]); });
}
#endif

// cos operation
void ew_cos_fp32(mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ src0, size_t size, int thread_count) {
  MLLM_CONDITIONAL_PARALLEL_FOR(thread_count > 1, thread_count, i, 0, size, 1, { dst[i] = std::cos(src0[i]); });
}

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
void ew_cos_fp16(mllm_fp16_t* __restrict__ dst, const mllm_fp16_t* __restrict__ src0, size_t size, int thread_count) {
  MLLM_CONDITIONAL_PARALLEL_FOR(thread_count > 1, thread_count, i, 0, size, 1, { dst[i] = std::cos(src0[i]); });
}
#endif

void clip_fp32(mllm_fp32_t* __restrict__ dst, const mllm_fp32_t* __restrict__ src, mllm_fp32_t min_val, mllm_fp32_t max_val,
               size_t size, int thread_count) {
  ParallelClipLoop<mllm_fp32_t, float32x4_t, __ScalarClip<mllm_fp32_t>, __VectorClip<float32x4_t>>::run(
      dst, src, min_val, max_val, size, thread_count);
}

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
void clip_fp16(mllm_fp16_t* __restrict__ dst, const mllm_fp16_t* __restrict__ src, mllm_fp16_t min_val, mllm_fp16_t max_val,
               size_t size, int thread_count) {
  ParallelClipLoop<mllm_fp16_t, float16x8_t, __ScalarClip<mllm_fp16_t>, __VectorClip<float16x8_t>>::run(
      dst, src, min_val, max_val, size, thread_count);
}
#endif

void clip_int8(mllm_int8_t* __restrict__ dst, const mllm_int8_t* __restrict__ src, mllm_int8_t min_val, mllm_int8_t max_val,
               size_t size, int thread_count) {
  ParallelClipLoop<mllm_int8_t, int8x16_t, __ScalarClip<mllm_int8_t>, __VectorClip<int8x16_t>>::run(dst, src, min_val, max_val,
                                                                                                    size, thread_count);
}

void clip_int16(mllm_int16_t* __restrict__ dst, const mllm_int16_t* __restrict__ src, mllm_int16_t min_val,
                mllm_int16_t max_val, size_t size, int thread_count) {
  ParallelClipLoop<mllm_int16_t, int16x8_t, __ScalarClip<mllm_int16_t>, __VectorClip<int16x8_t>>::run(
      dst, src, min_val, max_val, size, thread_count);
}

void clip_int32(mllm_int32_t* __restrict__ dst, const mllm_int32_t* __restrict__ src, mllm_int32_t min_val,
                mllm_int32_t max_val, size_t size, int thread_count) {
  ParallelClipLoop<mllm_int32_t, int32x4_t, __ScalarClip<mllm_int32_t>, __VectorClip<int32x4_t>>::run(
      dst, src, min_val, max_val, size, thread_count);
}

}  // namespace mllm::cpu::arm

#endif
