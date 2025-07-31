/**
 * @file reduce.cpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-31
 *
 */
#include "mllm/backends/cpu/kernels/arm/reduce.hpp"
#include "mllm/backends/cpu/kernels/arm/primitives.hpp"
#include "mllm/core/DataTypes.hpp"

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)

namespace mllm::cpu::arm {

void reduce_sum_fp32(mllm_fp32_t* dst, const mllm_fp32_t* src, size_t size, int32_t thread_count) {
  auto v = Reduce<mllm_fp32_t, float32x4_t, __ScalarAdd<float>, __VectorAdd<float32x4_t>, __VectorSumReduce<float32x4_t>>::run(
      src, size, thread_count);
  *dst = v;
}

void reduce_mul_fp32(mllm_fp32_t* dst, const mllm_fp32_t* src, size_t size, int32_t thread_count) {
  auto v = Reduce<mllm_fp32_t, float32x4_t, __ScalarMul<float>, __VectorMul<float32x4_t>, __VectorMulReduce<float32x4_t>>::run(
      src, size, thread_count);
  *dst = v;
}

void reduce_max_fp32(mllm_fp32_t* dst, const mllm_fp32_t* src, size_t size, int32_t thread_count) {
  auto v = Reduce<mllm_fp32_t, float32x4_t, __ScalarMax<float>, __VectorMax<float32x4_t>, __VectorMaxReduce<float32x4_t>>::run(
      src, size, thread_count);
  *dst = v;
}

void reduce_min_fp32(mllm_fp32_t* dst, const mllm_fp32_t* src, size_t size, int32_t thread_count) {
  auto v = Reduce<mllm_fp32_t, float32x4_t, __ScalarMin<float>, __VectorMin<float32x4_t>, __VectorMinReduce<float32x4_t>>::run(
      src, size, thread_count);
  *dst = v;
}

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
void reduce_sum_fp16(mllm_fp16_t* dst, const mllm_fp16_t* src, size_t size, int32_t thread_count) {
  auto v =
      Reduce<mllm_fp16_t, float16x8_t, __ScalarAdd<float16_t>, __VectorAdd<float16x8_t>, __VectorSumReduce<float16x8_t>>::run(
          src, size, thread_count);
  *dst = v;
}

void reduce_mul_fp16(mllm_fp16_t* dst, const mllm_fp16_t* src, size_t size, int32_t thread_count) {
  auto v =
      Reduce<mllm_fp16_t, float16x8_t, __ScalarMul<float16_t>, __VectorMul<float16x8_t>, __VectorMulReduce<float16x8_t>>::run(
          src, size, thread_count);
  *dst = v;
}

void reduce_max_fp16(mllm_fp16_t* dst, const mllm_fp16_t* src, size_t size, int32_t thread_count) {
  auto v =
      Reduce<mllm_fp16_t, float16x8_t, __ScalarMax<float16_t>, __VectorMax<float16x8_t>, __VectorMaxReduce<float16x8_t>>::run(
          src, size, thread_count);
  *dst = v;
}

void reduce_min_fp16(mllm_fp16_t* dst, const mllm_fp16_t* src, size_t size, int32_t thread_count) {
  auto v =
      Reduce<mllm_fp16_t, float16x8_t, __ScalarMin<float16_t>, __VectorMin<float16x8_t>, __VectorMinReduce<float16x8_t>>::run(
          src, size, thread_count);
  *dst = v;
}
#endif

void reduce_sum_int8(mllm_int8_t* dst, const mllm_int8_t* src, size_t size, int32_t thread_count) {
  auto v = Reduce<mllm_int8_t, int8x16_t, __ScalarAdd<int8_t>, __VectorAdd<int8x16_t>, __VectorSumReduce<int8x16_t>>::run(
      src, size, thread_count);
  *dst = v;
}

void reduce_mul_int8(mllm_int8_t* dst, const mllm_int8_t* src, size_t size, int32_t thread_count) {
  auto v = Reduce<mllm_int8_t, int8x16_t, __ScalarMul<int8_t>, __VectorMul<int8x16_t>, __VectorMulReduce<int8x16_t>>::run(
      src, size, thread_count);
  *dst = v;
}

void reduce_max_int8(mllm_int8_t* dst, const mllm_int8_t* src, size_t size, int32_t thread_count) {
  auto v = Reduce<mllm_int8_t, int8x16_t, __ScalarMax<int8_t>, __VectorMax<int8x16_t>, __VectorMaxReduce<int8x16_t>>::run(
      src, size, thread_count);
  *dst = v;
}

void reduce_min_int8(mllm_int8_t* dst, const mllm_int8_t* src, size_t size, int32_t thread_count) {
  auto v = Reduce<mllm_int8_t, int8x16_t, __ScalarMin<int8_t>, __VectorMin<int8x16_t>, __VectorMinReduce<int8x16_t>>::run(
      src, size, thread_count);
  *dst = v;
}

void reduce_sum_int16(mllm_int16_t* dst, const mllm_int16_t* src, size_t size, int32_t thread_count) {
  auto v = Reduce<mllm_int16_t, int16x8_t, __ScalarAdd<int16_t>, __VectorAdd<int16x8_t>, __VectorSumReduce<int16x8_t>>::run(
      src, size, thread_count);
  *dst = v;
}

void reduce_mul_int16(mllm_int16_t* dst, const mllm_int16_t* src, size_t size, int32_t thread_count) {
  auto v = Reduce<mllm_int16_t, int16x8_t, __ScalarMul<int16_t>, __VectorMul<int16x8_t>, __VectorMulReduce<int16x8_t>>::run(
      src, size, thread_count);
  *dst = v;
}

void reduce_max_int16(mllm_int16_t* dst, const mllm_int16_t* src, size_t size, int32_t thread_count) {
  auto v = Reduce<mllm_int16_t, int16x8_t, __ScalarMax<int16_t>, __VectorMax<int16x8_t>, __VectorMaxReduce<int16x8_t>>::run(
      src, size, thread_count);
  *dst = v;
}

void reduce_min_int16(mllm_int16_t* dst, const mllm_int16_t* src, size_t size, int32_t thread_count) {
  auto v = Reduce<mllm_int16_t, int16x8_t, __ScalarMin<int16_t>, __VectorMin<int16x8_t>, __VectorMinReduce<int16x8_t>>::run(
      src, size, thread_count);
  *dst = v;
}

void reduce_sum_int32(mllm_int32_t* dst, const mllm_int32_t* src, size_t size, int32_t thread_count) {
  auto v = Reduce<mllm_int32_t, int32x4_t, __ScalarAdd<int32_t>, __VectorAdd<int32x4_t>, __VectorSumReduce<int32x4_t>>::run(
      src, size, thread_count);
  *dst = v;
}

void reduce_mul_int32(mllm_int32_t* dst, const mllm_int32_t* src, size_t size, int32_t thread_count) {
  auto v = Reduce<mllm_int32_t, int32x4_t, __ScalarMul<int32_t>, __VectorMul<int32x4_t>, __VectorMulReduce<int32x4_t>>::run(
      src, size, thread_count);
  *dst = v;
}

void reduce_max_int32(mllm_int32_t* dst, const mllm_int32_t* src, size_t size, int32_t thread_count) {
  auto v = Reduce<mllm_int32_t, int32x4_t, __ScalarMax<int32_t>, __VectorMax<int32x4_t>, __VectorMaxReduce<int32x4_t>>::run(
      src, size, thread_count);
  *dst = v;
}

void reduce_min_int32(mllm_int32_t* dst, const mllm_int32_t* src, size_t size, int32_t thread_count) {
  auto v = Reduce<mllm_int32_t, int32x4_t, __ScalarMin<int32_t>, __VectorMin<int32x4_t>, __VectorMinReduce<int32x4_t>>::run(
      src, size, thread_count);
  *dst = v;
}

}  // namespace mllm::cpu::arm

#endif
