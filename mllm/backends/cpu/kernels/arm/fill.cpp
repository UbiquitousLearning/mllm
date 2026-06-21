// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/kernels/arm/fill.hpp"
#include "mllm/utils/CPUArchHelper.hpp"

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)

#include <arm_neon.h>

namespace mllm::cpu::arm {
void fill_zeros(mllm_fp32_t* __restrict dst, size_t size, int thread_count) {
  constexpr size_t vec_size = 4;  // 4 floats in NEON
  const float32x4_t zero_vec = vdupq_n_f32(0.0f);

  size_t vec_end = size / vec_size * vec_size;
  size_t i = 0;

  // Vectorized fill
  for (; i < vec_end; i += vec_size) { vst1q_f32(dst + i, zero_vec); }

  // Handle remaining elements
  for (; i < size; ++i) { dst[i] = 0.0f; }
}

void fill_ones(mllm_fp32_t* __restrict dst, size_t size, int thread_count) {
  constexpr size_t vec_size = 4;  // 4 floats in NEON
  const float32x4_t ones_vec = vdupq_n_f32(1.0f);

  size_t vec_end = size / vec_size * vec_size;
  size_t i = 0;

  // Vectorized fill
  for (; i < vec_end; i += vec_size) { vst1q_f32(dst + i, ones_vec); }

  // Handle remaining elements
  for (; i < size; ++i) { dst[i] = 1.0f; }
}

void fill_specific_value(mllm_fp32_t* __restrict dst, size_t size, float value, int thread_count) {
  constexpr size_t vec_size = 4;  // 4 floats in NEON
  const float32x4_t value_vec = vdupq_n_f32(value);

  size_t vec_end = size / vec_size * vec_size;
  size_t i = 0;

  // Vectorized fill
  for (; i < vec_end; i += vec_size) { vst1q_f32(dst + i, value_vec); }

  // Handle remaining elements
  for (; i < size; ++i) { dst[i] = value; }
}

void fill_arange(mllm_fp32_t* __restrict dst, size_t size, float start, float end, float step, int thread_count) {
  constexpr size_t vec_size = 4;  // 4 floats in NEON

  size_t vec_end = size / vec_size * vec_size;
  size_t i = 0;

  // Vectorized arange
  float current_value = start;
  for (; i < vec_end; i += vec_size) {
    float32x4_t val_vec = {current_value, current_value + step, current_value + 2 * step, current_value + 3 * step};
    vst1q_f32(dst + i, val_vec);
    current_value += step * vec_size;
  }

  // Handle remaining elements
  for (; i < size; ++i) { dst[i] = start + i * step; }
}

void fill_random(mllm_fp32_t* __restrict dst, size_t size, float start, float end, uint64_t seed, int thread_count) {
  uint64_t state = seed;
  const uint64_t multiplier = 1103515245ULL;
  const uint64_t increment = 12345ULL;
  const uint64_t modulus = 1ULL << 31;  // 2^31

  float range = end - start;

  for (size_t i = 0; i < size; ++i) {
    state = (multiplier * state + increment) % modulus;

    float random_value = static_cast<float>(state) / static_cast<float>(modulus - 1);
    dst[i] = start + random_value * range;
  }
}

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
void fill_zeros_fp16(mllm_fp16_t* __restrict dst, size_t size, int thread_count) {
  constexpr size_t vec_size = 8;  // 8 float16_t in NEON
  const float16x8_t zero_vec = vdupq_n_f16(0.0f);

  size_t vec_end = size / vec_size * vec_size;
  size_t i = 0;

  // Vectorized fill
  for (; i < vec_end; i += vec_size) { vst1q_f16(dst + i, zero_vec); }

  // Handle remaining elements
  for (; i < size; ++i) { dst[i] = 0.0f; }
}

void fill_ones_fp16(mllm_fp16_t* __restrict dst, size_t size, int thread_count) {
  constexpr size_t vec_size = 8;  // 8 float16_t in NEON
  const float16x8_t ones_vec = vdupq_n_f16(1.0f);

  size_t vec_end = size / vec_size * vec_size;
  size_t i = 0;

  // Vectorized fill
  for (; i < vec_end; i += vec_size) { vst1q_f16(dst + i, ones_vec); }

  // Handle remaining elements
  for (; i < size; ++i) { dst[i] = 1.0f; }
}

void fill_specific_value_fp16(mllm_fp16_t* __restrict dst, size_t size, float value, int thread_count) {
  constexpr size_t vec_size = 8;  // 8 float16_t in NEON
  const float16x8_t value_vec = vdupq_n_f16(value);

  size_t vec_end = size / vec_size * vec_size;
  size_t i = 0;

  // Vectorized fill
  for (; i < vec_end; i += vec_size) { vst1q_f16(dst + i, value_vec); }

  // Handle remaining elements
  for (; i < size; ++i) { dst[i] = value; }
}

void fill_arange_fp16(mllm_fp16_t* __restrict dst, size_t size, float start, float end, float step, int thread_count) {
  constexpr size_t vec_size = 8;  // 8 float16_t in NEON

  size_t vec_end = size / vec_size * vec_size;
  size_t i = 0;

  // Vectorized arange
  float current_value = start;
  for (; i < vec_end; i += vec_size) {
    float16x8_t val_vec = {static_cast<float16_t>(current_value),
                           static_cast<float16_t>(current_value + step),
                           static_cast<float16_t>(current_value + 2 * step),
                           static_cast<float16_t>(current_value + 3 * step),
                           static_cast<float16_t>(current_value + 4 * step),
                           static_cast<float16_t>(current_value + 5 * step),
                           static_cast<float16_t>(current_value + 6 * step),
                           static_cast<float16_t>(current_value + 7 * step)};
    vst1q_f16(dst + i, val_vec);
    current_value += step * vec_size;
  }

  // Handle remaining elements
  for (; i < size; ++i) { dst[i] = start + i * step; }
}

void fill_random_fp16(mllm_fp16_t* __restrict dst, size_t size, float start, float end, uint64_t seed, int thread_count) {
  uint64_t state = seed;
  const uint64_t multiplier = 1103515245ULL;
  const uint64_t increment = 12345ULL;
  const uint64_t modulus = 1ULL << 31;  // 2^31

  float range = end - start;

  for (size_t i = 0; i < size; ++i) {
    state = (multiplier * state + increment) % modulus;

    float random_value = static_cast<float>(state) / static_cast<float>(modulus - 1);
    dst[i] = static_cast<float16_t>(start + random_value * range);
  }
}
#endif
}  // namespace mllm::cpu::arm

#endif