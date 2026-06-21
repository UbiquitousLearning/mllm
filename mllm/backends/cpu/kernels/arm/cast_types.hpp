// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/DataTypes.hpp"
#include "mllm/utils/CPUArchHelper.hpp"

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)

#include <arm_neon.h>

namespace mllm::cpu::arm {

template<typename __From, typename __To>
struct CastAny {
  static inline void cast(const __From* __restrict src, __To* __restrict dst, int len, int thread_count) {
    for (int i = 0; i < len; i++) { dst[i] = static_cast<__To>(src[i]); }
  }
};

void int8_2_fp16(const mllm_int8_t* __restrict src, mllm_fp16_t* __restrict dst, int len, int thread_count);

void int32_2_fp16(const mllm_int32_t* __restrict src, mllm_fp16_t* __restrict dst, int len, int thread_count);

void int8_2_fp32(const mllm_int8_t* __restrict src, mllm_fp32_t* __restrict dst, int len, int thread_count);

void int32_2_fp32(const mllm_int32_t* __restrict src, mllm_fp32_t* __restrict dst, int len, int thread_count);

void fp32_2_fp16(const mllm_fp32_t* __restrict src, mllm_fp16_t* __restrict dst, int len, int thread_count);

void fp16_2_fp32(const mllm_fp16_t* __restrict src, mllm_fp32_t* __restrict dst, int len, int thread_count);

template<>
struct CastAny<mllm_fp16_t, mllm_fp32_t> {
  static inline void cast(const mllm_fp16_t* __restrict src, mllm_fp32_t* __restrict dst, int len, int thread_count) {
    fp16_2_fp32(src, dst, len, thread_count);
  }
};

template<>
struct CastAny<mllm_fp32_t, mllm_fp16_t> {
  static inline void cast(const mllm_fp32_t* __restrict src, mllm_fp16_t* __restrict dst, int len, int thread_count) {
    fp32_2_fp16(src, dst, len, thread_count);
  }
};

template<>
struct CastAny<int8_t, mllm_fp16_t> {
  static inline void cast(const int8_t* __restrict src, mllm_fp16_t* __restrict dst, int len, int thread_count) {
    int8_2_fp16(src, dst, len, thread_count);
  }
};

template<>
struct CastAny<int32_t, mllm_fp16_t> {
  static inline void cast(const int32_t* __restrict src, mllm_fp16_t* __restrict dst, int len, int thread_count) {
    int32_2_fp16(src, dst, len, thread_count);
  }
};

template<>
struct CastAny<int8_t, mllm_fp32_t> {
  static inline void cast(const int8_t* __restrict src, mllm_fp32_t* __restrict dst, int len, int thread_count) {
    int8_2_fp32(src, dst, len, thread_count);
  }
};

template<>
struct CastAny<int32_t, mllm_fp32_t> {
  static inline void cast(const int32_t* __restrict src, mllm_fp32_t* __restrict dst, int len, int thread_count) {
    int32_2_fp32(src, dst, len, thread_count);
  }
};

}  // namespace mllm::cpu::arm

#endif
