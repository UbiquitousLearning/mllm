// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <cassert>

#include "mllm/utils/Common.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/utils/CPUArchHelper.hpp"
#include "mllm/utils/UnsafeMacros.hpp"

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
#include <cmath>
#include <arm_neon.h>
#include "mllm/backends/cpu/kernels/arm/math.hpp"
#elif defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
#else
#endif

namespace mllm::cpu::paged_attn::details {

struct __AnyArchTag {};
using any_arch_tag = __AnyArchTag;

struct __X86ArchTag {};
using x86_arch_tag = __X86ArchTag;

struct __ArmArchTag {};
using arm_arch_tag = __ArmArchTag;

template<typename __ArchTag, DataTypes __LhsDataType, DataTypes __RhsDataType, DataTypes __OutDataType>
struct VectorDotProduct {
  static MLLM_FORCE_INLINE void run(const void* __lhs, const void* __rhs, void* __out, size_t len) {}
};

template<>
struct VectorDotProduct<any_arch_tag, kFloat32, kFloat32, kFloat32> {
  __MLLM_UNSAFE_OPT_BEGIN_O3  // Do not open fast math here
      static MLLM_FORCE_INLINE void
      run(const mllm_fp32_t* __restrict__ __lhs, const mllm_fp32_t* __restrict__ __rhs, mllm_fp32_t* __restrict__ __out,
          size_t len) {
    mllm_fp32_t sum = 0.0f;
    for (size_t i = 0; i < len; ++i) { sum += __lhs[i] * __rhs[i]; }
    *__out = sum;
  }
  __MLLM_UNSAFE_OPT_END
};

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
template<>
struct VectorDotProduct<arm_arch_tag, kFloat32, kFloat32, kFloat32> {
  static MLLM_FORCE_INLINE void run(const mllm_fp32_t* __restrict__ __lhs, const mllm_fp32_t* __restrict__ __rhs,
                                    mllm_fp32_t* __restrict__ __out, size_t len) {
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 3 < len; i += 4) {
      float32x4_t lhs_vec = vld1q_f32(__lhs + i);
      float32x4_t rhs_vec = vld1q_f32(__rhs + i);
      sum_vec = vmlaq_f32(sum_vec, lhs_vec, rhs_vec);
    }
    float sum = vaddvq_f32(sum_vec);
    for (; i < len; ++i) { sum += __lhs[i] * __rhs[i]; }
    *__out = sum;
  }
};
#elif defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
template<>
struct VectorDotProduct<x86_arch_tag, kFloat32, kFloat32, kFloat32> {
  static MLLM_FORCE_INLINE void run(const mllm_fp32_t* __restrict__ __lhs, const mllm_fp32_t* __restrict__ __rhs,
                                    mllm_fp32_t* __restrict__ __out, size_t len) {
    assert(false && "not impl");
  }
};
#else
#error "Use highway to impl your own dot product"
#endif

template<typename __ArchTag, DataTypes __DType, bool __HighPrecisionExp = false>
struct Softmax {
  static MLLM_FORCE_INLINE void run(void* __restrict__ __in, size_t len) {}
};

template<>
struct Softmax<any_arch_tag, kFloat32, false> {
  static MLLM_FORCE_INLINE void run(mllm_fp32_t* __restrict__ __in, size_t len) {
    // Find the maximum value to avoid overflow
    mllm_fp32_t max_val = __in[0];
    for (size_t i = 1; i < len; ++i) {
      if (__in[i] > max_val) { max_val = __in[i]; }
    }

    // Compute exp(x - max_val) for each element and sum them
    mllm_fp32_t sum = 0.0f;
    for (size_t i = 0; i < len; ++i) {
      mllm_fp32_t exp_val = expf(__in[i] - max_val);
      const_cast<mllm_fp32_t*>(__in)[i] = exp_val;
      sum += exp_val;
    }

    // Normalize by the sum
    mllm_fp32_t inv_sum = 1.0f / sum;
    for (size_t i = 0; i < len; ++i) { const_cast<mllm_fp32_t*>(__in)[i] *= inv_sum; }
  }
};

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)

template<>
struct Softmax<arm_arch_tag, kFloat32, false> {
  static MLLM_FORCE_INLINE void run(mllm_fp32_t* __restrict__ __in, size_t len) {
    // Find the maximum value to avoid overflow
    float32x4_t max_vec = vdupq_n_f32(std::numeric_limits<float>::lowest());
    size_t i = 0;

    // Process 4 elements at a time with NEON
    for (; i + 3 < len; i += 4) {
      float32x4_t data_vec = vld1q_f32(__in + i);
      max_vec = vmaxq_f32(max_vec, data_vec);
    }

    // Reduce to find the maximum among the 4 values
    float max_val = vmaxvq_f32(max_vec);

    // Handle remaining elements
    for (; i < len; ++i) { max_val = std::max(max_val, __in[i]); }

    // Calculate exp(x - max_val) and sum
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    float32x4_t max_vec_bcast = vdupq_n_f32(max_val);
    i = 0;

    // Process 4 elements at a time
    for (; i + 3 < len; i += 4) {
      float32x4_t data_vec = vld1q_f32(__in + i);
      float32x4_t sub_vec = vsubq_f32(data_vec, max_vec_bcast);
      float32x4_t exp_vec = mllm::cpu::arm::vexpq_fast_f32(sub_vec);
      sum_vec = vaddq_f32(sum_vec, exp_vec);
      vst1q_f32(__in + i, exp_vec);
    }

    // Reduce to get the sum
    float sum = vaddvq_f32(sum_vec);

    // Handle remaining elements
    for (; i < len; ++i) {
      float exp_val = expf(__in[i] - max_val);
      __in[i] = exp_val;
      sum += exp_val;
    }

    // Normalize by dividing by sum
    float inv_sum = 1.0f / sum;
    float32x4_t inv_sum_vec = vdupq_n_f32(inv_sum);
    i = 0;

    // Process 4 elements at a time
    for (; i + 3 < len; i += 4) {
      float32x4_t data_vec = vld1q_f32(__in + i);
      float32x4_t result_vec = vmulq_f32(data_vec, inv_sum_vec);
      vst1q_f32(__in + i, result_vec);
    }

    // Handle remaining elements
    for (; i < len; ++i) { __in[i] *= inv_sum; }
  }
};
#elif defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
template<>
struct Softmax<x86_arch_tag, kFloat32, false> {
  static MLLM_FORCE_INLINE void run(mllm_fp32_t* __in, size_t len) { assert(false && "not impl"); }
};
#else
#error "Use highway to impl your own softmax"
#endif

}  // namespace mllm::cpu::paged_attn::details
