// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/DataTypes.hpp"
#include "mllm/utils/CPUArchHelper.hpp"

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)

#include <cstddef>
#include <arm_neon.h>

namespace mllm::cpu::arm {

void permute_fp32(const mllm_fp32_t* __restrict__ input, mllm_fp32_t* __restrict__ output, const int* __restrict__ in_shape,
                  const int* __restrict__ perm, int ndim);

void permute_fp16(const mllm_fp16_t* __restrict__ input, mllm_fp16_t* __restrict__ output, const int* __restrict__ in_shape,
                  const int* __restrict__ perm, int ndim);

// Generic permute function for other data types
template<typename T>
void permute_generic(const T* __restrict__ input, T* __restrict__ output, const int* __restrict__ in_shape,
                     const int* __restrict__ perm, int ndim);

}  // namespace mllm::cpu::arm

#endif