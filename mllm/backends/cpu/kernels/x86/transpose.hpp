// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/DataTypes.hpp"
#include "mllm/utils/CPUArchHelper.hpp"

#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)

#include <cstddef>

namespace mllm::cpu::x86 {

void transpose_hw_wh_fp32(const mllm_fp32_t* __restrict X, mllm_fp32_t* __restrict Y, size_t H, size_t W);

void transpose_bshd_bhsd_fp32(const mllm_fp32_t* __restrict X, mllm_fp32_t* __restrict Y, size_t B, size_t S, size_t H,
                              size_t D);

void transpose_last_dims_fp32(const mllm_fp32_t* __restrict input, mllm_fp32_t* __restrict output, size_t batch, size_t dim0,
                              size_t dim1);

void transpose_hw_wh_int64(const mllm_int64_t* __restrict X, mllm_int64_t* __restrict Y, size_t H, size_t W);

void transpose_bshd_bhsd_int64(const mllm_int64_t* __restrict X, mllm_int64_t* __restrict Y, size_t B, size_t S, size_t H,
                               size_t D);

void transpose_last_dims_int64(const mllm_int64_t* __restrict input, mllm_int64_t* __restrict output, size_t batch, size_t dim0,
                               size_t dim1);

void permute_fp32(const mllm_fp32_t* __restrict input, mllm_fp32_t* __restrict output, const int* __restrict in_shape,
                  const int* __restrict perm, int ndim);

template<typename T>
void permute_generic(const T* __restrict input, T* __restrict output, const int* __restrict in_shape,
                     const int* __restrict perm, int ndim);

}  // namespace mllm::cpu::x86

#endif
