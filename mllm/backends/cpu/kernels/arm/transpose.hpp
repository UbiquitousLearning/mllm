// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/DataTypes.hpp"
#include "mllm/utils/CPUArchHelper.hpp"

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)

#include <cstddef>
#include <arm_neon.h>

namespace mllm::cpu::arm {

void transpose_hw_wh_fp32(const mllm_fp32_t* __restrict X, mllm_fp32_t* __restrict Y, size_t H, size_t W);

void transpose_bshd_bhsd_fp32(const mllm_fp32_t* __restrict X, mllm_fp32_t* __restrict Y, size_t B, size_t S, size_t H,
                              size_t D);

void transpose_last_dims_fp32(const mllm_fp32_t* __restrict input, mllm_fp32_t* __restrict output, size_t batch, size_t dim0,
                              size_t dim1);

void transpose_hw_wh_fp16(const mllm_fp16_t* __restrict X, mllm_fp16_t* __restrict Y, size_t H, size_t W);

void transpose_bshd_bhsd_fp16(const mllm_fp16_t* __restrict X, mllm_fp16_t* __restrict Y, size_t B, size_t S, size_t H,
                              size_t D);

void transpose_last_dims_fp16(const mllm_fp16_t* __restrict input, mllm_fp16_t* __restrict output, size_t batch, size_t dim0,
                              size_t dim1);

}  // namespace mllm::cpu::arm

#endif