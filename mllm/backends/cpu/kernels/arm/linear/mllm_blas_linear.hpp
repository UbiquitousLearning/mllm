// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/DataTypes.hpp"

namespace mllm::cpu::arm {

// Linear FP32 function using mllm_blas
// MxK
// NxK
void mllm_blas_linear_fp32(const int BATCH, const int M, const int in_channel, const int out_channel,
                           const int Dst_batch_stride, const int A_batch_stride, mllm_fp32_t* __restrict__ dst,
                           const mllm_fp32_t* __restrict__ A, const mllm_fp32_t* __restrict__ B,
                           const mllm_fp32_t* __restrict__ Bias, int thread_count);

}  // namespace mllm::cpu::arm
