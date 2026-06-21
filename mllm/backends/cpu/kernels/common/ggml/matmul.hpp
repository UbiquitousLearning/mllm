// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/Tensor.hpp"
namespace mllm::cpu::ggml {

/**
 * @brief Matmul implement using gemm & gemv in ggml
 * Migrated from src/backends/cpu/compute/Matmul.hpp in mllm v1
 */
void mat_mul(const Tensor& src0_, const Tensor& src1, Tensor& dst, bool support_bias, Tensor* bias = nullptr,
             bool transpose0 = false, bool transpose1 = true, int thread_count = 4);
}  // namespace mllm::cpu::ggml
