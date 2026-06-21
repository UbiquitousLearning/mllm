// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/utils/CPUArchHelper.hpp"

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)

#include <arm_neon.h>

namespace mllm::cpu::arm {

void conv3d_fp32_baseline(const float* input_data, const float* kernel_data, const float* bias, float* output_data,
                          int batch_size, int in_c, int in_d, int in_h, int in_w, int out_c, int k_d, int k_h, int k_w,
                          int stride_d, int stride_h, int stride_w);

}

#endif
