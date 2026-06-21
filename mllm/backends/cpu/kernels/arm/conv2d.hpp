// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/utils/CPUArchHelper.hpp"

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)

#include <arm_neon.h>

namespace mllm::cpu::arm {

//===----------------------------------------------------------------------===//
// Im2col.
//
// Reformat your inputs to im2col's input
// Reformat your weights to im2col's weight
// After those 2 parts, do gemm(weight, input)
//===----------------------------------------------------------------------===//
void conv2d_fp32_im2col_input(const float* input_data, const int channels, const int height, const int width,
                              const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h,
                              const int stride_w, const int dilation_h, const int dilation_w, float* col_data);

// Inputs weight format should in [Out_Channels, In_Channels, Kernel_H, Kernel_W]
// Output weight format should in [M x K]
//
//
// This kernel is not performance sensitive !!! We only need to pack weight once !
void conv2d_fp32_im2col_weight(const float* src_weight, float* packed_weight, int out_channels, int in_channels, int kernel_h,
                               int kernel_w);

}  // namespace mllm::cpu::arm
#endif
