// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include "mllm/backends/cpu/kernels/arm/conv2d.hpp"

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)

#include <arm_neon.h>

namespace mllm::cpu::arm {

void conv2d_fp32_im2col_input(const float* input_data, const int channels, const int height, const int width,
                              const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h,
                              const int stride_w, const int dilation_h, const int dilation_w, float* col_data) {
  const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;

  const float32x4_t vzero = vdupq_n_f32(0.0f);

  for (int channel = 0; channel < channels; ++channel) {
    for (int kernel_y = 0; kernel_y < kernel_h; ++kernel_y) {
      for (int kernel_x = 0; kernel_x < kernel_w; ++kernel_x) {
        const int input_start_y = -pad_h + kernel_y * dilation_h;
        const int input_start_x = -pad_w + kernel_x * dilation_w;

        for (int out_y = 0; out_y < output_h; ++out_y) {
          const int cur_input_y = input_start_y + out_y * stride_h;

          if (static_cast<unsigned>(cur_input_y) >= static_cast<unsigned>(height)) {
            for (int out_x = 0; out_x < output_w; out_x += 4) {
              if (out_x + 3 < output_w) {
                vst1q_f32(col_data, vzero);
                col_data += 4;
              } else {
                for (int i = 0; i < output_w - out_x; ++i) { *col_data++ = 0.0f; }
              }
            }
          } else {
            int out_x = 0;
            for (; out_x + 3 < output_w; out_x += 4) {
              const int input_x0 = input_start_x + (out_x + 0) * stride_w;
              const int input_x1 = input_start_x + (out_x + 1) * stride_w;
              const int input_x2 = input_start_x + (out_x + 2) * stride_w;
              const int input_x3 = input_start_x + (out_x + 3) * stride_w;

              const float val0 = (static_cast<unsigned>(input_x0) < static_cast<unsigned>(width))
                                     ? input_data[cur_input_y * width + input_x0]
                                     : 0.0f;
              const float val1 = (static_cast<unsigned>(input_x1) < static_cast<unsigned>(width))
                                     ? input_data[cur_input_y * width + input_x1]
                                     : 0.0f;
              const float val2 = (static_cast<unsigned>(input_x2) < static_cast<unsigned>(width))
                                     ? input_data[cur_input_y * width + input_x2]
                                     : 0.0f;
              const float val3 = (static_cast<unsigned>(input_x3) < static_cast<unsigned>(width))
                                     ? input_data[cur_input_y * width + input_x3]
                                     : 0.0f;

              float32x4_t v_data = {val0, val1, val2, val3};
              vst1q_f32(col_data, v_data);
              col_data += 4;
            }

            for (; out_x < output_w; ++out_x) {
              const int cur_input_x = input_start_x + out_x * stride_w;
              if (static_cast<unsigned>(cur_input_x) < static_cast<unsigned>(width)) {
                *col_data++ = input_data[cur_input_y * width + cur_input_x];
              } else {
                *col_data++ = 0.0f;
              }
            }
          }
        }
      }
    }
    input_data += channel_size;
  }
}

void conv2d_fp32_im2col_weight(const float* src_weight, float* packed_weight, int out_channels, int in_channels, int kernel_h,
                               int kernel_w) {
  // Original Weight: [Out, In, Kh, Kw]
  // Packed Weight: [Out, In*Kh*Kw]
  for (int o = 0; o < out_channels; ++o) {
    for (int i = 0; i < in_channels; ++i) {
      for (int h = 0; h < kernel_h; ++h) {
        for (int w = 0; w < kernel_w; ++w) {
          int src_idx = o * (in_channels * kernel_h * kernel_w) + i * (kernel_h * kernel_w) + h * kernel_w + w;
          int dst_idx = o * (in_channels * kernel_h * kernel_w) + i * (kernel_h * kernel_w) + h * kernel_w + w;
          packed_weight[dst_idx] = src_weight[src_idx];
        }
      }
    }
  }
}

}  // namespace mllm::cpu::arm

#endif
