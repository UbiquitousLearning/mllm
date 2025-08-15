// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include "mllm/backends/cpu/kernels/arm/conv3d.hpp"

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)

#include <arm_neon.h>

namespace mllm::cpu::arm {

void conv3d_fp32_baseline(const float* input_data, const float* kernel_data, const float* bias, float* output_data, int in_c,
                          int in_d, int in_h, int in_w, int out_c, int k_d, int k_h, int k_w, int stride_d, int stride_h,
                          int stride_w) {
  const int out_d = (in_d - k_d) / stride_d + 1;
  const int out_h = (in_h - k_h) / stride_h + 1;
  const int out_w = (in_w - k_w) / stride_w + 1;

  const int in_d_stride = in_h * in_w;
  const int in_c_stride = in_d * in_d_stride;

  const int out_d_stride = out_h * out_w;
  const int out_c_stride = out_d * out_d_stride;

  const int kernel_d_stride = k_h * k_w;
  const int kernel_ic_stride = k_d * kernel_d_stride;
  const int kernel_oc_stride = in_c * kernel_ic_stride;

  for (int oc = 0; oc < out_c; ++oc) {
    for (int od = 0; od < out_d; ++od) {
      for (int oh = 0; oh < out_h; ++oh) {
        if (stride_w == 1) {
          int ow = 0;
          for (; ow + 3 < out_w; ow += 4) {
            float32x4_t out_vec = vdupq_n_f32(0.0f);
            for (int ic = 0; ic < in_c; ++ic) {
              for (int kd = 0; kd < k_d; ++kd) {
                for (int kh = 0; kh < k_h; ++kh) {
                  const int id_start = od * stride_d + kd;
                  const int ih_start = oh * stride_h + kh;
                  const float* in_ptr_base = input_data + ic * in_c_stride + id_start * in_d_stride + ih_start * in_w + ow;

                  for (int kw = 0; kw < k_w; ++kw) {
                    const float k_val =
                        kernel_data[oc * kernel_oc_stride + ic * kernel_ic_stride + kd * kernel_d_stride + kh * k_w + kw];
                    float32x4_t k_vec = vdupq_n_f32(k_val);
                    float32x4_t in_vec = vld1q_f32(in_ptr_base + kw);
                    out_vec = vfmaq_f32(out_vec, in_vec, k_vec);
                  }
                }
              }
            }
            if (bias != nullptr) { out_vec = vaddq_f32(out_vec, vdupq_n_f32(bias[oc])); }
            float* out_ptr = output_data + oc * out_c_stride + od * out_d_stride + oh * out_w + ow;
            vst1q_f32(out_ptr, out_vec);
          }
          for (; ow < out_w; ++ow) {
            float sum = 0.0f;
            for (int ic = 0; ic < in_c; ++ic) {
              for (int kd = 0; kd < k_d; ++kd) {
                for (int kh = 0; kh < k_h; ++kh) {
                  for (int kw = 0; kw < k_w; ++kw) {
                    sum += input_data[ic * in_c_stride + (od * stride_d + kd) * in_d_stride + (oh * stride_h + kh) * in_w
                                      + (ow * stride_w + kw)]
                           * kernel_data[oc * kernel_oc_stride + ic * kernel_ic_stride + kd * kernel_d_stride + kh * k_w + kw];
                  }
                }
              }
            }
            if (bias != nullptr) sum += bias[oc];
            output_data[oc * out_c_stride + od * out_d_stride + oh * out_w + ow] = sum;
          }
        } else {
          for (int ow = 0; ow < out_w; ++ow) {
            float sum = 0.0f;
            for (int ic = 0; ic < in_c; ++ic) {
              for (int kd = 0; kd < k_d; ++kd) {
                for (int kh = 0; kh < k_h; ++kh) {
                  for (int kw = 0; kw < k_w; ++kw) {
                    const int id = od * stride_d + kd;
                    const int ih = oh * stride_h + kh;
                    const int iw = ow * stride_w + kw;
                    sum += input_data[ic * in_c_stride + id * in_d_stride + ih * in_w + iw]
                           * kernel_data[oc * kernel_oc_stride + ic * kernel_ic_stride + kd * kernel_d_stride + kh * k_w + kw];
                  }
                }
              }
            }
            if (bias != nullptr) { sum += bias[oc]; }
            output_data[oc * out_c_stride + od * out_d_stride + oh * out_w + ow] = sum;
          }
        }
      }
    }
  }
}

}  // namespace mllm::cpu::arm

#endif
