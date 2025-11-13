// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/kernels/arm/transpose.hpp"

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)

#include <cstddef>
#include <arm_neon.h>

namespace mllm::cpu::arm {

void transpose_hw_wh_fp32(const mllm_fp32_t* __restrict X, mllm_fp32_t* __restrict Y, size_t H, size_t W) {
  for (size_t i = 0; i + 4 <= H; i += 4) {
    for (size_t j = 0; j + 4 <= W; j += 4) {
      float32x4_t r0 = vld1q_f32(X + i * W + j);
      float32x4_t r1 = vld1q_f32(X + (i + 1) * W + j);
      float32x4_t r2 = vld1q_f32(X + (i + 2) * W + j);
      float32x4_t r3 = vld1q_f32(X + (i + 3) * W + j);

      float32x4x2_t r0r1 = vtrnq_f32(r0, r1);
      float32x4x2_t r2r3 = vtrnq_f32(r2, r3);

      float32x4_t col0 = vcombine_f32(vget_low_f32(r0r1.val[0]), vget_low_f32(r2r3.val[0]));
      float32x4_t col1 = vcombine_f32(vget_low_f32(r0r1.val[1]), vget_low_f32(r2r3.val[1]));
      float32x4_t col2 = vcombine_f32(vget_high_f32(r0r1.val[0]), vget_high_f32(r2r3.val[0]));
      float32x4_t col3 = vcombine_f32(vget_high_f32(r0r1.val[1]), vget_high_f32(r2r3.val[1]));

      vst1q_f32(Y + j * H + i, col0);
      vst1q_f32(Y + (j + 1) * H + i, col1);
      vst1q_f32(Y + (j + 2) * H + i, col2);
      vst1q_f32(Y + (j + 3) * H + i, col3);
    }

    size_t j_remain = W - (W % 4);
    for (size_t j = j_remain; j < W; ++j) {
      float32x4_t col = {X[i * W + j], X[(i + 1) * W + j], X[(i + 2) * W + j], X[(i + 3) * W + j]};
      vst1q_f32(Y + j * H + i, col);
    }
  }

  size_t i_remain = H - (H % 4);
  for (size_t j = 0; j < W; ++j) {
    for (size_t i = i_remain; i < H; ++i) { Y[j * H + i] = X[i * W + j]; }
  }
}

void transpose_bshd_bhsd_fp32(const mllm_fp32_t* __restrict X, mllm_fp32_t* __restrict Y, size_t B, size_t S, size_t H,
                              size_t D) {
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      for (int s = 0; s < S; ++s) {
        int d;
        for (d = 0; d <= D - 4; d += 4) {
          // B, S, H, D
          const mllm_fp32_t* src_ptr = X + b * S * H * D + s * H * D + h * D + d;

          // B, H, S, D
          mllm_fp32_t* dst_ptr = Y + b * H * S * D + h * S * D + s * D + d;

          float32x4_t data;
          data = vld1q_f32(src_ptr);
          vst1q_f32(dst_ptr, data);
        }
        for (; d < D; ++d) {
          const mllm_fp32_t* src_ptr = X + b * S * H * D + s * H * D + h * D + d;
          mllm_fp32_t* dst_ptr = Y + b * H * S * D + h * S * D + s * D + d;
          *dst_ptr = *src_ptr;
        }
      }
    }
  }
}

void transpose_hw_wh_fp16(const mllm_fp16_t* __restrict X, mllm_fp16_t* __restrict Y, size_t H, size_t W) {
  for (size_t i = 0; i + 4 <= H; i += 4) {
    for (size_t j = 0; j + 8 <= W; j += 8) {
      float16x8_t r0 = vld1q_f16(X + i * W + j);
      float16x8_t r1 = vld1q_f16(X + (i + 1) * W + j);
      float16x8_t r2 = vld1q_f16(X + (i + 2) * W + j);
      float16x8_t r3 = vld1q_f16(X + (i + 3) * W + j);

      float16x4_t r00 = vget_low_f16(r0);
      float16x4_t r01 = vget_high_f16(r0);
      float16x4_t r10 = vget_low_f16(r1);
      float16x4_t r11 = vget_high_f16(r1);
      float16x4_t r20 = vget_low_f16(r2);
      float16x4_t r21 = vget_high_f16(r2);
      float16x4_t r30 = vget_low_f16(r3);
      float16x4_t r31 = vget_high_f16(r3);

      float16x4x2_t tr00 = vtrn_f16(r00, r10);
      float16x4x2_t tr01 = vtrn_f16(r20, r30);
      float32x2x2_t tr00_32 = vtrn_f32(vreinterpret_f32_f16(tr00.val[0]), vreinterpret_f32_f16(tr01.val[0]));
      float32x2x2_t tr01_32 = vtrn_f32(vreinterpret_f32_f16(tr00.val[1]), vreinterpret_f32_f16(tr01.val[1]));

      float16x4x2_t tr10 = vtrn_f16(r01, r11);
      float16x4x2_t tr11 = vtrn_f16(r21, r31);
      float32x2x2_t tr10_32 = vtrn_f32(vreinterpret_f32_f16(tr10.val[0]), vreinterpret_f32_f16(tr11.val[0]));
      float32x2x2_t tr11_32 = vtrn_f32(vreinterpret_f32_f16(tr10.val[1]), vreinterpret_f32_f16(tr11.val[1]));

      float16x4_t col0 = vreinterpret_f16_f32(tr00_32.val[0]);
      float16x4_t col1 = vreinterpret_f16_f32(tr01_32.val[0]);
      float16x4_t col2 = vreinterpret_f16_f32(tr00_32.val[1]);
      float16x4_t col3 = vreinterpret_f16_f32(tr01_32.val[1]);
      float16x4_t col4 = vreinterpret_f16_f32(tr10_32.val[0]);
      float16x4_t col5 = vreinterpret_f16_f32(tr11_32.val[0]);
      float16x4_t col6 = vreinterpret_f16_f32(tr10_32.val[1]);
      float16x4_t col7 = vreinterpret_f16_f32(tr11_32.val[1]);

      vst1_f16(Y + (j)*H + i, col0);
      vst1_f16(Y + (j + 1) * H + i, col1);
      vst1_f16(Y + (j + 2) * H + i, col2);
      vst1_f16(Y + (j + 3) * H + i, col3);
      vst1_f16(Y + (j + 4) * H + i, col4);
      vst1_f16(Y + (j + 5) * H + i, col5);
      vst1_f16(Y + (j + 6) * H + i, col6);
      vst1_f16(Y + (j + 7) * H + i, col7);
    }

    size_t j_remain = W - (W % 8);
    for (size_t j = j_remain; j < W; ++j) {
      for (size_t ii = 0; ii < 4; ++ii) { Y[j * H + i + ii] = X[(i + ii) * W + j]; }
    }
  }

  size_t i_remain = H - (H % 4);
  for (size_t j = 0; j < W; ++j) {
    for (size_t i = i_remain; i < H; ++i) { Y[j * H + i] = X[i * W + j]; }
  }
}

void transpose_bshd_bhsd_fp16(const mllm_fp16_t* __restrict X, mllm_fp16_t* __restrict Y, size_t B, size_t S, size_t H,
                              size_t D) {
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      for (int s = 0; s < S; ++s) {
        int d;
        for (d = 0; d <= D - 8; d += 8) {
          // B, S, H, D
          const mllm_fp16_t* src_ptr = X + b * S * H * D + s * H * D + h * D + d;

          // B, H, S, D
          mllm_fp16_t* dst_ptr = Y + b * H * S * D + h * S * D + s * D + d;

          float16x8_t data;
          data = vld1q_f16(src_ptr);
          vst1q_f16(dst_ptr, data);
        }
        for (; d < D; ++d) {
          const mllm_fp16_t* src_ptr = X + b * S * H * D + s * H * D + h * D + d;
          mllm_fp16_t* dst_ptr = Y + b * H * S * D + h * S * D + s * D + d;
          *dst_ptr = *src_ptr;
        }
      }
    }
  }
}

void transpose_last_dims_fp32(const mllm_fp32_t* __restrict input, mllm_fp32_t* __restrict output, size_t batch, size_t dim0,
                              size_t dim1) {
  for (size_t b = 0; b < batch; b++) {
    const mllm_fp32_t* input_batch = input + b * dim0 * dim1;
    mllm_fp32_t* output_batch = output + b * dim0 * dim1;

    for (size_t i = 0; i + 4 <= dim0; i += 4) {
      for (size_t j = 0; j + 4 <= dim1; j += 4) {
        float32x4_t r0 = vld1q_f32(input_batch + i * dim1 + j);
        float32x4_t r1 = vld1q_f32(input_batch + (i + 1) * dim1 + j);
        float32x4_t r2 = vld1q_f32(input_batch + (i + 2) * dim1 + j);
        float32x4_t r3 = vld1q_f32(input_batch + (i + 3) * dim1 + j);

        float32x4x2_t r0r1 = vtrnq_f32(r0, r1);
        float32x4x2_t r2r3 = vtrnq_f32(r2, r3);

        float32x4_t col0 = vcombine_f32(vget_low_f32(r0r1.val[0]), vget_low_f32(r2r3.val[0]));
        float32x4_t col1 = vcombine_f32(vget_low_f32(r0r1.val[1]), vget_low_f32(r2r3.val[1]));
        float32x4_t col2 = vcombine_f32(vget_high_f32(r0r1.val[0]), vget_high_f32(r2r3.val[0]));
        float32x4_t col3 = vcombine_f32(vget_high_f32(r0r1.val[1]), vget_high_f32(r2r3.val[1]));

        vst1q_f32(output_batch + j * dim0 + i, col0);
        vst1q_f32(output_batch + (j + 1) * dim0 + i, col1);
        vst1q_f32(output_batch + (j + 2) * dim0 + i, col2);
        vst1q_f32(output_batch + (j + 3) * dim0 + i, col3);
      }

      size_t j_remain = dim1 - (dim1 % 4);
      for (size_t j = j_remain; j < dim1; ++j) {
        float32x4_t col = {input_batch[i * dim1 + j], input_batch[(i + 1) * dim1 + j], input_batch[(i + 2) * dim1 + j],
                           input_batch[(i + 3) * dim1 + j]};
        vst1q_f32(output_batch + j * dim0 + i, col);
      }
    }

    size_t i_remain = dim0 - (dim0 % 4);
    for (size_t j = 0; j < dim1; ++j) {
      for (size_t i = i_remain; i < dim0; ++i) { output_batch[j * dim0 + i] = input_batch[i * dim1 + j]; }
    }
  }
}

void transpose_last_dims_fp16(const mllm_fp16_t* __restrict input, mllm_fp16_t* __restrict output, size_t batch, size_t dim0,
                              size_t dim1) {
  for (size_t b = 0; b < batch; b++) {
    const mllm_fp16_t* input_batch = input + b * dim0 * dim1;
    mllm_fp16_t* output_batch = output + b * dim0 * dim1;

    for (size_t i = 0; i + 4 <= dim0; i += 4) {
      for (size_t j = 0; j + 8 <= dim1; j += 8) {
        float16x8_t r0 = vld1q_f16(input_batch + i * dim1 + j);
        float16x8_t r1 = vld1q_f16(input_batch + (i + 1) * dim1 + j);
        float16x8_t r2 = vld1q_f16(input_batch + (i + 2) * dim1 + j);
        float16x8_t r3 = vld1q_f16(input_batch + (i + 3) * dim1 + j);

        float16x4_t r00 = vget_low_f16(r0);
        float16x4_t r01 = vget_high_f16(r0);
        float16x4_t r10 = vget_low_f16(r1);
        float16x4_t r11 = vget_high_f16(r1);
        float16x4_t r20 = vget_low_f16(r2);
        float16x4_t r21 = vget_high_f16(r2);
        float16x4_t r30 = vget_low_f16(r3);
        float16x4_t r31 = vget_high_f16(r3);

        float16x4x2_t tr00 = vtrn_f16(r00, r10);
        float16x4x2_t tr01 = vtrn_f16(r20, r30);
        float32x2x2_t tr00_32 = vtrn_f32(vreinterpret_f32_f16(tr00.val[0]), vreinterpret_f32_f16(tr01.val[0]));
        float32x2x2_t tr01_32 = vtrn_f32(vreinterpret_f32_f16(tr00.val[1]), vreinterpret_f32_f16(tr01.val[1]));

        float16x4x2_t tr10 = vtrn_f16(r01, r11);
        float16x4x2_t tr11 = vtrn_f16(r21, r31);
        float32x2x2_t tr10_32 = vtrn_f32(vreinterpret_f32_f16(tr10.val[0]), vreinterpret_f32_f16(tr11.val[0]));
        float32x2x2_t tr11_32 = vtrn_f32(vreinterpret_f32_f16(tr10.val[1]), vreinterpret_f32_f16(tr11.val[1]));

        float16x4_t col0 = vreinterpret_f16_f32(tr00_32.val[0]);
        float16x4_t col1 = vreinterpret_f16_f32(tr01_32.val[0]);
        float16x4_t col2 = vreinterpret_f16_f32(tr00_32.val[1]);
        float16x4_t col3 = vreinterpret_f16_f32(tr01_32.val[1]);
        float16x4_t col4 = vreinterpret_f16_f32(tr10_32.val[0]);
        float16x4_t col5 = vreinterpret_f16_f32(tr11_32.val[0]);
        float16x4_t col6 = vreinterpret_f16_f32(tr10_32.val[1]);
        float16x4_t col7 = vreinterpret_f16_f32(tr11_32.val[1]);

        vst1_f16(output_batch + (j)*dim0 + i, col0);
        vst1_f16(output_batch + (j + 1) * dim0 + i, col1);
        vst1_f16(output_batch + (j + 2) * dim0 + i, col2);
        vst1_f16(output_batch + (j + 3) * dim0 + i, col3);
        vst1_f16(output_batch + (j + 4) * dim0 + i, col4);
        vst1_f16(output_batch + (j + 5) * dim0 + i, col5);
        vst1_f16(output_batch + (j + 6) * dim0 + i, col6);
        vst1_f16(output_batch + (j + 7) * dim0 + i, col7);
      }

      size_t j_remain = dim1 - (dim1 % 8);
      for (size_t j = j_remain; j < dim1; ++j) {
        for (size_t ii = 0; ii < 4; ++ii) { output_batch[j * dim0 + i + ii] = input_batch[(i + ii) * dim1 + j]; }
      }
    }

    size_t i_remain = dim0 - (dim0 % 4);
    for (size_t j = 0; j < dim1; ++j) {
      for (size_t i = i_remain; i < dim0; ++i) { output_batch[j * dim0 + i] = input_batch[i * dim1 + j]; }
    }
  }
}

void transpose_hw_wh_int64(const mllm_int64_t* __restrict__ X, mllm_int64_t* __restrict__ Y, size_t H, size_t W) {
  for (size_t i = 0; i + 2 <= H; i += 2) {
    for (size_t j = 0; j + 2 <= W; j += 2) {
      int64x2_t r0 = vld1q_s64(X + i * W + j);
      int64x2_t r1 = vld1q_s64(X + (i + 1) * W + j);

      int64x2_t col0 = vcombine_s64(vget_low_s64(r0), vget_low_s64(r1));
      int64x2_t col1 = vcombine_s64(vget_high_s64(r0), vget_high_s64(r1));

      vst1q_s64(Y + j * H + i, col0);
      vst1q_s64(Y + (j + 1) * H + i, col1);
    }

    size_t j_remain = W - (W % 2);
    for (size_t j = j_remain; j < W; ++j) {
      int64x2_t col = {X[i * W + j], X[(i + 1) * W + j]};
      vst1q_s64(Y + j * H + i, col);
    }
  }

  size_t i_remain = H - (H % 2);
  for (size_t j = 0; j < W; ++j) {
    for (size_t i = i_remain; i < H; ++i) { Y[j * H + i] = X[i * W + j]; }
  }
}

void transpose_bshd_bhsd_int64(const mllm_int64_t* __restrict__ X, mllm_int64_t* __restrict__ Y, size_t B, size_t S, size_t H,
                               size_t D) {
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      for (int s = 0; s < S; ++s) {
        int d;
        for (d = 0; d <= D - 2; d += 2) {
          // B, S, H, D
          const mllm_int64_t* src_ptr = X + b * S * H * D + s * H * D + h * D + d;

          // B, H, S, D
          mllm_int64_t* dst_ptr = Y + b * H * S * D + h * S * D + s * D + d;

          int64x2_t data;
          data = vld1q_s64(src_ptr);
          vst1q_s64(dst_ptr, data);
        }
        for (; d < D; ++d) {
          const mllm_int64_t* src_ptr = X + b * S * H * D + s * H * D + h * D + d;
          mllm_int64_t* dst_ptr = Y + b * H * S * D + h * S * D + s * D + d;
          *dst_ptr = *src_ptr;
        }
      }
    }
  }
}

void transpose_last_dims_int64(const mllm_int64_t* __restrict__ input, mllm_int64_t* __restrict__ output, size_t batch,
                               size_t dim0, size_t dim1) {
  for (size_t b = 0; b < batch; b++) {
    const mllm_int64_t* input_batch = input + b * dim0 * dim1;
    mllm_int64_t* output_batch = output + b * dim0 * dim1;

    for (size_t i = 0; i + 2 <= dim0; i += 2) {
      for (size_t j = 0; j + 2 <= dim1; j += 2) {
        int64x2_t r0 = vld1q_s64(input_batch + i * dim1 + j);
        int64x2_t r1 = vld1q_s64(input_batch + (i + 1) * dim1 + j);

        int64x2_t col0 = vcombine_s64(vget_low_s64(r0), vget_low_s64(r1));
        int64x2_t col1 = vcombine_s64(vget_high_s64(r0), vget_high_s64(r1));

        vst1q_s64(output_batch + j * dim0 + i, col0);
        vst1q_s64(output_batch + (j + 1) * dim0 + i, col1);
      }

      size_t j_remain = dim1 - (dim1 % 2);
      for (size_t j = j_remain; j < dim1; ++j) {
        int64x2_t col = {input_batch[i * dim1 + j], input_batch[(i + 1) * dim1 + j]};
        vst1q_s64(output_batch + j * dim0 + i, col);
      }
    }

    size_t i_remain = dim0 - (dim0 % 2);
    for (size_t j = 0; j < dim1; ++j) {
      for (size_t i = i_remain; i < dim0; ++i) { output_batch[j * dim0 + i] = input_batch[i * dim1 + j]; }
    }
  }
}

}  // namespace mllm::cpu::arm

#endif
