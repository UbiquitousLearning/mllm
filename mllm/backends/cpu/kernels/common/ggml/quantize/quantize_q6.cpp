/*
 * This code is based on ggml(https://github.com/ggerganov/ggml),
 * please see https://github.com/ggerganov/ggml/blob/master/src/ggml.c
 * ggml is licensed under MIT Copyright (c) 2022 Georgi Gerganov:
 *
 * MIT License
 * Copyright (c) 2022 Georgi Gerganov
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "mllm/backends/cpu/kernels/common/ggml/quantize/quantize.hpp"

namespace mllm::cpu {

// ====================== 6-bit (de)-quantization

void quantize_row_q6_K_reference(const float* __restrict x, block_q6_K* __restrict y, int k) {
  assert(k % QK_K == 0);
  const int nb = k / QK_K;

  int8_t L[QK_K];
  float scales[QK_K / 16];

  for (int i = 0; i < nb; i++) {
    float max_scale = 0;
    float max_abs_scale = 0;

    for (int ib = 0; ib < QK_K / 16; ++ib) {
      const float scale = make_qx_quants(16, 32, x + 16 * ib, L + 16 * ib, 1);
      scales[ib] = scale;

      const float abs_scale = fabsf(scale);
      if (abs_scale > max_abs_scale) {
        max_abs_scale = abs_scale;
        max_scale = scale;
      }
    }

    if (!max_abs_scale) {
      memset(&y[i], 0, sizeof(block_q6_K));
      y[i].d = MLLM_FP32_TO_FP16(0.f);
      x += QK_K;
      continue;
    }

    float iscale = -128.f / max_scale;
    y[i].d = MLLM_FP32_TO_FP16(1 / iscale);
    for (int ib = 0; ib < QK_K / 16; ++ib) { y[i].scales[ib] = MIN(127, nearest_int(iscale * scales[ib])); }

    for (int j = 0; j < QK_K / 16; ++j) {
      float d = MLLM_FP16_TO_FP32(y[i].d) * y[i].scales[j];
      if (!d) { continue; }
      for (int ii = 0; ii < 16; ++ii) {
        int l = nearest_int(x[16 * j + ii] / d);
        l = MAX(-32, MIN(31, l));
        L[16 * j + ii] = l + 32;
      }
    }

    uint8_t* __restrict ql = y[i].ql;
    uint8_t* __restrict qh = y[i].qh;
#if QK_K == 256
    for (int j = 0; j < QK_K; j += 128) {
      for (int l = 0; l < 32; ++l) {
        const uint8_t q1 = L[j + l + 0] & 0xF;
        const uint8_t q2 = L[j + l + 32] & 0xF;
        const uint8_t q3 = L[j + l + 64] & 0xF;
        const uint8_t q4 = L[j + l + 96] & 0xF;
        ql[l + 0] = q1 | (q3 << 4);
        ql[l + 32] = q2 | (q4 << 4);
        qh[l] = (L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6);
      }
      ql += 64;
      qh += 32;
    }
#else
    for (int l = 0; l < 32; ++l) {
      const uint8_t q1 = L[l + 0] & 0xF;
      const uint8_t q2 = L[l + 32] & 0xF;
      ql[l] = q1 | (q2 << 4);
    }
    for (int l = 0; l < 16; ++l) {
      qh[l] = (L[l] >> 4) | ((L[l + 16] >> 4) << 2) | ((L[l + 32] >> 4) << 4) | ((L[l + 48] >> 4) << 6);
    }
#endif

    x += QK_K;
  }
}

void dequantize_row_q6_K(const block_q6_K* __restrict x, float* __restrict y, int k) {
  assert(k % QK_K == 0);
  const int nb = k / QK_K;

  for (int i = 0; i < nb; i++) {
    const float d = MLLM_FP16_TO_FP32(x[i].d);

    const uint8_t* __restrict ql = x[i].ql;
    const uint8_t* __restrict qh = x[i].qh;
    const int8_t* __restrict sc = x[i].scales;

#if QK_K == 256
    for (int n = 0; n < QK_K; n += 128) {
      for (int l = 0; l < 32; ++l) {
        int is = l / 16;
        const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
        const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
        const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
        y[l + 0] = d * sc[is + 0] * q1;
        y[l + 32] = d * sc[is + 2] * q2;
        y[l + 64] = d * sc[is + 4] * q3;
        y[l + 96] = d * sc[is + 6] * q4;
      }
      y += 128;
      ql += 64;
      qh += 32;
      sc += 8;
    }
#else
    for (int l = 0; l < 16; ++l) {
      const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
      const int8_t q2 = (int8_t)((ql[l + 16] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
      const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
      const int8_t q4 = (int8_t)((ql[l + 16] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
      y[l + 0] = d * sc[0] * q1;
      y[l + 16] = d * sc[1] * q2;
      y[l + 32] = d * sc[2] * q3;
      y[l + 48] = d * sc[3] * q4;
    }
    y += 64;
#endif
  }
}

void quantize_row_q6_K(const float* __restrict x, void* __restrict vy, int k) {
  assert(k % QK_K == 0);
  block_q6_K* __restrict y = (block_q6_K*)vy;
  quantize_row_q6_K_reference(x, y, k);
}
}  // namespace mllm::cpu