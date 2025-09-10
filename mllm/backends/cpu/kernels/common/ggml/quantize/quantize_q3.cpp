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

//========================= 3-bit (de)-quantization

static float make_q3_quants(int n, int nmax, const float* __restrict x, int8_t* __restrict L, bool do_rmse) {
  float max = 0;
  float amax = 0;
  for (int i = 0; i < n; ++i) {
    float ax = fabsf(x[i]);
    if (ax > amax) {
      amax = ax;
      max = x[i];
    }
  }
  if (amax < GROUP_MAX_EPS) {  // all zero
    for (int i = 0; i < n; ++i) { L[i] = 0; }
    return 0.f;
  }
  float iscale = -nmax / max;
  if (do_rmse) {
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
      int l = nearest_int(iscale * x[i]);
      l = MAX(-nmax, MIN(nmax - 1, l));
      L[i] = l;
      float w = x[i] * x[i];
      sumlx += w * x[i] * l;
      suml2 += w * l * l;
    }
    for (int itry = 0; itry < 5; ++itry) {
      int n_changed = 0;
      for (int i = 0; i < n; ++i) {
        float w = x[i] * x[i];
        float slx = sumlx - w * x[i] * L[i];
        if (slx > 0) {
          float sl2 = suml2 - w * L[i] * L[i];
          int new_l = nearest_int(x[i] * sl2 / slx);
          new_l = MAX(-nmax, MIN(nmax - 1, new_l));
          if (new_l != L[i]) {
            slx += w * x[i] * new_l;
            sl2 += w * new_l * new_l;
            if (sl2 > 0 && slx * slx * suml2 > sumlx * sumlx * sl2) {
              L[i] = new_l;
              sumlx = slx;
              suml2 = sl2;
              ++n_changed;
            }
          }
        }
      }
      if (!n_changed) { break; }
    }
    for (int i = 0; i < n; ++i) { L[i] += nmax; }
    return sumlx / suml2;
  }
  for (int i = 0; i < n; ++i) {
    int l = nearest_int(iscale * x[i]);
    l = MAX(-nmax, MIN(nmax - 1, l));
    L[i] = l + nmax;
  }
  return 1 / iscale;
}
void quantize_row_q3_K_ref(const float* __restrict x, block_q3_K* __restrict y, int64_t k) {
  assert(k % QK_K == 0);
  const int nb = k / QK_K;

  int8_t L[QK_K];
  float scales[QK_K / 16];

  for (int i = 0; i < nb; i++) {
    float max_scale = 0;
    float amax = 0;
    for (int j = 0; j < QK_K / 16; ++j) {
      scales[j] = make_q3_quants(16, 4, x + 16 * j, L + 16 * j, true);
      float scale = fabsf(scales[j]);
      if (scale > amax) {
        amax = scale;
        max_scale = scales[j];
      }
    }

    memset(y[i].scales, 0, 12);
    if (max_scale) {
      float iscale = -32.f / max_scale;
      for (int j = 0; j < QK_K / 16; ++j) {
        int8_t l = nearest_int(iscale * scales[j]);
        l = MAX(-32, MIN(31, l)) + 32;
        if (j < 8) {
          y[i].scales[j] = l & 0xF;
        } else {
          y[i].scales[j - 8] |= ((l & 0xF) << 4);
        }
        l >>= 4;
        y[i].scales[j % 4 + 8] |= (l << (2 * (j / 4)));
      }
      y[i].d = MLLM_FP32_TO_FP16(1 / iscale);
    } else {
      y[i].d = MLLM_FP32_TO_FP16(0.f);
    }

    int8_t sc;
    for (int j = 0; j < QK_K / 16; ++j) {
      sc = j < 8 ? y[i].scales[j] & 0xF : y[i].scales[j - 8] >> 4;
      sc = (sc | (((y[i].scales[8 + j % 4] >> (2 * (j / 4))) & 3) << 4)) - 32;
      float d = MLLM_FP16_TO_FP32(y[i].d) * sc;
      if (!d) { continue; }
      for (int ii = 0; ii < 16; ++ii) {
        int l = nearest_int(x[16 * j + ii] / d);
        l = MAX(-4, MIN(3, l));
        L[16 * j + ii] = l + 4;
      }
    }

    memset(y[i].hmask, 0, QK_K / 8);
    // We put the high-bit for the 1st 8 quants into bit 0, the next 8 into bit 1, etc.
    int m = 0;
    uint8_t hm = 1;
    for (int j = 0; j < QK_K; ++j) {  // NOLINT
      if (L[j] > 3) {
        y[i].hmask[m] |= hm;
        L[j] -= 4;
      }
      if (++m == QK_K / 8) {
        m = 0;
        hm <<= 1;
      }
    }
    for (int j = 0; j < QK_K; j += 128) {
      for (int l = 0; l < 32; ++l) {
        y[i].qs[j / 4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
      }
    }

    x += QK_K;
  }
}

void quantize_row_q3_K(const float* __restrict x, void* __restrict vy, int k) {
  assert(k % QK_K == 0);
  block_q3_K* __restrict y = (block_q3_K*)vy;
  quantize_row_q3_K_ref(x, y, k);
}

void dequantize_row_q3_K(const block_q3_K* __restrict x, float* __restrict y, int64_t k) {
  assert(k % QK_K == 0);
  const int nb = k / QK_K;

  const uint32_t kmask1 = 0x03030303;
  const uint32_t kmask2 = 0x0f0f0f0f;

  uint32_t aux[4];
  const int8_t* scales = (const int8_t*)aux;

  for (int i = 0; i < nb; i++) {
    const float d_all = MLLM_FP16_TO_FP32(x[i].d);

    const uint8_t* __restrict q = x[i].qs;
    const uint8_t* __restrict hm = x[i].hmask;
    uint8_t m = 1;

    memcpy(aux, x[i].scales, 12);
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

    int is = 0;
    float dl;
    for (int n = 0; n < QK_K; n += 128) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        dl = d_all * (scales[is++] - 32);
        for (int l = 0; l < 16; ++l) { *y++ = dl * ((int8_t)((q[l + 0] >> shift) & 3) - ((hm[l + 0] & m) ? 0 : 4)); }

        dl = d_all * (scales[is++] - 32);
        for (int l = 0; l < 16; ++l) { *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4)); }

        shift += 2;
        m <<= 1;
      }
      q += 32;
    }
  }
}
}  // namespace mllm::cpu