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

// reference implementation for deterministic creation of model files
void quantize_row_q4_0_reference(const float* __restrict x, block_q4_0* __restrict y, int k) {
  static const int Qk = QK4_0;

  assert(k % Qk == 0);

  const int nb = k / Qk;

  for (int i = 0; i < nb; i++) {
    float amax = 0.0F;  // absolute max
    float max = 0.0F;

    for (int j = 0; j < Qk; j++) {
      const float v = x[i * Qk + j];
      if (amax < fabsf(v)) {
        amax = fabsf(v);
        max = v;
      }
    }

    const float d = max / -8;
    const float id = d ? 1.0F / d : 0.0F;

    y[i].d = MLLM_FP32_TO_FP16(d);

    for (int j = 0; j < Qk / 2; ++j) {
      const float x0 = x[i * Qk + 0 + j] * id;
      const float x1 = x[i * Qk + Qk / 2 + j] * id;

      const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5F));
      const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5F));

      y[i].qs[j] = xi0;
      y[i].qs[j] |= xi1 << 4;
    }
  }
}

void quantize_row_q4_0(const float* __restrict x, void* __restrict y, int k) {
  quantize_row_q4_0_reference(x, (block_q4_0*)y, k);
}

void dequantize_row_q4_0(const void* __restrict vx, float* __restrict y, int k) {
  static const int Qk = QK4_0;

  assert(k % Qk == 0);

  block_q4_0* __restrict x = (block_q4_0*)vx;
  const int nb = k / Qk;

  for (int i = 0; i < nb; i++) {
    const float d = MLLM_FP16_TO_FP32(x[i].d);

    for (int j = 0; j < Qk / 2; ++j) {
      const int x0 = (x[i].qs[j] & 0x0F) - 8;
      const int x1 = (x[i].qs[j] >> 4) - 8;

      y[i * Qk + j + 0] = x0 * d;
      y[i * Qk + j + Qk / 2] = x1 * d;
    }
  }
}

// ====================== 4-bit (de)-quantization

static float make_qkx2_quants(int n, int nmax, const float* __restrict x, const float* __restrict weights,
                              uint8_t* __restrict L, float* __restrict the_min, uint8_t* __restrict Laux, float rmin,
                              float rdelta, int nstep, bool use_mad) {
  float min = x[0];
  float max = x[0];
  float sum_w = weights[0];
  float sum_x = sum_w * x[0];
  for (int i = 1; i < n; ++i) {
    if (x[i] < min) min = x[i];
    if (x[i] > max) max = x[i];
    float w = weights[i];
    sum_w += w;
    sum_x += w * x[i];
  }
  if (min > 0) min = 0;
  if (max == min) {
    for (int i = 0; i < n; ++i) L[i] = 0;
    *the_min = -min;
    return 0.F;
  }
  float iscale = nmax / (max - min);
  float scale = 1 / iscale;
  float best_mad = 0;
  for (int i = 0; i < n; ++i) {
    int l = nearest_int(iscale * (x[i] - min));
    L[i] = MAX(0, MIN(nmax, l));
    float diff = scale * L[i] + min - x[i];
    diff = use_mad ? fabsf(diff) : diff * diff;
    float w = weights[i];
    best_mad += w * diff;
  }
  if (nstep < 1) {
    *the_min = -min;
    return scale;
  }
  for (int is = 0; is <= nstep; ++is) {
    iscale = (rmin + rdelta * is + nmax) / (max - min);
    float sum_l = 0;
    float sum_l2 = 0;
    float sum_xl = 0;
    for (int i = 0; i < n; ++i) {
      int l = nearest_int(iscale * (x[i] - min));
      l = MAX(0, MIN(nmax, l));
      Laux[i] = l;
      float w = weights[i];
      sum_l += w * l;
      sum_l2 += w * l * l;
      sum_xl += w * l * x[i];
    }
    float D = sum_w * sum_l2 - sum_l * sum_l;
    if (D > 0) {
      float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
      float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
      if (this_min > 0) {
        this_min = 0;
        this_scale = sum_xl / sum_l2;
      }
      float mad = 0;
      for (int i = 0; i < n; ++i) {
        float diff = this_scale * Laux[i] + this_min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        float w = weights[i];
        mad += w * diff;
      }
      if (mad < best_mad) {
        for (int i = 0; i < n; ++i) { L[i] = Laux[i]; }
        best_mad = mad;
        scale = this_scale;
        min = this_min;
      }
    }
  }
  *the_min = -min;
  return scale;
}
#if QK_K == 256
static inline void get_scale_min_k4(int j, const uint8_t* __restrict q, uint8_t* __restrict d, uint8_t* __restrict m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}
#endif

void quantize_row_q4_K_reference(const float* __restrict x, block_q4_K* __restrict y, int k) {
  assert(k % QK_K == 0);
  const int nb = k / QK_K;

  uint8_t L[QK_K];
  uint8_t Laux[32];
  float weights[32];
  float mins[QK_K / 32];
  float scales[QK_K / 32];

  for (int i = 0; i < nb; i++) {
    float max_scale = 0;  // as we are deducting the min, scales are always positive
    float max_min = 0;
    for (int j = 0; j < QK_K / 32; ++j) {
      // scales[j] = make_qkx1_quants(32, 15, x + 32*j, L + 32*j, &mins[j], 9, 0.5f);
      float sum_x2 = 0;
      for (int l = 0; l < 32; ++l) sum_x2 += x[32 * j + l] * x[32 * j + l];
      float av_x = sqrtf(sum_x2 / 32);
      for (int l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32 * j + l]);
      scales[j] = make_qkx2_quants(32, 15, x + 32 * j, weights, L + 32 * j, &mins[j], Laux, -1.F, 0.1F, 20, false);
      float scale = scales[j];
      if (scale > max_scale) { max_scale = scale; }
      float min = mins[j];
      if (min > max_min) { max_min = min; }
    }

#if QK_K == 256
    float inv_scale = max_scale > 0 ? 63.F / max_scale : 0.F;
    float inv_min = max_min > 0 ? 63.F / max_min : 0.F;
    for (int j = 0; j < QK_K / 32; ++j) {
      uint8_t ls = nearest_int(inv_scale * scales[j]);
      uint8_t lm = nearest_int(inv_min * mins[j]);
      ls = MIN(63, ls);
      lm = MIN(63, lm);
      if (j < 4) {
        y[i].scales[j] = ls;
        y[i].scales[j + 4] = lm;
      } else {
        y[i].scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
        y[i].scales[j - 4] |= ((ls >> 4) << 6);
        y[i].scales[j - 0] |= ((lm >> 4) << 6);
      }
    }
    y[i].d = MLLM_FP32_TO_FP16(max_scale / 63.F);
    y[i].dmin = MLLM_FP32_TO_FP16(max_min / 63.F);

    uint8_t sc;
    uint8_t m;
    for (int j = 0; j < QK_K / 32; ++j) {
      get_scale_min_k4(j, y[i].scales, &sc, &m);
      const float d = MLLM_FP16_TO_FP32(y[i].d) * sc;
      if (d == 0.0F) continue;
      const float dm = MLLM_FP16_TO_FP32(y[i].dmin) * m;
      for (int ii = 0; ii < 32; ++ii) {
        int l = nearest_int((x[32 * j + ii] + dm) / d);
        l = MAX(0, MIN(15, l));
        L[32 * j + ii] = l;
      }
    }
#else
    const float s_factor = 15.f;
    float inv_scale = max_scale > 0 ? s_factor / max_scale : 0.f;
    float inv_min = max_min > 0 ? s_factor / max_min : 0.f;
    int d1 = nearest_int(inv_scale * scales[0]);
    int m1 = nearest_int(inv_min * mins[0]);
    int d2 = nearest_int(inv_scale * scales[1]);
    int m2 = nearest_int(inv_min * mins[1]);
    y[i].scales[0] = d1 | (m1 << 4);
    y[i].scales[1] = d2 | (m2 << 4);
    y[i].d[0] = MLLM_FP32_TO_FP16(max_scale / s_factor);
    y[i].d[1] = MLLM_FP32_TO_FP16(max_min / s_factor);

    float sumlx = 0;
    int suml2 = 0;
    for (int j = 0; j < QK_K / 32; ++j) {
      const uint8_t sd = y[i].scales[j] & 0xF;
      const uint8_t sm = y[i].scales[j] >> 4;
      const float d = MLLM_FP16_TO_FP32(y[i].d[0]) * sd;
      if (!d) continue;
      const float m = MLLM_FP16_TO_FP32(y[i].d[1]) * sm;
      for (int ii = 0; ii < 32; ++ii) {
        int l = nearest_int((x[32 * j + ii] + m) / d);
        l = MAX(0, MIN(15, l));
        L[32 * j + ii] = l;
        sumlx += (x[32 * j + ii] + m) * l * sd;
        suml2 += l * l * sd * sd;
      }
    }
    if (suml2) { y[i].d[0] = MLLM_FP32_TO_FP16(sumlx / suml2); }
#endif
    uint8_t* q = y[i].qs;
    for (int j = 0; j < QK_K; j += 64) {
      for (int l = 0; l < 32; ++l) q[l] = L[j + l] | (L[j + l + 32] << 4);
      q += 32;
    }

    x += QK_K;
  }
}

void dequantize_row_q4_K(const mllm_block_q4_K_t* __restrict x, float* __restrict y, int k) {
  assert(k % QK_K == 0);
  const int nb = k / QK_K;

  for (int i = 0; i < nb; i++) {
    const uint8_t* q = x[i].qs;

#if QK_K == 256

    const float d = MLLM_FP16_TO_FP32(x[i].d);
    const float min = MLLM_FP16_TO_FP32(x[i].dmin);

    int is = 0;
    uint8_t sc;
    uint8_t m;
    for (int j = 0; j < QK_K; j += 64) {
      get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
      const float d1 = d * sc;
      const float m1 = min * m;
      get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
      const float d2 = d * sc;
      const float m2 = min * m;
      for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
      for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
      q += 32;
      is += 2;
    }
#else
    const float dall = MLLM_FP16_TO_FP32(x[i].d[0]);
    const float mall = MLLM_FP16_TO_FP32(x[i].d[1]);
    const float d1 = dall * (x[i].scales[0] & 0xF), m1 = mall * (x[i].scales[0] >> 4);
    const float d2 = dall * (x[i].scales[1] & 0xF), m2 = mall * (x[i].scales[1] >> 4);
    for (int l = 0; l < 32; ++l) {
      y[l + 0] = d1 * (q[l] & 0xF) - m1;
      y[l + 32] = d2 * (q[l] >> 4) - m2;
    }
    y += QK_K;
#endif
  }
}

void quantize_row_q4_K(const float* __restrict x, void* __restrict vy, int k) {
  assert(k % QK_K == 0);
  block_q4_K* __restrict y = (block_q4_K*)vy;
  quantize_row_q4_K_reference(x, y, k);
}
}  // namespace mllm::cpu
