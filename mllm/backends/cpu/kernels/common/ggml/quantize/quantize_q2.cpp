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
    return 0.f;
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
    float sum_l = 0, sum_l2 = 0, sum_xl = 0;
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

void quantize_row_q2_K_ref(const float* __restrict x, block_q2_K* __restrict y, int64_t k) {
  assert(k % QK_K == 0);
  const int nb = k / QK_K;

  uint8_t L[QK_K];
  uint8_t Laux[16];
  float weights[16];
  float mins[QK_K / 16];
  float scales[QK_K / 16];

  const float q4scale = 15.f;

  for (int i = 0; i < nb; i++) {
    float max_scale = 0;  // as we are deducting the min, scales are always positive
    float max_min = 0;
    for (int j = 0; j < QK_K / 16; ++j) {
      for (int l = 0; l < 16; ++l) weights[l] = fabsf(x[16 * j + l]);
      scales[j] = make_qkx2_quants(16, 3, x + 16 * j, weights, L + 16 * j, &mins[j], Laux, -0.5f, 0.1f, 15, true);
      float scale = scales[j];
      if (scale > max_scale) { max_scale = scale; }
      float min = mins[j];
      if (min > max_min) { max_min = min; }
    }

    if (max_scale > 0) {
      float iscale = q4scale / max_scale;
      for (int j = 0; j < QK_K / 16; ++j) {
        int l = nearest_int(iscale * scales[j]);
        y[i].scales[j] = l;
      }
      y[i].d = MLLM_FP32_TO_FP16(max_scale / q4scale);
    } else {
      for (unsigned char& scale : y[i].scales) scale = 0;
      y[i].d = MLLM_FP32_TO_FP16(0.f);
    }
    if (max_min > 0) {
      float iscale = q4scale / max_min;
      for (int j = 0; j < QK_K / 16; ++j) {
        int l = nearest_int(iscale * mins[j]);
        y[i].scales[j] |= (l << 4);
      }
      y[i].dmin = MLLM_FP32_TO_FP16(max_min / q4scale);
    } else {
      y[i].dmin = MLLM_FP32_TO_FP16(0.f);
    }
    for (int j = 0; j < QK_K / 16; ++j) {
      const float d = MLLM_FP16_TO_FP32(y[i].d) * (y[i].scales[j] & 0xF);
      if (!d) continue;
      const float dm = MLLM_FP16_TO_FP32(y[i].dmin) * (y[i].scales[j] >> 4);
      for (int ii = 0; ii < 16; ++ii) {
        int l = nearest_int((x[16 * j + ii] + dm) / d);
        l = MAX(0, MIN(3, l));
        L[16 * j + ii] = l;
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

void quantize_row_q2_K(const float* __restrict x, void* __restrict y, int k) { quantize_row_q2_K_ref(x, (block_q2_K*)y, k); }

void dequantize_row_q2_K(const block_q2_K* __restrict x, float* __restrict y, int64_t k) {
  assert(k % QK_K == 0);
  const int nb = k / QK_K;

  for (int i = 0; i < nb; i++) {
    const float d = MLLM_FP16_TO_FP32(x[i].d);
    const float min = MLLM_FP16_TO_FP32(x[i].dmin);

    const uint8_t* q = x[i].qs;

    int is = 0;
    float dl, ml;
    for (int n = 0; n < QK_K; n += 128) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        uint8_t sc = x[i].scales[is++];
        dl = d * (sc & 0xF);
        ml = min * (sc >> 4);
        for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;

        sc = x[i].scales[is++];
        dl = d * (sc & 0xF);
        ml = min * (sc >> 4);
        for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3)) - ml;

        shift += 2;
      }
      q += 32;
    }
  }
}

static inline int iq2_data_index(enum DataTypes type) {
  assert(type == MLLM_TYPE_IQ2_XXS || type == MLLM_TYPE_IQ2_XS || type == MLLM_TYPE_IQ1_S || type == MLLM_TYPE_IQ1_M
         || type == MLLM_TYPE_IQ2_S);
  return type == MLLM_TYPE_IQ2_XXS                            ? 0
         : type == MLLM_TYPE_IQ2_XS                           ? 1
         : type == MLLM_TYPE_IQ1_S || type == MLLM_TYPE_IQ1_M ? 2
                                                              : 3;
}

using iq2_entry_t = struct {
  uint64_t* grid;
  int* map;
  uint16_t* neighbours;
};

static iq2_entry_t iq2_data[4] = {
    {nullptr, nullptr, nullptr},
    {nullptr, nullptr, nullptr},
    {nullptr, nullptr, nullptr},
    {nullptr, nullptr, nullptr},
};

static float make_qp_quants(int n, int nmax, const float* __restrict x, uint8_t* __restrict L, const float* quant_weights) {
  float max = 0;
  for (int i = 0; i < n; ++i) { max = MAX(max, x[i]); }
  if (!max) {  // all zero
    for (int i = 0; i < n; ++i) { L[i] = 0; }
    return 0.f;
  }
  float iscale = nmax / max;
  for (int i = 0; i < n; ++i) { L[i] = nearest_int(iscale * x[i]); }
  float scale = 1 / iscale;
  float best_mse = 0;
  for (int i = 0; i < n; ++i) {
    float diff = x[i] - scale * L[i];
    float w = quant_weights[i];
    best_mse += w * diff * diff;
  }
  for (int is = -4; is <= 4; ++is) {
    if (is == 0) continue;
    float iscale_is = (0.1f * is + nmax) / max;
    float scale_is = 1 / iscale_is;
    float mse = 0;
    for (int i = 0; i < n; ++i) {
      int l = nearest_int(iscale_is * x[i]);
      l = MIN(nmax, l);
      float diff = x[i] - scale_is * l;
      float w = quant_weights[i];
      mse += w * diff * diff;
    }
    if (mse < best_mse) {
      best_mse = mse;
      iscale = iscale_is;
    }
  }
  float sumlx = 0;
  float suml2 = 0;
  for (int i = 0; i < n; ++i) {
    int l = nearest_int(iscale * x[i]);
    l = MIN(nmax, l);
    L[i] = l;
    float w = quant_weights[i];
    sumlx += w * x[i] * l;
    suml2 += w * l * l;
  }
  for (int itry = 0; itry < 5; ++itry) {
    int n_changed = 0;
    for (int i = 0; i < n; ++i) {
      float w = quant_weights[i];
      float slx = sumlx - w * x[i] * L[i];
      float sl2 = suml2 - w * L[i] * L[i];
      if (slx > 0 && sl2 > 0) {
        int new_l = nearest_int(x[i] * sl2 / slx);
        new_l = MIN(nmax, new_l);
        if (new_l != L[i]) {
          slx += w * x[i] * new_l;
          sl2 += w * new_l * new_l;
          if (slx * slx * suml2 > sumlx * sumlx * sl2) {
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
  return sumlx / suml2;
}

static int iq2_find_best_neighbour(const uint16_t* __restrict neighbours, const uint64_t* __restrict grid,
                                   const float* __restrict xval, const float* __restrict weight, float scale,
                                   int8_t* __restrict L) {
  int num_neighbors = neighbours[0];
  assert(num_neighbors > 0);
  float best_d2 = FLT_MAX;
  int grid_index = -1;
  for (int j = 1; j <= num_neighbors; ++j) {
    const int8_t* pg = (const int8_t*)(grid + neighbours[j]);
    float d2 = 0;
    for (int i = 0; i < 8; ++i) {
      float q = pg[i];
      float diff = scale * q - xval[i];
      d2 += weight[i] * diff * diff;
    }
    if (d2 < best_d2) {
      best_d2 = d2;
      grid_index = neighbours[j];
    }
  }
  assert(grid_index >= 0);
  const int8_t* pg = (const int8_t*)(grid + grid_index);
  for (int i = 0; i < 8; ++i) L[i] = (pg[i] - 1) / 2;
  return grid_index;
}

static void quantize_row_iq2_xxs_impl(const float* __restrict x, void* __restrict vy, int64_t n,
                                      const float* __restrict quant_weights) {
  const int gindex = iq2_data_index(MLLM_TYPE_IQ2_XXS);

  const uint64_t* kgrid_q2xs = iq2_data[gindex].grid;
  const int* kmap_q2xs = iq2_data[gindex].map;
  const uint16_t* kneighbors_q2xs = iq2_data[gindex].neighbours;

  assert(quant_weights && "missing quantization weights");
  assert(kgrid_q2xs && "forgot to call ggml_quantize_init()?");
  assert(kmap_q2xs && "forgot to call ggml_quantize_init()?");
  assert(kneighbors_q2xs && "forgot to call ggml_quantize_init()?");
  assert(n % QK_K == 0);

  const int kMaxQ = 3;

  const int64_t nbl = n / QK_K;

  block_iq2_xxs* y = (block_iq2_xxs*)vy;

  float scales[QK_K / 32];
  float weight[32];
  float xval[32];
  int8_t L[32];
  int8_t Laux[32];
  float waux[32];
  uint8_t block_signs[4];
  uint32_t q2[2 * (QK_K / 32)];

  for (int ibl = 0; ibl < nbl; ++ibl) {
    y[ibl].d = MLLM_FP32_TO_FP16(0.f);
    memset(q2, 0, QK_K / 4);

    float max_scale = 0;

    const float* xbl = x + QK_K * ibl;
    float sumx2 = 0;
    for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i] * xbl[i];
    float sigma2 = sumx2 / QK_K;

    for (int ib = 0; ib < QK_K / 32; ++ib) {
      const float* xb = xbl + 32 * ib;
      const float* qw = quant_weights + QK_K * ibl + 32 * ib;
      for (int i = 0; i < 32; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i] * xb[i]);
      for (int i = 0; i < 32; ++i) waux[i] = sqrtf(weight[i]);
      for (int k = 0; k < 4; ++k) {
        int nflip = 0;
        uint8_t s = 0;
        for (int i = 0; i < 8; ++i) {
          if (xb[8 * k + i] >= 0) {
            xval[8 * k + i] = xb[8 * k + i];
          } else {
            xval[8 * k + i] = -xb[8 * k + i];
            ++nflip;
            s |= (1 << i);
          }
        }
        if (nflip % 2) {
          int imin = 0;
          float min = weight[8 * k + imin] * xb[8 * k + imin] * xb[8 * k + imin];
          for (int i = 1; i < 8; ++i) {
            float ax = weight[8 * k + i] * xb[8 * k + i] * xb[8 * k + i];
            if (ax < min) {
              min = ax;
              imin = i;
            }
          }
          xval[8 * k + imin] = -xval[8 * k + imin];
          s ^= (1 << imin);
        }
        block_signs[k] = s & 127;
      }
      float max = xval[0];
      for (int i = 1; i < 32; ++i) max = MAX(max, xval[i]);
      if (max < GROUP_MAX_EPS) {
        scales[ib] = 0;
        memset(L, 0, 32);
        continue;
      }
      float scale = make_qp_quants(32, kMaxQ + 1, xval, (uint8_t*)L, weight);
      float eff_max = scale * kMaxQ;
      float best = 0;
      for (int is = -6; is <= 6; ++is) {
        float id = (2 * kMaxQ - 1 + is * 0.1f) / eff_max;
        float this_scale = 1 / id;
        for (int k = 0; k < 4; ++k) {
          for (int i = 0; i < 8; ++i) {
            int l = nearest_int(0.5f * (id * xval[8 * k + i] - 1));
            Laux[8 * k + i] = MAX(0, MIN(kMaxQ - 1, l));
          }
          uint16_t u = 0;
          for (int i = 0; i < 8; ++i) u |= (Laux[8 * k + i] << 2 * i);
          int grid_index = kmap_q2xs[u];
          if (grid_index < 0) {
            const uint16_t* neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
            grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8 * k, waux + 8 * k, this_scale, Laux + 8 * k);
          }
        }
        float sumqx = 0, sumq2 = 0;
        for (int i = 0; i < 32; ++i) {
          float w = weight[i];
          float q = 2 * Laux[i] + 1;
          sumqx += w * xval[i] * q;
          sumq2 += w * q * q;
        }
        if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
          scale = sumqx / sumq2;
          best = scale * sumqx;
          memcpy(L, Laux, 32);
        }
      }
      if (scale > 0) {
        float id = 1 / scale;
        for (int k = 0; k < 4; ++k) {
          uint16_t u = 0;
          for (int i = 0; i < 8; ++i) {
            int l = nearest_int(0.5f * (id * xval[8 * k + i] - 1));
            l = MAX(0, MIN(kMaxQ - 1, l));
            u |= (l << 2 * i);
          }
          int grid_index = kmap_q2xs[u];
          if (grid_index < 0) {
            const uint16_t* neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
            grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8 * k, waux + 8 * k, scale, L + 8 * k);
          }
          const int8_t* pg = (const int8_t*)(kgrid_q2xs + grid_index);
          for (int i = 0; i < 8; ++i) L[8 * k + i] = (pg[i] - 1) / 2;
        }
        float sumqx = 0, sumq2 = 0;
        for (int i = 0; i < 32; ++i) {
          float w = weight[i];
          float q = 2 * L[i] + 1;
          sumqx += w * xval[i] * q;
          sumq2 += w * q * q;
        }
        if (sumq2 > 0) scale = sumqx / sumq2;
      }
      if (scale < 0) {
        // This should never happen, but just in case, flip scale so that it is positive (we use uint's to encode the scale)
        // and correspondingly flip quant signs.
        scale = -scale;
        for (int k = 0; k < 4; ++k) block_signs[k] = (~block_signs[k]) & 127;  // NOLINT
      }
      for (int k = 0; k < 4; ++k) {
        uint16_t u = 0;
        for (int i = 0; i < 8; ++i) u |= (L[8 * k + i] << 2 * i);
        int grid_index = kmap_q2xs[u];
        if (grid_index < 0) {
          printf("Oops: found point %u not on grid:", u);
          for (int i = 0; i < 8; ++i) printf(" %d", L[8 * k + i]);
          printf("\n");
          // abort("fatal error");
          abort();
        }
        q2[2 * ib + 0] |= ((uint32_t)grid_index << 8 * k);
        q2[2 * ib + 1] |= (block_signs[k] << 7 * k);
      }
      assert(scale >= 0);
      scales[ib] = scale;
      max_scale = MAX(max_scale, scale);
    }

    if (!max_scale) {
      memset(y[ibl].qs, 0, QK_K / 4);
      continue;
    }

    float d = max_scale / 31;
    y[ibl].d = MLLM_FP32_TO_FP16(d);
    float id = 1 / d;
    for (int ib = 0; ib < QK_K / 32; ++ib) {
      int l = nearest_int(0.5f * (id * scales[ib] - 1));
      l = MAX(0, MIN(15, l));
      q2[2 * ib + 1] |= ((uint32_t)l << 28);
    }
    memcpy(y[ibl].qs, q2, QK_K / 4);
  }
}

size_t quantize_iq2_xxs(const float* __restrict src, void* __restrict dst, int64_t nrow, int64_t n_per_row,
                        const float* quant_weights) {
  assert(n_per_row % QK_K == 0);
  int64_t nblock = n_per_row / QK_K;
  char* qrow = (char*)dst;
  for (int64_t row = 0; row < nrow; ++row) {
    quantize_row_iq2_xxs_impl(src, qrow, n_per_row, quant_weights);
    src += n_per_row;
    qrow += nblock * sizeof(block_iq2_xxs);
  }
  return nrow * nblock * sizeof(block_iq2_xxs);
}

void dequantize_row_iq2_xxs(const block_iq2_xxs* __restrict x, float* __restrict y, int64_t k) {
  assert(k % QK_K == 0);
  const int64_t nb = k / QK_K;

  uint32_t aux32[2];
  const uint8_t* aux8 = (const uint8_t*)aux32;

  for (int i = 0; i < nb; i++) {
    const float d = MLLM_FP16_TO_FP32(x[i].d);

    for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
      memcpy(aux32, x[i].qs + 4 * ib32, 2 * sizeof(uint32_t));
      const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
      for (int l = 0; l < 4; ++l) {
        const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + aux8[l]);
        const uint8_t signs = ksigns_iq2xs[(aux32[1] >> 7 * l) & 127];
        for (int j = 0; j < 8; ++j) { y[j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f); }
        y += 8;
      }
    }
  }
}
}  // namespace mllm::cpu