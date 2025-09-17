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

#include "mllm/backends/cpu/kernels/common/ggml/vec_dot.hpp"
#include "mllm/backends/cpu/kernels/common/ggml/quantize/quantize.hpp"

namespace mllm::cpu::ggml {

#ifdef __AVX2__
static void vec_dot_fp32_avx2(const int n, float* __restrict s, const float* __restrict x, const float* __restrict y) {
  float sumf = 0.0F;
  const int np = (n & ~(MLLM_F32_STEP - 1));

  MLLM_F32_VEC sum[MLLM_F32_ARR] = {MLLM_F32_VEC_ZERO};

  MLLM_F32_VEC ax[MLLM_F32_ARR];
  MLLM_F32_VEC ay[MLLM_F32_ARR];

  for (int i = 0; i < np; i += MLLM_F32_STEP) {
    for (int j = 0; j < MLLM_F32_ARR; j++) {
      ax[j] = MLLM_F32_VEC_LOAD(x + i + j * MLLM_F32_EPR);
      ay[j] = MLLM_F32_VEC_LOAD(y + i + j * MLLM_F32_EPR);

      sum[j] = MLLM_F32_VEC_FMA(sum[j], ax[j], ay[j]);
    }
  }

  // reduce sum0..sum3 to sum0
  MLLM_F32_VEC_REDUCE(sumf, sum);

  // leftovers
  for (int i = np; i < n; ++i) { sumf += x[i] * y[i]; }

  *s = sumf;
}
#endif

#ifdef __ARM_NEON
static void vec_dot_fp32_arm(const int n, float* __restrict s, const float* __restrict x, const float* __restrict y) {
  float sumf = 0.0F;
  const int np = (n & ~(16 - 1));

  F32_VEC sum[4] = {vdupq_n_f32(0.0F)};

  F32_VEC ax[F32_ARR];
  F32_VEC ay[F32_ARR];

  for (int i = 0; i < np; i += F32_STEP) {
    for (int j = 0; j < F32_ARR; j++) {
      ax[j] = vld1q_f32(x + i + j * F32_REG);
      ay[j] = vld1q_f32(y + i + j * F32_REG);
      sum[j] = vfmaq_f32(sum[j], ax[j], ay[j]);
      // sum[j] = vmlaq_lane_f32(sum[j], ax[j], ay[0],
    }
  }

  // reduce sum0..sum3 to sum0
  F32_VEC_REDUCE(sumf, sum);

  // leftovers
  for (int i = np; i < n; ++i) { sumf += x[i] * y[i]; }

  *s = sumf;
}
#endif

void vec_dot_fp32(const int n, float* __restrict s, const float* __restrict vx, const float* __restrict vy) {
#ifdef __AVX2__
  vec_dot_fp32_avx2(n, s, vx, vy);
#elif defined(__ARM_NEON)
  vec_dot_fp32_arm(n, s, vx, vy);
#endif
}

void vec_dot_fp16(const int n, float* __restrict s, const mllm_fp16_t* __restrict vx, const mllm_fp16_t* __restrict vy) {
  float sumf = 0.0;

#if defined(__AVX2__) || defined(__ARM_NEON)
  const int np = (n & ~(MLLM_F16_STEP - 1));

  MLLM_F16_VEC sum[MLLM_F16_ARR] = {MLLM_F16_VEC_ZERO};

  MLLM_F16_VEC ax[MLLM_F16_ARR];
  MLLM_F16_VEC ay[MLLM_F16_ARR];

  for (int i = 0; i < np; i += MLLM_F16_STEP) {
    for (int j = 0; j < MLLM_F16_ARR; j++) {
      ax[j] = MLLM_F16_VEC_LOAD(vx + i + j * MLLM_F16_EPR, j);
      ay[j] = MLLM_F16_VEC_LOAD(vy + i + j * MLLM_F16_EPR, j);

      sum[j] = MLLM_F16_VEC_FMA(sum[j], ax[j], ay[j]);
    }
  }

  // reduce sum0..sum3 to sum0
  MLLM_F16_VEC_REDUCE(sumf, sum);

  // leftovers
  for (int i = np; i < n; ++i) { sumf += (float)(MLLM_FP16_TO_FP32(vx[i]) * MLLM_FP16_TO_FP32(vy[i])); }
#else
  for (int i = 0; i < n; ++i) { sumf += (float)(MLLM_FP16_TO_FP32(vx[i]) * MLLM_FP16_TO_FP32(vy[i])); }
#endif

  *s = sumf;
}

#ifdef __AVX2__
static void vec_dot_q4_0_q8_0_avx(const int n, float* __restrict s, const void* __restrict vx, const void* __restrict vy) {
  const int qk = QK8_0;
  const int nb = n / qk;

  assert(n % qk == 0);

  const block_q4_0* __restrict x = (block_q4_0*)vx;
  const block_q8_0* __restrict y = (block_q8_0*)vy;
  // Initialize accumulator with zeros
  __m256 acc = _mm256_setzero_ps();

  // Main loop
  for (int i = 0; i < nb; ++i) {
    /* Compute combined scale for the block */
    const __m256 d = _mm256_set1_ps(MLLM_FP16_TO_FP32(x[i].d) * MLLM_FP16_TO_FP32(y[i].d));

    __m256i bx = bytes_from_nibbles_32(x[i].qs);

    // Now we have a vector with bytes in [ 0 .. 15 ] interval. Offset them into [ -8 .. +7 ] interval.
    const __m256i off = _mm256_set1_epi8(8);
    bx = _mm256_sub_epi8(bx, off);

    __m256i by = _mm256_loadu_si256((const __m256i*)y[i].qs);

    const __m256 q = mul_sum_i8_pairs_float(bx, by);

    /* Multiply q with scale and accumulate */
    acc = _mm256_fmadd_ps(d, q, acc);
  }
  *s = hsum_float_8(acc);
}
#endif

#ifdef __ARM_NEON
// COPY FROMN
static void vec_dot_q4_0_q8_0_arm(const int n, float* __restrict s, const void* __restrict vx, const void* __restrict vy) {
  const int qk = QK8_0;
  const int nb = n / qk;

  assert(n % qk == 0);

  const block_q4_0* __restrict x = (block_q4_0*)vx;
  const block_q8_0* __restrict y = (block_q8_0*)vy;

#if defined(__ARM_FEATURE_MATMUL_INT8)
  {
    size_t bs = 0;
    size_t bx = 0;
    size_t by = 0;
    const block_q4_0* __restrict vx0 = (const block_q4_0*)vx;
    const block_q4_0* __restrict vx1 = (const block_q4_0*)((const uint8_t*)vx + bx);
    const block_q8_0* __restrict vy0 = (const block_q8_0*)vy;
    const block_q8_0* __restrict vy1 = (const block_q8_0*)((const uint8_t*)vy + by);

    float32x4_t sumv0 = vdupq_n_f32(0.0f);

    for (int i = 0; i < nb; i++) {
      const block_q4_0* __restrict b_x0 = &vx0[i];
      const block_q4_0* __restrict b_x1 = &vx1[i];
      const block_q8_0* __restrict b_y0 = &vy0[i];
      const block_q8_0* __restrict b_y1 = &vy1[i];

      const uint8x16_t m4b = vdupq_n_u8(0x0F);
      const int8x16_t s8b = vdupq_n_s8(0x8);

      const uint8x16_t v0_0 = vld1q_u8(b_x0->qs);
      const uint8x16_t v0_1 = vld1q_u8(b_x1->qs);

      // 4-bit -> 8-bit
      const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8(v0_0, m4b));
      const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
      const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8(v0_1, m4b));
      const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

      // sub 8
      const int8x16_t x0_l = vsubq_s8(v0_0l, s8b);
      const int8x16_t x0_h = vsubq_s8(v0_0h, s8b);
      const int8x16_t x1_l = vsubq_s8(v0_1l, s8b);
      const int8x16_t x1_h = vsubq_s8(v0_1h, s8b);

      // load y
      const int8x16_t y0_l = vld1q_s8(b_y0->qs);
      const int8x16_t y0_h = vld1q_s8(b_y0->qs + 16);
      const int8x16_t y1_l = vld1q_s8(b_y1->qs);
      const int8x16_t y1_h = vld1q_s8(b_y1->qs + 16);

      float32_t _scale[4] = {
          MLLM_FP16_TO_FP32(b_x0->d) * MLLM_FP16_TO_FP32(b_y0->d), MLLM_FP16_TO_FP32(b_x0->d) * MLLM_FP16_TO_FP32(b_y1->d),
          MLLM_FP16_TO_FP32(b_x1->d) * MLLM_FP16_TO_FP32(b_y0->d), MLLM_FP16_TO_FP32(b_x1->d) * MLLM_FP16_TO_FP32(b_y1->d)};
      float32x4_t scale = vld1q_f32(_scale);

      int8x16_t l0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));
      int8x16_t l1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));

      int8x16_t l2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));
      int8x16_t l3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));

      int8x16_t r0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));
      int8x16_t r1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));

      int8x16_t r2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));
      int8x16_t r3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));

      sumv0 = vmlaq_f32(
          sumv0,
          (vcvtq_f32_s32(vmmlaq_s32((vmmlaq_s32((vmmlaq_s32((vmmlaq_s32(vdupq_n_s32(0), l0, r0)), l1, r1)), l2, r2)), l3, r3))),
          scale);
    }

    float32x4_t sumv1 = vextq_f32(sumv0, sumv0, 2);
    float32x4_t sumv2 = vzip1q_f32(sumv0, sumv1);

    vst1_f32(s, vget_low_f32(sumv2));
    vst1_f32(s + bs, vget_high_f32(sumv2));

    return;
  }
#endif

  float32x4_t sumv0 = vdupq_n_f32(0.0F);
  float32x4_t sumv1 = vdupq_n_f32(0.0F);

  assert(nb % 2 == 0);  // TODO: handle odd nb
  for (int i = 0; i < nb; i += 2) {
    const block_q4_0* __restrict x0 = &x[i + 0];
    const block_q4_0* __restrict x1 = &x[i + 1];
    const block_q8_0* __restrict y0 = &y[i + 0];
    const block_q8_0* __restrict y1 = &y[i + 1];

    const uint8x16_t m4b = vdupq_n_u8(0x0F);
    const int8x16_t s8b = vdupq_n_s8(0x8);

    const uint8x16_t v0_0 = vld1q_u8(x0->qs);
    const uint8x16_t v0_1 = vld1q_u8(x1->qs);

    // 4-bit -> 8-bit
    const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8(v0_0, m4b));
    const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
    const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8(v0_1, m4b));
    const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

    // sub 8
    const int8x16_t v0_0ls = vsubq_s8(v0_0l, s8b);
    const int8x16_t v0_0hs = vsubq_s8(v0_0h, s8b);
    const int8x16_t v0_1ls = vsubq_s8(v0_1l, s8b);
    const int8x16_t v0_1hs = vsubq_s8(v0_1h, s8b);

    // load y
    const int8x16_t v1_0l = vld1q_s8(y0->qs);
    const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);
    const int8x16_t v1_1l = vld1q_s8(y1->qs);
    const int8x16_t v1_1h = vld1q_s8(y1->qs + 16);

#if defined(__ARM_FEATURE_DOTPROD)
    // dot product into int32x4_t
    const int32x4_t p_0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), v0_0ls, v1_0l), v0_0hs, v1_0h);
    const int32x4_t p_1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), v0_1ls, v1_1l), v0_1hs, v1_1h);

    sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), MLLM_FP16_TO_FP32(x0->d) * MLLM_FP16_TO_FP32(y0->d));
    sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), MLLM_FP16_TO_FP32(x1->d) * MLLM_FP16_TO_FP32(y1->d));
#else
    const int16x8_t pl0l = vmull_s8(vget_low_s8(v0_0ls), vget_low_s8(v1_0l));
    const int16x8_t pl0h = vmull_s8(vget_high_s8(v0_0ls), vget_high_s8(v1_0l));
    const int16x8_t ph0l = vmull_s8(vget_low_s8(v0_0hs), vget_low_s8(v1_0h));
    const int16x8_t ph0h = vmull_s8(vget_high_s8(v0_0hs), vget_high_s8(v1_0h));

    const int16x8_t pl1l = vmull_s8(vget_low_s8(v0_1ls), vget_low_s8(v1_1l));
    const int16x8_t pl1h = vmull_s8(vget_high_s8(v0_1ls), vget_high_s8(v1_1l));
    const int16x8_t ph1l = vmull_s8(vget_low_s8(v0_1hs), vget_low_s8(v1_1h));
    const int16x8_t ph1h = vmull_s8(vget_high_s8(v0_1hs), vget_high_s8(v1_1h));

    const int32x4_t pl0 = vaddq_s32(vpaddlq_s16(pl0l), vpaddlq_s16(pl0h));
    const int32x4_t ph0 = vaddq_s32(vpaddlq_s16(ph0l), vpaddlq_s16(ph0h));
    const int32x4_t pl1 = vaddq_s32(vpaddlq_s16(pl1l), vpaddlq_s16(pl1h));
    const int32x4_t ph1 = vaddq_s32(vpaddlq_s16(ph1l), vpaddlq_s16(ph1h));

    sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(pl0, ph0)), MLLM_FP16_TO_FP32(x0->d) * MLLM_FP16_TO_FP32(y0->d));
    sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(pl1, ph1)), MLLM_FP16_TO_FP32(x1->d) * MLLM_FP16_TO_FP32(y1->d));
#endif
  }

  *s = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
}
#endif

void vec_dot_q4_0_q8_0(const int n, float* __restrict s, const void* __restrict vx, const void* __restrict vy) {
#ifdef __AVX2__
  vec_dot_q4_0_q8_0_avx(n, s, vx, vy);
#elif defined(__ARM_NEON)
  vec_dot_q4_0_q8_0_arm(n, s, vx, vy);
#endif
}

#if QK_K == 256
void vec_dot_q4_K_q8_K(const int n, float* __restrict s, const void* __restrict vx, const void* __restrict vy) {
  assert(n % QK_K == 0);

  const block_q4_K* __restrict x = (block_q4_K*)vx;
  const block_q8_K* __restrict y = (block_q8_K*)vy;

  const int nb = n / QK_K;

  static const uint32_t kmask1 = 0x3f3f3f3f;
  static const uint32_t kmask2 = 0x0f0f0f0f;
  static const uint32_t kmask3 = 0x03030303;

  uint32_t utmp[4];

#ifdef __ARM_FEATURE_SVE
  float sumf = 0;
  for (int i = 0; i < nb; ++i) {
    const float d = y[i].d * MLLM_FP16_TO_FP32(x[i].d);
    const float dmin = y[i].d * MLLM_FP16_TO_FP32(x[i].dmin);

    const int16x8_t q8sums = vpaddq_s16(vld1q_s16(y[i].bsums), vld1q_s16(y[i].bsums + 8));

    memcpy(utmp, x[i].scales, K_SCALE_SIZE);

    uint32x2_t mins8 = {0};
    mins8 = vset_lane_u32(utmp[1] & kmask1, mins8, 0);
    mins8 = vset_lane_u32(((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4), mins8, 1);

    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[0] &= kmask1;

    const int16x8_t mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins8)));
    const int32x4_t prod =
        vaddq_s32(vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins)), vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)));
    sumf -= dmin * vaddvq_s32(prod);

    const uint8_t* scales = (const uint8_t*)utmp;

    const uint8_t* __restrict q4 = (const uint8_t*)x[i].qs;
    const int8_t* __restrict q8 = (const int8_t*)y[i].qs;

    const int vector_length = mllm_cpu_get_sve_cnt() * 8;
    const svuint8_t m4b = svdup_n_u8(0xf);
    const svint32_t mzero = svdup_n_s32(0);
    svint32_t sumi1 = svdup_n_s32(0);
    svint32_t sumi1_1 = svdup_n_s32(0);
    svint32_t sumi1_2 = svdup_n_s32(0);
    svint32_t sumi2 = svdup_n_s32(0);
    svint32_t sumi2_1 = svdup_n_s32(0);
    svint32_t sumi2_2 = svdup_n_s32(0);
    switch (vector_length) {
      case 128: {
        for (int j = 0; j < QK_K / 64; ++j) {
          svint8_t q4bytes = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svld1_u8(svptrue_b8(), q4), m4b));
          svint8_t q8bytes = svld1_s8(svptrue_b8(), q8);
          q8 += 16;
          sumi1_1 = svmla_n_s32_x(svptrue_b32(), sumi1_1, svdot_s32(mzero, q4bytes, q8bytes), scales[2 * j + 0]);
          q4bytes = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svld1_u8(svptrue_b8(), q4 + 16), m4b));
          q8bytes = svld1_s8(svptrue_b8(), q8);
          q8 += 16;
          sumi1_2 = svmla_n_s32_x(svptrue_b32(), sumi1_2, svdot_s32(mzero, q4bytes, q8bytes), scales[2 * j + 0]);

          q4bytes = svreinterpret_s8_u8(svlsr_n_u8_x(svptrue_b8(), svld1_u8(svptrue_b8(), q4), 4));
          q8bytes = svld1_s8(svptrue_b8(), q8);
          q8 += 16;
          sumi2_1 = svmla_n_s32_x(svptrue_b32(), sumi2_1, svdot_s32(mzero, q4bytes, q8bytes), scales[2 * j + 1]);
          q4bytes = svreinterpret_s8_u8(svlsr_n_u8_x(svptrue_b8(), svld1_u8(svptrue_b8(), q4 + 16), 4));
          q8bytes = svld1_s8(svptrue_b8(), q8);
          q8 += 16;
          sumi2_2 = svmla_n_s32_x(svptrue_b32(), sumi2_2, svdot_s32(mzero, q4bytes, q8bytes), scales[2 * j + 1]);
          q4 += 32;
        }
        sumi1 = svadd_s32_x(svptrue_b32(), sumi1_1, sumi1_2);
        sumi2 = svadd_s32_x(svptrue_b32(), sumi2_1, sumi2_2);
        sumf += d * (svaddv_s32(svptrue_b32(), svadd_s32_x(svptrue_b32(), sumi1, sumi2)));
      } break;
      case 256:
      case 512: {
        for (int j = 0; j < QK_K / 64; ++j) {
          const svuint8_t q4bits = svld1_u8(svptrue_pat_b8(SV_VL32), q4);
          q4 += 32;
          svint8_t q4bytes = svreinterpret_s8_u8(svand_u8_x(svptrue_pat_b8(SV_VL32), q4bits, m4b));
          svint8_t q8bytes = svld1_s8(svptrue_pat_b8(SV_VL32), q8);
          q8 += 32;
          sumi1 = svmla_n_s32_x(svptrue_pat_b32(SV_VL8), sumi1, svdot_s32(mzero, q4bytes, q8bytes), scales[2 * j + 0]);

          q4bytes = svreinterpret_s8_u8(svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), q4bits, 4));
          q8bytes = svld1_s8(svptrue_pat_b8(SV_VL32), q8);
          q8 += 32;
          sumi2 = svmla_n_s32_x(svptrue_pat_b32(SV_VL8), sumi2, svdot_s32(mzero, q4bytes, q8bytes), scales[2 * j + 1]);
        }
        sumf += d * (svaddv_s32(svptrue_pat_b32(SV_VL8), svadd_s32_x(svptrue_pat_b32(SV_VL8), sumi1, sumi2)));
      } break;
      default: assert(false && "Unsupported vector length"); break;
    }
  }
  *s = sumf;
#elif defined __ARM_NEON

  const uint8x16_t m4b = vdupq_n_u8(0xf);
#ifdef __ARM_FEATURE_DOTPROD
  const int32x4_t mzero = vdupq_n_s32(0);
#endif

  int8x16x2_t q4bytes;
  int8x16x2_t q8bytes;

  float sumf = 0;

  for (int i = 0; i < nb; ++i) {
    const float d = y[i].d * MLLM_FP16_TO_FP32(x[i].d);
    const float dmin = y[i].d * MLLM_FP16_TO_FP32(x[i].dmin);

    const int16x8_t q8sums = vpaddq_s16(vld1q_s16(y[i].bsums), vld1q_s16(y[i].bsums + 8));

    memcpy(utmp, x[i].scales, 12);

    const uint32x2_t mins8 = {utmp[1] & kmask1, ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4)};
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[0] &= kmask1;

    const int16x8_t mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins8)));
    const int32x4_t prod =
        vaddq_s32(vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins)), vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)));
    sumf -= dmin * vaddvq_s32(prod);

    const uint8_t* scales = (const uint8_t*)utmp;

    const uint8_t* __restrict q4 = x[i].qs;
    const int8_t* __restrict q8 = y[i].qs;

    // int32x4_t isum = mzero;

    int32_t sumi1 = 0;
    int32_t sumi2 = 0;

    for (int j = 0; j < QK_K / 64; ++j) {
      const uint8x16x2_t q4bits = vld1q_u8_x2(q4);
      q4 += 32;

#ifdef __ARM_FEATURE_DOTPROD
      q8bytes = vld1q_s8_x2(q8);
      q8 += 32;
      q4bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(q4bits.val[0], m4b));
      q4bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(q4bits.val[1], m4b));

      const int32x4_t p1 = vdotq_s32(vdotq_s32(mzero, q4bytes.val[0], q8bytes.val[0]), q4bytes.val[1], q8bytes.val[1]);
      sumi1 += vaddvq_s32(p1) * scales[2 * j + 0];

      q8bytes = vld1q_s8_x2(q8);
      q8 += 32;
      q4bytes.val[0] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[0], 4));
      q4bytes.val[1] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[1], 4));

      const int32x4_t p2 = vdotq_s32(vdotq_s32(mzero, q4bytes.val[0], q8bytes.val[0]), q4bytes.val[1], q8bytes.val[1]);

      sumi2 += vaddvq_s32(p2) * scales[2 * j + 1];
#else
      q8bytes = vld1q_s8_x2(q8);
      q8 += 32;
      q4bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(q4bits.val[0], m4b));
      q4bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(q4bits.val[1], m4b));
      const int16x8_t p0 = vaddq_s16(vmull_s8(vget_low_s8(q4bytes.val[0]), vget_low_s8(q8bytes.val[0])),
                                     vmull_s8(vget_high_s8(q4bytes.val[0]), vget_high_s8(q8bytes.val[0])));
      const int16x8_t p1 = vaddq_s16(vmull_s8(vget_low_s8(q4bytes.val[1]), vget_low_s8(q8bytes.val[1])),
                                     vmull_s8(vget_high_s8(q4bytes.val[1]), vget_high_s8(q8bytes.val[1])));
      sumi1 += vaddvq_s16(vaddq_s16(p0, p1)) * scales[2 * j + 0];

      q8bytes = vld1q_s8_x2(q8);
      q8 += 32;
      q4bytes.val[0] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[0], 4));
      q4bytes.val[1] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[1], 4));
      const int16x8_t p2 = vaddq_s16(vmull_s8(vget_low_s8(q4bytes.val[0]), vget_low_s8(q8bytes.val[0])),
                                     vmull_s8(vget_high_s8(q4bytes.val[0]), vget_high_s8(q8bytes.val[0])));
      const int16x8_t p3 = vaddq_s16(vmull_s8(vget_low_s8(q4bytes.val[1]), vget_low_s8(q8bytes.val[1])),
                                     vmull_s8(vget_high_s8(q4bytes.val[1]), vget_high_s8(q8bytes.val[1])));
      sumi2 += vaddvq_s16(vaddq_s16(p2, p3)) * scales[2 * j + 1];

#endif
    }

    sumf += d * (sumi1 + sumi2);
  }

  *s = sumf;

#elif defined __AVX2__

  const __m256i m4 = _mm256_set1_epi8(0xF);

  __m256 acc = _mm256_setzero_ps();
  __m128 acc_m = _mm_setzero_ps();

  for (int i = 0; i < nb; ++i) {
    const float d = y[i].d * MLLM_FP16_TO_FP32(x[i].d);
    const float dmin = -y[i].d * MLLM_FP16_TO_FP32(x[i].dmin);

    memcpy(utmp, x[i].scales, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;

    const uint8_t* __restrict q4 = x[i].qs;
    const int8_t* __restrict q8 = y[i].qs;

    const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));

    const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
    const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
    const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
    acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

    const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
    const __m256i scales = MM256_SET_M128I(sc128, sc128);

    __m256i sumi = _mm256_setzero_si256();

    for (int j = 0; j < QK_K / 64; ++j) {
      const __m256i scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
      const __m256i scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));

      const __m256i q4bits = _mm256_loadu_si256((const __m256i*)q4);
      q4 += 32;
      const __m256i q4l = _mm256_and_si256(q4bits, m4);
      const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

      const __m256i q8l = _mm256_loadu_si256((const __m256i*)q8);
      q8 += 32;
      __m256i p16l = _mm256_maddubs_epi16(q4l, q8l);
      p16l = _mm256_madd_epi16(scale_l, p16l);

      const __m256i q8h = _mm256_loadu_si256((const __m256i*)q8);
      q8 += 32;
      __m256i p16h = _mm256_maddubs_epi16(q4h, q8h);
      p16h = _mm256_madd_epi16(scale_h, p16h);
      const __m256i sumj = _mm256_add_epi32(p16l, p16h);

      sumi = _mm256_add_epi32(sumi, sumj);
    }

    __m256 vd = _mm256_set1_ps(d);
    acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);
  }

  acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
  acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));

  *s = hsum_float_8(acc) + _mm_cvtss_f32(acc_m);

#else
  const uint8_t* scales = (const uint8_t*)&utmp[0];
  const uint8_t* mins = (const uint8_t*)&utmp[2];

  int8_t aux8[QK_K];
  int16_t aux16[8];
  float sums[8];
  int32_t aux32[8];
  memset(sums, 0, 8 * sizeof(float));

  float sumf = 0;
  for (int i = 0; i < nb; ++i) {
    const uint8_t* __restrict q4 = x[i].qs;
    const int8_t* __restrict q8 = y[i].qs;
    memset(aux32, 0, 8 * sizeof(int32_t));
    int8_t* __restrict a = aux8;
    for (int j = 0; j < QK_K / 64; ++j) {
      for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
      a += 32;
      for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] >> 4);
      a += 32;
      q4 += 32;
    }
    memcpy(utmp, x[i].scales, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;

    int sumi = 0;
    for (int j = 0; j < QK_K / 16; ++j) sumi += y[i].bsums[j] * mins[j / 2];
    a = aux8;
    int is = 0;
    for (int j = 0; j < QK_K / 32; ++j) {
      int32_t scale = scales[is++];
      for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
      for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
      q8 += 8;
      a += 8;
      for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
      for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
      q8 += 8;
      a += 8;
      for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
      for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
      q8 += 8;
      a += 8;
      for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
      for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
      q8 += 8;
      a += 8;
    }
    const float d = MLLM_FP16_TO_FP32(x[i].d) * y[i].d;
    for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    const float dmin = MLLM_FP16_TO_FP32(x[i].dmin) * y[i].d;
    sumf -= dmin * sumi;
  }
  for (int l = 0; l < 8; ++l) sumf += sums[l];
  *s = sumf;
#endif
}
#else
void vec_dot_q4_K_q8_K(const int n, float* __restrict s, const void* __restrict vx, const void* __restrict vy) {
  assert(n % QK_K == 0);

  const block_q4_K* __restrict x = (block_q4_K*)vx;
  const block_q8_K* __restrict y = (block_q8_K*)vy;

  const int nb = n / QK_K;

#ifdef __ARM_NEON

  const uint8x16_t m4b = vdupq_n_u8(0xf);

#ifdef __ARM_FEATURE_DOTPROD
  const int32x4_t mzero = vdupq_n_s32(0);
#endif

  float sumf = 0;

  int8x16x2_t q4bytes;
  int8x16x4_t q8bytes;

  float sum_mins = 0.f;

  uint16_t aux16[2];
  const uint8_t* __restrict scales = (const uint8_t*)aux16;

  for (int i = 0; i < nb; ++i) {
    const uint8_t* __restrict q4 = x[i].qs;
    const int8_t* __restrict q8 = y[i].qs;

    const uint16_t* __restrict a = (const uint16_t*)x[i].scales;
    aux16[0] = a[0] & 0x0f0f;
    aux16[1] = (a[0] >> 4) & 0x0f0f;

    const int32_t summi = scales[2] * (y[i].bsums[0] + y[i].bsums[1]) + scales[3] * (y[i].bsums[2] + y[i].bsums[3]);
    sum_mins += y[i].d * (float)x[i].d[1] * summi;

    const float d = y[i].d * (float)x[i].d[0];

    const uint8x16x2_t q4bits = vld1q_u8_x2(q4);

#ifdef __ARM_FEATURE_DOTPROD
    q8bytes = vld1q_s8_x4(q8);
    q4bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(q4bits.val[0], m4b));
    q4bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(q4bits.val[1], m4b));

    const int32x4_t p1 = vdotq_s32(vdotq_s32(mzero, q4bytes.val[0], q8bytes.val[0]), q4bytes.val[1], q8bytes.val[1]);
    const int32_t sumi1 = vaddvq_s32(p1) * scales[0];

    q4bytes.val[0] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[0], 4));
    q4bytes.val[1] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[1], 4));

    const int32x4_t p2 = vdotq_s32(vdotq_s32(mzero, q4bytes.val[0], q8bytes.val[2]), q4bytes.val[1], q8bytes.val[3]);
    const int32_t sumi2 = vaddvq_s32(p2) * scales[1];

#else
    q8bytes = vld1q_s8_x4(q8);
    q4bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(q4bits.val[0], m4b));
    q4bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(q4bits.val[1], m4b));
    const int16x8_t p0 = vaddq_s16(vmull_s8(vget_low_s8(q4bytes.val[0]), vget_low_s8(q8bytes.val[0])),
                                   vmull_s8(vget_high_s8(q4bytes.val[0]), vget_high_s8(q8bytes.val[0])));
    const int16x8_t p1 = vaddq_s16(vmull_s8(vget_low_s8(q4bytes.val[1]), vget_low_s8(q8bytes.val[1])),
                                   vmull_s8(vget_high_s8(q4bytes.val[1]), vget_high_s8(q8bytes.val[1])));
    int32_t sumi1 = vaddvq_s16(vaddq_s16(p0, p1)) * scales[0];

    q4bytes.val[0] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[0], 4));
    q4bytes.val[1] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[1], 4));
    const int16x8_t p2 = vaddq_s16(vmull_s8(vget_low_s8(q4bytes.val[0]), vget_low_s8(q8bytes.val[2])),
                                   vmull_s8(vget_high_s8(q4bytes.val[0]), vget_high_s8(q8bytes.val[2])));
    const int16x8_t p3 = vaddq_s16(vmull_s8(vget_low_s8(q4bytes.val[1]), vget_low_s8(q8bytes.val[3])),
                                   vmull_s8(vget_high_s8(q4bytes.val[1]), vget_high_s8(q8bytes.val[3])));
    int32_t sumi2 = vaddvq_s16(vaddq_s16(p2, p3)) * scales[1];

#endif
    sumf += d * (sumi1 + sumi2);
  }

  *s = sumf - sum_mins;

#elif defined __AVX2__

  const __m256i m4 = _mm256_set1_epi8(0xF);

  __m256 acc = _mm256_setzero_ps();

  float summs = 0;

  uint16_t aux16[2];
  const uint8_t* scales = (const uint8_t*)aux16;

  for (int i = 0; i < nb; ++i) {
    const float d = MLLM_FP16_TO_FP32(x[i].d[0]) * y[i].d;
    const float m = MLLM_FP16_TO_FP32(x[i].d[1]) * y[i].d;
    const __m256 vd = _mm256_set1_ps(d);

    const uint16_t* a = (const uint16_t*)x[i].scales;
    aux16[0] = a[0] & 0x0f0f;
    aux16[1] = (a[0] >> 4) & 0x0f0f;

    summs += m * (scales[2] * (y[i].bsums[0] + y[i].bsums[1]) + scales[3] * (y[i].bsums[2] + y[i].bsums[3]));

    const uint8_t* __restrict q4 = x[i].qs;
    const int8_t* __restrict q8 = y[i].qs;

    const __m256i q4bits = _mm256_loadu_si256((const __m256i*)q4);
    const __m256i q4l = _mm256_and_si256(q4bits, m4);
    const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

    const __m256i q8l = _mm256_loadu_si256((const __m256i*)(q8 + 0));
    const __m256i q8h = _mm256_loadu_si256((const __m256i*)(q8 + 32));

    const __m256i p16l = _mm256_maddubs_epi16(q4l, q8l);
    const __m256i p16h = _mm256_maddubs_epi16(q4h, q8h);

    const __m256i p32l = _mm256_madd_epi16(_mm256_set1_epi16(scales[0]), p16l);
    acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(p32l), acc);

    const __m256i p32h = _mm256_madd_epi16(_mm256_set1_epi16(scales[1]), p16h);
    acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(p32h), acc);
  }

  *s = hsum_float_8(acc) - summs;

#else

  uint8_t aux8[QK_K];
  int16_t aux16[16];
  float sums[8];
  memset(sums, 0, 8 * sizeof(float));

  uint16_t s16[2];
  const uint8_t* __restrict scales = (const uint8_t*)s16;

  float sumf = 0;
  for (int i = 0; i < nb; ++i) {
    const uint8_t* __restrict q4 = x[i].qs;
    const int8_t* __restrict q8 = y[i].qs;
    uint8_t* __restrict a = aux8;
    for (int l = 0; l < 32; ++l) a[l + 0] = q4[l] & 0xF;
    for (int l = 0; l < 32; ++l) a[l + 32] = q4[l] >> 4;

    const uint16_t* __restrict b = (const uint16_t*)x[i].scales;
    s16[0] = b[0] & 0x0f0f;
    s16[1] = (b[0] >> 4) & 0x0f0f;

    sumf -= y[i].d * MLLM_FP16_TO_FP32(x[i].d[1])
            * (scales[2] * (y[i].bsums[0] + y[i].bsums[1]) + scales[3] * (y[i].bsums[2] + y[i].bsums[3]));

    const float d = y[i].d * MLLM_FP16_TO_FP32(x[i].d[0]);

    for (int j = 0; j < QK_K / 32; ++j) {
      for (int l = 0; l < 16; ++l) aux16[l] = q8[l] * a[l];
      q8 += 16;
      a += 16;
      for (int l = 0; l < 16; ++l) aux16[l] += q8[l] * a[l];
      q8 += 16;
      a += 16;
      const float dl = d * scales[j];
      for (int l = 0; l < 8; ++l) sums[l] += dl * (aux16[l] + aux16[l + 8]);
    }
  }
  for (int l = 0; l < 8; ++l) sumf += sums[l];
  *s = sumf;
#endif
}
#endif

#if QK_K == 256
void vec_dot_q6_K_q8_K(const int n, float* __restrict s, const void* __restrict vx, const void* __restrict vy) {
  assert(n % QK_K == 0);

  const block_q6_K* __restrict x = (block_q6_K*)vx;
  const block_q8_K* __restrict y = (block_q8_K*)vy;

  const int nb = n / QK_K;

#ifdef __ARM_NEON

  float sum = 0;

  const uint8x16_t m4b = vdupq_n_u8(0xF);
#if defined(__ARM_FEATURE_DOTPROD)
  const int32x4_t vzero = vdupq_n_s32(0);
#endif
  // const int8x16_t  m32s = vdupq_n_s8(32);

  const uint8x16_t mone = vdupq_n_u8(3);

  int8x16x4_t q6bytes;
  uint8x16x4_t q6h;

  for (int i = 0; i < nb; ++i) {
    const float d_all = MLLM_FP16_TO_FP32(x[i].d);

    const uint8_t* __restrict q6 = x[i].ql;
    const uint8_t* __restrict qh = x[i].qh;
    const int8_t* __restrict q8 = y[i].qs;

    const int8_t* __restrict scale = x[i].scales;

    const int16x8x2_t q8sums = vld1q_s16_x2(y[i].bsums);
    const int8x16_t scales = vld1q_s8(scale);
    const int16x8x2_t q6scales = {vmovl_s8(vget_low_s8(scales)), vmovl_s8(vget_high_s8(scales))};

    const int32x4_t prod = vaddq_s32(vaddq_s32(vmull_s16(vget_low_s16(q8sums.val[0]), vget_low_s16(q6scales.val[0])),
                                               vmull_s16(vget_high_s16(q8sums.val[0]), vget_high_s16(q6scales.val[0]))),
                                     vaddq_s32(vmull_s16(vget_low_s16(q8sums.val[1]), vget_low_s16(q6scales.val[1])),
                                               vmull_s16(vget_high_s16(q8sums.val[1]), vget_high_s16(q6scales.val[1]))));
    int32_t isum_mins = vaddvq_s32(prod);

    int32_t isum = 0;

    for (int j = 0; j < QK_K / 128; ++j) {
      uint8x16x2_t qhbits = vld1q_u8_x2(qh);
      qh += 32;
      uint8x16x4_t q6bits = vld1q_u8_x4(q6);
      q6 += 64;
      int8x16x4_t q8bytes = vld1q_s8_x4(q8);
      q8 += 64;

      q6h.val[0] = vshlq_n_u8(vandq_u8(mone, qhbits.val[0]), 4);
      q6h.val[1] = vshlq_n_u8(vandq_u8(mone, qhbits.val[1]), 4);
      uint8x16_t shifted = vshrq_n_u8(qhbits.val[0], 2);
      q6h.val[2] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
      shifted = vshrq_n_u8(qhbits.val[1], 2);
      q6h.val[3] = vshlq_n_u8(vandq_u8(mone, shifted), 4);

      // q6bytes.val[0] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[0], m4b), q6h.val[0])), m32s);
      // q6bytes.val[1] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[1], m4b), q6h.val[1])), m32s);
      // q6bytes.val[2] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[2], m4b), q6h.val[2])), m32s);
      // q6bytes.val[3] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[3], m4b), q6h.val[3])), m32s);
      q6bytes.val[0] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[0], m4b), q6h.val[0]));
      q6bytes.val[1] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[1], m4b), q6h.val[1]));
      q6bytes.val[2] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[2], m4b), q6h.val[2]));
      q6bytes.val[3] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[3], m4b), q6h.val[3]));

#if defined(__ARM_FEATURE_DOTPROD)

      isum += vaddvq_s32(vdotq_s32(vzero, q6bytes.val[0], q8bytes.val[0])) * scale[0]
              + vaddvq_s32(vdotq_s32(vzero, q6bytes.val[1], q8bytes.val[1])) * scale[1]
              + vaddvq_s32(vdotq_s32(vzero, q6bytes.val[2], q8bytes.val[2])) * scale[2]
              + vaddvq_s32(vdotq_s32(vzero, q6bytes.val[3], q8bytes.val[3])) * scale[3];
      scale += 4;

#else

      int16x8_t p0 = vaddq_s16(vmull_s8(vget_low_s8(q6bytes.val[0]), vget_low_s8(q8bytes.val[0])),
                               vmull_s8(vget_high_s8(q6bytes.val[0]), vget_high_s8(q8bytes.val[0])));
      int16x8_t p1 = vaddq_s16(vmull_s8(vget_low_s8(q6bytes.val[1]), vget_low_s8(q8bytes.val[1])),
                               vmull_s8(vget_high_s8(q6bytes.val[1]), vget_high_s8(q8bytes.val[1])));
      isum += vaddvq_s16(p0) * scale[0] + vaddvq_s16(p1) * scale[1];
      scale += 2;

      int16x8_t p2 = vaddq_s16(vmull_s8(vget_low_s8(q6bytes.val[2]), vget_low_s8(q8bytes.val[2])),
                               vmull_s8(vget_high_s8(q6bytes.val[2]), vget_high_s8(q8bytes.val[2])));
      int16x8_t p3 = vaddq_s16(vmull_s8(vget_low_s8(q6bytes.val[3]), vget_low_s8(q8bytes.val[3])),
                               vmull_s8(vget_high_s8(q6bytes.val[3]), vget_high_s8(q8bytes.val[3])));
      isum += vaddvq_s16(p2) * scale[0] + vaddvq_s16(p3) * scale[1];
      scale += 2;
#endif

      q8bytes = vld1q_s8_x4(q8);
      q8 += 64;

      shifted = vshrq_n_u8(qhbits.val[0], 4);
      q6h.val[0] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
      shifted = vshrq_n_u8(qhbits.val[1], 4);
      q6h.val[1] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
      shifted = vshrq_n_u8(qhbits.val[0], 6);
      q6h.val[2] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
      shifted = vshrq_n_u8(qhbits.val[1], 6);
      q6h.val[3] = vshlq_n_u8(vandq_u8(mone, shifted), 4);

      // q6bytes.val[0] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[0], 4), q6h.val[0])), m32s);
      // q6bytes.val[1] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[1], 4), q6h.val[1])), m32s);
      // q6bytes.val[2] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[2], 4), q6h.val[2])), m32s);
      // q6bytes.val[3] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[3], 4), q6h.val[3])), m32s);
      q6bytes.val[0] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[0], 4), q6h.val[0]));
      q6bytes.val[1] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[1], 4), q6h.val[1]));
      q6bytes.val[2] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[2], 4), q6h.val[2]));
      q6bytes.val[3] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[3], 4), q6h.val[3]));

#if defined(__ARM_FEATURE_DOTPROD)

      isum += vaddvq_s32(vdotq_s32(vzero, q6bytes.val[0], q8bytes.val[0])) * scale[0]
              + vaddvq_s32(vdotq_s32(vzero, q6bytes.val[1], q8bytes.val[1])) * scale[1]
              + vaddvq_s32(vdotq_s32(vzero, q6bytes.val[2], q8bytes.val[2])) * scale[2]
              + vaddvq_s32(vdotq_s32(vzero, q6bytes.val[3], q8bytes.val[3])) * scale[3];
      scale += 4;

      // for (int l = 0; l < 4; ++l) {
      //     const int32x4_t p = vdotq_s32(vzero, q6bytes.val[l], q8bytes.val[l]);
      //     isum += vaddvq_s32(p) * *scale++;
      // }
#else
      p0 = vaddq_s16(vmull_s8(vget_low_s8(q6bytes.val[0]), vget_low_s8(q8bytes.val[0])),
                     vmull_s8(vget_high_s8(q6bytes.val[0]), vget_high_s8(q8bytes.val[0])));
      p1 = vaddq_s16(vmull_s8(vget_low_s8(q6bytes.val[1]), vget_low_s8(q8bytes.val[1])),
                     vmull_s8(vget_high_s8(q6bytes.val[1]), vget_high_s8(q8bytes.val[1])));
      isum += vaddvq_s16(p0) * scale[0] + vaddvq_s16(p1) * scale[1];
      scale += 2;

      p2 = vaddq_s16(vmull_s8(vget_low_s8(q6bytes.val[2]), vget_low_s8(q8bytes.val[2])),
                     vmull_s8(vget_high_s8(q6bytes.val[2]), vget_high_s8(q8bytes.val[2])));
      p3 = vaddq_s16(vmull_s8(vget_low_s8(q6bytes.val[3]), vget_low_s8(q8bytes.val[3])),
                     vmull_s8(vget_high_s8(q6bytes.val[3]), vget_high_s8(q8bytes.val[3])));
      isum += vaddvq_s16(p2) * scale[0] + vaddvq_s16(p3) * scale[1];
      scale += 2;
#endif
    }
    // sum += isum * d_all * y[i].d;
    sum += d_all * y[i].d * (isum - 32 * isum_mins);
  }
  *s = sum;

#elif defined __AVX2__

  const __m256i m4 = _mm256_set1_epi8(0xF);
  const __m256i m2 = _mm256_set1_epi8(3);
  const __m256i m32s = _mm256_set1_epi8(32);

  __m256 acc = _mm256_setzero_ps();

  for (int i = 0; i < nb; ++i) {
    const float d = y[i].d * MLLM_FP16_TO_FP32(x[i].d);

    const uint8_t* __restrict q4 = x[i].ql;
    const uint8_t* __restrict qh = x[i].qh;
    const int8_t* __restrict q8 = y[i].qs;

    const __m128i scales = _mm_loadu_si128((const __m128i*)x[i].scales);

    __m256i sumi = _mm256_setzero_si256();

    int is = 0;

    for (int j = 0; j < QK_K / 128; ++j) {
      const __m128i scale_0 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 0));
      const __m128i scale_1 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 1));
      const __m128i scale_2 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 2));
      const __m128i scale_3 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 3));
      is += 4;

      const __m256i q4bits1 = _mm256_loadu_si256((const __m256i*)q4);
      q4 += 32;
      const __m256i q4bits2 = _mm256_loadu_si256((const __m256i*)q4);
      q4 += 32;
      const __m256i q4bitsH = _mm256_loadu_si256((const __m256i*)qh);
      qh += 32;

      const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m2), 4);
      const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 2), m2), 4);
      const __m256i q4h_2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 4), m2), 4);
      const __m256i q4h_3 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 6), m2), 4);

      const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q4h_0);
      const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m4), q4h_1);
      const __m256i q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4), q4h_2);
      const __m256i q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m4), q4h_3);

      const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8);
      q8 += 32;
      const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8);
      q8 += 32;
      const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8);
      q8 += 32;
      const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8);
      q8 += 32;

      __m256i q8s_0 = _mm256_maddubs_epi16(m32s, q8_0);
      __m256i q8s_1 = _mm256_maddubs_epi16(m32s, q8_1);
      __m256i q8s_2 = _mm256_maddubs_epi16(m32s, q8_2);
      __m256i q8s_3 = _mm256_maddubs_epi16(m32s, q8_3);

      __m256i p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
      __m256i p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
      __m256i p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
      __m256i p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

      p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
      p16_1 = _mm256_sub_epi16(p16_1, q8s_1);
      p16_2 = _mm256_sub_epi16(p16_2, q8s_2);
      p16_3 = _mm256_sub_epi16(p16_3, q8s_3);

      p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
      p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
      p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
      p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);

      sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
      sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
    }

    acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
  }

#if defined(_MSC_VER) || defined(__MINGW32__)
  float arr[8];
  _mm256_storeu_ps(arr, acc);

  // for(float i : arr) {
  //     printf("%f ", i);
  // }
  // printf("\n");
#endif

  *s = hsum_float_8(acc);

#elif defined __AVX__
  const __m128i m4 = _mm_set1_epi8(0xF);
  const __m128i m3 = _mm_set1_epi8(3);
  const __m128i m32s = _mm_set1_epi8(32);
  const __m128i m2 = _mm_set1_epi8(2);

  __m256 acc = _mm256_setzero_ps();

  for (int i = 0; i < nb; ++i) {
    const float d = y[i].d * MLLM_FP16_TO_FP32(x[i].d);

    const uint8_t* __restrict q4 = x[i].ql;
    const uint8_t* __restrict qh = x[i].qh;
    const int8_t* __restrict q8 = y[i].qs;

    const __m128i scales = _mm_loadu_si128((const __m128i*)x[i].scales);

    __m128i sumi_0 = _mm_setzero_si128();
    __m128i sumi_1 = _mm_setzero_si128();

    __m128i shuffle = _mm_set_epi64x(0x0101010101010101, 0x0000000000000000);
    for (int j = 0; j < QK_K / 128; ++j) {
      const __m128i q4bitsH_0 = _mm_loadu_si128((const __m128i*)qh);
      qh += 16;
      const __m128i q4bitsH_1 = _mm_loadu_si128((const __m128i*)qh);
      qh += 16;

      const __m128i q4h_0 = _mm_slli_epi16(_mm_and_si128(q4bitsH_0, m3), 4);
      const __m128i q4h_1 = _mm_slli_epi16(_mm_and_si128(q4bitsH_1, m3), 4);
      const __m128i q4h_2 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bitsH_0, 2), m3), 4);
      const __m128i q4h_3 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bitsH_1, 2), m3), 4);
      const __m128i q4h_4 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bitsH_0, 4), m3), 4);
      const __m128i q4h_5 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bitsH_1, 4), m3), 4);
      const __m128i q4h_6 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bitsH_0, 6), m3), 4);
      const __m128i q4h_7 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bitsH_1, 6), m3), 4);

      const __m128i q4bits1_0 = _mm_loadu_si128((const __m128i*)q4);
      q4 += 16;
      const __m128i q4bits1_1 = _mm_loadu_si128((const __m128i*)q4);
      q4 += 16;
      const __m128i q4bits2_0 = _mm_loadu_si128((const __m128i*)q4);
      q4 += 16;
      const __m128i q4bits2_1 = _mm_loadu_si128((const __m128i*)q4);
      q4 += 16;

      const __m128i q4_0 = _mm_or_si128(_mm_and_si128(q4bits1_0, m4), q4h_0);
      const __m128i q4_1 = _mm_or_si128(_mm_and_si128(q4bits1_1, m4), q4h_1);
      const __m128i q4_2 = _mm_or_si128(_mm_and_si128(q4bits2_0, m4), q4h_2);
      const __m128i q4_3 = _mm_or_si128(_mm_and_si128(q4bits2_1, m4), q4h_3);
      const __m128i q4_4 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits1_0, 4), m4), q4h_4);
      const __m128i q4_5 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits1_1, 4), m4), q4h_5);
      const __m128i q4_6 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits2_0, 4), m4), q4h_6);
      const __m128i q4_7 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits2_1, 4), m4), q4h_7);

      const __m128i q8_0 = _mm_loadu_si128((const __m128i*)q8);
      q8 += 16;
      const __m128i q8_1 = _mm_loadu_si128((const __m128i*)q8);
      q8 += 16;
      const __m128i q8_2 = _mm_loadu_si128((const __m128i*)q8);
      q8 += 16;
      const __m128i q8_3 = _mm_loadu_si128((const __m128i*)q8);
      q8 += 16;
      const __m128i q8_4 = _mm_loadu_si128((const __m128i*)q8);
      q8 += 16;
      const __m128i q8_5 = _mm_loadu_si128((const __m128i*)q8);
      q8 += 16;
      const __m128i q8_6 = _mm_loadu_si128((const __m128i*)q8);
      q8 += 16;
      const __m128i q8_7 = _mm_loadu_si128((const __m128i*)q8);
      q8 += 16;

      __m128i q8s_0 = _mm_maddubs_epi16(m32s, q8_0);
      __m128i q8s_1 = _mm_maddubs_epi16(m32s, q8_1);
      __m128i q8s_2 = _mm_maddubs_epi16(m32s, q8_2);
      __m128i q8s_3 = _mm_maddubs_epi16(m32s, q8_3);
      __m128i q8s_4 = _mm_maddubs_epi16(m32s, q8_4);
      __m128i q8s_5 = _mm_maddubs_epi16(m32s, q8_5);
      __m128i q8s_6 = _mm_maddubs_epi16(m32s, q8_6);
      __m128i q8s_7 = _mm_maddubs_epi16(m32s, q8_7);

      __m128i p16_0 = _mm_maddubs_epi16(q4_0, q8_0);
      __m128i p16_1 = _mm_maddubs_epi16(q4_1, q8_1);
      __m128i p16_2 = _mm_maddubs_epi16(q4_2, q8_2);
      __m128i p16_3 = _mm_maddubs_epi16(q4_3, q8_3);
      __m128i p16_4 = _mm_maddubs_epi16(q4_4, q8_4);
      __m128i p16_5 = _mm_maddubs_epi16(q4_5, q8_5);
      __m128i p16_6 = _mm_maddubs_epi16(q4_6, q8_6);
      __m128i p16_7 = _mm_maddubs_epi16(q4_7, q8_7);

      p16_0 = _mm_sub_epi16(p16_0, q8s_0);
      p16_1 = _mm_sub_epi16(p16_1, q8s_1);
      p16_2 = _mm_sub_epi16(p16_2, q8s_2);
      p16_3 = _mm_sub_epi16(p16_3, q8s_3);
      p16_4 = _mm_sub_epi16(p16_4, q8s_4);
      p16_5 = _mm_sub_epi16(p16_5, q8s_5);
      p16_6 = _mm_sub_epi16(p16_6, q8s_6);
      p16_7 = _mm_sub_epi16(p16_7, q8s_7);

      const __m128i scale_0 = _mm_shuffle_epi8(scales, shuffle);
      shuffle = _mm_add_epi8(shuffle, m2);
      const __m128i scale_1 = _mm_shuffle_epi8(scales, shuffle);
      shuffle = _mm_add_epi8(shuffle, m2);
      const __m128i scale_2 = _mm_shuffle_epi8(scales, shuffle);
      shuffle = _mm_add_epi8(shuffle, m2);
      const __m128i scale_3 = _mm_shuffle_epi8(scales, shuffle);
      shuffle = _mm_add_epi8(shuffle, m2);

      p16_0 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_0), p16_0);
      p16_1 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_unpackhi_epi64(scale_0, scale_0)), p16_1);
      p16_2 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_1), p16_2);
      p16_3 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_unpackhi_epi64(scale_1, scale_1)), p16_3);
      p16_4 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_2), p16_4);
      p16_5 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_unpackhi_epi64(scale_2, scale_2)), p16_5);
      p16_6 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_3), p16_6);
      p16_7 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_unpackhi_epi64(scale_3, scale_3)), p16_7);

      sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p16_0, p16_2));
      sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p16_1, p16_3));
      sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p16_4, p16_6));
      sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p16_5, p16_7));
    }

    __m256i sumi = MM256_SET_M128I(sumi_1, sumi_0);
    acc = _mm256_add_ps(_mm256_mul_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi)), acc);
  }

  *s = hsum_float_8(acc);

#else

  int8_t aux8[QK_K];
  int16_t aux16[8];
  float sums[8];
  int32_t aux32[8];
  memset(sums, 0, 8 * sizeof(float));

  float sumf = 0;
  for (int i = 0; i < nb; ++i) {
    const uint8_t* __restrict q4 = x[i].ql;
    const uint8_t* __restrict qh = x[i].qh;
    const int8_t* __restrict q8 = y[i].qs;
    memset(aux32, 0, 8 * sizeof(int32_t));
    int8_t* __restrict a = aux8;
    for (int j = 0; j < QK_K; j += 128) {
      for (int l = 0; l < 32; ++l) {
        a[l + 0] = (int8_t)((q4[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
        a[l + 32] = (int8_t)((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
        a[l + 64] = (int8_t)((q4[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        a[l + 96] = (int8_t)((q4[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
      }
      a += 128;
      q4 += 64;
      qh += 32;
    }
    a = aux8;
    int is = 0;
    for (int j = 0; j < QK_K / 16; ++j) {
      int scale = x[i].scales[is++];
      for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
      for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
      q8 += 8;
      a += 8;
      for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
      for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
      q8 += 8;
      a += 8;
    }
    const float d = MLLM_FP16_TO_FP32(x[i].d) * y[i].d;
    for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
  }
  for (int l = 0; l < 8; ++l) sumf += sums[l];
  *s = sumf;
#endif
}

#else

void vec_dot_q6_K_q8_K(const int n, float* __restrict s, const void* __restrict vx, const void* __restrict vy) {
  assert(n % QK_K == 0);

  const block_q6_K* __restrict x = vx;
  const block_q8_K* __restrict y = vy;

  const int nb = n / QK_K;

#ifdef __ARM_NEON

  float sum = 0;

  const uint8x16_t m4b = vdupq_n_u8(0xF);
  const int8x16_t m32s = vdupq_n_s8(32);
#if defined(__ARM_FEATURE_DOTPROD)
  const int32x4_t vzero = vdupq_n_s32(0);
#endif

  const uint8x16_t mone = vdupq_n_u8(3);

  int8x16x4_t q6bytes;
  uint8x16x4_t q6h;

  for (int i = 0; i < nb; ++i) {
    const float d_all = (float)x[i].d;

    const uint8_t* __restrict q6 = x[i].ql;
    const uint8_t* __restrict qh = x[i].qh;
    const int8_t* __restrict q8 = y[i].qs;

    const int8_t* __restrict scale = x[i].scales;

    int32_t isum = 0;

    uint8x16_t qhbits = vld1q_u8(qh);
    uint8x16x2_t q6bits = vld1q_u8_x2(q6);
    int8x16x4_t q8bytes = vld1q_s8_x4(q8);

    q6h.val[0] = vshlq_n_u8(vandq_u8(mone, qhbits), 4);
    uint8x16_t shifted = vshrq_n_u8(qhbits, 2);
    q6h.val[1] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
    shifted = vshrq_n_u8(qhbits, 4);
    q6h.val[2] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
    shifted = vshrq_n_u8(qhbits, 6);
    q6h.val[3] = vshlq_n_u8(vandq_u8(mone, shifted), 4);

    q6bytes.val[0] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[0], m4b), q6h.val[0])), m32s);
    q6bytes.val[1] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[1], m4b), q6h.val[1])), m32s);
    q6bytes.val[2] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[0], 4), q6h.val[2])), m32s);
    q6bytes.val[3] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[1], 4), q6h.val[3])), m32s);

#if defined(__ARM_FEATURE_DOTPROD)

    isum += vaddvq_s32(vdotq_s32(vzero, q6bytes.val[0], q8bytes.val[0])) * scale[0]
            + vaddvq_s32(vdotq_s32(vzero, q6bytes.val[1], q8bytes.val[1])) * scale[1]
            + vaddvq_s32(vdotq_s32(vzero, q6bytes.val[2], q8bytes.val[2])) * scale[2]
            + vaddvq_s32(vdotq_s32(vzero, q6bytes.val[3], q8bytes.val[3])) * scale[3];
#else

    int16x8_t p0 = vaddq_s16(vmull_s8(vget_low_s8(q6bytes.val[0]), vget_low_s8(q8bytes.val[0])),
                             vmull_s8(vget_high_s8(q6bytes.val[0]), vget_high_s8(q8bytes.val[0])));
    int16x8_t p1 = vaddq_s16(vmull_s8(vget_low_s8(q6bytes.val[1]), vget_low_s8(q8bytes.val[1])),
                             vmull_s8(vget_high_s8(q6bytes.val[1]), vget_high_s8(q8bytes.val[1])));
    isum += vaddvq_s16(p0) * scale[0] + vaddvq_s16(p1) * scale[1];

    int16x8_t p2 = vaddq_s16(vmull_s8(vget_low_s8(q6bytes.val[2]), vget_low_s8(q8bytes.val[2])),
                             vmull_s8(vget_high_s8(q6bytes.val[2]), vget_high_s8(q8bytes.val[2])));
    int16x8_t p3 = vaddq_s16(vmull_s8(vget_low_s8(q6bytes.val[3]), vget_low_s8(q8bytes.val[3])),
                             vmull_s8(vget_high_s8(q6bytes.val[3]), vget_high_s8(q8bytes.val[3])));
    isum += vaddvq_s16(p2) * scale[2] + vaddvq_s16(p3) * scale[3];
#endif

    sum += isum * d_all * y[i].d;
  }
  *s = sum;

#elif defined __AVX2__

  const __m256i m4 = _mm256_set1_epi8(0xF);
  const __m256i m2 = _mm256_set1_epi8(3);
  const __m256i m32s = _mm256_set1_epi8(32);

  __m256 acc = _mm256_setzero_ps();

  for (int i = 0; i < nb; ++i) {
    const float d = y[i].d * MLLM_FP16_TO_FP32(x[i].d);

    const uint8_t* __restrict q4 = x[i].ql;
    const uint8_t* __restrict qh = x[i].qh;
    const int8_t* __restrict q8 = y[i].qs;

    const __m64 scales_1 = _mm_set1_pi8(x[i].scales[0]);
    const __m64 scales_2 = _mm_set1_pi8(x[i].scales[1]);
    const __m64 scales_3 = _mm_set1_pi8(x[i].scales[2]);
    const __m64 scales_4 = _mm_set1_pi8(x[i].scales[3]);

    __m256i sumi = _mm256_setzero_si256();

    const __m128i scale_0 = _mm_set_epi64(scales_2, scales_1);
    const __m128i scale_1 = _mm_set_epi64(scales_4, scales_3);

    const __m256i q4bits1 = _mm256_loadu_si256((const __m256i*)q4);
    const __m128i q4bitsH = _mm_loadu_si128((const __m128i*)qh);

    const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(q4bitsH, 2), q4bitsH), m2), 4);
    const __m256i q4h_1 =
        _mm256_slli_epi16(_mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(q4bitsH, 6), _mm_srli_epi16(q4bitsH, 4)), m2), 4);

    const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q4h_0);
    const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4), q4h_1);

    const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)(q8 + 0));
    const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)(q8 + 32));

    __m256i q8s_0 = _mm256_maddubs_epi16(m32s, q8_0);
    __m256i q8s_1 = _mm256_maddubs_epi16(m32s, q8_1);

    __m256i p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
    __m256i p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);

    p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
    p16_1 = _mm256_sub_epi16(p16_1, q8s_1);

    p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
    p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);

    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));

    acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
  }

  *s = hsum_float_8(acc);

#elif defined __AVX__

  const __m128i m4 = _mm_set1_epi8(0xF);
  const __m128i m2 = _mm_set1_epi8(3);
  const __m128i m32s = _mm_set1_epi8(32);

  __m256 acc = _mm256_setzero_ps();

  for (int i = 0; i < nb; ++i) {
    const float d = y[i].d * MLLM_FP16_TO_FP32(x[i].d);

    const uint8_t* __restrict q4 = x[i].ql;
    const uint8_t* __restrict qh = x[i].qh;
    const int8_t* __restrict q8 = y[i].qs;

    const __m64 scales_1 = _mm_set1_pi8(x[i].scales[0]);
    const __m64 scales_2 = _mm_set1_pi8(x[i].scales[1]);
    const __m64 scales_3 = _mm_set1_pi8(x[i].scales[2]);
    const __m64 scales_4 = _mm_set1_pi8(x[i].scales[3]);

    __m128i sumi_0 = _mm_setzero_si128();
    __m128i sumi_1 = _mm_setzero_si128();

    const __m128i scale_0 = _mm_set_epi64(scales_2, scales_1);
    const __m128i scale_1 = _mm_set_epi64(scales_4, scales_3);

    const __m256i q4bits1 = _mm256_loadu_si256((const __m256i*)q4);
    const __m128i q4bitsH = _mm_loadu_si128((const __m128i*)qh);

    const __m128i q4h_0 = _mm_slli_epi16(_mm_and_si128(q4bitsH, m2), 4);
    const __m128i q4h_1 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bitsH, 2), m2), 4);
    const __m128i q4h_2 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bitsH, 4), m2), 4);
    const __m128i q4h_3 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bitsH, 6), m2), 4);

    const __m128i q4_0 = _mm_or_si128(_mm_and_si128(_mm256_extractf128_si256(q4bits1, 0), m4), q4h_0);
    const __m128i q4_1 = _mm_or_si128(_mm_and_si128(_mm256_extractf128_si256(q4bits1, 1), m4), q4h_1);
    const __m128i q4_2 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(_mm256_extractf128_si256(q4bits1, 0), 4), m4), q4h_2);
    const __m128i q4_3 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(_mm256_extractf128_si256(q4bits1, 1), 4), m4), q4h_3);

    const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)(q8 + 0));
    const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)(q8 + 32));

    __m128i q8s_0 = _mm_maddubs_epi16(m32s, _mm256_extractf128_si256(q8_0, 0));
    __m128i q8s_1 = _mm_maddubs_epi16(m32s, _mm256_extractf128_si256(q8_0, 1));
    __m128i q8s_2 = _mm_maddubs_epi16(m32s, _mm256_extractf128_si256(q8_1, 0));
    __m128i q8s_3 = _mm_maddubs_epi16(m32s, _mm256_extractf128_si256(q8_1, 1));

    __m128i p16_0 = _mm_maddubs_epi16(q4_0, _mm256_extractf128_si256(q8_0, 0));
    __m128i p16_1 = _mm_maddubs_epi16(q4_1, _mm256_extractf128_si256(q8_0, 1));
    __m128i p16_2 = _mm_maddubs_epi16(q4_2, _mm256_extractf128_si256(q8_1, 0));
    __m128i p16_3 = _mm_maddubs_epi16(q4_3, _mm256_extractf128_si256(q8_1, 1));

    p16_0 = _mm_sub_epi16(p16_0, q8s_0);
    p16_1 = _mm_sub_epi16(p16_1, q8s_1);
    p16_2 = _mm_sub_epi16(p16_2, q8s_2);
    p16_3 = _mm_sub_epi16(p16_3, q8s_3);

    p16_0 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_0), p16_0);
    p16_1 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_unpackhi_epi64(scale_0, scale_0)), p16_1);
    p16_2 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_1), p16_2);
    p16_3 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_unpackhi_epi64(scale_1, scale_1)), p16_3);

    sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p16_0, p16_2));
    sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p16_1, p16_3));

    acc = _mm256_add_ps(_mm256_mul_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(MM256_SET_M128I(sumi_1, sumi_0))), acc);
  }

  *s = hsum_float_8(acc);

#else

  int8_t aux8[QK_K];
  int16_t aux16[8];
  float sums[8];
  int32_t aux32[8];
  memset(sums, 0, 8 * sizeof(float));

  float sumf = 0;
  for (int i = 0; i < nb; ++i) {
    const uint8_t* __restrict q4 = x[i].ql;
    const uint8_t* __restrict qh = x[i].qh;
    const int8_t* __restrict q8 = y[i].qs;
    memset(aux32, 0, 8 * sizeof(int32_t));
    int8_t* __restrict a = aux8;
    for (int l = 0; l < 16; ++l) {
      a[l + 0] = (int8_t)((q4[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
      a[l + 16] = (int8_t)((q4[l + 16] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
      a[l + 32] = (int8_t)((q4[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
      a[l + 48] = (int8_t)((q4[l + 16] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
    }
    int is = 0;
    for (int j = 0; j < QK_K / 16; ++j) {
      int scale = x[i].scales[is++];
      for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
      for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
      q8 += 8;
      a += 8;
      for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
      for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
      q8 += 8;
      a += 8;
    }
    const float d = MLLM_FP16_TO_FP32(x[i].d) * y[i].d;
    for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
  }
  for (int l = 0; l < 8; ++l) sumf += sums[l];
  *s = sumf;
#endif
}
#endif

void vec_dot_q8_0_q8_0(int n, float* __restrict s, const void* __restrict vx, const void* __restrict vy, size_t bs, size_t bx,
                       size_t by) {
  const int qk = QK8_0;
  const int nb = n / qk;  // number of blocks

  assert(n % qk == 0);

  const auto* __restrict x = static_cast<const block_q8_0*>(vx);
  const auto* __restrict y = static_cast<const block_q8_0*>(vy);

#if defined(__ARM_FEATURE_MATMUL_INT8)
  // if (nrc == 2)
  {
    const block_q8_0* __restrict vx0 = (const block_q8_0*)vx;
    const block_q8_0* __restrict vx1 = (const block_q8_0*)((const uint8_t*)vx + bx);
    const block_q8_0* __restrict vy0 = (const block_q8_0*)vy;
    const block_q8_0* __restrict vy1 = (const block_q8_0*)((const uint8_t*)vy + by);

    float32x4_t sumv0 = vdupq_n_f32(0.0f);

    for (int i = 0; i < nb; i++) {
      const block_q8_0* __restrict b_x0 = &vx0[i];
      const block_q8_0* __restrict b_y0 = &vy0[i];

      const block_q8_0* __restrict b_x1 = &vx1[i];
      const block_q8_0* __restrict b_y1 = &vy1[i];

      const int8x16_t x0_l = vld1q_s8(b_x0->qs);
      const int8x16_t x0_h = vld1q_s8(b_x0->qs + 16);
      const int8x16_t x1_l = vld1q_s8(b_x1->qs);
      const int8x16_t x1_h = vld1q_s8(b_x1->qs + 16);

      // load y
      const int8x16_t y0_l = vld1q_s8(b_y0->qs);
      const int8x16_t y0_h = vld1q_s8(b_y0->qs + 16);
      const int8x16_t y1_l = vld1q_s8(b_y1->qs);
      const int8x16_t y1_h = vld1q_s8(b_y1->qs + 16);

      float32_t _scale[4] = {
          MLLM_FP16_TO_FP32(b_x0->d) * MLLM_FP16_TO_FP32(b_y0->d), MLLM_FP16_TO_FP32(b_x0->d) * MLLM_FP16_TO_FP32(b_y1->d),
          MLLM_FP16_TO_FP32(b_x1->d) * MLLM_FP16_TO_FP32(b_y0->d), MLLM_FP16_TO_FP32(b_x1->d) * MLLM_FP16_TO_FP32(b_y1->d)};
      float32x4_t scale = vld1q_f32(_scale);

      int8x16_t l0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));
      int8x16_t l1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));

      int8x16_t l2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));
      int8x16_t l3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));

      int8x16_t r0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));
      int8x16_t r1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));

      int8x16_t r2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));
      int8x16_t r3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));

      sumv0 = vmlaq_f32(
          sumv0,
          (vcvtq_f32_s32(vmmlaq_s32((vmmlaq_s32((vmmlaq_s32((vmmlaq_s32(vdupq_n_s32(0), l0, r0)), l1, r1)), l2, r2)), l3, r3))),
          scale);
    }

    float32x4_t sumv1 = vextq_f32(sumv0, sumv0, 2);
    float32x4_t sumv2 = vzip1q_f32(sumv0, sumv1);

    vst1_f32(s, vget_low_f32(sumv2));
    vst1_f32(s + bs, vget_high_f32(sumv2));

    return;
  }
#elif defined(__ARM_NEON)
  float32x4_t sumv0 = vdupq_n_f32(0.0f);
  float32x4_t sumv1 = vdupq_n_f32(0.0f);

  assert(nb % 2 == 0);  // TODO: handle odd nb

  for (int i = 0; i < nb; i += 2) {
    const block_q8_0* x0 = &x[i + 0];
    const block_q8_0* x1 = &x[i + 1];
    const block_q8_0* y0 = &y[i + 0];
    const block_q8_0* y1 = &y[i + 1];

    const int8x16_t x0_0 = vld1q_s8(x0->qs);
    const int8x16_t x0_1 = vld1q_s8(x0->qs + 16);
    const int8x16_t x1_0 = vld1q_s8(x1->qs);
    const int8x16_t x1_1 = vld1q_s8(x1->qs + 16);

    // load y
    const int8x16_t y0_0 = vld1q_s8(y0->qs);
    const int8x16_t y0_1 = vld1q_s8(y0->qs + 16);
    const int8x16_t y1_0 = vld1q_s8(y1->qs);
    const int8x16_t y1_1 = vld1q_s8(y1->qs + 16);

    sumv0 = vmlaq_n_f32(
        sumv0, vcvtq_f32_s32(vaddq_s32(mllm_vdotq_s32(vdupq_n_s32(0), x0_0, y0_0), mllm_vdotq_s32(vdupq_n_s32(0), x0_1, y0_1))),
        MLLM_FP16_TO_FP32(x0->d) * MLLM_FP16_TO_FP32(y0->d));

    sumv1 = vmlaq_n_f32(
        sumv1, vcvtq_f32_s32(vaddq_s32(mllm_vdotq_s32(vdupq_n_s32(0), x1_0, y1_0), mllm_vdotq_s32(vdupq_n_s32(0), x1_1, y1_1))),
        MLLM_FP16_TO_FP32(x1->d) * MLLM_FP16_TO_FP32(y1->d));
  }

  *s = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
#elif defined(__AVX2__) || defined(__AVX__)
  // Initialize accumulator with zeros
  __m256 acc = _mm256_setzero_ps();

  // Main loop
  for (int i = 0; i < nb; ++i) {
    // Compute combined scale for the block
    const __m256 d = _mm256_set1_ps(MLLM_FP16_TO_FP32(x[i].d) * MLLM_FP16_TO_FP32(y[i].d));
    __m256i bx = _mm256_loadu_si256((const __m256i*)x[i].qs);
    __m256i by = _mm256_loadu_si256((const __m256i*)y[i].qs);

    const __m256 q = mul_sum_i8_pairs_float(bx, by);

    // Multiply q with scale and accumulate
#if defined(__AVX2__)
    acc = _mm256_fmadd_ps(d, q, acc);
#else
    acc = _mm256_add_ps(_mm256_mul_ps(d, q), acc);
#endif
  }

  *s = hsum_float_8(acc);
#endif
}

void vec_dot_i8_i8(const int n, float* __restrict s, const void* __restrict vx, const void* __restrict vy, float scale1,
                   float scale2) {
  const int qk = QK8_0;
  const int nb = n / qk;

  const float scale = scale1 * scale2;

  assert(n % qk == 0);

  const block_q8_per_tensor* __restrict x = (block_q8_per_tensor*)vx;
  const block_q8_per_tensor* __restrict y = (block_q8_per_tensor*)vy;

#if defined(__ARM_NEON)
  float32x4_t sumv0 = vdupq_n_f32(0.0f);
  float32x4_t sumv1 = vdupq_n_f32(0.0f);

  assert(nb % 2 == 0);  // TODO: handle odd nb

  for (int i = 0; i < nb; i += 2) {
    const block_q8_per_tensor* __restrict x0 = &x[i + 0];
    const block_q8_per_tensor* __restrict x1 = &x[i + 1];
    const block_q8_per_tensor* __restrict y0 = &y[i + 0];
    const block_q8_per_tensor* __restrict y1 = &y[i + 1];

    const int8x16_t x0_0 = vld1q_s8(x0->qs);
    const int8x16_t x0_1 = vld1q_s8(x0->qs + 16);
    const int8x16_t x1_0 = vld1q_s8(x1->qs);
    const int8x16_t x1_1 = vld1q_s8(x1->qs + 16);

    // load y
    const int8x16_t y0_0 = vld1q_s8(y0->qs);
    const int8x16_t y0_1 = vld1q_s8(y0->qs + 16);
    const int8x16_t y1_0 = vld1q_s8(y1->qs);
    const int8x16_t y1_1 = vld1q_s8(y1->qs + 16);

    sumv0 = vmlaq_n_f32(
        sumv0, vcvtq_f32_s32(vaddq_s32(mllm_vdotq_s32(vdupq_n_s32(0), x0_0, y0_0), mllm_vdotq_s32(vdupq_n_s32(0), x0_1, y0_1))),
        scale);

    sumv1 = vmlaq_n_f32(
        sumv1, vcvtq_f32_s32(vaddq_s32(mllm_vdotq_s32(vdupq_n_s32(0), x1_0, y1_0), mllm_vdotq_s32(vdupq_n_s32(0), x1_1, y1_1))),
        scale);
  }

  *s = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
#elif defined(__AVX2__) || defined(__AVX__)
  // Initialize accumulator with zeros
  __m256 acc = _mm256_setzero_ps();

  // Main loop
  for (int i = 0; i < nb; ++i) {
    // Compute combined scale for the block
    const __m256 d = _mm256_set1_ps(scale);
    __m256i qx = _mm256_loadu_si256((const __m256i*)x[i].qs);
    __m256i qy = _mm256_loadu_si256((const __m256i*)y[i].qs);

    const __m256 q = mul_sum_i8_pairs_float(qx, qy);

    // Multiply q with scale and accumulate
#if defined(__AVX2__)
    acc = _mm256_fmadd_ps(d, q, acc);
#else
    acc = _mm256_add_ps(_mm256_mul_ps(d, q), acc);
#endif
  }

  *s = hsum_float_8(acc);
#else
  // scalar
  float sumf = 0.0;

  for (int i = 0; i < nb; i++) {
    int sumi = 0;

    for (int j = 0; j < qk; j++) { sumi += x[i].qs[j] * y[i].qs[j]; }

    sumf += sumi * scale;
  }

  *s = sumf;
#endif
}

#ifdef __AVX2__
static void vec_value_dot_fp32_avx2(const int n, float* __restrict s, const float* __restrict x, const float* __restrict y,
                                    bool addition) {
  float sumf = 0.0F;
  const int np = (n & ~(MLLM_F32_STEP - 1));

  MLLM_F32_VEC sum[MLLM_F32_ARR] = {MLLM_F32_VEC_ZERO};

  MLLM_F32_VEC ax[MLLM_F32_ARR];
  MLLM_F32_VEC ay[MLLM_F32_ARR];

  for (int i = 0; i < np; i += MLLM_F32_STEP) {
    for (int j = 0; j < MLLM_F32_ARR; j++) {
      ax[j] = MLLM_F32_VEC_LOAD(x + i + j * MLLM_F32_EPR);
      ay[j] = MLLM_F32_VEC_LOAD(y + i + j * MLLM_F32_EPR);

      sum[j] = MLLM_F32_VEC_FMA(sum[j], ax[j], ay[j]);
    }
  }

  // reduce sum0..sum3 to sum0
  MLLM_F32_VEC_REDUCE(sumf, sum);

  // leftovers
  for (int i = np; i < n; ++i) { sumf += x[i] * y[i]; }

  *s = sumf;
}
#endif

#ifdef __ARM_NEON
// s:vector k
// x:value
// y:vector k
static void vec_value_dot_fp32_arm(const int n, float* __restrict s, const float x, const float* __restrict y, bool addition) {
  int i;
  float32x4_t vec_x;
  float32x4_t vec_y;
  float32x4_t vec_s;

  vec_x = vdupq_n_f32(x);

  int n_aligned = n & -4;

  if (addition) {
    for (i = 0; i < n_aligned; i += 4) {
      vec_y = vld1q_f32(y + i);
      vec_s = vmulq_f32(vec_x, vec_y);
      vec_s = vaddq_f32(vec_s, vld1q_f32(s + i));
      vst1q_f32(s + i, vec_s);
    }
  } else {
    for (i = 0; i < n_aligned; i += 4) {
      vec_y = vld1q_f32(y + i);
      vec_s = vmulq_f32(vec_x, vec_y);
      vst1q_f32(s + i, vec_s);
    }
  }
  for (; i < n; ++i) {
    if (addition) {
      s[i] += x * y[i];
    } else {
      s[i] = x * y[i];
    }
  }
}
#endif

#ifdef __AVX2__
void vec_value_dot_fp32(const int n, float* __restrict s, const float* x, const float* __restrict vy, bool addition) {
  vec_value_dot_fp32_avx2(n, s, x, vy, addition);
}
#elif defined(__ARM_NEON)
void vec_value_dot_fp32(const int n, float* __restrict s, const float x, const float* __restrict vy, bool addition) {
  vec_value_dot_fp32_arm(n, s, x, vy, addition);
}
#endif

void vec_dot_q2_K_q8_K(int n, float* __restrict s, const void* __restrict vx, const void* __restrict vy) {
  const block_q2_K* __restrict x = (block_q2_K*)vx;
  const block_q8_K* __restrict y = (block_q8_K*)vy;

  const int nb = n / QK_K;

#ifdef __ARM_FEATURE_SVE
  const int vector_length = svcntb() * 8;
  const svuint8_t m3s = svdup_n_u8(0x3);
  const svuint32_t m4s = svdup_n_u32(0xF);
  const svint32_t vzero_sv = svdup_n_s32(0);
  svfloat32_t acc_sum = svdup_n_f32(0);
  svbool_t pred_s32 = svptrue_pat_b32(SV_VL4);

  switch (vector_length) {
    case 128:
      for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * MLLM_FP16_TO_FP32(x[i].d);
        svfloat32_t d_broad = svdup_n_f32((float32_t)d);
        const float dmin = -y[i].d * MLLM_FP16_TO_FP32(x[i].dmin);
        svfloat32_t dmin_broad = svdup_n_f32((float32_t)dmin);

        const uint8_t* __restrict q2 = x[i].qs;
        const int8_t* __restrict q8_sv = y[i].qs;
        const uint8_t* __restrict sc = x[i].scales;

        svuint32_t mins_and_scales_sve = svld1ub_u32(svptrue_b32(), sc);
        const svint32_t mins_sv_1 = svreinterpret_s32_u32(svlsr_n_u32_x(svptrue_b32(), mins_and_scales_sve, 4));

        mins_and_scales_sve = svld1ub_u32(svptrue_b32(), sc + 4);
        const svint32_t mins_sv_2 = svreinterpret_s32_u32(svlsr_n_u32_x(svptrue_b32(), mins_and_scales_sve, 4));

        svint32_t q8sums_sv_1 = svld1sh_s32(svptrue_b32(), y[i].bsums);
        svint32_t q8sums_sv_2 = svld1sh_s32(svptrue_b32(), y[i].bsums + 4);

        const svint32_t s0 = svadd_s32_x(svptrue_b32(), svmul_s32_x(svptrue_b32(), mins_sv_1, q8sums_sv_1),
                                         svmul_s32_x(svptrue_b32(), mins_sv_2, q8sums_sv_2));

        mins_and_scales_sve = svld1ub_u32(svptrue_b32(), sc + 8);
        const svint32_t mins_sv_3 = svreinterpret_s32_u32(svlsr_n_u32_x(svptrue_b32(), mins_and_scales_sve, 4));

        mins_and_scales_sve = svld1ub_u32(svptrue_b32(), sc + 12);
        const svint32_t mins_sv_4 = svreinterpret_s32_u32(svlsr_n_u32_x(svptrue_b32(), mins_and_scales_sve, 4));

        q8sums_sv_1 = svld1sh_s32(svptrue_b32(), y[i].bsums + 8);
        q8sums_sv_2 = svld1sh_s32(svptrue_b32(), y[i].bsums + 12);

        svint32_t s1 = svadd_s32_x(svptrue_b32(), svmul_s32_x(svptrue_b32(), mins_sv_3, q8sums_sv_1),
                                   svmul_s32_x(svptrue_b32(), mins_sv_4, q8sums_sv_2));

        svfloat32_t temp = svcvt_f32_s32_x(svptrue_b32(), svadd_s32_x(svptrue_b32(), s0, s1));

        acc_sum = svmla_f32_m(svptrue_b32(), acc_sum, temp, dmin_broad);

        svint32_t sumi1 = svdup_n_s32(0);

        {
          const svuint8_t q2bits_1 = svld1_u8(svptrue_b8(), q2);
          svint8_t q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), q2bits_1, m3s));
          svint8_t q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv);
          q8_sv += 16;
          const svint32_t scales_sv = svreinterpret_s32_u32(svand_u32_m(svptrue_b32(), svld1ub_u32(svptrue_b32(), sc), m4s));

          sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv, 0));

          const svuint8_t q2bits_3 = svld1_u8(svptrue_b8(), q2 + 16);
          q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), q2bits_3, m3s));
          q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv);
          q8_sv += 16;

          sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv, 1));

          q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_1, 2), m3s));
          q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv);
          q8_sv += 16;

          sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv, 2));

          q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_3, 2), m3s));
          q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv);
          q8_sv += 16;

          sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv, 3));

          const svint32_t scales_sv_1 =
              svreinterpret_s32_u32(svand_u32_m(svptrue_b32(), svld1ub_u32(svptrue_b32(), sc + 4), m4s));

          q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_1, 4), m3s));
          q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv);
          q8_sv += 16;

          sumi1 =
              svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_1, 0));

          q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_3, 4), m3s));
          q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv);
          q8_sv += 16;

          sumi1 =
              svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_1, 1));

          q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_1, 6), m3s));
          q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv);
          q8_sv += 16;

          sumi1 =
              svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_1, 2));

          q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_3, 6), m3s));
          q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv);
          q8_sv += 16;

          sumi1 =
              svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_1, 3));

          //-------------------------------

          q2 += 32;
          const svint32_t scales_sv_2 =
              svreinterpret_s32_u32(svand_u32_m(svptrue_b32(), svld1ub_u32(svptrue_b32(), sc + 8), m4s));
          const svuint8_t q2bits_2 = svld1_u8(svptrue_b8(), q2);

          q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), q2bits_2, m3s));
          q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv);
          q8_sv += 16;

          sumi1 =
              svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_2, 0));

          const svuint8_t q2bits_4 = svld1_u8(svptrue_b8(), q2 + 16);
          q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), q2bits_4, m3s));
          q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv);
          q8_sv += 16;

          sumi1 =
              svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_2, 1));

          q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_2, 2), m3s));
          q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv);
          q8_sv += 16;

          sumi1 =
              svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_2, 2));

          q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_4, 2), m3s));
          q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv);
          q8_sv += 16;

          sumi1 =
              svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_2, 3));

          const svint32_t scales_sv_3 =
              svreinterpret_s32_u32(svand_u32_m(svptrue_b32(), svld1ub_u32(svptrue_b32(), sc + 12), m4s));

          q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_2, 4), m3s));
          q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv);
          q8_sv += 16;

          sumi1 =
              svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_3, 0));

          q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_4, 4), m3s));
          q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv);
          q8_sv += 16;

          sumi1 =
              svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_3, 1));

          q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_2, 6), m3s));
          q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv);
          q8_sv += 16;

          sumi1 =
              svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_3, 2));

          q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_4, 6), m3s));
          q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv);
          q8_sv += 16;

          sumi1 =
              svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_3, 3));
        }
        acc_sum = svmla_f32_m(svptrue_b32(), acc_sum, svcvt_f32_s32_x(svptrue_b32(), sumi1), d_broad);
      }
      *s = svaddv_f32(svptrue_b32(), acc_sum);
      break;

    case 256:
    case 512:
      for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * MLLM_FP16_TO_FP32(x[i].d);
        svfloat32_t d_broad = svdup_n_f32((float32_t)d);
        const float dmin = -y[i].d * MLLM_FP16_TO_FP32(x[i].dmin);
        svfloat32_t dmin_broad = svdup_n_f32((float32_t)dmin);

        const uint8_t* __restrict q2 = x[i].qs;
        const int8_t* __restrict q8_sv = y[i].qs;
        const uint8_t* __restrict sc = x[i].scales;

        const svuint32_t mins_and_scales_sve = svld1ub_u32(svptrue_pat_b32(SV_VL8), sc);
        sc += 8;
        const svint32_t scales_sv = svreinterpret_s32_u32(svand_u32_m(svptrue_pat_b32(SV_VL8), mins_and_scales_sve, m4s));
        const svint32_t mins_sv_1 = svreinterpret_s32_u32(svlsr_n_u32_x(svptrue_pat_b32(SV_VL8), mins_and_scales_sve, 4));
        svint32_t q8sums_sv_1 = svld1sh_s32(svptrue_pat_b32(SV_VL8), y[i].bsums);

        const svuint32_t mins_and_scales_sve_1 = svld1ub_u32(svptrue_pat_b32(SV_VL8), sc);
        const svint32_t scales_sv_1 = svreinterpret_s32_u32(svand_u32_m(svptrue_pat_b32(SV_VL8), mins_and_scales_sve_1, m4s));
        const svint32_t mins_sv_2 = svreinterpret_s32_u32(svlsr_n_u32_x(svptrue_pat_b32(SV_VL8), mins_and_scales_sve_1, 4));

        svint32_t q8sums_sv_2 = svld1sh_s32(svptrue_pat_b32(SV_VL8), y[i].bsums + 8);

        svfloat32_t temp =
            svcvt_f32_s32_x(svptrue_pat_b32(SV_VL8),
                            svadd_s32_x(svptrue_pat_b32(SV_VL8), svmul_s32_x(svptrue_pat_b32(SV_VL8), mins_sv_1, q8sums_sv_1),
                                        svmul_s32_x(svptrue_pat_b32(SV_VL8), mins_sv_2, q8sums_sv_2)));

        acc_sum = svmla_f32_m(svptrue_pat_b32(SV_VL8), acc_sum, temp, dmin_broad);

        svint32_t sumi1 = svdup_n_s32(0);

        {
          const svuint8_t q2bits_1 = svld1_u8(svptrue_pat_b8(SV_VL32), q2);
          svint8_t q2bytes_sv = svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), q2bits_1, m3s));
          svint8_t q8bytes_sv = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv);
          q8_sv += 32;

          svint32_t scale_1 = svsel(pred_s32, svdup_lane_s32(scales_sv, 0), svdup_lane_s32(scales_sv, 1));
          sumi1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), scale_1);

          q2bytes_sv =
              svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), q2bits_1, 2), m3s));
          q8bytes_sv = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv);
          q8_sv += 32;

          svint32_t scale_2 = svsel(pred_s32, svdup_lane_s32(scales_sv, 2), svdup_lane_s32(scales_sv, 3));
          sumi1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1, svdot_s32(svdup_n_s32(0), q2bytes_sv, q8bytes_sv), scale_2);

          q2bytes_sv =
              svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), q2bits_1, 4), m3s));
          q8bytes_sv = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv);
          q8_sv += 32;

          scale_1 = svsel(pred_s32, svdup_lane_s32(scales_sv, 4), svdup_lane_s32(scales_sv, 5));
          sumi1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), scale_1);

          q2bytes_sv =
              svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), q2bits_1, 6), m3s));
          q8bytes_sv = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv);
          q8_sv += 32;

          scale_2 = svsel(pred_s32, svdup_lane_s32(scales_sv, 6), svdup_lane_s32(scales_sv, 7));
          sumi1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), scale_2);

          q2 += 32;

          const svuint8_t q2bits_2 = svld1_u8(svptrue_pat_b8(SV_VL32), q2);
          q2bytes_sv = svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), q2bits_2, m3s));
          q8bytes_sv = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv);
          q8_sv += 32;

          scale_1 = svsel(pred_s32, svdup_lane_s32(scales_sv_1, 0), svdup_lane_s32(scales_sv_1, 1));
          sumi1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), scale_1);

          q2bytes_sv =
              svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), q2bits_2, 2), m3s));
          q8bytes_sv = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv);
          q8_sv += 32;

          scale_2 = svsel(pred_s32, svdup_lane_s32(scales_sv_1, 2), svdup_lane_s32(scales_sv_1, 3));
          sumi1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), scale_2);

          q2bytes_sv =
              svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), q2bits_2, 4), m3s));
          q8bytes_sv = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv);
          q8_sv += 32;

          scale_1 = svsel(pred_s32, svdup_lane_s32(scales_sv_1, 4), svdup_lane_s32(scales_sv_1, 5));
          sumi1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), scale_1);

          q2bytes_sv =
              svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), q2bits_2, 6), m3s));
          q8bytes_sv = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv);
          q8_sv += 32;

          scale_2 = svsel(pred_s32, svdup_lane_s32(scales_sv_1, 6), svdup_lane_s32(scales_sv_1, 7));
          sumi1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), scale_2);
        }
        acc_sum = svmla_f32_m(svptrue_pat_b32(SV_VL8), acc_sum, svcvt_f32_s32_x(svptrue_pat_b32(SV_VL8), sumi1), d_broad);
      }
      *s = svaddv_f32(svptrue_pat_b32(SV_VL8), acc_sum);
      break;

    default: assert(false && "Unsupported vector length"); break;
  }

#elif __ARM_NEON
  const uint8x16_t m3 = vdupq_n_u8(0x3);
  const uint8x16_t m4 = vdupq_n_u8(0xF);

  const int32x4_t vzero = vdupq_n_s32(0);

  mllm_int8x16x2_t q2bytes;
  uint8_t aux[16];

  float sum = 0;

  for (int i = 0; i < nb; ++i) {
    const float d = y[i].d * MLLM_FP16_TO_FP32(x[i].d);
    const float dmin = -y[i].d * MLLM_FP16_TO_FP32(x[i].dmin);

    const uint8_t* __restrict q2 = x[i].qs;
    const int8_t* __restrict q8 = y[i].qs;
    const uint8_t* __restrict sc = x[i].scales;

    const uint8x16_t mins_and_scales = vld1q_u8(sc);
    const uint8x16_t scales = vandq_u8(mins_and_scales, m4);
    vst1q_u8(aux, scales);

    const uint8x16_t mins = vshrq_n_u8(mins_and_scales, 4);
    const mllm_int16x8x2_t q8sums = mllm_vld1q_s16_x2(y[i].bsums);
    const mllm_int16x8x2_t mins16 = {
        {vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mins))), vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(mins)))}};
    const int32x4_t s0 = vaddq_s32(vmull_s16(vget_low_s16(mins16.val[0]), vget_low_s16(q8sums.val[0])),
                                   vmull_s16(vget_high_s16(mins16.val[0]), vget_high_s16(q8sums.val[0])));
    const int32x4_t s1 = vaddq_s32(vmull_s16(vget_low_s16(mins16.val[1]), vget_low_s16(q8sums.val[1])),
                                   vmull_s16(vget_high_s16(mins16.val[1]), vget_high_s16(q8sums.val[1])));
    sum += dmin * vaddvq_s32(vaddq_s32(s0, s1));

    int isum = 0;
    int is = 0;

    // We use this macro instead of a function call because for some reason
    // the code runs 2-3% slower, even if the function is declared inline
#define MULTIPLY_ACCUM_WITH_SCALE(index)                                                         \
  isum += vaddvq_s32(mllm_vdotq_s32(vzero, q2bytes.val[0], q8bytes.val[0])) * aux[is + (index)]; \
  isum += vaddvq_s32(mllm_vdotq_s32(vzero, q2bytes.val[1], q8bytes.val[1])) * aux[is + 1 + (index)];

#define SHIFT_MULTIPLY_ACCUM_WITH_SCALE(shift, index)                                     \
  q8bytes = mllm_vld1q_s8_x2(q8);                                                         \
  q8 += 32;                                                                               \
  q2bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.val[0], (shift)), m3)); \
  q2bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.val[1], (shift)), m3)); \
  MULTIPLY_ACCUM_WITH_SCALE((index));

    for (int j = 0; j < QK_K / 128; ++j) {
      const mllm_uint8x16x2_t q2bits = mllm_vld1q_u8_x2(q2);
      q2 += 32;

      mllm_int8x16x2_t q8bytes = mllm_vld1q_s8_x2(q8);
      q8 += 32;
      q2bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(q2bits.val[0], m3));
      q2bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(q2bits.val[1], m3));

      MULTIPLY_ACCUM_WITH_SCALE(0);

      SHIFT_MULTIPLY_ACCUM_WITH_SCALE(2, 2);
      SHIFT_MULTIPLY_ACCUM_WITH_SCALE(4, 4);
      SHIFT_MULTIPLY_ACCUM_WITH_SCALE(6, 6);

      is += 8;
    }

    sum += d * isum;
  }

  *s = sum;

#elif defined __AVX2__

  const __m256i m3 = _mm256_set1_epi8(3);
  const __m128i m4 = _mm_set1_epi8(0xF);

  __m256 acc = _mm256_setzero_ps();

  for (int i = 0; i < nb; ++i) {
    const float d = y[i].d * MLLM_FP16_TO_FP32(x[i].d);
    const float dmin = -y[i].d * MLLM_FP16_TO_FP32(x[i].dmin);

    const uint8_t* __restrict q2 = x[i].qs;
    const int8_t* __restrict q8 = y[i].qs;

    const __m128i mins_and_scales = _mm_loadu_si128((const __m128i*)x[i].scales);
    const __m128i scales8 = _mm_and_si128(mins_and_scales, m4);
    const __m128i mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
    const __m256i mins = _mm256_cvtepi8_epi16(mins8);
    const __m256i prod = _mm256_madd_epi16(mins, _mm256_loadu_si256((const __m256i*)y[i].bsums));

    acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&dmin), _mm256_cvtepi32_ps(prod), acc);

    const __m256i all_scales = _mm256_cvtepi8_epi16(scales8);
    const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
    const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
    const __m256i scales[2] = {MM256_SET_M128I(l_scales, l_scales), MM256_SET_M128I(h_scales, h_scales)};

    __m256i sumi = _mm256_setzero_si256();

    for (int j = 0; j < QK_K / 128; ++j) {
      const __m256i q2bits = _mm256_loadu_si256((const __m256i*)q2);
      q2 += 32;

      const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8);
      q8 += 32;
      const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8);
      q8 += 32;
      const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8);
      q8 += 32;
      const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8);
      q8 += 32;

      const __m256i q2_0 = _mm256_and_si256(q2bits, m3);
      const __m256i q2_1 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 2), m3);
      const __m256i q2_2 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 4), m3);
      const __m256i q2_3 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 6), m3);

      __m256i p0 = _mm256_maddubs_epi16(q2_0, q8_0);
      __m256i p1 = _mm256_maddubs_epi16(q2_1, q8_1);
      __m256i p2 = _mm256_maddubs_epi16(q2_2, q8_2);
      __m256i p3 = _mm256_maddubs_epi16(q2_3, q8_3);

      p0 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(0)), p0);
      p1 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(1)), p1);
      p2 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(2)), p2);
      p3 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(3)), p3);

      p0 = _mm256_add_epi32(p0, p1);
      p2 = _mm256_add_epi32(p2, p3);

      sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p0, p2));
    }

    acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
  }

  *s = hsum_float_8(acc);

#elif defined __AVX__

  const __m128i m3 = _mm_set1_epi8(0x3);
  const __m128i m4 = _mm_set1_epi8(0xF);
  const __m128i m2 = _mm_set1_epi8(0x2);

  __m256 acc = _mm256_setzero_ps();

  for (int i = 0; i < nb; ++i) {
    const float dall = y[i].d * MLLM_FP16_TO_FP32(x[i].d);
    const float dmin = -y[i].d * MLLM_FP16_TO_FP32(x[i].dmin);

    const uint8_t* __restrict q2 = x[i].qs;
    const int8_t* __restrict q8 = y[i].qs;

    // load mins and scales from block_q2_K.scales[QK_K/16]
    const __m128i mins_and_scales = _mm_loadu_si128((const __m128i*)x[i].scales);
    const __m128i scales16 = _mm_and_si128(mins_and_scales, m4);
    const __m128i mins16 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
    const __m128i mins_0 = _mm_cvtepi8_epi16(mins16);
    const __m128i mins_1 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(mins16, mins16));

    // summs = y[i].bsums * (x[i].scales >> 4) in 16bits*8*2 to 32bits*4*2
    const __m128i summs_0 = _mm_madd_epi16(mins_0, _mm_loadu_si128((const __m128i*)&y[i].bsums[0]));
    const __m128i summs_1 = _mm_madd_epi16(mins_1, _mm_loadu_si128((const __m128i*)&y[i].bsums[8]));

    // sumf += -dmin * summs in 32bits*8
    acc = _mm256_add_ps(_mm256_mul_ps(_mm256_broadcast_ss(&dmin), _mm256_cvtepi32_ps(MM256_SET_M128I(summs_1, summs_0))), acc);

    const __m128i scales_0 = _mm_cvtepi8_epi16(scales16);
    const __m128i scales_1 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(scales16, scales16));
    const __m128i scales[2] = {scales_0, scales_1};

    __m128i sumi_0 = _mm_setzero_si128();
    __m128i sumi_1 = _mm_setzero_si128();

    for (int j = 0; j < QK_K / 128; ++j) {
      // load Q8 quants int8*16*8 from block_q8_K.qs[QK_K]
      const __m128i q8_0 = _mm_loadu_si128((const __m128i*)q8);
      q8 += 16;
      const __m128i q8_1 = _mm_loadu_si128((const __m128i*)q8);
      q8 += 16;
      const __m128i q8_2 = _mm_loadu_si128((const __m128i*)q8);
      q8 += 16;
      const __m128i q8_3 = _mm_loadu_si128((const __m128i*)q8);
      q8 += 16;
      const __m128i q8_4 = _mm_loadu_si128((const __m128i*)q8);
      q8 += 16;
      const __m128i q8_5 = _mm_loadu_si128((const __m128i*)q8);
      q8 += 16;
      const __m128i q8_6 = _mm_loadu_si128((const __m128i*)q8);
      q8 += 16;
      const __m128i q8_7 = _mm_loadu_si128((const __m128i*)q8);
      q8 += 16;

      // load 2bits*16*8 from block_q2_K.qs[QK_K/4]
      __m128i q2bits = _mm_loadu_si128((const __m128i*)q2);
      q2 += 16;
      const __m128i q2_0 = _mm_and_si128(q2bits, m3);
      const __m128i q2_2 = _mm_and_si128(_mm_srli_epi16(q2bits, 2), m3);
      const __m128i q2_4 = _mm_and_si128(_mm_srli_epi16(q2bits, 4), m3);
      const __m128i q2_6 = _mm_and_si128(_mm_srli_epi16(q2bits, 6), m3);
      q2bits = _mm_loadu_si128((const __m128i*)q2);
      q2 += 16;
      const __m128i q2_1 = _mm_and_si128(q2bits, m3);
      const __m128i q2_3 = _mm_and_si128(_mm_srli_epi16(q2bits, 2), m3);
      const __m128i q2_5 = _mm_and_si128(_mm_srli_epi16(q2bits, 4), m3);
      const __m128i q2_7 = _mm_and_si128(_mm_srli_epi16(q2bits, 6), m3);

      // isuml = q8[l] * ((q2[l] >> shift) & 3) in 8bits*16*8 to 16bits*8*8
      __m128i p0 = _mm_maddubs_epi16(q2_0, q8_0);
      __m128i p1 = _mm_maddubs_epi16(q2_1, q8_1);
      __m128i p2 = _mm_maddubs_epi16(q2_2, q8_2);
      __m128i p3 = _mm_maddubs_epi16(q2_3, q8_3);
      __m128i p4 = _mm_maddubs_epi16(q2_4, q8_4);
      __m128i p5 = _mm_maddubs_epi16(q2_5, q8_5);
      __m128i p6 = _mm_maddubs_epi16(q2_6, q8_6);
      __m128i p7 = _mm_maddubs_epi16(q2_7, q8_7);

      // isum += (x[i].scales[is++] & 0xF) * isuml in 16bits*8*8 to 32bits*4*8
      __m128i shuffle = _mm_set1_epi16(0x0100);
      p0 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p0);
      shuffle = _mm_add_epi16(shuffle, m2);
      p1 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p1);
      shuffle = _mm_add_epi16(shuffle, m2);
      p2 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p2);
      shuffle = _mm_add_epi16(shuffle, m2);
      p3 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p3);
      shuffle = _mm_add_epi16(shuffle, m2);
      p4 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p4);
      shuffle = _mm_add_epi16(shuffle, m2);
      p5 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p5);
      shuffle = _mm_add_epi16(shuffle, m2);
      p6 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p6);
      shuffle = _mm_add_epi16(shuffle, m2);
      p7 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p7);

      p0 = _mm_add_epi32(p0, p1);
      p2 = _mm_add_epi32(p2, p3);
      p4 = _mm_add_epi32(p4, p5);
      p6 = _mm_add_epi32(p6, p7);

      // isum in 32bits*4*2
      sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p0, p2));
      sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p4, p6));
    }

    // sumf += dall * isum - dmin * summs in 32bits
    __m256i sumi = MM256_SET_M128I(sumi_1, sumi_0);
    acc = _mm256_add_ps(_mm256_mul_ps(_mm256_broadcast_ss(&dall), _mm256_cvtepi32_ps(sumi)), acc);
  }

  *s = hsum_float_8(acc);

#elif defined __wasm_simd128__
  float sumf = 0;

  for (int i = 0; i < nb; ++i) {
    const uint8_t* q2 = x[i].qs;
    const int8_t* q8 = y[i].qs;
    const uint8_t* sc = x[i].scales;

    // Vectorized summs calculation
    v128_t summs_vec = wasm_i32x4_splat(0);
    {
      v128_t sc_vec = wasm_v128_load(sc);
      v128_t sc_upper = wasm_u8x16_shr(sc_vec, 4);

      v128_t sc_low = wasm_u16x8_extend_low_u8x16(sc_upper);
      v128_t sc_high = wasm_u16x8_extend_high_u8x16(sc_upper);

      v128_t bsums1 = wasm_v128_load(&y[i].bsums[0]);
      v128_t bsums2 = wasm_v128_load(&y[i].bsums[8]);

      summs_vec = wasm_i32x4_add(wasm_i32x4_add(wasm_i32x4_dot_i16x8(sc_low, bsums1), wasm_i32x4_dot_i16x8(sc_high, bsums2)),
                                 summs_vec);

      summs_vec = wasm_i32x4_add(summs_vec, wasm_i32x4_shuffle(summs_vec, summs_vec, 2, 3, 0, 1));
      summs_vec = wasm_i32x4_add(summs_vec, wasm_i32x4_shuffle(summs_vec, summs_vec, 1, 0, 3, 2));
    }
    int32_t summs = wasm_i32x4_extract_lane(summs_vec, 0);

    // Vectorized isum calculation
    int32_t isum = 0;
    const uint8_t* sc_ptr = sc;
    const int k_iters = QK_K / 128;

    for (int k = 0; k < k_iters; ++k) {
      v128_t isum_vec = wasm_i32x4_splat(0);
      int shift = 0;

      for (int j = 0; j < 4; ++j) {
        const int d0 = (sc_ptr[0] & 0xF);
        const int d1 = (sc_ptr[1] & 0xF);
        sc_ptr += 2;

        // Process first 16 elements
        v128_t q2_0 = wasm_v128_load(q2);
        v128_t q8_0 = wasm_v128_load(q8);
        v128_t q2_shift_0 = wasm_u8x16_shr(q2_0, shift);
        v128_t q2_bits_0 = wasm_v128_and(q2_shift_0, wasm_i8x16_splat(0x03));

        // Process next 16 elements
        v128_t q2_1 = wasm_v128_load(q2 + 16);
        v128_t q8_1 = wasm_v128_load(q8 + 16);
        v128_t q2_shift_1 = wasm_u8x16_shr(q2_1, shift);
        v128_t q2_bits_1 = wasm_v128_and(q2_shift_1, wasm_i8x16_splat(0x03));

        // Calculate dot products
        v128_t p0 = wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(q8_0), wasm_i16x8_extend_low_i8x16(q2_bits_0));
        v128_t p1 = wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(q8_0), wasm_i16x8_extend_high_i8x16(q2_bits_0));
        v128_t p2 = wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(q8_1), wasm_i16x8_extend_low_i8x16(q2_bits_1));
        v128_t p3 = wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(q8_1), wasm_i16x8_extend_high_i8x16(q2_bits_1));

        // Accumulate scaled results
        v128_t scaled = wasm_i32x4_add(wasm_i32x4_mul(wasm_i32x4_add(p0, p1), wasm_i32x4_splat(d0)),
                                       wasm_i32x4_mul(wasm_i32x4_add(p2, p3), wasm_i32x4_splat(d1)));

        isum_vec = wasm_i32x4_add(isum_vec, scaled);
        q8 += 32;
        shift += 2;
      }
      q2 += 32;

      // Horizontal sum of isum_vec
      isum_vec = wasm_i32x4_add(isum_vec, wasm_i32x4_shuffle(isum_vec, isum_vec, 2, 3, 0, 1));
      isum_vec = wasm_i32x4_add(isum_vec, wasm_i32x4_shuffle(isum_vec, isum_vec, 1, 0, 3, 2));
      isum += wasm_i32x4_extract_lane(isum_vec, 0);
    }

    const float dall = MLLM_FP16_TO_FP32(x[i].d) * y[i].d;
    const float dmin = MLLM_FP16_TO_FP32(x[i].dmin) * y[i].d;
    sumf += dall * isum - dmin * summs;
  }

  *s = sumf;

#elif defined __riscv_v_intrinsic

  const int vector_length = __riscv_vlenb() * 8;
  float sumf = 0;

  uint8_t temp_01[32] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  uint8_t atmp[16];

  switch (vector_length) {
    case 256:
      for (int i = 0; i < nb; ++i) {
        const uint8_t* q2 = x[i].qs;
        const int8_t* q8 = y[i].qs;
        const uint8_t* sc = x[i].scales;

        const float dall = y[i].d * MLLM_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * MLLM_FP16_TO_FP32(x[i].dmin);

        size_t vl = 16;

        vuint8m1_t scales = __riscv_vle8_v_u8m1(sc, vl);
        vuint8m1_t aux = __riscv_vand_vx_u8m1(scales, 0x0F, vl);

        vint16m1_t q8sums = __riscv_vle16_v_i16m1(y[i].bsums, vl);

        vuint8mf2_t scales_2 = __riscv_vle8_v_u8mf2(sc, vl);
        vuint8mf2_t mins8 = __riscv_vsrl_vx_u8mf2(scales_2, 0x4, vl);
        vint16m1_t mins = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2_u16m1(mins8, vl));
        vint32m2_t prod = __riscv_vwmul_vv_i32m2(q8sums, mins, vl);
        vint32m1_t vsums = __riscv_vredsum_vs_i32m2_i32m1(prod, __riscv_vmv_v_x_i32m1(0, 1), vl);

        sumf += dmin * __riscv_vmv_x_s_i32m1_i32(vsums);

        vl = 32;

        vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);
        vuint8m1_t v_b = __riscv_vle8_v_u8m1(temp_01, vl);

        uint8_t is = 0;
        int isum = 0;

        for (int j = 0; j < QK_K / 128; ++j) {
          // load Q2
          vuint8m1_t q2_x = __riscv_vle8_v_u8m1(q2, vl);

          vuint8m1_t q2_0 = __riscv_vand_vx_u8m1(q2_x, 0x03, vl);
          vuint8m1_t q2_1 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(q2_x, 0x2, vl), 0x03, vl);
          vuint8m1_t q2_2 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(q2_x, 0x4, vl), 0x03, vl);
          vuint8m1_t q2_3 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(q2_x, 0x6, vl), 0x03, vl);

          // duplicate scale elements for product
          vuint8m1_t sc0 = __riscv_vrgather_vv_u8m1(aux, __riscv_vadd_vx_u8m1(v_b, 0 + is, vl), vl);
          vuint8m1_t sc1 = __riscv_vrgather_vv_u8m1(aux, __riscv_vadd_vx_u8m1(v_b, 2 + is, vl), vl);
          vuint8m1_t sc2 = __riscv_vrgather_vv_u8m1(aux, __riscv_vadd_vx_u8m1(v_b, 4 + is, vl), vl);
          vuint8m1_t sc3 = __riscv_vrgather_vv_u8m1(aux, __riscv_vadd_vx_u8m1(v_b, 6 + is, vl), vl);

          vint16m2_t p0 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vwmulu_vv_u16m2(q2_0, sc0, vl));
          vint16m2_t p1 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vwmulu_vv_u16m2(q2_1, sc1, vl));
          vint16m2_t p2 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vwmulu_vv_u16m2(q2_2, sc2, vl));
          vint16m2_t p3 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vwmulu_vv_u16m2(q2_3, sc3, vl));

          // load Q8
          vint8m1_t q8_0 = __riscv_vle8_v_i8m1(q8, vl);
          vint8m1_t q8_1 = __riscv_vle8_v_i8m1(q8 + 32, vl);
          vint8m1_t q8_2 = __riscv_vle8_v_i8m1(q8 + 64, vl);
          vint8m1_t q8_3 = __riscv_vle8_v_i8m1(q8 + 96, vl);

          vint32m4_t s0 = __riscv_vwmul_vv_i32m4(p0, __riscv_vwcvt_x_x_v_i16m2(q8_0, vl), vl);
          vint32m4_t s1 = __riscv_vwmul_vv_i32m4(p1, __riscv_vwcvt_x_x_v_i16m2(q8_1, vl), vl);
          vint32m4_t s2 = __riscv_vwmul_vv_i32m4(p2, __riscv_vwcvt_x_x_v_i16m2(q8_2, vl), vl);
          vint32m4_t s3 = __riscv_vwmul_vv_i32m4(p3, __riscv_vwcvt_x_x_v_i16m2(q8_3, vl), vl);

          vint32m1_t isum0 = __riscv_vredsum_vs_i32m4_i32m1(__riscv_vadd_vv_i32m4(s0, s1, vl), vzero, vl);
          vint32m1_t isum1 = __riscv_vredsum_vs_i32m4_i32m1(__riscv_vadd_vv_i32m4(s2, s3, vl), isum0, vl);

          isum += __riscv_vmv_x_s_i32m1_i32(isum1);

          q2 += 32;
          q8 += 128;
          is = 8;
        }

        sumf += dall * isum;
      }
      break;
    case 128:
      for (int i = 0; i < nb; ++i) {
        const uint8_t* q2 = x[i].qs;
        const int8_t* q8 = y[i].qs;
        const uint8_t* sc = x[i].scales;
        const float dall = y[i].d * MLLM_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * MLLM_FP16_TO_FP32(x[i].dmin);
        uint8_t* patmp = atmp;
        int vsums;
        int tmp;
        __asm__ __volatile__("vsetivli zero, 16, e8, m1\n\t"
                             "vmv.v.x v8, zero\n\t"
                             "vle8.v v1, (%[sc])\n\t"
                             "vand.vi v0, v1, 0xF\n\t"
                             "vsrl.vi v1, v1, 4\n\t"
                             "vse8.v v0, (%[scale])\n\t"
                             "vsetivli zero, 16, e16, m2\n\t"
                             "vle16.v v2, (%[bsums])\n\t"
                             "vzext.vf2 v0, v1\n\t"
                             "vwmul.vv v4, v0, v2\n\t"
                             "vsetivli zero, 16, e32, m4\n\t"
                             "vredsum.vs v8, v4, v8\n\t"
                             "vmv.x.s %[vsums], v8"
                             : [tmp] "=&r"(tmp), [vsums] "=&r"(vsums)
                             : [sc] "r"(sc), [scale] "r"(atmp), [bsums] "r"(y[i].bsums)
                             : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13",
                               "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
                               "v28", "v29", "v30", "v31");
        sumf += dmin * vsums;
        int isum = 0;

        for (int j = 0; j < QK_K / 128; ++j) {
          __asm__ __volatile__(
              "vsetvli zero, %[vl32], e8, m2\n\t"
              "vle8.v v0, (%[q2])\n\t"
              "vsrl.vi v2, v0, 2\n\t"
              "vsrl.vi v4, v0, 4\n\t"
              "vsrl.vi v6, v0, 6\n\t"
              "vand.vi v0, v0, 0x3\n\t"
              "vand.vi v2, v2, 0x3\n\t"
              "vand.vi v4, v4, 0x3\n\t"
              "vsetvli zero, %[vl128], e8, m8\n\t"
              "vle8.v v8, (%[q8])\n\t"
              "vsetvli zero, %[vl64], e8, m4\n\t"
              "vwmul.vv v16, v0, v8\n\t"
              "vwmul.vv v24, v4, v12\n\t"
              "vsetivli zero, 16, e16, m2\n\t"
              "vmv.v.x v0, zero\n\t"
              "vwredsum.vs v10, v16, v0\n\t"
              "vwredsum.vs v9, v18, v0\n\t"
              "vwredsum.vs v8, v20, v0\n\t"
              "vwredsum.vs v7, v22, v0\n\t"
              "vwredsum.vs v11, v24, v0\n\t"
              "vwredsum.vs v12, v26, v0\n\t"
              "vwredsum.vs v13, v28, v0\n\t"
              "vwredsum.vs v14, v30, v0\n\t"
              "vsetivli zero, 4, e32, m1\n\t"
              "vslideup.vi v10, v9, 1\n\t"
              "vslideup.vi v8, v7, 1\n\t"
              "vslideup.vi v11, v12, 1\n\t"
              "vslideup.vi v13, v14, 1\n\t"
              "vslideup.vi v10, v8, 2\n\t"
              "vslideup.vi v11, v13, 2\n\t"
              "vsetivli zero, 8, e32, m2\n\t"
              "vle8.v v15, (%[scale])\n\t"
              "vzext.vf4 v12, v15\n\t"
              "vmul.vv v10, v10, v12\n\t"
              "vredsum.vs v0, v10, v0\n\t"
              "vmv.x.s %[tmp], v0\n\t"
              "add %[isum], %[isum], %[tmp]"
              : [tmp] "=&r"(tmp), [isum] "+&r"(isum)
              : [q2] "r"(q2), [scale] "r"(patmp), [q8] "r"(q8), [vl32] "r"(32), [vl64] "r"(64), [vl128] "r"(128)
              : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
          q2 += 32;
          q8 += 128;
          patmp += 8;
        }

        sumf += dall * isum;
      }
      break;
    default: assert(false && "Unsupported vector length"); break;
  }

  *s = sumf;

#elif defined(__POWER9_VECTOR__)
  const vector signed char lowMask = vec_splats((signed char)0x3);
  const vector signed char lowScaleMask = vec_splats((signed char)0xF);
  const vector int v0 = vec_splats((int32_t)0);
  const vector unsigned char v2 = vec_splats((unsigned char)0x2);
  const vector unsigned char v6 = vec_splats((unsigned char)0x6);
  const vector unsigned char v4 = vec_splats((unsigned char)0x4);

  vector float vsumf0 = vec_splats(0.0f);
  vector float vsumf1 = vec_splats(0.0f);
  vector float vsumf2 = vec_splats(0.0f);
  vector float vsumf3 = vec_splats(0.0f);

  for (int i = 0; i < nb; ++i) {
    vector float vxd = vec_splats(MLLM_FP16_TO_FP32(x[i].d));
    vector float vyd = vec_splats(y[i].d);
    vector float vd = vec_mul(vxd, vyd);

    vector float vxmin = vec_splats(MLLM_FP16_TO_FP32(x[i].dmin));
    vector float vdmin = vec_mul(vxmin, vyd);

    vector signed short q8ysums0 = vec_xl(0, y[i].bsums);
    vector signed short q8ysums1 = vec_xl(16, y[i].bsums);

    vector signed char q2xmins = (vector signed char)vec_xl(0, x[i].scales);
    vector signed char vscales = vec_and(q2xmins, lowScaleMask);

    q2xmins = vec_sr(q2xmins, v4);
    vector signed short q2xmins0 = vec_unpackh(q2xmins);
    vector signed short q2xmins1 = vec_unpackl(q2xmins);

    vector signed int prod0 = vec_mule(q2xmins0, q8ysums0);
    vector signed int prod1 = vec_mulo(q2xmins0, q8ysums0);
    vector signed int prod2 = vec_mule(q2xmins1, q8ysums1);
    vector signed int prod3 = vec_mulo(q2xmins1, q8ysums1);

    vsumf0 = vec_nmsub(vec_ctf(prod0, 0), vdmin, vsumf0);
    vsumf1 = vec_nmsub(vec_ctf(prod1, 0), vdmin, vsumf1);
    vsumf2 = vec_nmsub(vec_ctf(prod2, 0), vdmin, vsumf2);
    vsumf3 = vec_nmsub(vec_ctf(prod3, 0), vdmin, vsumf3);

    vector signed int vsumi0 = v0;
    vector signed int vsumi1 = v0;
    vector signed int vsumi2 = v0;
    vector signed int vsumi3 = v0;
    vector signed int vsumi4 = v0;
    vector signed int vsumi5 = v0;
    vector signed int vsumi6 = v0;
    vector signed int vsumi7 = v0;

    const uint8_t* __restrict q2 = x[i].qs;
    const int8_t* __restrict q8 = y[i].qs;

    for (int j = 0; j < QK_K / 128; ++j) {
      __builtin_prefetch(q2, 0, 1);
      __builtin_prefetch(q8, 0, 1);

      vector signed char qxs0 = (vector signed char)vec_xl(0, q2);
      vector signed char qxs1 = (vector signed char)vec_xl(16, q2);
      q2 += 32;

      vector unsigned char q2x00 = (vector unsigned char)vec_and(qxs0, lowMask);
      vector unsigned char q2x01 = (vector unsigned char)vec_and(vec_sr(qxs0, v2), lowMask);
      vector unsigned char q2x02 = (vector unsigned char)vec_and(vec_sr(qxs0, v4), lowMask);
      vector unsigned char q2x03 = (vector unsigned char)vec_and(vec_sr(qxs0, v6), lowMask);
      vector unsigned char q2x10 = (vector unsigned char)vec_and(qxs1, lowMask);
      vector unsigned char q2x11 = (vector unsigned char)vec_and(vec_sr(qxs1, v2), lowMask);
      vector unsigned char q2x12 = (vector unsigned char)vec_and(vec_sr(qxs1, v4), lowMask);
      vector unsigned char q2x13 = (vector unsigned char)vec_and(vec_sr(qxs1, v6), lowMask);

      vector signed char q8y00 = vec_xl(0, q8);
      vector signed char q8y10 = vec_xl(16, q8);
      vector signed char q8y01 = vec_xl(32, q8);
      vector signed char q8y11 = vec_xl(48, q8);
      vector signed char q8y02 = vec_xl(64, q8);
      vector signed char q8y12 = vec_xl(80, q8);
      vector signed char q8y03 = vec_xl(96, q8);
      vector signed char q8y13 = vec_xl(112, q8);
      q8 += 128;

      vector signed int qv0 = vec_msum(q8y00, q2x00, v0);
      vector signed int qv1 = vec_msum(q8y01, q2x01, v0);
      vector signed int qv2 = vec_msum(q8y02, q2x02, v0);
      vector signed int qv3 = vec_msum(q8y03, q2x03, v0);
      vector signed int qv4 = vec_msum(q8y10, q2x10, v0);
      vector signed int qv5 = vec_msum(q8y11, q2x11, v0);
      vector signed int qv6 = vec_msum(q8y12, q2x12, v0);
      vector signed int qv7 = vec_msum(q8y13, q2x13, v0);

      vector signed short vscales_07 = vec_unpackh(vscales);
      vector signed int vscales_03 = vec_unpackh(vscales_07);
      vector signed int vscales_47 = vec_unpackl(vscales_07);
      vector signed int vs0 = vec_splat(vscales_03, 0);
      vector signed int vs1 = vec_splat(vscales_03, 1);
      vector signed int vs2 = vec_splat(vscales_03, 2);
      vector signed int vs3 = vec_splat(vscales_03, 3);
      vector signed int vs4 = vec_splat(vscales_47, 0);
      vector signed int vs5 = vec_splat(vscales_47, 1);
      vector signed int vs6 = vec_splat(vscales_47, 2);
      vector signed int vs7 = vec_splat(vscales_47, 3);
      vscales = vec_sld(vscales, vscales, 8);

      vsumi0 = vec_add(vec_mul(qv0, vs0), vsumi0);
      vsumi1 = vec_add(vec_mul(qv1, vs2), vsumi1);
      vsumi2 = vec_add(vec_mul(qv2, vs4), vsumi2);
      vsumi3 = vec_add(vec_mul(qv3, vs6), vsumi3);
      vsumi4 = vec_add(vec_mul(qv4, vs1), vsumi4);
      vsumi5 = vec_add(vec_mul(qv5, vs3), vsumi5);
      vsumi6 = vec_add(vec_mul(qv6, vs5), vsumi6);
      vsumi7 = vec_add(vec_mul(qv7, vs7), vsumi7);
    }

    vsumi0 = vec_add(vsumi0, vsumi4);
    vsumi1 = vec_add(vsumi1, vsumi5);
    vsumi2 = vec_add(vsumi2, vsumi6);
    vsumi3 = vec_add(vsumi3, vsumi7);

    vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
    vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
    vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
  }

  vsumf0 = vec_add(vsumf0, vsumf2);
  vsumf1 = vec_add(vsumf1, vsumf3);

  vsumf0 = vec_add(vsumf0, vsumf1);

  vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
  vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

  *s = vec_extract(vsumf0, 0);

#elif defined __loongarch_asx

  __m256 acc = (__m256)__lasx_xvldi(0);

  for (int i = 0; i < nb; ++i) {
    const float d = y[i].d * MLLM_FP16_TO_FP32(x[i].d);
    const float dmin = -y[i].d * MLLM_FP16_TO_FP32(x[i].dmin);

    const uint8_t* __restrict q2 = x[i].qs;
    const int8_t* __restrict q8 = y[i].qs;

    const __m128i mins_and_scales128 = __lsx_vld((const __m128i*)x[i].scales, 0);
    const __m128i scales128 = __lsx_vandi_b(mins_and_scales128, 0xf);
    const __m256i mins = lasx_ext8_16(__lsx_vsrli_b(mins_and_scales128, 4));
    const __m256i prod = lasx_madd_h(mins, __lasx_xvld((const __m256i*)y[i].bsums, 0));

    acc = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(dmin), __lasx_xvffint_s_w(prod), acc);

    const v16i8 shuffle_mask = {0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};
    const __m256i scales_shuffled = lasx_ext8_16(__lsx_vshuf_b(scales128, scales128, (__m128i)shuffle_mask));

    __m256i sumi = __lasx_xvldi(0);

    for (int j = 0; j < QK_K / 128; ++j) {
      const __m256i q2bits = __lasx_xvld((const __m256i*)q2, 0);
      q2 += 32;

      const __m256i q8_0 = __lasx_xvld((const __m256i*)q8, 0);
      q8 += 32;
      const __m256i q8_1 = __lasx_xvld((const __m256i*)q8, 0);
      q8 += 32;
      const __m256i q8_2 = __lasx_xvld((const __m256i*)q8, 0);
      q8 += 32;
      const __m256i q8_3 = __lasx_xvld((const __m256i*)q8, 0);
      q8 += 32;

      const __m256i q2_0 = __lasx_xvandi_b(q2bits, 3);
      const __m256i q2_1 = __lasx_xvandi_b(__lasx_xvsrli_b(q2bits, 2), 3);
      const __m256i q2_2 = __lasx_xvandi_b(__lasx_xvsrli_b(q2bits, 4), 3);
      const __m256i q2_3 = __lasx_xvsrli_b(q2bits, 6);

      __m256i p0 = lasx_madd_h_b(q2_0, q8_0);
      __m256i p1 = lasx_madd_h_b(q2_1, q8_1);
      __m256i p2 = lasx_madd_h_b(q2_2, q8_2);
      __m256i p3 = lasx_madd_h_b(q2_3, q8_3);

      p0 = lasx_madd_h(lasx_xvrepl128vei_h(scales_shuffled, 4 * j + 0), p0);
      p1 = lasx_madd_h(lasx_xvrepl128vei_h(scales_shuffled, 4 * j + 1), p1);
      p2 = lasx_madd_h(lasx_xvrepl128vei_h(scales_shuffled, 4 * j + 2), p2);
      p3 = lasx_madd_h(lasx_xvrepl128vei_h(scales_shuffled, 4 * j + 3), p3);

      p0 = __lasx_xvadd_w(p0, p1);
      p2 = __lasx_xvadd_w(p2, p3);

      sumi = __lasx_xvadd_w(sumi, __lasx_xvadd_w(p0, p2));
    }

    acc = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(d), __lasx_xvffint_s_w(sumi), acc);
  }

  *s = hsum_float_8(acc);

#else

  float sumf = 0;

  for (int i = 0; i < nb; ++i) {
    const uint8_t* q2 = x[i].qs;
    const int8_t* q8 = y[i].qs;
    const uint8_t* sc = x[i].scales;

    int summs = 0;
    for (int j = 0; j < 16; ++j) { summs += y[i].bsums[j] * (sc[j] >> 4); }

    const float dall = y[i].d * MLLM_FP16_TO_FP32(x[i].d);
    const float dmin = y[i].d * MLLM_FP16_TO_FP32(x[i].dmin);

    int isum = 0;
    int is = 0;
    int d;
    for (int k = 0; k < QK_K / 128; ++k) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        d = sc[is++] & 0xF;
        int isuml = 0;
        for (int l = 0; l < 16; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
        isum += d * isuml;
        d = sc[is++] & 0xF;
        isuml = 0;
        for (int l = 16; l < 32; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
        isum += d * isuml;
        shift += 2;
        q8 += 32;
      }
      q2 += 32;
    }
    sumf += dall * isum - dmin * summs;
  }
  *s = sumf;
#endif
}

void vec_dot_q3_K_q8_K(int n, float* __restrict s, const void* __restrict vx, const void* __restrict vy) {
  assert(n % QK_K == 0);

  const uint32_t kmask1 = 0x03030303;
  const uint32_t kmask2 = 0x0f0f0f0f;

  const block_q3_K* __restrict x = (block_q3_K*)vx;
  const block_q8_K* __restrict y = (block_q8_K*)vy;

  const int nb = n / QK_K;

#if defined(__ARM_FEATURE_SVE)

  uint32_t aux[3];
  uint32_t utmp[4];

  const int8_t m32 = 32;
  const int vector_length = svcntb() * 8;
  const svuint8_t m3b_sv = svdup_n_u8(0x3);
  const svint32_t vzero_sv = svdup_n_s32(0);

  const svuint8_t m0_sv = svdup_n_u8(1);
  const svuint8_t m1_sv = svlsl_n_u8_x(svptrue_b8(), m0_sv, 1);
  const svuint8_t m2_sv = svlsl_n_u8_x(svptrue_b8(), m0_sv, 2);
  const svuint8_t m3_sv = svlsl_n_u8_x(svptrue_b8(), m0_sv, 3);

  float sum = 0;

  for (int i = 0; i < nb; ++i) {
    const float d = y[i].d * MLLM_FP16_TO_FP32(x[i].d);

    const uint8_t* __restrict q3_sv = x[i].qs;
    const uint8_t* __restrict qh_sv = x[i].hmask;
    const int8_t* __restrict q8_sv = y[i].qs;

    // Set up scales
    memcpy(aux, x[i].scales, 12);
    utmp[3] = ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4);
    utmp[2] = ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4);
    utmp[1] = (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4);
    utmp[0] = (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4);

    int8_t* scale = (int8_t*)utmp;

    for (int j = 0; j < 16; ++j) scale[j] -= m32;

    switch (vector_length) {
      case 128: {
        svuint8_t qhbits_sv_1 = svld1_u8(svptrue_b8(), qh_sv);
        svuint8_t qhbits_sv_2 = svld1_u8(svptrue_b8(), qh_sv + 16);
        svuint8_t q3h_sv;

        svint32_t sumi1_1 = svdup_n_s32(0);
        svint8_t q3bytes_sv;

        for (int j = 0; j < QK_K / 128; ++j) {
          const svuint8_t q3bits_sv = svld1_u8(svptrue_b8(), q3_sv);
          q3_sv += 16;
          const svuint8_t q3bits_sv_1 = svld1_u8(svptrue_b8(), q3_sv);
          q3_sv += 16;
          svint8_t q8bytes_1_sv_1 = svld1_s8(svptrue_b8(), q8_sv);
          q8_sv += 16;
          svint8_t q8bytes_1_sv_2 = svld1_s8(svptrue_b8(), q8_sv);
          q8_sv += 16;

          q3h_sv = svlsl_n_u8_x(svptrue_b8(), svbic_u8_x(svptrue_b8(), m0_sv, qhbits_sv_1), 2);
          q3bytes_sv = svsub_s8_x(svptrue_b8(), svreinterpret_s8_u8(svand_u8_m(svptrue_b8(), q3bits_sv, m3b_sv)),
                                  svreinterpret_s8_u8(q3h_sv));

          sumi1_1 = svmla_s32_m(svptrue_b32(), sumi1_1, svdot_s32(vzero_sv, q3bytes_sv, q8bytes_1_sv_1),
                                svdup_n_s32((int32_t)scale[0]));

          q3h_sv = svlsl_n_u8_x(svptrue_b8(), svbic_u8_x(svptrue_b8(), m0_sv, qhbits_sv_2), 2);
          q3bytes_sv = svsub_s8_x(svptrue_b8(), svreinterpret_s8_u8(svand_u8_m(svptrue_b8(), q3bits_sv_1, m3b_sv)),
                                  svreinterpret_s8_u8(q3h_sv));

          sumi1_1 = svmla_s32_m(svptrue_b32(), sumi1_1, svdot_s32(vzero_sv, q3bytes_sv, q8bytes_1_sv_2),
                                svdup_n_s32((int32_t)scale[1]));

          q8bytes_1_sv_1 = svld1_s8(svptrue_b8(), q8_sv);
          q8_sv += 16;
          q8bytes_1_sv_2 = svld1_s8(svptrue_b8(), q8_sv);
          q8_sv += 16;

          q3h_sv = svlsl_n_u8_x(svptrue_b8(), svbic_u8_x(svptrue_b8(), m1_sv, qhbits_sv_1), 1);
          q3bytes_sv = svsub_s8_x(
              svptrue_b8(), svreinterpret_s8_u8(svand_u8_m(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q3bits_sv, 2), m3b_sv)),
              svreinterpret_s8_u8(q3h_sv));

          sumi1_1 = svmla_s32_m(svptrue_b32(), sumi1_1, svdot_s32(vzero_sv, q3bytes_sv, q8bytes_1_sv_1),
                                svdup_n_s32((int32_t)scale[2]));

          q3h_sv = svlsl_n_u8_x(svptrue_b8(), svbic_u8_x(svptrue_b8(), m1_sv, qhbits_sv_2), 1);
          q3bytes_sv = svsub_s8_x(
              svptrue_b8(), svreinterpret_s8_u8(svand_u8_m(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q3bits_sv_1, 2), m3b_sv)),
              svreinterpret_s8_u8(q3h_sv));

          sumi1_1 = svmla_s32_m(svptrue_b32(), sumi1_1, svdot_s32(vzero_sv, q3bytes_sv, q8bytes_1_sv_2),
                                svdup_n_s32((int32_t)scale[3]));

          scale += 4;
          q8bytes_1_sv_1 = svld1_s8(svptrue_b8(), q8_sv);
          q8_sv += 16;
          q8bytes_1_sv_2 = svld1_s8(svptrue_b8(), q8_sv);
          q8_sv += 16;

          q3h_sv = svbic_u8_x(svptrue_b8(), m2_sv, qhbits_sv_1);
          q3bytes_sv = svsub_s8_x(
              svptrue_b8(), svreinterpret_s8_u8(svand_u8_m(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q3bits_sv, 4), m3b_sv)),
              svreinterpret_s8_u8(q3h_sv));

          sumi1_1 = svmla_s32_m(svptrue_b32(), sumi1_1, svdot_s32(vzero_sv, q3bytes_sv, q8bytes_1_sv_1),
                                svdup_n_s32((int32_t)scale[0]));

          q3h_sv = svbic_u8_x(svptrue_b8(), m2_sv, qhbits_sv_2);
          q3bytes_sv = svsub_s8_x(
              svptrue_b8(), svreinterpret_s8_u8(svand_u8_m(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q3bits_sv_1, 4), m3b_sv)),
              svreinterpret_s8_u8(q3h_sv));

          sumi1_1 = svmla_s32_m(svptrue_b32(), sumi1_1, svdot_s32(vzero_sv, q3bytes_sv, q8bytes_1_sv_2),
                                svdup_n_s32((int32_t)scale[1]));

          q8bytes_1_sv_1 = svld1_s8(svptrue_b8(), q8_sv);
          q8_sv += 16;
          q8bytes_1_sv_2 = svld1_s8(svptrue_b8(), q8_sv);
          q8_sv += 16;

          q3h_sv = svlsr_n_u8_x(svptrue_b8(), svbic_u8_x(svptrue_b8(), m3_sv, qhbits_sv_1), 1);
          q3bytes_sv = svsub_s8_x(
              svptrue_b8(), svreinterpret_s8_u8(svand_u8_m(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q3bits_sv, 6), m3b_sv)),
              svreinterpret_s8_u8(q3h_sv));

          sumi1_1 = svmla_s32_m(svptrue_b32(), sumi1_1, svdot_s32(vzero_sv, q3bytes_sv, q8bytes_1_sv_1),
                                svdup_n_s32((int32_t)scale[2]));

          q3h_sv = svlsr_n_u8_x(svptrue_b8(), svbic_u8_x(svptrue_b8(), m3_sv, qhbits_sv_2), 1);
          q3bytes_sv = svsub_s8_x(
              svptrue_b8(), svreinterpret_s8_u8(svand_u8_m(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q3bits_sv_1, 6), m3b_sv)),
              svreinterpret_s8_u8(q3h_sv));

          sumi1_1 = svmla_s32_m(svptrue_b32(), sumi1_1, svdot_s32(vzero_sv, q3bytes_sv, q8bytes_1_sv_2),
                                svdup_n_s32((int32_t)scale[3]));

          if (j == 0) {
            qhbits_sv_1 = svlsr_n_u8_x(svptrue_b8(), qhbits_sv_1, 4);
            qhbits_sv_2 = svlsr_n_u8_x(svptrue_b8(), qhbits_sv_2, 4);
          }

          scale += 4;
        }

        sum += d * (svaddv_s32(svptrue_b32(), sumi1_1));
      } break;
      case 256:
      case 512: {
        svuint8_t qhbits_sv = svld1_u8(svptrue_pat_b8(SV_VL32), qh_sv);
        svuint8_t q3h_sv;

        svint32_t sumi1_1 = svdup_n_s32(0);
        svint8_t q3bytes_sv;

        for (int j = 0; j < QK_K / 128; ++j) {
          const svuint8_t q3bits_sv = svld1_u8(svptrue_pat_b8(SV_VL32), q3_sv);
          q3_sv += 32;
          svint8_t q8bytes_1_sv_1 = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv);
          q8_sv += 32;
          svint8_t q8bytes_1_sv_2 = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv);
          q8_sv += 32;

          q3h_sv = svlsl_n_u8_x(svptrue_pat_b8(SV_VL32), svbic_u8_x(svptrue_pat_b8(SV_VL32), m0_sv, qhbits_sv), 2);
          q3bytes_sv =
              svsub_s8_x(svptrue_pat_b8(SV_VL32), svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), q3bits_sv, m3b_sv)),
                         svreinterpret_s8_u8(q3h_sv));

          svint32_t scale_1 =
              svsel_s32(svptrue_pat_b32(SV_VL4), svdup_n_s32((int32_t)scale[0]), svdup_n_s32((int32_t)scale[1]));
          sumi1_1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1_1, svdot_s32(vzero_sv, q3bytes_sv, q8bytes_1_sv_1), scale_1);

          q3h_sv = svlsl_n_u8_x(svptrue_pat_b8(SV_VL32), svbic_u8_x(svptrue_pat_b8(SV_VL32), m1_sv, qhbits_sv), 1);
          q3bytes_sv = svsub_s8_x(svptrue_pat_b8(SV_VL32),
                                  svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32),
                                                                 svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), q3bits_sv, 2), m3b_sv)),
                                  svreinterpret_s8_u8(q3h_sv));

          scale_1 = svsel_s32(svptrue_pat_b32(SV_VL4), svdup_n_s32((int32_t)scale[2]), svdup_n_s32((int32_t)scale[3]));
          sumi1_1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1_1, svdot_s32(vzero_sv, q3bytes_sv, q8bytes_1_sv_2), scale_1);

          scale += 4;
          q8bytes_1_sv_1 = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv);
          q8_sv += 32;
          q8bytes_1_sv_2 = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv);
          q8_sv += 32;

          q3h_sv = svbic_u8_x(svptrue_pat_b8(SV_VL32), m2_sv, qhbits_sv);
          q3bytes_sv = svsub_s8_x(svptrue_pat_b8(SV_VL32),
                                  svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32),
                                                                 svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), q3bits_sv, 4), m3b_sv)),
                                  svreinterpret_s8_u8(q3h_sv));

          scale_1 = svsel_s32(svptrue_pat_b32(SV_VL4), svdup_n_s32((int32_t)scale[0]), svdup_n_s32((int32_t)scale[1]));
          sumi1_1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1_1, svdot_s32(vzero_sv, q3bytes_sv, q8bytes_1_sv_1), scale_1);

          q3h_sv = svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), svbic_u8_x(svptrue_pat_b8(SV_VL32), m3_sv, qhbits_sv), 1);
          q3bytes_sv = svsub_s8_x(svptrue_pat_b8(SV_VL32),
                                  svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32),
                                                                 svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), q3bits_sv, 6), m3b_sv)),
                                  svreinterpret_s8_u8(q3h_sv));

          scale_1 = svsel_s32(svptrue_pat_b32(SV_VL4), svdup_n_s32((int32_t)scale[2]), svdup_n_s32((int32_t)scale[3]));
          sumi1_1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1_1, svdot_s32(vzero_sv, q3bytes_sv, q8bytes_1_sv_2), scale_1);

          if (j == 0) { qhbits_sv = svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), qhbits_sv, 4); }

          scale += 4;
        }

        sum += d * (svaddv_s32(svptrue_pat_b32(SV_VL8), sumi1_1));
      } break;
      default: assert(false && "Unsupported vector length"); break;
    }
  }
  *s = sum;

#elif __ARM_NEON

  uint32_t aux[3];
  uint32_t utmp[4];

  const uint8x16_t m3b = vdupq_n_u8(0x3);
  const int32x4_t vzero = vdupq_n_s32(0);

  const uint8x16_t m0 = vdupq_n_u8(1);
  const uint8x16_t m1 = vshlq_n_u8(m0, 1);
  const uint8x16_t m2 = vshlq_n_u8(m0, 2);
  const uint8x16_t m3 = vshlq_n_u8(m0, 3);
  const int8_t m32 = 32;

  mllm_int8x16x4_t q3bytes;

  float sum = 0;

  for (int i = 0; i < nb; ++i) {
    const float d = y[i].d * MLLM_FP16_TO_FP32(x[i].d);

    const uint8_t* __restrict q3 = x[i].qs;
    const uint8_t* __restrict qh = x[i].hmask;
    const int8_t* __restrict q8 = y[i].qs;

    mllm_uint8x16x2_t qhbits = mllm_vld1q_u8_x2(qh);

    mllm_uint8x16x4_t q3h;

    int32_t isum = 0;

    // Set up scales
    memcpy(aux, x[i].scales, 12);
    utmp[3] = ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4);
    utmp[2] = ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4);
    utmp[1] = (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4);
    utmp[0] = (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4);

    int8_t* scale = (int8_t*)utmp;
    for (int j = 0; j < 16; ++j) scale[j] -= m32;

    for (int j = 0; j < QK_K / 128; ++j) {
      const mllm_uint8x16x2_t q3bits = mllm_vld1q_u8_x2(q3);
      q3 += 32;
      const mllm_int8x16x4_t q8bytes_1 = mllm_vld1q_s8_x4(q8);
      q8 += 64;
      const mllm_int8x16x4_t q8bytes_2 = mllm_vld1q_s8_x4(q8);
      q8 += 64;

      q3h.val[0] = vshlq_n_u8(vbicq_u8(m0, qhbits.val[0]), 2);
      q3h.val[1] = vshlq_n_u8(vbicq_u8(m0, qhbits.val[1]), 2);
      q3h.val[2] = vshlq_n_u8(vbicq_u8(m1, qhbits.val[0]), 1);
      q3h.val[3] = vshlq_n_u8(vbicq_u8(m1, qhbits.val[1]), 1);

      q3bytes.val[0] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q3bits.val[0], m3b)), vreinterpretq_s8_u8(q3h.val[0]));
      q3bytes.val[1] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q3bits.val[1], m3b)), vreinterpretq_s8_u8(q3h.val[1]));
      q3bytes.val[2] =
          vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[0], 2), m3b)), vreinterpretq_s8_u8(q3h.val[2]));
      q3bytes.val[3] =
          vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[1], 2), m3b)), vreinterpretq_s8_u8(q3h.val[3]));

      isum += vaddvq_s32(mllm_vdotq_s32(vzero, q3bytes.val[0], q8bytes_1.val[0])) * scale[0];
      isum += vaddvq_s32(mllm_vdotq_s32(vzero, q3bytes.val[1], q8bytes_1.val[1])) * scale[1];
      isum += vaddvq_s32(mllm_vdotq_s32(vzero, q3bytes.val[2], q8bytes_1.val[2])) * scale[2];
      isum += vaddvq_s32(mllm_vdotq_s32(vzero, q3bytes.val[3], q8bytes_1.val[3])) * scale[3];

      scale += 4;

      q3h.val[0] = vbicq_u8(m2, qhbits.val[0]);
      q3h.val[1] = vbicq_u8(m2, qhbits.val[1]);
      q3h.val[2] = vshrq_n_u8(vbicq_u8(m3, qhbits.val[0]), 1);
      q3h.val[3] = vshrq_n_u8(vbicq_u8(m3, qhbits.val[1]), 1);

      q3bytes.val[0] =
          vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[0], 4), m3b)), vreinterpretq_s8_u8(q3h.val[0]));
      q3bytes.val[1] =
          vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[1], 4), m3b)), vreinterpretq_s8_u8(q3h.val[1]));
      q3bytes.val[2] =
          vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[0], 6), m3b)), vreinterpretq_s8_u8(q3h.val[2]));
      q3bytes.val[3] =
          vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[1], 6), m3b)), vreinterpretq_s8_u8(q3h.val[3]));

      isum += vaddvq_s32(mllm_vdotq_s32(vzero, q3bytes.val[0], q8bytes_2.val[0])) * scale[0];
      isum += vaddvq_s32(mllm_vdotq_s32(vzero, q3bytes.val[1], q8bytes_2.val[1])) * scale[1];
      isum += vaddvq_s32(mllm_vdotq_s32(vzero, q3bytes.val[2], q8bytes_2.val[2])) * scale[2];
      isum += vaddvq_s32(mllm_vdotq_s32(vzero, q3bytes.val[3], q8bytes_2.val[3])) * scale[3];

      scale += 4;

      if (j == 0) {
        qhbits.val[0] = vshrq_n_u8(qhbits.val[0], 4);
        qhbits.val[1] = vshrq_n_u8(qhbits.val[1], 4);
      }
    }
    sum += d * isum;
  }

  *s = sum;

#elif defined __AVX2__

  const __m256i m3 = _mm256_set1_epi8(3);
  const __m256i mone = _mm256_set1_epi8(1);
  const __m128i m32 = _mm_set1_epi8(32);

  __m256 acc = _mm256_setzero_ps();

  uint32_t aux[3];

  for (int i = 0; i < nb; ++i) {
    const float d = y[i].d * MLLM_FP16_TO_FP32(x[i].d);

    const uint8_t* __restrict q3 = x[i].qs;
    const int8_t* __restrict q8 = y[i].qs;

    // Set up scales
    memcpy(aux, x[i].scales, 12);
    __m128i scales128 = _mm_set_epi32(
        ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4), ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4),
        (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4), (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4));
    scales128 = _mm_sub_epi8(scales128, m32);
    const __m256i all_scales = _mm256_cvtepi8_epi16(scales128);
    const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
    const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
    const __m256i scales[2] = {MM256_SET_M128I(l_scales, l_scales), MM256_SET_M128I(h_scales, h_scales)};

    // high bit
    const __m256i hbits = _mm256_loadu_si256((const __m256i*)x[i].hmask);

    // integer accumulator
    __m256i sumi = _mm256_setzero_si256();

    int bit = 0;
    int is = 0;

    for (int j = 0; j < QK_K / 128; ++j) {
      // load low 2 bits
      const __m256i q3bits = _mm256_loadu_si256((const __m256i*)q3);
      q3 += 32;

      // prepare low and high bits
      const __m256i q3l_0 = _mm256_and_si256(q3bits, m3);
      const __m256i q3h_0 =
          _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
      ++bit;

      const __m256i q3l_1 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 2), m3);
      const __m256i q3h_1 =
          _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
      ++bit;

      const __m256i q3l_2 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 4), m3);
      const __m256i q3h_2 =
          _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
      ++bit;

      const __m256i q3l_3 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 6), m3);
      const __m256i q3h_3 =
          _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
      ++bit;

      // load Q8 quants
      const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8);
      q8 += 32;
      const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8);
      q8 += 32;
      const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8);
      q8 += 32;
      const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8);
      q8 += 32;

      // Dot product: we multiply the 2 low bits and 1 high bit part separately, so we can use _mm256_maddubs_epi16,
      // and then subtract. The high bit part has the 2 already subtracted (and so, it is zero if the high bit was not set,
      // and 2 if the high bit was set)
      __m256i q8s_0 = _mm256_maddubs_epi16(q3h_0, q8_0);
      __m256i q8s_1 = _mm256_maddubs_epi16(q3h_1, q8_1);
      __m256i q8s_2 = _mm256_maddubs_epi16(q3h_2, q8_2);
      __m256i q8s_3 = _mm256_maddubs_epi16(q3h_3, q8_3);

      __m256i p16_0 = _mm256_maddubs_epi16(q3l_0, q8_0);
      __m256i p16_1 = _mm256_maddubs_epi16(q3l_1, q8_1);
      __m256i p16_2 = _mm256_maddubs_epi16(q3l_2, q8_2);
      __m256i p16_3 = _mm256_maddubs_epi16(q3l_3, q8_3);

      p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
      p16_1 = _mm256_sub_epi16(p16_1, q8s_1);
      p16_2 = _mm256_sub_epi16(p16_2, q8s_2);
      p16_3 = _mm256_sub_epi16(p16_3, q8s_3);

      // multiply with scales
      p16_0 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 0)), p16_0);
      p16_1 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 1)), p16_1);
      p16_2 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 2)), p16_2);
      p16_3 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 3)), p16_3);

      // accumulate
      p16_0 = _mm256_add_epi32(p16_0, p16_1);
      p16_2 = _mm256_add_epi32(p16_2, p16_3);
      sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_2));
    }

    // multiply with block scale and accumulate
    acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
  }

  *s = hsum_float_8(acc);

#elif defined __AVX__

  const __m128i m3 = _mm_set1_epi8(3);
  const __m128i mone = _mm_set1_epi8(1);
  const __m128i m32 = _mm_set1_epi8(32);
  const __m128i m2 = _mm_set1_epi8(2);

  __m256 acc = _mm256_setzero_ps();

  const uint32_t* aux;

  for (int i = 0; i < nb; ++i) {
    const float d = y[i].d * MLLM_FP16_TO_FP32(x[i].d);

    const uint8_t* __restrict q3 = x[i].qs;
    const int8_t* __restrict q8 = y[i].qs;

    // Set up scales
    aux = (const uint32_t*)x[i].scales;
    __m128i scales128 = _mm_set_epi32(
        ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4), ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4),
        (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4), (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4));
    scales128 = _mm_sub_epi8(scales128, m32);
    const __m128i scales_0 = _mm_cvtepi8_epi16(scales128);
    const __m128i scales_1 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(scales128, scales128));
    const __m128i scales[2] = {scales_0, scales_1};

    // high bit *128*2 from block_q3_K.hmask[QK_K/8]
    const __m128i hbits_0 = _mm_loadu_si128((const __m128i*)&x[i].hmask[0]);
    const __m128i hbits_1 = _mm_loadu_si128((const __m128i*)&x[i].hmask[16]);

    // integer accumulator
    __m128i sumi_0 = _mm_setzero_si128();
    __m128i sumi_1 = _mm_setzero_si128();

    for (int j = 0; j < QK_K / 128; ++j) {
      // load low 2 bits *64*2 from block_q3_K.qs[QK_K/4]
      const __m128i q3bits_0 = _mm_loadu_si128((const __m128i*)q3);
      q3 += 16;
      const __m128i q3bits_1 = _mm_loadu_si128((const __m128i*)q3);
      q3 += 16;

      // prepare low and high bits
      const int bit = j << 2;

      const __m128i q3l_0 = _mm_and_si128(q3bits_0, m3);
      const __m128i q3l_1 = _mm_and_si128(q3bits_1, m3);
      const __m128i q3h_0 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_0, _mm_slli_epi16(mone, bit)), bit), 2);
      const __m128i q3h_1 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_1, _mm_slli_epi16(mone, bit)), bit), 2);

      const __m128i q3l_2 = _mm_and_si128(_mm_srli_epi16(q3bits_0, 2), m3);
      const __m128i q3l_3 = _mm_and_si128(_mm_srli_epi16(q3bits_1, 2), m3);
      const __m128i q3h_2 =
          _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_0, _mm_slli_epi16(mone, bit + 1)), bit + 1), 2);
      const __m128i q3h_3 =
          _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_1, _mm_slli_epi16(mone, bit + 1)), bit + 1), 2);

      const __m128i q3l_4 = _mm_and_si128(_mm_srli_epi16(q3bits_0, 4), m3);
      const __m128i q3l_5 = _mm_and_si128(_mm_srli_epi16(q3bits_1, 4), m3);
      const __m128i q3h_4 =
          _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_0, _mm_slli_epi16(mone, bit + 2)), bit + 2), 2);
      const __m128i q3h_5 =
          _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_1, _mm_slli_epi16(mone, bit + 2)), bit + 2), 2);

      const __m128i q3l_6 = _mm_and_si128(_mm_srli_epi16(q3bits_0, 6), m3);
      const __m128i q3l_7 = _mm_and_si128(_mm_srli_epi16(q3bits_1, 6), m3);
      const __m128i q3h_6 =
          _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_0, _mm_slli_epi16(mone, bit + 3)), bit + 3), 2);
      const __m128i q3h_7 =
          _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_1, _mm_slli_epi16(mone, bit + 3)), bit + 3), 2);

      // load Q8 quants from block_q8_K.qs[QK_K]
      const __m128i q8_0 = _mm_loadu_si128((const __m128i*)q8);
      q8 += 16;
      const __m128i q8_1 = _mm_loadu_si128((const __m128i*)q8);
      q8 += 16;
      const __m128i q8_2 = _mm_loadu_si128((const __m128i*)q8);
      q8 += 16;
      const __m128i q8_3 = _mm_loadu_si128((const __m128i*)q8);
      q8 += 16;
      const __m128i q8_4 = _mm_loadu_si128((const __m128i*)q8);
      q8 += 16;
      const __m128i q8_5 = _mm_loadu_si128((const __m128i*)q8);
      q8 += 16;
      const __m128i q8_6 = _mm_loadu_si128((const __m128i*)q8);
      q8 += 16;
      const __m128i q8_7 = _mm_loadu_si128((const __m128i*)q8);
      q8 += 16;

      // Dot product: we multiply the 2 low bits and 1 high bit part separately, so we can use _mm256_maddubs_epi16,
      // and then subtract. The high bit part has the 2 already subtracted (and so, it is zero if the high bit was not set,
      // and 2 if the high bit was set)
      __m128i q8s_0 = _mm_maddubs_epi16(q3h_0, q8_0);
      __m128i q8s_1 = _mm_maddubs_epi16(q3h_1, q8_1);
      __m128i q8s_2 = _mm_maddubs_epi16(q3h_2, q8_2);
      __m128i q8s_3 = _mm_maddubs_epi16(q3h_3, q8_3);
      __m128i q8s_4 = _mm_maddubs_epi16(q3h_4, q8_4);
      __m128i q8s_5 = _mm_maddubs_epi16(q3h_5, q8_5);
      __m128i q8s_6 = _mm_maddubs_epi16(q3h_6, q8_6);
      __m128i q8s_7 = _mm_maddubs_epi16(q3h_7, q8_7);

      __m128i p16_0 = _mm_maddubs_epi16(q3l_0, q8_0);
      __m128i p16_1 = _mm_maddubs_epi16(q3l_1, q8_1);
      __m128i p16_2 = _mm_maddubs_epi16(q3l_2, q8_2);
      __m128i p16_3 = _mm_maddubs_epi16(q3l_3, q8_3);
      __m128i p16_4 = _mm_maddubs_epi16(q3l_4, q8_4);
      __m128i p16_5 = _mm_maddubs_epi16(q3l_5, q8_5);
      __m128i p16_6 = _mm_maddubs_epi16(q3l_6, q8_6);
      __m128i p16_7 = _mm_maddubs_epi16(q3l_7, q8_7);

      p16_0 = _mm_sub_epi16(p16_0, q8s_0);
      p16_1 = _mm_sub_epi16(p16_1, q8s_1);
      p16_2 = _mm_sub_epi16(p16_2, q8s_2);
      p16_3 = _mm_sub_epi16(p16_3, q8s_3);
      p16_4 = _mm_sub_epi16(p16_4, q8s_4);
      p16_5 = _mm_sub_epi16(p16_5, q8s_5);
      p16_6 = _mm_sub_epi16(p16_6, q8s_6);
      p16_7 = _mm_sub_epi16(p16_7, q8s_7);

      // multiply with scales
      __m128i shuffle = _mm_set1_epi16(0x0100);
      p16_0 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_0);
      shuffle = _mm_add_epi16(shuffle, m2);
      p16_1 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_1);
      shuffle = _mm_add_epi16(shuffle, m2);
      p16_2 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_2);
      shuffle = _mm_add_epi16(shuffle, m2);
      p16_3 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_3);
      shuffle = _mm_add_epi16(shuffle, m2);
      p16_4 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_4);
      shuffle = _mm_add_epi16(shuffle, m2);
      p16_5 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_5);
      shuffle = _mm_add_epi16(shuffle, m2);
      p16_6 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_6);
      shuffle = _mm_add_epi16(shuffle, m2);
      p16_7 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_7);

      // accumulate
      p16_0 = _mm_add_epi32(p16_0, p16_1);
      p16_2 = _mm_add_epi32(p16_2, p16_3);
      p16_4 = _mm_add_epi32(p16_4, p16_5);
      p16_6 = _mm_add_epi32(p16_6, p16_7);
      sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p16_0, p16_2));
      sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p16_4, p16_6));
    }

    // multiply with block scale and accumulate
    __m256i sumi = MM256_SET_M128I(sumi_1, sumi_0);
    acc = _mm256_add_ps(_mm256_mul_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi)), acc);
  }

  *s = hsum_float_8(acc);

#elif defined __wasm_simd128__
  int8_t aux8[QK_K];
  float sums[8] = {0};
  uint32_t auxs[4];

  float sumf = 0;
  for (int i = 0; i < nb; ++i) {
    const uint8_t* __restrict q3 = x[i].qs;
    const uint8_t* __restrict hm = x[i].hmask;
    const int8_t* __restrict q8 = y[i].qs;

    // Process blocks with SIMD
    int8_t* a = aux8;
    uint8_t m = 1;
    for (int j = 0; j < QK_K; j += 128) {
      for (int shift = 0; shift <= 6; shift += 2) {
        v128_t v_m = wasm_i8x16_splat(m);
        for (int l = 0; l < 32; l += 16) {
          v128_t v_q3 = wasm_v128_load(q3 + l);
          v128_t v_shift = wasm_i8x16_shr(v_q3, shift);
          v128_t v_low2 = wasm_v128_and(v_shift, wasm_i8x16_splat(0x03));

          v128_t v_hm = wasm_v128_load(hm + l);
          v128_t v_mask = wasm_v128_and(v_hm, v_m);
          v_mask = wasm_i8x16_ne(v_mask, wasm_i8x16_splat(0));

          v_low2 = wasm_i8x16_sub(v_low2, wasm_v128_and(wasm_i8x16_splat(4), wasm_v128_not(v_mask)));
          wasm_v128_store(a + l, v_low2);
        }
        a += 32;
        m <<= 1;
      }
      q3 += 32;
    }

    // Extract scales
    memcpy(auxs, x[i].scales, 12);
    uint32_t tmp = auxs[2];
    auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const int8_t* scales = (const int8_t*)auxs;

    // SIMD dot product with register accumulators
    v128_t v_acc0 = wasm_i32x4_splat(0);
    v128_t v_acc1 = wasm_i32x4_splat(0);
    a = aux8;
    for (int j = 0; j < QK_K / 16; ++j) {
      const v128_t v_scale = wasm_i16x8_splat(scales[j] - 32);

      // Process 16 elements per iteration
      for (int k = 0; k < 2; ++k) {
        const v128_t v_q8 = wasm_i16x8_load8x8(q8);
        const v128_t v_a = wasm_i16x8_load8x8(a);

        v128_t v_prod = wasm_i16x8_mul(v_q8, v_a);
        v_prod = wasm_i16x8_mul(v_prod, v_scale);

        v_acc0 = wasm_i32x4_add(v_acc0, wasm_i32x4_extend_low_i16x8(v_prod));
        v_acc1 = wasm_i32x4_add(v_acc1, wasm_i32x4_extend_high_i16x8(v_prod));

        q8 += 8;
        a += 8;
      }
    }

    // Accumulate results
    const float d = MLLM_FP16_TO_FP32(x[i].d) * y[i].d;
    const v128_t v_d = wasm_f32x4_splat(d);
    v128_t v_sum = wasm_f32x4_add(wasm_f32x4_mul(wasm_f32x4_convert_i32x4(v_acc0), v_d),
                                  wasm_f32x4_mul(wasm_f32x4_convert_i32x4(v_acc1), v_d));

    // Accumulate into sums vector
    wasm_v128_store(sums, wasm_f32x4_add(wasm_v128_load(sums), v_sum));
  }

  // Horizontal sum
  v128_t v_sum = wasm_f32x4_add(wasm_v128_load(sums), wasm_v128_load(sums + 4));
  sumf = wasm_f32x4_extract_lane(v_sum, 0) + wasm_f32x4_extract_lane(v_sum, 1) + wasm_f32x4_extract_lane(v_sum, 2)
         + wasm_f32x4_extract_lane(v_sum, 3);

  *s = sumf;

#elif defined __riscv_v_intrinsic

  uint32_t aux[3];
  uint32_t utmp[4];

  const int vector_length = __riscv_vlenb() * 8;
  float sumf = 0;

  switch (vector_length) {
    case 256:
      for (int i = 0; i < nb; ++i) {
        const uint8_t* __restrict q3 = x[i].qs;
        const uint8_t* __restrict qh = x[i].hmask;
        const int8_t* __restrict q8 = y[i].qs;

        memcpy(aux, x[i].scales, 12);
        utmp[3] = ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4);
        utmp[2] = ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4);
        utmp[1] = (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4);
        utmp[0] = (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4);

        int8_t* scale = (int8_t*)utmp;
        for (int j = 0; j < 16; ++j) scale[j] -= 32;

        size_t vl = 32;
        uint8_t m = 1;

        vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);
        vuint8m1_t vqh = __riscv_vle8_v_u8m1(qh, vl);

        int sum_t = 0;

        for (int j = 0; j < QK_K; j += 128) {
          vl = 32;

          // load Q3
          vuint8m1_t q3_x = __riscv_vle8_v_u8m1(q3, vl);

          vint8m1_t q3_0 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(q3_x, 0x03, vl));
          vint8m1_t q3_1 =
              __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(q3_x, 0x2, vl), 0x03, vl));
          vint8m1_t q3_2 =
              __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(q3_x, 0x4, vl), 0x03, vl));
          vint8m1_t q3_3 =
              __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(q3_x, 0x6, vl), 0x03, vl));

          // compute mask for subtraction
          vuint8m1_t qh_m0 = __riscv_vand_vx_u8m1(vqh, m, vl);
          vbool8_t vmask_0 = __riscv_vmseq_vx_u8m1_b8(qh_m0, 0, vl);
          vint8m1_t q3_m0 = __riscv_vsub_vx_i8m1_mu(vmask_0, q3_0, q3_0, 0x4, vl);
          m <<= 1;

          vuint8m1_t qh_m1 = __riscv_vand_vx_u8m1(vqh, m, vl);
          vbool8_t vmask_1 = __riscv_vmseq_vx_u8m1_b8(qh_m1, 0, vl);
          vint8m1_t q3_m1 = __riscv_vsub_vx_i8m1_mu(vmask_1, q3_1, q3_1, 0x4, vl);
          m <<= 1;

          vuint8m1_t qh_m2 = __riscv_vand_vx_u8m1(vqh, m, vl);
          vbool8_t vmask_2 = __riscv_vmseq_vx_u8m1_b8(qh_m2, 0, vl);
          vint8m1_t q3_m2 = __riscv_vsub_vx_i8m1_mu(vmask_2, q3_2, q3_2, 0x4, vl);
          m <<= 1;

          vuint8m1_t qh_m3 = __riscv_vand_vx_u8m1(vqh, m, vl);
          vbool8_t vmask_3 = __riscv_vmseq_vx_u8m1_b8(qh_m3, 0, vl);
          vint8m1_t q3_m3 = __riscv_vsub_vx_i8m1_mu(vmask_3, q3_3, q3_3, 0x4, vl);
          m <<= 1;

          // load Q8 and take product with Q3
          vint16m2_t a0 = __riscv_vwmul_vv_i16m2(q3_m0, __riscv_vle8_v_i8m1(q8, vl), vl);
          vint16m2_t a1 = __riscv_vwmul_vv_i16m2(q3_m1, __riscv_vle8_v_i8m1(q8 + 32, vl), vl);
          vint16m2_t a2 = __riscv_vwmul_vv_i16m2(q3_m2, __riscv_vle8_v_i8m1(q8 + 64, vl), vl);
          vint16m2_t a3 = __riscv_vwmul_vv_i16m2(q3_m3, __riscv_vle8_v_i8m1(q8 + 96, vl), vl);

          vl = 16;

          // retrieve lane to multiply with scale
          vint32m2_t aux0_0 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a0, 0), (scale[0]), vl);
          vint32m2_t aux0_1 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a0, 1), (scale[1]), vl);
          vint32m2_t aux1_0 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a1, 0), (scale[2]), vl);
          vint32m2_t aux1_1 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a1, 1), (scale[3]), vl);
          vint32m2_t aux2_0 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a2, 0), (scale[4]), vl);
          vint32m2_t aux2_1 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a2, 1), (scale[5]), vl);
          vint32m2_t aux3_0 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a3, 0), (scale[6]), vl);
          vint32m2_t aux3_1 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a3, 1), (scale[7]), vl);

          vint32m1_t isum0 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(aux0_0, aux0_1, vl), vzero, vl);
          vint32m1_t isum1 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(aux1_0, aux1_1, vl), isum0, vl);
          vint32m1_t isum2 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(aux2_0, aux2_1, vl), isum1, vl);
          vint32m1_t isum3 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(aux3_0, aux3_1, vl), isum2, vl);

          sum_t += __riscv_vmv_x_s_i32m1_i32(isum3);

          q3 += 32;
          q8 += 128;
          scale += 8;
        }

        const float d = MLLM_FP16_TO_FP32(x[i].d) * y[i].d;

        sumf += d * sum_t;
      }
      break;
    case 128:
      for (int i = 0; i < nb; ++i) {
        const uint8_t* restrict q3 = x[i].qs;
        const uint8_t* restrict qh = x[i].hmask;
        const int8_t* restrict q8 = y[i].qs;

        int8_t* scale = (int8_t*)utmp;
        int tmp;
        __asm__ __volatile__("vsetivli zero, 12, e8, m1\n\t"
                             "vle8.v v0, (%[s6b])\n\t"
                             "vmv1r.v v2, v0\n\t"
                             "vsetivli zero, 2, e64, m1\n\t"
                             "vmv.v.x v9, %[sh]\n\t"
                             "vslidedown.vi v1, v0, 1\n\t"
                             "vslide1up.vx v8, v9, zero\n\t"  // {0, 0, 4, 4}
                             "vslideup.vi v0, v2, 1\n\t"      // {aux[0], aux[1], aux[0], aux[1]}
                             "vsetivli zero, 4, e32, m1\n\t"
                             "vid.v v9\n\t"
                             "vmv.x.s %[tmp], v1\n\t"
                             "vsll.vi v9, v9, 1\n\t"   // {0, 2, 4, 6}
                             "vmv.v.x v1, %[tmp]\n\t"  // {aux[2], aux[2], aux[2], aux[2]}
                             "vsrl.vv v4, v1, v9\n\t"
                             "vsrl.vv v2, v0, v8\n\t"
                             "vand.vx v5, v4, %[kmask1]\n\t"
                             "vand.vx v3, v2, %[kmask2]\n\t"
                             "vsll.vi v6, v5, 4\n\t"
                             "vor.vv v7, v6, v3\n\t"
                             "vsetivli zero, 16, e8, m1\n\t"
                             "vsub.vx v0, v7, %[c]\n\t"
                             "vse8.v v0, (%[scale])"
                             : [tmp] "=&r"(tmp)
                             : [sh] "r"(0x0000000400000004), [s6b] "r"(x[i].scales), [c] "r"(32), [scale] "r"(scale),
                               [kmask1] "r"(kmask1), [kmask2] "r"(kmask2)
                             : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13",
                               "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
                               "v28", "v29", "v30", "v31");

        uint8_t m = 1;
        int isum = 0;
        for (int j = 0; j < QK_K; j += 128) {
          __asm__ __volatile__(
              "vsetvli zero, %[vl32], e8, m2, ta, mu\n\t"
              "vle8.v v8, (%[q3])\n\t"
              "vsrl.vi v10, v8, 2\n\t"
              "vsrl.vi v12, v8, 4\n\t"
              "vsrl.vi v14, v8, 6\n\t"
              "vand.vi v8, v8, 3\n\t"
              "vand.vi v10, v10, 3\n\t"
              "vand.vi v12, v12, 3\n\t"
              "vle8.v v2, (%[qh])\n\t"
              "vand.vx v4, v2, %[m]\n\t"
              "slli %[m], %[m], 1\n\t"
              "vmseq.vx v0, v4, zero\n\t"
              "vadd.vi v8, v8, -4, v0.t\n\t"
              "vand.vx v4, v2, %[m]\n\t"
              "slli %[m], %[m], 1\n\t"
              "vmseq.vx v0, v4, zero\n\t"
              "vadd.vi v10, v10, -4, v0.t\n\t"
              "vand.vx v4, v2, %[m]\n\t"
              "slli %[m], %[m], 1\n\t"
              "vmseq.vx v0, v4, zero\n\t"
              "vadd.vi v12, v12, -4, v0.t\n\t"
              "vand.vx v4, v2, %[m]\n\t"
              "slli %[m], %[m], 1\n\t"
              "vmseq.vx v0, v4, zero\n\t"
              "vadd.vi v14, v14, -4, v0.t\n\t"
              "vsetvli zero, %[vl128], e8, m8\n\t"
              "vle8.v v0, (%[q8])\n\t"
              "vsetvli zero, %[vl64], e8, m4\n\t"
              "vwmul.vv v16, v0, v8\n\t"
              "vwmul.vv v24, v4, v12\n\t"
              "vsetivli zero, 16, e16, m2\n\t"
              "vmv.v.x v0, zero\n\t"
              "vwredsum.vs v10, v16, v0\n\t"
              "vwredsum.vs v9, v18, v0\n\t"
              "vwredsum.vs v8, v20, v0\n\t"
              "vwredsum.vs v7, v22, v0\n\t"
              "vwredsum.vs v11, v24, v0\n\t"
              "vwredsum.vs v12, v26, v0\n\t"
              "vwredsum.vs v13, v28, v0\n\t"
              "vwredsum.vs v14, v30, v0\n\t"
              "vsetivli zero, 4, e32, m1\n\t"
              "vslideup.vi v10, v9, 1\n\t"
              "vslideup.vi v8, v7, 1\n\t"
              "vslideup.vi v11, v12, 1\n\t"
              "vslideup.vi v13, v14, 1\n\t"
              "vslideup.vi v10, v8, 2\n\t"
              "vslideup.vi v11, v13, 2\n\t"
              "vsetivli zero, 8, e32, m2\n\t"
              "vle8.v v15, (%[scale])\n\t"
              "vsext.vf4 v12, v15\n\t"
              "vmul.vv v10, v10, v12\n\t"
              "vredsum.vs v0, v10, v0\n\t"
              "vmv.x.s %[tmp], v0\n\t"
              "add %[isum], %[isum], %[tmp]"
              : [tmp] "=&r"(tmp), [m] "+&r"(m), [isum] "+&r"(isum)
              : [vl128] "r"(128), [vl64] "r"(64), [vl32] "r"(32), [q3] "r"(q3), [qh] "r"(qh), [scale] "r"(scale), [q8] "r"(q8)
              : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
          q3 += 32;
          q8 += 128;
          scale += 8;
        }

        const float d = MLLM_FP16_TO_FP32(x[i].d) * y[i].d;
        sumf += d * isum;
      }
      break;
    default: assert(false && "Unsupported vector length"); break;
  }

  *s = sumf;

#elif defined(__POWER9_VECTOR__)
  const vector signed char lowMask = vec_splats((signed char)0x3);
  const vector signed char lowMask1 = vec_splats((int8_t)0xf);
  const vector signed char lowMask2 = vec_splats((int8_t)0x30);
  const vector int v0 = vec_splats((int32_t)0);
  const vector signed char v1 = vec_splats((signed char)0x1);
  const vector unsigned char v2 = vec_splats((unsigned char)0x2);
  const vector unsigned char v3 = vec_splats((unsigned char)0x3);
  const vector unsigned char v4 = vec_splats((unsigned char)0x4);
  const vector unsigned char v6 = vec_splats((unsigned char)0x6);
  const vector signed char off = vec_splats((signed char)0x20);

  vector float vsumf0 = vec_splats(0.0f);
  vector float vsumf1 = vec_splats(0.0f);
  vector float vsumf2 = vec_splats(0.0f);
  vector float vsumf3 = vec_splats(0.0f);

  for (int i = 0; i < nb; ++i) {
    vector float vxd = vec_splats(MLLM_FP16_TO_FP32(x[i].d));
    vector float vyd = vec_splats(y[i].d);
    vector float vd = vec_mul(vxd, vyd);

    UNUSED(kmask1);
    UNUSED(kmask2);

    vector signed char u0 = (vector signed char)vec_xl_len(x[i].scales, 8);
    vector signed char u1 = vec_and(u0, lowMask1);
    vector signed char u2 = (vector signed char)vec_xl_len(x[i].scales + 8, 4);
    vector signed char u3 = (vector signed char)vec_mergeh((vector signed int)u2, (vector signed int)vec_sr(u2, v2));
    vector signed char u30 = vec_sl(vec_and(u3, lowMask), v4);
    vector signed char u31 = vec_and(u3, lowMask2);

    u1 = vec_or(u1, u30);
    u2 = vec_or(vec_sr(u0, v4), u31);

    vector signed char vscales = (vector signed char)vec_mergeh((vector signed long long)u1, (vector signed long long)u2);
    vector signed char qxhs0 = (vector signed char)vec_xl(0, x[i].hmask);
    vector signed char qxhs1 = (vector signed char)vec_xl(16, x[i].hmask);

    vscales = vec_sub(vscales, off);

    vector signed int vsumi0 = v0;
    vector signed int vsumi1 = v0;
    vector signed int vsumi2 = v0;
    vector signed int vsumi3 = v0;
    vector signed int vsumi4 = v0;
    vector signed int vsumi5 = v0;
    vector signed int vsumi6 = v0;
    vector signed int vsumi7 = v0;

    const uint8_t* __restrict q3 = x[i].qs;
    const int8_t* __restrict q8 = y[i].qs;

    for (int j = 0; j < QK_K / 128; ++j) {
      __builtin_prefetch(q3, 0, 1);
      __builtin_prefetch(q8, 0, 1);

      vector signed char qxs0 = (vector signed char)vec_xl(0, q3);
      vector signed char qxs1 = (vector signed char)vec_xl(16, q3);
      q3 += 32;

      // the low 2 bits
      vector signed char qxs00 = vec_and(qxs0, lowMask);
      vector signed char qxs01 = vec_and(vec_sr(qxs0, v2), lowMask);
      vector signed char qxs02 = vec_and(vec_sr(qxs0, v4), lowMask);
      vector signed char qxs03 = vec_and(vec_sr(qxs0, v6), lowMask);
      vector signed char qxs10 = vec_and(qxs1, lowMask);
      vector signed char qxs11 = vec_and(vec_sr(qxs1, v2), lowMask);
      vector signed char qxs12 = vec_and(vec_sr(qxs1, v4), lowMask);
      vector signed char qxs13 = vec_and(vec_sr(qxs1, v6), lowMask);

      // the 3rd bit
      vector signed char qxh00 = vec_sl(vec_andc(v1, qxhs0), v2);
      vector signed char qxh01 = vec_sl(vec_andc(v1, vec_sr(qxhs0, (vector unsigned char)v1)), v2);
      vector signed char qxh02 = vec_sl(vec_andc(v1, vec_sr(qxhs0, v2)), v2);
      vector signed char qxh03 = vec_sl(vec_andc(v1, vec_sr(qxhs0, v3)), v2);
      vector signed char qxh10 = vec_sl(vec_andc(v1, qxhs1), v2);
      vector signed char qxh11 = vec_sl(vec_andc(v1, vec_sr(qxhs1, (vector unsigned char)v1)), v2);
      vector signed char qxh12 = vec_sl(vec_andc(v1, vec_sr(qxhs1, v2)), v2);
      vector signed char qxh13 = vec_sl(vec_andc(v1, vec_sr(qxhs1, v3)), v2);
      qxhs0 = vec_sr(qxhs0, v4);
      qxhs1 = vec_sr(qxhs1, v4);

      vector signed char q3x00 = vec_sub(qxs00, qxh00);
      vector signed char q3x01 = vec_sub(qxs01, qxh01);
      vector signed char q3x02 = vec_sub(qxs02, qxh02);
      vector signed char q3x03 = vec_sub(qxs03, qxh03);
      vector signed char q3x10 = vec_sub(qxs10, qxh10);
      vector signed char q3x11 = vec_sub(qxs11, qxh11);
      vector signed char q3x12 = vec_sub(qxs12, qxh12);
      vector signed char q3x13 = vec_sub(qxs13, qxh13);

      vector signed char q8y00 = vec_xl(0, q8);
      vector signed char q8y10 = vec_xl(16, q8);
      vector signed char q8y01 = vec_xl(32, q8);
      vector signed char q8y11 = vec_xl(48, q8);
      vector signed char q8y02 = vec_xl(64, q8);
      vector signed char q8y12 = vec_xl(80, q8);
      vector signed char q8y03 = vec_xl(96, q8);
      vector signed char q8y13 = vec_xl(112, q8);
      q8 += 128;

      vector signed short vscales_h = vec_unpackh(vscales);
      vector signed short vs0 = vec_splat(vscales_h, 0);
      vector signed short vs1 = vec_splat(vscales_h, 1);
      vector signed short vs2 = vec_splat(vscales_h, 2);
      vector signed short vs3 = vec_splat(vscales_h, 3);
      vector signed short vs4 = vec_splat(vscales_h, 4);
      vector signed short vs5 = vec_splat(vscales_h, 5);
      vector signed short vs6 = vec_splat(vscales_h, 6);
      vector signed short vs7 = vec_splat(vscales_h, 7);
      vscales = vec_sld(vscales, vscales, 8);

      vector signed short qv00 = vec_add(vec_mule(q3x00, q8y00), vec_mulo(q3x00, q8y00));
      vector signed short qv01 = vec_add(vec_mule(q3x01, q8y01), vec_mulo(q3x01, q8y01));
      vector signed short qv02 = vec_add(vec_mule(q3x02, q8y02), vec_mulo(q3x02, q8y02));
      vector signed short qv03 = vec_add(vec_mule(q3x03, q8y03), vec_mulo(q3x03, q8y03));
      vector signed short qv10 = vec_add(vec_mule(q3x10, q8y10), vec_mulo(q3x10, q8y10));
      vector signed short qv11 = vec_add(vec_mule(q3x11, q8y11), vec_mulo(q3x11, q8y11));
      vector signed short qv12 = vec_add(vec_mule(q3x12, q8y12), vec_mulo(q3x12, q8y12));
      vector signed short qv13 = vec_add(vec_mule(q3x13, q8y13), vec_mulo(q3x13, q8y13));

      vsumi0 = vec_msum(qv00, vs0, vsumi0);
      vsumi1 = vec_msum(qv01, vs2, vsumi1);
      vsumi2 = vec_msum(qv02, vs4, vsumi2);
      vsumi3 = vec_msum(qv03, vs6, vsumi3);
      vsumi4 = vec_msum(qv10, vs1, vsumi4);
      vsumi5 = vec_msum(qv11, vs3, vsumi5);
      vsumi6 = vec_msum(qv12, vs5, vsumi6);
      vsumi7 = vec_msum(qv13, vs7, vsumi7);
    }

    vsumi0 = vec_add(vsumi0, vsumi4);
    vsumi1 = vec_add(vsumi1, vsumi5);
    vsumi2 = vec_add(vsumi2, vsumi6);
    vsumi3 = vec_add(vsumi3, vsumi7);

    vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
    vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
    vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
  }

  vsumf0 = vec_add(vsumf0, vsumf2);
  vsumf1 = vec_add(vsumf1, vsumf3);

  vsumf0 = vec_add(vsumf0, vsumf1);

  vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
  vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

  *s = vec_extract(vsumf0, 0);

#elif defined __loongarch_asx

  const __m128i m32 = __lsx_vreplgr2vr_b(32);

  __m256 acc = (__m256)__lasx_xvldi(0);

  uint32_t aux[3];

  for (int i = 0; i < nb; ++i) {
    const float d = y[i].d * MLLM_FP16_TO_FP32(x[i].d);
    const uint8_t* __restrict q3 = x[i].qs;
    const int8_t* __restrict q8 = y[i].qs;
    // Set up scales
    memcpy(aux, x[i].scales, 12);
    __m128i scales128 = lsx_set_w(
        ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4), ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4),
        (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4), (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4));
    scales128 = __lsx_vsub_b(scales128, m32);

    const v16i8 shuffle_mask = {0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};
    const __m256i scales_shuffled = lasx_ext8_16(__lsx_vshuf_b(scales128, scales128, (__m128i)shuffle_mask));

    // high bit
    const __m256i hbits = __lasx_xvld((const __m256i*)x[i].hmask, 0);

    // integer accumulator
    __m256i sumi = __lasx_xvldi(0);

    for (int j = 0; j < QK_K / 128; ++j) {
      // load low 2 bits
      const __m256i q3bits = __lasx_xvld((const __m256i*)q3, 0);
      q3 += 32;

      // prepare low and high bits
      const __m256i q3l_0 = __lasx_xvandi_b(q3bits, 3);
      const __m256i q3l_1 = __lasx_xvandi_b(__lasx_xvsrli_b(q3bits, 2), 3);
      const __m256i q3l_2 = __lasx_xvandi_b(__lasx_xvsrli_b(q3bits, 4), 3);
      const __m256i q3l_3 = __lasx_xvsrli_b(q3bits, 6);
      const __m256i q3h_0 = __lasx_xvslli_b(__lasx_xvseqi_b(lasx_xvandi_b_bit(hbits, 4 * j + 0), 0), 2);
      const __m256i q3h_1 = __lasx_xvslli_b(__lasx_xvseqi_b(lasx_xvandi_b_bit(hbits, 4 * j + 1), 0), 2);
      const __m256i q3h_2 = __lasx_xvslli_b(__lasx_xvseqi_b(lasx_xvandi_b_bit(hbits, 4 * j + 2), 0), 2);
      const __m256i q3h_3 = __lasx_xvslli_b(__lasx_xvseqi_b(lasx_xvandi_b_bit(hbits, 4 * j + 3), 0), 2);
      const __m256i q3_0 = __lasx_xvor_v(q3h_0, q3l_0);
      const __m256i q3_1 = __lasx_xvor_v(q3h_1, q3l_1);
      const __m256i q3_2 = __lasx_xvor_v(q3h_2, q3l_2);
      const __m256i q3_3 = __lasx_xvor_v(q3h_3, q3l_3);

      // load Q8 quants
      const __m256i q8_0 = __lasx_xvld((const __m256i*)q8, 0);
      q8 += 32;
      const __m256i q8_1 = __lasx_xvld((const __m256i*)q8, 0);
      q8 += 32;
      const __m256i q8_2 = __lasx_xvld((const __m256i*)q8, 0);
      q8 += 32;
      const __m256i q8_3 = __lasx_xvld((const __m256i*)q8, 0);
      q8 += 32;

      __m256i p16_0 = lasx_madd_h_b(q8_0, q3_0);
      __m256i p16_1 = lasx_madd_h_b(q8_1, q3_1);
      __m256i p16_2 = lasx_madd_h_b(q8_2, q3_2);
      __m256i p16_3 = lasx_madd_h_b(q8_3, q3_3);

      // multiply with scales
      p16_0 = lasx_madd_h(lasx_xvrepl128vei_h(scales_shuffled, 4 * j + 0), p16_0);
      p16_1 = lasx_madd_h(lasx_xvrepl128vei_h(scales_shuffled, 4 * j + 1), p16_1);
      p16_2 = lasx_madd_h(lasx_xvrepl128vei_h(scales_shuffled, 4 * j + 2), p16_2);
      p16_3 = lasx_madd_h(lasx_xvrepl128vei_h(scales_shuffled, 4 * j + 3), p16_3);

      // accumulate
      p16_0 = __lasx_xvadd_w(p16_0, p16_1);
      p16_2 = __lasx_xvadd_w(p16_2, p16_3);
      sumi = __lasx_xvadd_w(sumi, __lasx_xvadd_w(p16_0, p16_2));
    }
    // multiply with block scale and accumulate
    acc = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(d), __lasx_xvffint_s_w(sumi), acc);
  }

  *s = hsum_float_8(acc);
#elif defined(__VXE__) || defined(__VXE2__)
  uint32_t aux[3];
  uint32_t utmp[4];

  const int32x4_t v_z = vec_splat_s32(0);
  const uint8x16_t v_3m = vec_splat_u8(0x03);

  const uint8x16_t v_0c = vec_splat_u8(1);
  const uint8x16_t v_1c = vec_sl(v_0c, 1);
  const uint8x16_t v_2c = vec_sl(v_0c, 2);
  const uint8x16_t v_3c = vec_sl(v_0c, 3);

  uint8x16_t q3h[4];
  uint8x16_t q3b[2];
  int8x16_t q3bytes[4];
  int8x16_t q8bytes[4];
  uint8x16_t qhbits[2];

  float sum = 0;

  for (int i = 0; i < nb; ++i) {
    const float d = y[i].d * MLLM_FP16_TO_FP32(x[i].d);

    const uint8_t* restrict x0l = x[i].qs;
    const uint8_t* restrict x0h = x[i].hmask;
    const int8_t* restrict y0 = y[i].qs;

    qhbits[0] = vec_xl(0, x0h);
    qhbits[1] = vec_xl(16, x0h);

    int32_t isum = 0;

    memcpy(aux, x[i].scales, 12);
    utmp[3] = ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4);
    utmp[2] = ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4);
    utmp[1] = (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4);
    utmp[0] = (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4);

    int8_t* scale = (int8_t*)utmp;
    for (int j = 0; j < 16; ++j) scale[j] -= 32;

    for (int j = 0; j < QK_K / 128; ++j) {
      int32x4_t isum0, isum1, isum2, isum3;

      q3b[0] = vec_xl(0, x0l);
      q3b[1] = vec_xl(16, x0l);
      x0l += 32;

      q8bytes[0] = vec_xl(0, y0);
      q8bytes[1] = vec_xl(16, y0);
      q8bytes[2] = vec_xl(32, y0);
      q8bytes[3] = vec_xl(48, y0);
      q8bytes[4] = vec_xl(64, y0);
      q8bytes[5] = vec_xl(80, y0);
      q8bytes[6] = vec_xl(96, y0);
      q8bytes[7] = vec_xl(112, y0);
      y0 += 128;

      q3h[0] = vec_sl(vec_andc(v_0c, qhbits[0]), 2);
      q3h[1] = vec_sl(vec_andc(v_0c, qhbits[1]), 2);
      q3h[2] = vec_sl(vec_andc(v_1c, qhbits[0]), 1);
      q3h[3] = vec_sl(vec_andc(v_1c, qhbits[1]), 1);

      q3bytes[0] = vec_sub((int8x16_t)vec_and(q3b[0], v_3m), (int8x16_t)q3h[0]);
      q3bytes[1] = vec_sub((int8x16_t)vec_and(q3b[1], v_3m), (int8x16_t)q3h[1]);
      q3bytes[2] = vec_sub((int8x16_t)vec_and(vec_sr(q3b[0], 2), v_3m), (int8x16_t)q3h[2]);
      q3bytes[3] = vec_sub((int8x16_t)vec_and(vec_sr(q3b[1], 2), v_3m), (int8x16_t)q3h[3]);

      isum0 = mllm_vec_dot(v_z, q3bytes[0], q8bytes[0]);
      isum1 = mllm_vec_dot(v_z, q3bytes[1], q8bytes[1]);
      isum2 = mllm_vec_dot(v_z, q3bytes[2], q8bytes[2]);
      isum3 = mllm_vec_dot(v_z, q3bytes[3], q8bytes[3]);

      isum += (isum0[0] + isum0[1] + isum0[2] + isum0[3]) * scale[0];
      isum += (isum1[0] + isum1[1] + isum1[2] + isum1[3]) * scale[1];
      isum += (isum2[0] + isum2[1] + isum2[2] + isum2[3]) * scale[2];
      isum += (isum3[0] + isum3[1] + isum3[2] + isum3[3]) * scale[3];

      scale += 4;

      q3h[0] = vec_andc(v_2c, qhbits[0]);
      q3h[1] = vec_andc(v_2c, qhbits[1]);
      q3h[2] = vec_sr(vec_andc(v_3c, qhbits[0]), 1);
      q3h[3] = vec_sr(vec_andc(v_3c, qhbits[1]), 1);

      q3bytes[0] = vec_sub((int8x16_t)vec_and(vec_sr(q3b[0], 4), v_3m), (int8x16_t)q3h[0]);
      q3bytes[1] = vec_sub((int8x16_t)vec_and(vec_sr(q3b[1], 4), v_3m), (int8x16_t)q3h[1]);
      q3bytes[2] = vec_sub((int8x16_t)vec_and(vec_sr(q3b[0], 6), v_3m), (int8x16_t)q3h[2]);
      q3bytes[3] = vec_sub((int8x16_t)vec_and(vec_sr(q3b[1], 6), v_3m), (int8x16_t)q3h[3]);

      isum0 = mllm_vec_dot(v_z, q3bytes[0], q8bytes[4]);
      isum1 = mllm_vec_dot(v_z, q3bytes[1], q8bytes[5]);
      isum2 = mllm_vec_dot(v_z, q3bytes[2], q8bytes[6]);
      isum3 = mllm_vec_dot(v_z, q3bytes[3], q8bytes[7]);

      isum += (isum0[0] + isum0[1] + isum0[2] + isum0[3]) * scale[0];
      isum += (isum1[0] + isum1[1] + isum1[2] + isum1[3]) * scale[1];
      isum += (isum2[0] + isum2[1] + isum2[2] + isum2[3]) * scale[2];
      isum += (isum3[0] + isum3[1] + isum3[2] + isum3[3]) * scale[3];

      scale += 4;

      if (j == 0) {
        qhbits[0] = vec_sr(qhbits[0], 4);
        qhbits[1] = vec_sr(qhbits[1], 4);
      }
    }

    sum += d * isum;
  }

  *s = sum;
#else
  // scalar version
  // This function is written like this so the compiler can manage to vectorize most of it
  // Using -Ofast, GCC and clang manage to produce code that is within a factor of 2 or so from the
  // manually vectorized version above. Every other version I tried would run at least 4 times slower.
  // The ideal situation would be if we could just write the code once, and the compiler would
  // automatically produce the best possible set of machine instructions, instead of us having to manually
  // write vectorized versions for AVX, ARM_NEON, etc.

  int8_t aux8[QK_K];
  int16_t aux16[8];
  float sums[8];
  int32_t aux32[8];
  memset(sums, 0, 8 * sizeof(float));

  uint32_t auxs[4];
  const int8_t* scales = (const int8_t*)auxs;

  float sumf = 0;
  for (int i = 0; i < nb; ++i) {
    const uint8_t* __restrict q3 = x[i].qs;
    const uint8_t* __restrict hm = x[i].hmask;
    const int8_t* __restrict q8 = y[i].qs;
    memset(aux32, 0, 8 * sizeof(int32_t));
    int8_t* __restrict a = aux8;
    uint8_t m = 1;
    for (int j = 0; j < QK_K; j += 128) {
      for (int l = 0; l < 32; ++l) a[l] = q3[l] & 3;
      for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
      a += 32;
      m <<= 1;
      for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 2) & 3;
      for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
      a += 32;
      m <<= 1;
      for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 4) & 3;
      for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
      a += 32;
      m <<= 1;
      for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 6) & 3;
      for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
      a += 32;
      m <<= 1;
      q3 += 32;
    }
    a = aux8;

    memcpy(auxs, x[i].scales, 12);
    uint32_t tmp = auxs[2];
    auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    for (int j = 0; j < QK_K / 16; ++j) {
      for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
      for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
      q8 += 8;
      a += 8;
      for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
      for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
      q8 += 8;
      a += 8;
    }
    const float d = MLLM_FP16_TO_FP32(x[i].d) * y[i].d;
    for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
  }
  for (int l = 0; l < 8; ++l) sumf += sums[l];
  *s = sumf;

#endif
}

void vec_dot_iq2_xxs_q8_K(int n, float* __restrict s, const void* __restrict vx, const void* __restrict vy) {
  assert(n % QK_K == 0);

  const block_iq2_xxs* __restrict x = (block_iq2_xxs*)vx;
  const block_q8_K* __restrict y = (block_q8_K*)vy;

  const int nb = n / QK_K;

#if defined(__ARM_NEON)

  const uint64_t* signs64 = (const uint64_t*)keven_signs_q2xs;

  uint32_t aux32[4];
  const uint8_t* aux8 = (const uint8_t*)aux32;

  mllm_int8x16x4_t q2u;
  mllm_int8x16x4_t q2s;
  mllm_int8x16x4_t q8b;

  float sumf = 0;
  for (int i = 0; i < nb; ++i) {
    const float d = MLLM_FP16_TO_FP32(x[i].d) * y[i].d;
    const uint16_t* __restrict q2 = x[i].qs;
    const int8_t* __restrict q8 = y[i].qs;
    float sumf1 = 0, sumf2 = 0;
    for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
      q8b = mllm_vld1q_s8_x4(q8);
      q8 += 64;
      memcpy(aux32, q2, 4 * sizeof(uint32_t));
      q2 += 8;
      q2u.val[0] =
          vcombine_s8(vld1_s8((const int8_t*)(iq2xxs_grid + aux8[0])), vld1_s8((const int8_t*)(iq2xxs_grid + aux8[1])));
      q2u.val[1] =
          vcombine_s8(vld1_s8((const int8_t*)(iq2xxs_grid + aux8[2])), vld1_s8((const int8_t*)(iq2xxs_grid + aux8[3])));
      q2u.val[2] =
          vcombine_s8(vld1_s8((const int8_t*)(iq2xxs_grid + aux8[8])), vld1_s8((const int8_t*)(iq2xxs_grid + aux8[9])));
      q2u.val[3] =
          vcombine_s8(vld1_s8((const int8_t*)(iq2xxs_grid + aux8[10])), vld1_s8((const int8_t*)(iq2xxs_grid + aux8[11])));
      q2s.val[0] = vcombine_s8(vld1_s8((const int8_t*)(signs64 + ((aux32[1] >> 0) & 127))),
                               vld1_s8((const int8_t*)(signs64 + ((aux32[1] >> 7) & 127))));
      q2s.val[1] = vcombine_s8(vld1_s8((const int8_t*)(signs64 + ((aux32[1] >> 14) & 127))),
                               vld1_s8((const int8_t*)(signs64 + ((aux32[1] >> 21) & 127))));
      q2s.val[2] = vcombine_s8(vld1_s8((const int8_t*)(signs64 + ((aux32[3] >> 0) & 127))),
                               vld1_s8((const int8_t*)(signs64 + ((aux32[3] >> 7) & 127))));
      q2s.val[3] = vcombine_s8(vld1_s8((const int8_t*)(signs64 + ((aux32[3] >> 14) & 127))),
                               vld1_s8((const int8_t*)(signs64 + ((aux32[3] >> 21) & 127))));
      q2u.val[0] = vmulq_s8(q2u.val[0], q2s.val[0]);
      q2u.val[1] = vmulq_s8(q2u.val[1], q2s.val[1]);
      q2u.val[2] = vmulq_s8(q2u.val[2], q2s.val[2]);
      q2u.val[3] = vmulq_s8(q2u.val[3], q2s.val[3]);
      const int32x4_t p1 = mllm_vdotq_s32(mllm_vdotq_s32(vdupq_n_s32(0), q2u.val[0], q8b.val[0]), q2u.val[1], q8b.val[1]);
      const int32x4_t p2 = mllm_vdotq_s32(mllm_vdotq_s32(vdupq_n_s32(0), q2u.val[2], q8b.val[2]), q2u.val[3], q8b.val[3]);
      sumf1 += vaddvq_s32(p1) * (0.5f + (aux32[1] >> 28));
      sumf2 += vaddvq_s32(p2) * (0.5f + (aux32[3] >> 28));
    }
    sumf += d * (sumf1 + sumf2);
  }
  *s = 0.25f * sumf;

#elif defined(__AVX2__)

  const uint64_t* signs64 = (const uint64_t*)keven_signs_q2xs;

  uint32_t aux32[4];
  const uint8_t* aux8 = (const uint8_t*)aux32;

  __m256 accumf = _mm256_setzero_ps();
  for (int i = 0; i < nb; ++i) {
    const float d = MLLM_FP16_TO_FP32(x[i].d) * y[i].d;
    const uint16_t* __restrict q2 = x[i].qs;
    const int8_t* __restrict q8 = y[i].qs;
    __m256i sumi1 = _mm256_setzero_si256();
    __m256i sumi2 = _mm256_setzero_si256();
    for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
      const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8);
      q8 += 32;
      const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8);
      q8 += 32;
      memcpy(aux32, q2, 4 * sizeof(uint32_t));
      q2 += 8;
      const __m256i q2_1 =
          _mm256_set_epi64x(iq2xxs_grid[aux8[3]], iq2xxs_grid[aux8[2]], iq2xxs_grid[aux8[1]], iq2xxs_grid[aux8[0]]);
      const __m256i q2_2 =
          _mm256_set_epi64x(iq2xxs_grid[aux8[11]], iq2xxs_grid[aux8[10]], iq2xxs_grid[aux8[9]], iq2xxs_grid[aux8[8]]);
      const __m256i s2_1 = _mm256_set_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127],
                                             signs64[(aux32[1] >> 7) & 127], signs64[(aux32[1] >> 0) & 127]);
      const __m256i s2_2 = _mm256_set_epi64x(signs64[(aux32[3] >> 21) & 127], signs64[(aux32[3] >> 14) & 127],
                                             signs64[(aux32[3] >> 7) & 127], signs64[(aux32[3] >> 0) & 127]);
      const __m256i q8s_1 = _mm256_sign_epi8(q8_1, s2_1);
      const __m256i q8s_2 = _mm256_sign_epi8(q8_2, s2_2);
      const __m256i dot1 = _mm256_maddubs_epi16(q2_1, q8s_1);
      const __m256i dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);
      const uint16_t ls1 = aux32[1] >> 28;
      const uint16_t ls2 = aux32[3] >> 28;
      const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16(2 * ls1 + 1));
      const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16(2 * ls2 + 1));
      sumi1 = _mm256_add_epi32(sumi1, p1);
      sumi2 = _mm256_add_epi32(sumi2, p2);
    }

    accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
  }

  *s = 0.125f * hsum_float_8(accumf);

#elif defined(__AVX__)
  const uint64_t* signs64 = (const uint64_t*)keven_signs_q2xs;

  uint32_t aux32[4];
  const uint8_t* aux8 = (const uint8_t*)aux32;

  __m256 accumf = _mm256_setzero_ps();
  for (int i = 0; i < nb; ++i) {
    const float d = MLLM_FP16_TO_FP32(x[i].d) * y[i].d;
    const uint16_t* __restrict q2 = x[i].qs;
    const int8_t* __restrict q8 = y[i].qs;
    __m128i sumi1_0 = _mm_setzero_si128();
    __m128i sumi1_1 = _mm_setzero_si128();
    __m128i sumi2_0 = _mm_setzero_si128();
    __m128i sumi2_1 = _mm_setzero_si128();
    for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
      const __m128i q8_1_0 = _mm_loadu_si128((const __m128i*)q8);
      q8 += 16;
      const __m128i q8_1_1 = _mm_loadu_si128((const __m128i*)q8);
      q8 += 16;
      const __m128i q8_2_0 = _mm_loadu_si128((const __m128i*)q8);
      q8 += 16;
      const __m128i q8_2_1 = _mm_loadu_si128((const __m128i*)q8);
      q8 += 16;
      memcpy(aux32, q2, 4 * sizeof(uint32_t));
      q2 += 8;
      const __m128i q2_1_0 = _mm_set_epi64x(iq2xxs_grid[aux8[1]], iq2xxs_grid[aux8[0]]);
      const __m128i q2_1_1 = _mm_set_epi64x(iq2xxs_grid[aux8[3]], iq2xxs_grid[aux8[2]]);
      const __m128i q2_2_0 = _mm_set_epi64x(iq2xxs_grid[aux8[9]], iq2xxs_grid[aux8[8]]);
      const __m128i q2_2_1 = _mm_set_epi64x(iq2xxs_grid[aux8[11]], iq2xxs_grid[aux8[10]]);
      const __m128i s2_1_0 = _mm_set_epi64x(signs64[(aux32[1] >> 7) & 127], signs64[(aux32[1] >> 0) & 127]);
      const __m128i s2_1_1 = _mm_set_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127]);
      const __m128i s2_2_0 = _mm_set_epi64x(signs64[(aux32[3] >> 7) & 127], signs64[(aux32[3] >> 0) & 127]);
      const __m128i s2_2_1 = _mm_set_epi64x(signs64[(aux32[3] >> 21) & 127], signs64[(aux32[3] >> 14) & 127]);
      const __m128i q8s_1_0 = _mm_sign_epi8(q8_1_0, s2_1_0);
      const __m128i q8s_1_1 = _mm_sign_epi8(q8_1_1, s2_1_1);
      const __m128i q8s_2_0 = _mm_sign_epi8(q8_2_0, s2_2_0);
      const __m128i q8s_2_1 = _mm_sign_epi8(q8_2_1, s2_2_1);
      const __m128i dot1_0 = _mm_maddubs_epi16(q2_1_0, q8s_1_0);
      const __m128i dot1_1 = _mm_maddubs_epi16(q2_1_1, q8s_1_1);
      const __m128i dot2_0 = _mm_maddubs_epi16(q2_2_0, q8s_2_0);
      const __m128i dot2_1 = _mm_maddubs_epi16(q2_2_1, q8s_2_1);
      const uint16_t ls1 = aux32[1] >> 28;
      const uint16_t ls2 = aux32[3] >> 28;
      const __m128i p1_0 = _mm_madd_epi16(dot1_0, _mm_set1_epi16(2 * ls1 + 1));
      const __m128i p1_1 = _mm_madd_epi16(dot1_1, _mm_set1_epi16(2 * ls1 + 1));
      const __m128i p2_0 = _mm_madd_epi16(dot2_0, _mm_set1_epi16(2 * ls2 + 1));
      const __m128i p2_1 = _mm_madd_epi16(dot2_1, _mm_set1_epi16(2 * ls2 + 1));
      sumi1_0 = _mm_add_epi32(sumi1_0, p1_0);
      sumi1_1 = _mm_add_epi32(sumi1_1, p1_1);
      sumi2_0 = _mm_add_epi32(sumi2_0, p2_0);
      sumi2_1 = _mm_add_epi32(sumi2_1, p2_1);
    }

    accumf = _mm256_add_ps(
        _mm256_mul_ps(_mm256_set1_ps(d),
                      _mm256_cvtepi32_ps(MM256_SET_M128I(_mm_add_epi32(sumi1_1, sumi2_1), _mm_add_epi32(sumi1_0, sumi2_0)))),
        accumf);
  }

  *s = 0.125f * hsum_float_8(accumf);

#elif defined(__POWER9_VECTOR__)
  const vector int v0 = vec_splats((int32_t)0);
  vector float vsumf0 = vec_splats(0.0f);
  vector float vsumf1 = vec_splats(0.0f);
  vector float vsumf2 = vec_splats(0.0f);
  vector float vsumf3 = vec_splats(0.0f);

  const uint64_t* signs64 = (const uint64_t*)keven_signs_q2xs;

  for (int i = 0; i < nb; ++i) {
    vector float vxd = vec_splats(MLLM_FP16_TO_FP32(x[i].d));
    vector float vyd = vec_splats(y[i].d);
    vector float vd = vec_mul(vxd, vyd);

    vector signed int vsumi0 = v0;
    vector signed int vsumi1 = v0;
    vector signed int vsumi2 = v0;
    vector signed int vsumi3 = v0;

    const uint16_t* __restrict q2 = x[i].qs;
    const int8_t* __restrict q8 = y[i].qs;

    for (int j = 0; j < QK_K / 32; j += 2) {
      __builtin_prefetch(q2, 0, 1);
      __builtin_prefetch(q8, 0, 1);

      uint32_t aux32[4];
      const uint8_t* aux8 = (const uint8_t*)aux32;

      memcpy(aux32, q2, 4 * sizeof(uint32_t));
      q2 += 8;

      vector signed long long aux64x2_0 = {*(const int64_t*)(iq2xxs_grid + aux8[0]), *(const int64_t*)(iq2xxs_grid + aux8[1])};
      vector signed long long aux64x2_1 = {*(const int64_t*)(iq2xxs_grid + aux8[2]), *(const int64_t*)(iq2xxs_grid + aux8[3])};
      vector signed long long aux64x2_2 = {*(const int64_t*)(iq2xxs_grid + aux8[8]), *(const int64_t*)(iq2xxs_grid + aux8[9])};
      vector signed long long aux64x2_3 = {*(const int64_t*)(iq2xxs_grid + aux8[10]),
                                           *(const int64_t*)(iq2xxs_grid + aux8[11])};

      vector signed long long vsigns0 = {*(const int64_t*)(signs64 + ((aux32[1] >> 0) & 127)),
                                         *(const int64_t*)(signs64 + ((aux32[1] >> 7) & 127))};
      vector signed long long vsigns1 = {*(const int64_t*)(signs64 + ((aux32[1] >> 14) & 127)),
                                         *(const int64_t*)(signs64 + ((aux32[1] >> 21) & 127))};
      vector signed long long vsigns2 = {*(const int64_t*)(signs64 + ((aux32[3] >> 0) & 127)),
                                         *(const int64_t*)(signs64 + ((aux32[3] >> 7) & 127))};
      vector signed long long vsigns3 = {*(const int64_t*)(signs64 + ((aux32[3] >> 14) & 127)),
                                         *(const int64_t*)(signs64 + ((aux32[3] >> 21) & 127))};

      vector signed char q2x0 = (vector signed char)vec_mul((vector signed char)vsigns0, (vector signed char)aux64x2_0);
      vector signed char q2x1 = (vector signed char)vec_mul((vector signed char)vsigns1, (vector signed char)aux64x2_1);
      vector signed char q2x2 = (vector signed char)vec_mul((vector signed char)vsigns2, (vector signed char)aux64x2_2);
      vector signed char q2x3 = (vector signed char)vec_mul((vector signed char)vsigns3, (vector signed char)aux64x2_3);

      vector signed char q8y0 = vec_xl(0, q8);
      vector signed char q8y1 = vec_xl(16, q8);
      vector signed char q8y2 = vec_xl(32, q8);
      vector signed char q8y3 = vec_xl(48, q8);
      q8 += 64;

      vector signed short qv0 = vec_add(vec_mule(q2x0, q8y0), vec_mulo(q2x0, q8y0));
      vector signed short qv1 = vec_add(vec_mule(q2x1, q8y1), vec_mulo(q2x1, q8y1));
      vector signed short qv2 = vec_add(vec_mule(q2x2, q8y2), vec_mulo(q2x2, q8y2));
      vector signed short qv3 = vec_add(vec_mule(q2x3, q8y3), vec_mulo(q2x3, q8y3));

      const uint16_t ls0 = aux32[1] >> 28;
      const uint16_t ls1 = aux32[3] >> 28;

      vector signed short vscales01 = vec_splats((int16_t)(2 * ls0 + 1));
      vector signed short vscales23 = vec_splats((int16_t)(2 * ls1 + 1));

      vsumi0 = vec_msum(qv0, vscales01, vsumi0);
      vsumi1 = vec_msum(qv1, vscales01, vsumi1);
      vsumi2 = vec_msum(qv2, vscales23, vsumi2);
      vsumi3 = vec_msum(qv3, vscales23, vsumi3);
    }

    vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
    vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
    vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
  }

  vsumf0 = vec_add(vsumf0, vsumf2);
  vsumf1 = vec_add(vsumf1, vsumf3);

  vsumf0 = vec_add(vsumf0, vsumf1);

  vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
  vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

  *s = 0.125f * vec_extract(vsumf0, 0);

#elif defined(__loongarch_asx)

  const uint64_t* signs64 = (const uint64_t*)keven_signs_q2xs;

  uint32_t aux32[4];
  const uint8_t* aux8 = (const uint8_t*)aux32;

  __m256 accumf = (__m256)__lasx_xvldi(0);
  for (int i = 0; i < nb; ++i) {
    const float d = MLLM_FP16_TO_FP32(x[i].d) * y[i].d;
    const uint16_t* __restrict q2 = x[i].qs;
    const int8_t* __restrict q8 = y[i].qs;
    __m256i sumi1 = __lasx_xvldi(0);
    __m256i sumi2 = __lasx_xvldi(0);
    for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
      const __m256i q8_1 = __lasx_xvld((const __m256i*)q8, 0);
      q8 += 32;
      const __m256i q8_2 = __lasx_xvld((const __m256i*)q8, 0);
      q8 += 32;
      memcpy(aux32, q2, 4 * sizeof(uint32_t));
      q2 += 8;

      const __m256i q2_1 = lasx_set_d(iq2xxs_grid[aux8[3]], iq2xxs_grid[aux8[2]], iq2xxs_grid[aux8[1]], iq2xxs_grid[aux8[0]]);
      const __m256i q2_2 = lasx_set_d(iq2xxs_grid[aux8[11]], iq2xxs_grid[aux8[10]], iq2xxs_grid[aux8[9]], iq2xxs_grid[aux8[8]]);
      const __m256i s2_1 = lasx_set_d(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127],
                                      signs64[(aux32[1] >> 7) & 127], signs64[(aux32[1] >> 0) & 127]);
      const __m256i s2_2 = lasx_set_d(signs64[(aux32[3] >> 21) & 127], signs64[(aux32[3] >> 14) & 127],
                                      signs64[(aux32[3] >> 7) & 127], signs64[(aux32[3] >> 0) & 127]);
      const __m256i q8s_1 = __lasx_xvsigncov_b(s2_1, q8_1);
      const __m256i q8s_2 = __lasx_xvsigncov_b(s2_2, q8_2);
      const __m256i dot1 = lasx_maddubs_h(q2_1, q8s_1);
      const __m256i dot2 = lasx_maddubs_h(q2_2, q8s_2);
      const uint16_t ls1 = aux32[1] >> 28;
      const uint16_t ls2 = aux32[3] >> 28;
      const __m256i p1 = lasx_madd_h(dot1, __lasx_xvreplgr2vr_h(2 * ls1 + 1));
      const __m256i p2 = lasx_madd_h(dot2, __lasx_xvreplgr2vr_h(2 * ls2 + 1));
      sumi1 = __lasx_xvadd_w(sumi1, p1);
      sumi2 = __lasx_xvadd_w(sumi2, p2);
    }

    accumf = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(d), __lasx_xvffint_s_w(__lasx_xvadd_w(sumi1, sumi2)), accumf);
  }

  *s = 0.125f * hsum_float_8(accumf);
// #elif defined(__VXE__) || defined(__VXE2__)
//     const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;
//
//     uint32_t aux32[4];
//     const uint8_t * aux8 = (const uint8_t *)aux32;
//
//     float sumf = 0;
//
//     for (int i = 0; i < nb; ++i) {
//         const float d = MLLM_FP16_TO_FP32(x[i].d) * y[i].d;
//         const uint16_t * __restrict q2 = x[i].qs;
//         const int8_t   * __restrict q8 = y[i].qs;
//
//         float sumf1 = 0, sumf2 = 0;
//
//         for (int ib32 = 0; ib32 < QK_K/32; ib += 2) {
//             int8x16_t q8b0 = vec_xl( 0, q8);
//             int8x16_t qb81 = vec_xl(16, q8);
//             int8x16_t q8b2 = vec_xl(32, q8);
//             int8x16_t q8b3 = vec_xl(48, q8);
//             q8 += 64;
//
//             memcpy(aux32, q2, 4 * sizeof(uint32_t));
//             q2 += 8;
//
//             int8x16_t q2u0 = { *(const int64_t *)(iq2xxs_grid + aux8[ 0]), *(const int64_t *)(iq2xxs_grid + aux8[ 1]) };
//             int8x16_t q2u1 = { *(const int64_t *)(iq2xxs_grid + aux8[ 2]), *(const int64_t *)(iq2xxs_grid + aux8[ 3]) };
//             int8x16_t q2u2 = { *(const int64_t *)(iq2xxs_grid + aux8[ 8]), *(const int64_t *)(iq2xxs_grid + aux8[ 9]) };
//             int8x16_t q2u3 = { *(const int64_t *)(iq2xxs_grid + aux8[10]), *(const int64_t *)(iq2xxs_grid + aux8[11]) };
//
//             int8x16_t q2s0 = { *(const int64_t *)(signs64 + ((aux32[1] >>  0) & 127)), *(const int64_t *)(signs64 +
//             ((aux32[1] >>  7) & 127)) }; int8x16_t q2s1 = { *(const int64_t *)(signs64 + ((aux32[1] >> 14) & 127)), *(const
//             int64_t *)(signs64 + ((aux32[1] >> 21) & 127)) }; int8x16_t q2s2 = { *(const int64_t *)(signs64 + ((aux32[3] >>
//             0) & 127)), *(const int64_t *)(signs64 + ((aux32[3] >>  7) & 127)) }; int8x16_t q2s3 = { *(const int64_t
//             *)(signs64 + ((aux32[3] >> 14) & 127)), *(const int64_t *)(signs64 + ((aux32[3] >> 21) & 127)) };
//
//             q2u0 = vec_mul(q2u0, q2s0);
//             q2u1 = vec_mul(q2u1, q2s1);
//             q2u2 = vec_mul(q2u2, q2s2);
//             q2u3 = vec_mul(q2u3, q2s3);
//
//             const int32x4_t p1 = mllm_vec_dot(mllm_vec_dot(vec_splat_s32(0), q2u0, q8b0), q2u1, q8b1);
//             const int32x4_t p2 = mllm_vec_dot(mllm_vec_dot(vec_splat_s32(0), q2u2, q8b2), q2u3, q8b3);
//
//             sumf1 += (p1[0] + p1[1] + p1[2] + p1[3]) * (0.5f + (aux32[1] >> 28));
//             sumf2 += (p2[0] + p2[1] + p2[2] + p2[3]) * (0.5f + (aux32[3] >> 28));
//         }
//
//         sumf += d * (sumf1 + sumf2);
//     }
//
//     *s = 0.25f * sumf;
#else

  uint32_t aux32[2];
  const uint8_t* aux8 = (const uint8_t*)aux32;

  float sumf = 0.f;
  for (int i = 0; i < nb; ++i) {
    const float d = MLLM_FP16_TO_FP32(x[i].d) * y[i].d;
    const uint16_t* __restrict q2 = x[i].qs;
    const int8_t* __restrict q8 = y[i].qs;
    int32_t bsum = 0;
    for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
      memcpy(aux32, q2, 2 * sizeof(uint32_t));
      q2 += 4;
      const uint32_t ls = 2 * (aux32[1] >> 28) + 1;
      int32_t sumi = 0;
      for (int l = 0; l < 4; ++l) {
        const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + aux8[l]);
        const uint8_t signs = ksigns_iq2xs[(aux32[1] >> 7 * l) & 127];
        for (int j = 0; j < 8; ++j) { sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1); }
        q8 += 8;
      }
      bsum += sumi * ls;
    }
    sumf += d * bsum;
  }
  *s = 0.125f * sumf;
#endif
}

}  // namespace mllm::cpu::ggml
