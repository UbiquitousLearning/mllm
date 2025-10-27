#ifndef SAGE_ATTENTION_KVQ8_HPP
#define SAGE_ATTENTION_KVQ8_HPP

#include "Types.hpp"
#include <omp.h>
#include <vector>
#include <memory>
#include <algorithm>
#include <limits>
#include <cstring>

#ifdef __AVX2__
#include <immintrin.h>
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
#include <arm_neon.h>
#endif

namespace seq_attn_kvq8 {

const int QK_K_BLOCK_SIZE = QK8_0F;
#define NEG_INF std::numeric_limits<float>::lowest()

namespace { // 匿名空间，限制作用域
#ifdef __AVX2__
inline float _mm256_hmax_ps(__m256 x) {
    __m128 lo = _mm256_castps256_ps128(x);
    __m128 hi = _mm256_extractf128_ps(x, 1);
    __m128 max_val = _mm_max_ps(lo, hi);
    max_val = _mm_max_ps(max_val, _mm_shuffle_ps(max_val, max_val, _MM_SHUFFLE(0, 0, 2, 2)));
    max_val = _mm_max_ps(max_val, _mm_shuffle_ps(max_val, max_val, _MM_SHUFFLE(0, 0, 0, 1)));
    return _mm_cvtss_f32(max_val);
}

inline int32_t hsum_i32(__m256i v) {
    __m128i vlo = _mm256_castsi256_si128(v);
    __m128i vhi = _mm256_extracti128_si256(v, 1);
    __m128i vsum = _mm_add_epi32(vlo, vhi);
    vsum = _mm_add_epi32(vsum, _mm_shuffle_epi32(vsum, _MM_SHUFFLE(1, 0, 3, 2)));
    vsum = _mm_add_epi32(vsum, _mm_shuffle_epi32(vsum, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(vsum);
}
#endif // __AVX2__
} // namespace

#if (defined(__ARM_NEON) || defined(__ARM_NEON__))
inline float32x4_t exp_ps_f32(float32x4_t x) {
    float32x4_t a = vdupq_n_f32(12102203.0f);   // (1 << 23) / ln(2)
    float32x4_t b = vdupq_n_f32(1065353216.0f); // (1 << 23) * (0.5 - 0.04165) + (127 << 23)
    int32x4_t m = vdupq_n_s32(0x7f);
    float32x4_t y = vmlaq_f32(b, a, x);
    int32x4_t r = vreinterpretq_s32_f32(y);
    r = vandq_s32(r, vdupq_n_s32(0xffffff));
    r = vorrq_s32(r, vdupq_n_s32(0x3f800000));
    return vreinterpretq_f32_s32(r);
}
#endif

class WorkspaceManager {
public:
    WorkspaceManager() = default;
    ~WorkspaceManager() {
        for (auto &p : workspace_)
            if (p) free(p);
    }
    void **get_workspace(const std::vector<size_t> &s) {
        if (workspace_.empty()) {
            workspace_.resize(s.size(), nullptr);
            current_sizes_.resize(s.size(), 0);
        }
        for (size_t i = 0; i < s.size(); ++i) {
            if (s[i] > current_sizes_[i]) {
                if (workspace_[i]) free(workspace_[i]);
                posix_memalign(&workspace_[i], 64, s[i]);
                current_sizes_[i] = s[i];
            }
        }
        return workspace_.data();
    }

private:
    std::vector<void *> workspace_;
    std::vector<size_t> current_sizes_;
};

struct SAGE_CPU_IMPL_KVQ8 {
    using TQ = float;
    using TO = float;
    int32_t Br, Bc, Q_Head, KV_Head, threads;
    float *acc_o, *acc_s, *logsum, *scoremax, *scoremax_prev, *score_scale, *score_sum;
    int8_t *q_quant;
    float *q_scale, *q_scaled_buf;
    int8_t *p_quant;
    float *p_scale;

    void configure(int32_t Br_, int32_t Bc_, int32_t Q_H, int32_t KV_H, int32_t T) {
        Br = Br_;
        Bc = Bc_;
        Q_Head = Q_H;
        KV_Head = KV_H;
        threads = T;
    }

    void init_workspace(void **ws) {
        acc_o = static_cast<float *>(ws[0]);
        acc_s = static_cast<float *>(ws[1]);
        logsum = static_cast<float *>(ws[2]);
        scoremax = static_cast<float *>(ws[3]);
        scoremax_prev = static_cast<float *>(ws[4]);
        score_scale = static_cast<float *>(ws[5]);
        score_sum = static_cast<float *>(ws[6]);
        q_quant = static_cast<int8_t *>(ws[7]);
        q_scale = static_cast<float *>(ws[8]);
        q_scaled_buf = static_cast<float *>(ws[9]);
        p_quant = static_cast<int8_t *>(ws[10]);
        p_scale = static_cast<float *>(ws[11]);
    }

    void init_temp(float *l, float *sm, float *o, int Br_f, int D) {
        for (int i = 0; i < Br_f; ++i) {
            l[i] = 0.0f;
            sm[i] = -std::numeric_limits<float>::infinity();
        }
        if (o) memset(o, 0, Br_f * D * sizeof(float));
    }

    void quantize_q_row(const float *float_row, int8_t *int8_row, float *scales, int dim_size, float sm_scale, float *temp_buf) {
        const int num_groups = dim_size / QK_K_BLOCK_SIZE;
        for (int g = 0; g < num_groups; ++g) {
            const int group_start_idx = g * QK_K_BLOCK_SIZE;
            for (int d = 0; d < QK_K_BLOCK_SIZE; ++d) temp_buf[d] = float_row[group_start_idx + d] * sm_scale;
            float max_abs_val = 0.0f;
#if defined(__AVX2__)
            __m256 max_vec = _mm256_setzero_ps();
            const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
            int d = 0;
            for (; d <= QK_K_BLOCK_SIZE - 8; d += 8) max_vec = _mm256_max_ps(max_vec, _mm256_and_ps(_mm256_loadu_ps(temp_buf + d), abs_mask));
            max_abs_val = _mm256_hmax_ps(max_vec);
            for (; d < QK_K_BLOCK_SIZE; ++d) max_abs_val = std::max(max_abs_val, fabsf(temp_buf[d]));
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
            float32x4_t max_vec = vdupq_n_f32(0.0f);
            int d = 0;
            for (; d <= QK_K_BLOCK_SIZE - 4; d += 4) max_vec = vmaxq_f32(max_vec, vabsq_f32(vld1q_f32(temp_buf + d)));
            max_abs_val = vmaxvq_f32(max_vec);
            for (; d < QK_K_BLOCK_SIZE; ++d) max_abs_val = std::max(max_abs_val, fabsf(temp_buf[d]));
#else
            for (int d = 0; d < QK_K_BLOCK_SIZE; ++d) max_abs_val = std::max(max_abs_val, fabsf(temp_buf[d]));
#endif
            const float scale = (max_abs_val > 1e-9f) ? max_abs_val / 127.0f : 0.0f;
            scales[g] = scale;
            const float inv_scale = (scale > 1e-9f) ? 1.0f / scale : 0.0f;
            int8_t *group_int8_row = int8_row + group_start_idx;
#if defined(__AVX2__)
            __m256 inv_scale_vec = _mm256_set1_ps(inv_scale);
            d = 0;
            for (; d <= QK_K_BLOCK_SIZE - 8; d += 8) {
                __m256i val_i32 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(temp_buf + d), inv_scale_vec));
                __m128i val_i16 = _mm_packs_epi32(_mm256_castsi256_si128(val_i32), _mm256_extracti128_si256(val_i32, 1));
                __m128i val_i8 = _mm_packs_epi16(val_i16, val_i16);
                *(int64_t *)(group_int8_row + d) = _mm_cvtsi128_si64(val_i8);
            }
            for (; d < QK_K_BLOCK_SIZE; ++d) group_int8_row[d] = static_cast<int8_t>(roundf(temp_buf[d] * inv_scale));
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
            float32x4_t inv_scale_vec = vdupq_n_f32(inv_scale);
            d = 0;
            for (; d <= QK_K_BLOCK_SIZE - 16; d += 16) {
                int32x4_t i32_0 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(temp_buf + d + 0), inv_scale_vec));
                int32x4_t i32_1 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(temp_buf + d + 4), inv_scale_vec));
                int32x4_t i32_2 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(temp_buf + d + 8), inv_scale_vec));
                int32x4_t i32_3 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(temp_buf + d + 12), inv_scale_vec));
                int16x8_t i16_0 = vcombine_s16(vqmovn_s32(i32_0), vqmovn_s32(i32_1));
                int16x8_t i16_1 = vcombine_s16(vqmovn_s32(i32_2), vqmovn_s32(i32_3));
                vst1q_s8(group_int8_row + d, vcombine_s8(vqmovn_s16(i16_0), vqmovn_s16(i16_1)));
            }
            for (; d < QK_K_BLOCK_SIZE; ++d) group_int8_row[d] = static_cast<int8_t>(roundf(temp_buf[d] * inv_scale));
#else
            for (int d = 0; d < QK_K_BLOCK_SIZE; ++d) group_int8_row[d] = static_cast<int8_t>(roundf(temp_buf[d] * inv_scale));
#endif
        }
    }

    void mma0_sdot(int Br_f, int Bc_f, const int8_t *q_q_tile, const block_q8_0f *k_cache_tile, float *s, const float *q_s_tile, int D, int grs, int gcs, bool causal) {
#if defined(__AVX2__)
        const int num_k_blocks = D / QK_K_BLOCK_SIZE;
        for (int r = 0; r < Br_f; ++r) {
            for (int c = 0; c < Bc_f; ++c) {
                if (causal && (gcs + c) > (grs + r)) {
                    s[r * Bc + c] = NEG_INF;
                    continue;
                }
                const int8_t *q_quant_line = q_q_tile + r * D;
                const block_q8_0f *k_block_line = k_cache_tile + c * KV_Head * num_k_blocks;
                const float *q_scale_line = q_s_tile + r * num_k_blocks;
                float total_f32 = 0.0f;

                for (int g = 0; g < num_k_blocks; ++g) {
                    const int g_start = g * QK_K_BLOCK_SIZE;
                    const int8_t *q_group_ptr = q_quant_line + g_start;
                    const int8_t *k_group_ptr = k_block_line[g].qs;
                    const float k_group_scale = k_block_line[g].scale;
                    const float q_group_scale = q_scale_line[g];

                    int32_t g_dot = 0;
                    __m256i acc_i32_v = _mm256_setzero_si256();
                    int d = 0;
                    for (; d <= QK_K_BLOCK_SIZE - 16; d += 16) {
                        __m128i q_i8_v = _mm_loadu_si128((const __m128i *)(q_group_ptr + d));
                        __m128i k_i8_v = _mm_loadu_si128((const __m128i *)(k_group_ptr + d));
                        __m256i q_i16_v = _mm256_cvtepi8_epi16(q_i8_v);
                        __m256i k_i16_v = _mm256_cvtepi8_epi16(k_i8_v);
                        __m256i prod_i32_v = _mm256_madd_epi16(k_i16_v, q_i16_v);
                        acc_i32_v = _mm256_add_epi32(acc_i32_v, prod_i32_v);
                    }
                    g_dot = hsum_i32(acc_i32_v);
                    for (; d < QK_K_BLOCK_SIZE; ++d) g_dot += q_group_ptr[d] * k_group_ptr[d];

                    total_f32 += (float)g_dot * q_group_scale * k_group_scale;
                }
                s[r * Bc + c] = total_f32;
            }
        }

#elif (defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(__ARM_FEATURE_DOTPROD)
        const int num_k_blocks = D / QK_K_BLOCK_SIZE;

        for (int br_base = 0; br_base < Br_f; br_base += 4) {
            for (int bc_base = 0; bc_base < Bc_f; bc_base += 4) {
                int br_limit = std::min(4, Br_f - br_base);
                int bc_limit = std::min(4, Bc_f - bc_base);
                float accumulators[16] = {0.0f};

                for (int g = 0; g < num_k_blocks; ++g) {
                    const int g_start = g * QK_K_BLOCK_SIZE;
                    const int8_t *q_rows[4];
                    float q_scales[4];
                    for (int i = 0; i < br_limit; ++i) {
                        q_rows[i] = q_q_tile + (br_base + i) * D + g_start;
                        q_scales[i] = q_s_tile[(br_base + i) * num_k_blocks + g];
                    }

                    const int8_t *k_rows[4];
                    float k_scales[4];
                    for (int i = 0; i < bc_limit; ++i) {
                        k_rows[i] = k_cache_tile[(bc_base + i) * KV_Head * num_k_blocks + g].qs;
                        k_scales[i] = k_cache_tile[(bc_base + i) * KV_Head * num_k_blocks + g].scale;
                    }

                    for (int r_i = 0; r_i < br_limit; ++r_i) {
                        for (int c_i = 0; c_i < bc_limit; ++c_i) {
                            int32x4_t acc_s32 = vdupq_n_s32(0);
                            int d = 0;
                            for (; d <= QK_K_BLOCK_SIZE - 16; d += 16) {
                                int8x16_t q_s8 = vld1q_s8(q_rows[r_i] + d);
                                int8x16_t k_s8 = vld1q_s8(k_rows[c_i] + d);
                                acc_s32 = vdotq_s32(acc_s32, q_s8, k_s8);
                            }
                            int32_t dot_prod = vaddvq_s32(acc_s32);
                            for (; d < QK_K_BLOCK_SIZE; ++d) { dot_prod += q_rows[r_i][d] * k_rows[c_i][d]; }

                            accumulators[r_i * 4 + c_i] += (float)dot_prod * q_scales[r_i] * k_scales[c_i];
                        }
                    }
                }

                for (int r_i = 0; r_i < br_limit; ++r_i) {
                    for (int c_i = 0; c_i < bc_limit; ++c_i) {
                        const int gr = grs + br_base + r_i;
                        const int gc = gcs + bc_base + c_i;
                        if (causal && gc > gr) {
                            s[(br_base + r_i) * Bc + (bc_base + c_i)] = NEG_INF;
                        } else {
                            s[(br_base + r_i) * Bc + (bc_base + c_i)] = accumulators[r_i * 4 + c_i];
                        }
                    }
                }
            }
        }

#else
        const int num_k_blocks = D / QK_K_BLOCK_SIZE;
        for (int r = 0; r < Br_f; ++r) {
            for (int c = 0; c < Bc_f; ++c) {
                if (causal && (gcs + c) > (grs + r)) {
                    s[r * Bc + c] = NEG_INF;
                    continue;
                }
                const int8_t *q_quant_line = q_q_tile + r * D;
                const block_q8_0f *k_block_line = k_cache_tile + c * KV_Head * num_k_blocks;
                const float *q_scale_line = q_s_tile + r * num_k_blocks;
                float total_f32 = 0.0f;

                for (int g = 0; g < num_k_blocks; ++g) {
                    const int g_start = g * QK_K_BLOCK_SIZE;
                    const int8_t *q_group_ptr = q_quant_line + g_start;
                    const int8_t *k_group_ptr = k_block_line[g].qs;
                    const float k_group_scale = k_block_line[g].scale;
                    const float q_group_scale = q_scale_line[g];

                    int32_t g_dot = 0;
                    for (int d = 0; d < QK_K_BLOCK_SIZE; ++d) {
                        g_dot += q_group_ptr[d] * k_group_ptr[d];
                    }
                    total_f32 += (float)g_dot * q_group_scale * k_group_scale;
                }
                s[r * Bc + c] = total_f32;
            }
        }
#endif
    }

    void quantize_p_rows(int Br_f, int Bc_f, const float *p_float_block, int8_t *p_quant_block, float *p_scale_block) {
        for (int r = 0; r < Br_f; ++r) {
            const float *p_float_row = p_float_block + r * Bc;
            int8_t *p_quant_row = p_quant_block + r * Bc;

            float max_abs_val = 0.0f;
            for (int c = 0; c < Bc_f; ++c) {
                max_abs_val = std::max(max_abs_val, fabsf(p_float_row[c]));
            }

            const float scale = (max_abs_val > 1e-9f) ? max_abs_val / 127.0f : 0.0f;
            p_scale_block[r] = scale;
            const float inv_scale = (scale > 1e-9f) ? 1.0f / scale : 0.0f;

            for (int c = 0; c < Bc_f; ++c) {
                p_quant_row[c] = static_cast<int8_t>(roundf(p_float_row[c] * inv_scale));
            }
        }
    }

    void mma1(int Br_f, int Bc_f, const int8_t *p_quant_block, const float *p_scale_block, const block_q8_0f *v_cache_tile, float *acc_o, int D) {
#if defined(__AVX2__) && defined(__FMA__)
        const int num_v_blocks = D / QK_K_BLOCK_SIZE;
        for (int r = 0; r < Br_f; ++r) {
            const float p_row_scale = p_scale_block[r];
            if (fabsf(p_row_scale) < 1e-9) continue;

            float *o_row = acc_o + r * D;
            const int8_t *p_quant_row = p_quant_block + r * Bc;

            for (int c = 0; c < Bc_f; ++c) {
                const int8_t p_quant_scalar = p_quant_row[c];
                if (p_quant_scalar == 0) continue;

                const float p_dequant_val = (float)p_quant_scalar * p_row_scale;
                const block_q8_0f *v_block_line = v_cache_tile + c * KV_Head * num_v_blocks;

                const __m256 p_vec = _mm256_set1_ps(p_dequant_val);
                for (int g = 0; g < num_v_blocks; ++g) {
                    const int g_start = g * QK_K_BLOCK_SIZE;
                    const float v_scale_scalar = v_block_line[g].scale;
                    const __m256 v_scale_vec = _mm256_set1_ps(v_scale_scalar);
                    for (int d_group = 0; d_group < QK_K_BLOCK_SIZE; d_group += 8) {
                        const int d = g_start + d_group;
                        __m128i v_i8_vec_part = _mm_loadl_epi64((const __m128i *)(v_block_line[g].qs + d_group));
                        __m256i v_i32_vec = _mm256_cvtepi8_epi32(v_i8_vec_part);
                        __m256 v_f32_vec = _mm256_cvtepi32_ps(v_i32_vec);
                        __m256 dequant_v_vec = _mm256_mul_ps(v_f32_vec, v_scale_vec);
                        __m256 o_vec = _mm256_loadu_ps(o_row + d);
                        o_vec = _mm256_fmadd_ps(p_vec, dequant_v_vec, o_vec);
                        _mm256_storeu_ps(o_row + d, o_vec);
                    }
                }
            }
        }

#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
        const int num_v_blocks = D / QK_K_BLOCK_SIZE;

        for (int r = 0; r < Br_f; ++r) {
            float *o_row = acc_o + r * D;
            const float p_row_scale = p_scale_block[r];
            if (fabsf(p_row_scale) < 1e-9) continue;

            for (int g = 0; g < num_v_blocks; ++g) {
                const int g_start = g * QK_K_BLOCK_SIZE;

                for (int d = 0; d < QK_K_BLOCK_SIZE; d += 8) {
                    float32x4_t acc0 = vld1q_f32(o_row + g_start + d);
                    float32x4_t acc1 = vld1q_f32(o_row + g_start + d + 4);
                    for (int c = 0; c < Bc_f; ++c) {
                        const int8_t p_quant_scalar = p_quant_block[r * Bc + c];
                        if (p_quant_scalar == 0) continue;

                        const float32x4_t p_vec = vdupq_n_f32((float)p_quant_scalar * p_row_scale);

                        const block_q8_0f *v_block = v_cache_tile + c * KV_Head * num_v_blocks + g;
                        const float32x4_t v_scale_vec = vdupq_n_f32(v_block->scale);
                        const int8_t *v_qs = v_block->qs + d;

                        int8x8_t v_s8 = vld1_s8(v_qs);
                        int16x8_t v_s16 = vmovl_s8(v_s8);

                        float32x4_t v_f32_lo = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_s16))), v_scale_vec);
                        float32x4_t v_f32_hi = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_s16))), v_scale_vec);

                        acc0 = vfmaq_f32(acc0, p_vec, v_f32_lo);
                        acc1 = vfmaq_f32(acc1, p_vec, v_f32_hi);
                    }

                    vst1q_f32(o_row + g_start + d, acc0);
                    vst1q_f32(o_row + g_start + d + 4, acc1);
                }
            }
        }

#else
        const int num_v_blocks = D / QK_K_BLOCK_SIZE;
        for (int r = 0; r < Br_f; ++r) {
            const float p_row_scale = p_scale_block[r];
            if (fabsf(p_row_scale) < 1e-9) continue;

            float *o_row = acc_o + r * D;
            const int8_t *p_quant_row = p_quant_block + r * Bc;

            for (int c = 0; c < Bc_f; ++c) {
                const int8_t p_quant_scalar = p_quant_row[c];
                if (p_quant_scalar == 0) continue;

                const float p_dequant_val = (float)p_quant_scalar * p_row_scale;
                const block_q8_0f *v_block_line = v_cache_tile + c * KV_Head * num_v_blocks;

                for (int g = 0; g < num_v_blocks; ++g) {
                    const int g_start = g * QK_K_BLOCK_SIZE;
                    const float v_s = v_block_line[g].scale;
                    const int8_t *v_qs = v_block_line[g].qs;
                    for (int d = 0; d < QK_K_BLOCK_SIZE; ++d) {
                        o_row[g_start + d] += p_dequant_val * ((float)v_qs[d] * v_s);
                    }
                }
            }
        }
#endif
    }

    void softmax(int Br_f, int Bc_f, float *acc_s, float *sm, float *sm_p, float *ss, float *sum, float *l) {
        memcpy(sm_p, sm, Br_f * sizeof(float));
        for (int r = 0; r < Br_f; ++r) {
            float *row = acc_s + r * Bc, cmax = sm[r];
            for (int c = 0; c < Bc_f; ++c) cmax = std::max(cmax, row[c]);
            sm[r] = cmax;
        }
        for (int r = 0; r < Br_f; ++r) ss[r] = expf(sm_p[r] - sm[r]);
        for (int r = 0; r < Br_f; ++r) {
            float *row = acc_s + r * Bc;
            float smax = sm[r], s = 0.f;
            for (int c = 0; c < Bc_f; ++c) row[c] = (row[c] > NEG_INF / 2) ? (s += row[c] = expf(row[c] - smax), row[c]) : 0.f;
            sum[r] = s;
        }
        for (int r = 0; r < Br_f; ++r) l[r] = l[r] * ss[r] + sum[r];
    }

    /*
    void softmax(int Br_f, int Bc_f, float *acc_s, float *sm, float *sm_p, float *ss, float *sum, float *l) {
        memcpy(sm_p, sm, Br_f * sizeof(float));

        for (int r = 0; r < Br_f; ++r) {
            float *row = acc_s + r * Bc;
            float cmax = sm[r];
#if (defined(__ARM_NEON) || defined(__ARM_NEON__))
            float32x4_t max_vec = vdupq_n_f32(cmax);
            int c = 0;
            for (; c <= Bc_f - 4; c += 4) {
                max_vec = vmaxq_f32(max_vec, vld1q_f32(row + c));
            }
            cmax = vmaxvq_f32(max_vec);
            for (; c < Bc_f; ++c) cmax = std::max(cmax, row[c]);
#else
            for (int c = 0; c < Bc_f; ++c) cmax = std::max(cmax, row[c]);
#endif
            sm[r] = cmax;
        }

        for (int r = 0; r < Br_f; ++r) ss[r] = expf(sm_p[r] - sm[r]);

        for (int r = 0; r < Br_f; ++r) {
            float *row = acc_s + r * Bc;
            float smax = sm[r];
            float current_sum = 0.f;
#if (defined(__ARM_NEON) || defined(__ARM_NEON__))
            float32x4_t sum_vec = vdupq_n_f32(0.0f);
            const float32x4_t smax_vec = vdupq_n_f32(smax);
            int c = 0;
            for (; c <= Bc_f - 4; c += 4) {
                float32x4_t val_vec = vld1q_f32(row + c);
                val_vec = vsubq_f32(val_vec, smax_vec);
                val_vec = exp_ps_f32(val_vec); // 使用快速exp
                vst1q_f32(row + c, val_vec);
                sum_vec = vaddq_f32(sum_vec, val_vec);
            }
            current_sum = vaddvq_f32(sum_vec);
            for (; c < Bc_f; ++c)
                if (row[c] > NEG_INF / 2)
                    current_sum += row[c] = expf(row[c] - smax);
                else
                    row[c] = 0.f;
#else
            for (int c = 0; c < Bc_f; ++c)
                if (row[c] > NEG_INF / 2)
                    current_sum += row[c] = expf(row[c] - smax);
                else
                    row[c] = 0.f;
#endif
            sum[r] = current_sum;
        }
        for (int r = 0; r < Br_f; ++r) l[r] = l[r] * ss[r] + sum[r];
    }
    */

    void rescale(int Br_f, float *acc_o, const float *ss, int D) {
        for (int r = 0; r < Br_f; ++r) {
            float s_val = ss[r], *r_ptr = acc_o + r * D;
#if defined(__AVX2__)
            __m256 s_vec = _mm256_set1_ps(s_val);
            int d = 0;
            for (; d <= D - 8; d += 8) _mm256_storeu_ps(r_ptr + d, _mm256_mul_ps(_mm256_loadu_ps(r_ptr + d), s_vec));
            for (; d < D; ++d) r_ptr[d] *= s_val;
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
            float32x4_t s_vec = vdupq_n_f32(s_val);
            int d = 0;
            for (; d <= D - 4; d += 4) vst1q_f32(r_ptr + d, vmulq_f32(vld1q_f32(r_ptr + d), s_vec));
            for (; d < D; ++d) r_ptr[d] *= s_val;
#else
            for (int d = 0; d < D; ++d) r_ptr[d] *= s_val;
#endif
        }
    }

    void scale_and_store(int Br_f, const float *acc_o, const float *logsum, const float *v_mean, TO *O, int H, int D) {
        int o_stride = H * D;
        for (int r = 0; r < Br_f; ++r) {
            float inv_logsum = (logsum[r] > 1e-9f) ? 1.f / logsum[r] : 0.f;
            const float *o_row = acc_o + r * D;
            float *O_row = O + (size_t)r * o_stride;
#if defined(__AVX2__) && defined(__FMA__)
            const __m256 inv_l_vec = _mm256_set1_ps(inv_logsum);
            int d = 0;
            for (; d <= D - 8; d += 8) {
                const __m256 o_vec = _mm256_loadu_ps(o_row + d);
                const __m256 vm_vec = _mm256_loadu_ps(v_mean + d);
                _mm256_storeu_ps(O_row + d, _mm256_fmadd_ps(o_vec, inv_l_vec, vm_vec));
            }
            for (; d < D; ++d) O_row[d] = o_row[d] * inv_logsum + v_mean[d];
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
            const float32x4_t inv_l_vec = vdupq_n_f32(inv_logsum);
            int d = 0;
            for (; d <= D - 4; d += 4) {
                const float32x4_t o_vec = vld1q_f32(o_row + d);
                const float32x4_t vm_vec = vld1q_f32(v_mean + d);
                vst1q_f32(O_row + d, vfmaq_f32(vm_vec, o_vec, inv_l_vec));
            }
            for (; d < D; ++d) O_row[d] = o_row[d] * inv_logsum + v_mean[d];
#else
            for (int d = 0; d < D; ++d) O_row[d] = o_row[d] * inv_logsum + v_mean[d];
#endif
        }
    }

    void sage_attn_prefill(const TQ *Q, const block_q8_0f *K_cache, const block_q8_0f *V_cache, TO *O, const float *K_mean, const float *V_mean, int32_t batch_size, int32_t head_size, int32_t seq_size_q, int32_t seq_size_k, int32_t dim_size, bool causal) {
        const int32_t Tr = (seq_size_q + Br - 1) / Br, Tc = (seq_size_k + Bc - 1) / Bc;
        const float local_scale = 1.0f / sqrtf(static_cast<float>(dim_size));
        const int32_t kv_group = (Q_Head > 0 && KV_Head > 0) ? Q_Head / KV_Head : 1;
        const int32_t num_k_blocks = dim_size / QK_K_BLOCK_SIZE;

#pragma omp parallel for num_threads(threads) schedule(dynamic, 1)
        for (int32_t b_h = 0; b_h < batch_size * head_size; ++b_h) {
            int b = b_h / head_size;
            int h = b_h % head_size;
            const int32_t tid = omp_get_thread_num();
            const int32_t kvh = h / kv_group;
            float *po = acc_o + tid * Br * dim_size, *ps = acc_s + tid * Br * Bc;
            float *plog = logsum + tid * Br, *pmax = scoremax + tid * Br, *pmax_p = scoremax_prev + tid * Br;
            float *pscale = score_scale + tid * Br, *psum = score_sum + tid * Br;
            int8_t *p_q_q = q_quant + tid * Br * dim_size;
            float *p_q_s = q_scale + tid * Br * num_k_blocks;
            float *p_q_scaled = q_scaled_buf + tid * dim_size;

            int8_t *p_p_q = p_quant + tid * Br * Bc;
            float *p_p_s = p_scale + tid * Br;

            const float *p_V_m = V_mean + (b * KV_Head + kvh) * dim_size;

            for (int32_t tr = 0; tr < Tr; ++tr) {
                int32_t Br_f = std::min(Br, seq_size_q - tr * Br);
                init_temp(plog, pmax, po, Br_f, dim_size);
                const TQ *tile_q_bshd = Q + (size_t)b * seq_size_q * head_size * dim_size + (size_t)tr * Br * head_size * dim_size + (size_t)h * dim_size;

                for (int r = 0; r < Br_f; ++r) {
                    quantize_q_row(tile_q_bshd + r * head_size * dim_size, p_q_q + r * dim_size, p_q_s + r * num_k_blocks, dim_size, local_scale, p_q_scaled);
                }

                for (int32_t tc = 0; tc < Tc; ++tc) {
                    int32_t Bc_f = std::min(Bc, seq_size_k - tc * Bc);
                    const int kv_offset = seq_size_k - seq_size_q;

                    const block_q8_0f *k_cache_tile = K_cache + ((size_t)b * seq_size_k * KV_Head + (tc * Bc) * KV_Head + kvh) * num_k_blocks;
                    const block_q8_0f *v_cache_tile = V_cache + ((size_t)b * seq_size_k * KV_Head + (tc * Bc) * KV_Head + kvh) * num_k_blocks;

                    mma0_sdot(Br_f, Bc_f, p_q_q, k_cache_tile, ps, p_q_s, dim_size, tr * Br + kv_offset, tc * Bc, causal);
                    softmax(Br_f, Bc_f, ps, pmax, pmax_p, pscale, psum, plog);
                    rescale(Br_f, po, pscale, dim_size);

                    quantize_p_rows(Br_f, Bc_f, ps, p_p_q, p_p_s);

                    mma1(Br_f, Bc_f, p_p_q, p_p_s, v_cache_tile, po, dim_size);
                }
                TO *tile_o_bshd = O + (size_t)b * seq_size_q * head_size * dim_size + (size_t)tr * Br * head_size * dim_size + (size_t)h * dim_size;
                scale_and_store(Br_f, po, plog, p_V_m, tile_o_bshd, head_size, dim_size);
            }
        }
    }
    void sage_attn_decode(const TQ *Q, const block_q8_0f *K_cache, const block_q8_0f *V_cache, TO *O, const float *K_mean, const float *V_mean, int32_t batch_size, int32_t head_size, int32_t seq_size_k, int32_t dim_size, bool causal) {
        const int32_t Tc = (seq_size_k + Bc - 1) / Bc;
        const float local_scale = 1.0f / sqrtf(static_cast<float>(dim_size));
        const int32_t kv_group = (Q_Head > 0 && KV_Head > 0) ? Q_Head / KV_Head : 1;
        const int32_t num_k_blocks = dim_size / QK_K_BLOCK_SIZE;

#pragma omp parallel for num_threads(threads) schedule(dynamic, 1)
        for (int32_t b_h = 0; b_h < batch_size * head_size; ++b_h) {
            int b = b_h / head_size;
            int h = b_h % head_size;
            const int32_t Br_f = 1;
            const int32_t tid = omp_get_thread_num();
            const int32_t kvh = h / kv_group;
            float *po = acc_o + tid * Br_f * dim_size, *ps = acc_s + tid * Br_f * Bc;
            float *plog = logsum + tid * Br_f, *pmax = scoremax + tid * Br_f, *pmax_p = scoremax_prev + tid * Br_f;
            float *pscale = score_scale + tid * Br_f, *psum = score_sum + tid * Br_f;
            int8_t *p_q_q = q_quant + tid * Br_f * dim_size;
            float *p_q_s = q_scale + tid * Br_f * num_k_blocks;
            float *p_q_scaled = q_scaled_buf + tid * dim_size;
            const float *p_V_m = V_mean + (b * KV_Head + kvh) * dim_size;

            int8_t *p_p_q = p_quant + tid * Br_f * Bc;
            float *p_p_s = p_scale + tid * Br_f;

            const TQ *tile_q_decode = Q + (size_t)b * head_size * dim_size + (size_t)h * dim_size;
            quantize_q_row(tile_q_decode, p_q_q, p_q_s, dim_size, local_scale, p_q_scaled);
            init_temp(plog, pmax, po, Br_f, dim_size);

            for (int32_t tc = 0; tc < Tc; ++tc) {
                int32_t Bc_f = std::min(Bc, seq_size_k - tc * Bc);
                const block_q8_0f *k_cache_tile = K_cache + ((size_t)b * seq_size_k * KV_Head + (tc * Bc) * KV_Head + kvh) * num_k_blocks;
                const block_q8_0f *v_cache_tile = V_cache + ((size_t)b * seq_size_k * KV_Head + (tc * Bc) * KV_Head + kvh) * num_k_blocks;
                mma0_sdot(Br_f, Bc_f, p_q_q, k_cache_tile, ps, p_q_s, dim_size, seq_size_k - 1, tc * Bc, causal);
                softmax(Br_f, Bc_f, ps, pmax, pmax_p, pscale, psum, plog);
                rescale(Br_f, po, pscale, dim_size);

                quantize_p_rows(Br_f, Bc_f, ps, p_p_q, p_p_s);

                mma1(Br_f, Bc_f, p_p_q, p_p_s, v_cache_tile, po, dim_size);
            }
            TO *tile_o_bshd = O + (size_t)b * head_size * dim_size + (size_t)h * dim_size;
            scale_and_store(Br_f, po, plog, p_V_m, tile_o_bshd, head_size, dim_size);
        }
    }
};

inline void sage_attention_forward_cpu_dispatch(
    const float *Q, const void *K_in, const void *V_in, const float *K_mean_ext,
    const float *V_mean_ext, float *O, int32_t batch_size, int32_t q_head,
    int32_t kv_head, int32_t seq_size_q, int32_t seq_size_k, int32_t dim_size,
    bool causal_mask, int32_t threads, int32_t br, int32_t bc) {
    if (dim_size % QK_K_BLOCK_SIZE != 0) {
        std::cerr << "Error: dim_size must be divisible by QK_K_BLOCK_SIZE" << std::endl;
        return;
    }
    const int32_t num_k_blocks = dim_size / QK_K_BLOCK_SIZE;

    thread_local WorkspaceManager manager;
    SAGE_CPU_IMPL_KVQ8 op;
    op.configure(br, bc, q_head, kv_head, threads);

    const int32_t current_br = (seq_size_q > 1) ? br : 1;
    const std::vector<size_t> ws_sizes = {
        (size_t)threads * current_br * dim_size * sizeof(float),     // 0: acc_o
        (size_t)threads * current_br * bc * sizeof(float),           // 1: acc_s
        (size_t)threads * current_br * sizeof(float),                // 2: logsum
        (size_t)threads * current_br * sizeof(float),                // 3: scoremax
        (size_t)threads * current_br * sizeof(float),                // 4: scoremax_prev
        (size_t)threads * current_br * sizeof(float),                // 5: score_scale
        (size_t)threads * current_br * sizeof(float),                // 6: score_sum
        (size_t)threads * current_br * dim_size * sizeof(int8_t),    // 7: q_quant
        (size_t)threads * current_br * num_k_blocks * sizeof(float), // 8: q_scale
        (size_t)threads * dim_size * sizeof(float),                  // 9: q_scaled_buf (for quantize_q_row)
        (size_t)threads * current_br * bc * sizeof(int8_t),          // 10: p_quant [NEW]
        (size_t)threads * current_br * sizeof(float),                // 11: p_scale [NEW]
    };
    void **workspace = manager.get_workspace(ws_sizes);
    op.init_workspace(workspace);

    if (seq_size_q > 1) {
        op.sage_attn_prefill(Q, (const block_q8_0f *)K_in, (const block_q8_0f *)V_in, O, K_mean_ext, V_mean_ext, batch_size, q_head,
                             seq_size_q, seq_size_k, dim_size, causal_mask);
    } else {
        op.sage_attn_decode(Q, (const block_q8_0f *)K_in, (const block_q8_0f *)V_in, O, K_mean_ext, V_mean_ext, batch_size, q_head,
                            seq_size_k, dim_size, causal_mask);
    }
}

} // namespace seq_attn_kvq8
#endif // SAGE_ATTENTION_KVQ8_HPP