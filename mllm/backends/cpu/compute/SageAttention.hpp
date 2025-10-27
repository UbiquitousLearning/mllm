#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>
#include <cstdint>
#include <limits>
#include <cstring>
#include <omp.h>
#include <algorithm>
#include <type_traits>

// --- SIMD Intrinsics ---
#ifdef __AVX2__
#include <immintrin.h>
#include <immintrin.h>
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
#include <arm_neon.h>
#if defined(__ARM_FP16_FORMAT_IEEE) && !defined(_MSC_VER)
#include <arm_fp16.h>
#endif
#endif

#include "Types.hpp"
#include "backends/cpu/third_party/ggml/QuantizeFP16.hpp"
#include "../compute/SageQuantize.hpp"

#define SAGE_V_I8

#ifdef SAGE_V_I8

namespace sage_attn_cpu {
const int QK_K_BLOCK_SIZE = QK8_0F;
#define NEG_INF std::numeric_limits<float>::lowest()

template <typename T>
inline float to_float(T val);
template <>
inline float to_float<float>(float val) {
    return val;
}
template <>
inline float to_float<mllm_fp16_t>(mllm_fp16_t val) {
    return MLLM_FP16_TO_FP32(val);
}

#if defined(_WIN32)
#include <malloc.h>
inline void aligned_alloc(void **ptr, size_t r, size_t a) {
    *ptr = _aligned_malloc(r, a);
}
inline void aligned_free(void *ptr) {
    _aligned_free(ptr);
}
#else
inline void aligned_alloc(void **ptr, size_t r, size_t a) {
    if (a % sizeof(void *) != 0 || (a & (a - 1)) != 0 || posix_memalign(ptr, a, r) != 0) *ptr = nullptr;
}
inline void aligned_free(void *ptr) {
    free(ptr);
}
#endif

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
inline __m256 load_and_convert_to_fp32_vec(const float *ptr) {
    return _mm256_loadu_ps(ptr);
}
#ifdef __F16C__
inline __m256 load_and_convert_to_fp32_vec(const mllm_fp16_t *ptr) {
    return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)ptr));
}
#endif
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
inline void load_and_convert_to_fp32x4x2(const float *ptr, float32x4_t &out_lo, float32x4_t &out_hi) {
    out_lo = vld1q_f32(ptr);
    out_hi = vld1q_f32(ptr + 4);
}
#if defined(__ARM_FP16_FORMAT_IEEE)
inline void load_and_convert_to_fp32x4x2(const mllm_fp16_t *ptr, float32x4_t &out_lo, float32x4_t &out_hi) {
    float16x8_t v_f16 = vld1q_f16(reinterpret_cast<const float16_t *>(ptr));
    out_lo = vcvt_f32_f16(vget_low_f16(v_f16));
    out_hi = vcvt_f32_f16(vget_high_f16(v_f16));
}
#endif
#endif

void quantize_row_per_group_simd(const float *float_row, int8_t *int8_row, float *scales, int dim_size, float sm_scale, float *temp_buf) {
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

template <typename T>
void compute_mean_and_quantize_tensor(
    const T *tensor_bshd, float *mean_tensor_bhd, int8_t *quant_global_bhsd,
    float *scale_global_bhsn, int batch_size, int head_size, int seq_size,
    int dim_size, int threads, float *temp_sum, float *temp_smoothed,
    float *temp_head_buffer) {
#pragma omp parallel for num_threads(threads) collapse(2)
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < head_size; ++h) {
            const int thread_id = omp_get_thread_num();
            float *thread_sum_buf = temp_sum + thread_id * dim_size;
            float *thread_smoothed_buf = temp_smoothed + thread_id * dim_size;
            float *thread_head_buf_bhsd = temp_head_buffer + thread_id * seq_size * dim_size;

            float *target_mean = mean_tensor_bhd + (b * head_size + h) * dim_size;
            const int num_blocks = dim_size / QK_K_BLOCK_SIZE;
            int8_t *target_quant_bhsd = quant_global_bhsd + (b * head_size + h) * seq_size * dim_size;
            float *target_scale_bhsn = scale_global_bhsn + (b * head_size + h) * seq_size * num_blocks;

            memset(thread_sum_buf, 0, dim_size * sizeof(float));

            for (int s = 0; s < seq_size; ++s) {
                const T *row_global_bshd = tensor_bshd + (size_t)b * seq_size * head_size * dim_size + (size_t)s * head_size * dim_size + (size_t)h * dim_size;
                float *row_buffered_bhsd = thread_head_buf_bhsd + s * dim_size;
#if defined(__AVX2__)
                int d = 0;
                for (; d <= dim_size - 8; d += 8) {
                    const __m256 val_vec = load_and_convert_to_fp32_vec(row_global_bshd + d);
                    _mm256_storeu_ps(row_buffered_bhsd + d, val_vec);
                    __m256 sum_vec = _mm256_loadu_ps(thread_sum_buf + d);
                    sum_vec = _mm256_add_ps(sum_vec, val_vec);
                    _mm256_storeu_ps(thread_sum_buf + d, sum_vec);
                }
                for (; d < dim_size; ++d) {
                    float val = to_float(row_global_bshd[d]);
                    row_buffered_bhsd[d] = val;
                    thread_sum_buf[d] += val;
                }
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
                int d = 0;
                for (; d <= dim_size - 8; d += 8) {
                    float32x4_t val_lo, val_hi;
                    load_and_convert_to_fp32x4x2(row_global_bshd + d, val_lo, val_hi);
                    vst1q_f32(row_buffered_bhsd + d, val_lo);
                    vst1q_f32(row_buffered_bhsd + d + 4, val_hi);
                    float32x4_t sum_lo = vld1q_f32(thread_sum_buf + d);
                    float32x4_t sum_hi = vld1q_f32(thread_sum_buf + d + 4);
                    vst1q_f32(thread_sum_buf + d, vaddq_f32(sum_lo, val_lo));
                    vst1q_f32(thread_sum_buf + d + 4, vaddq_f32(sum_hi, val_hi));
                }
                for (; d < dim_size; ++d) {
                    float val = to_float(row_global_bshd[d]);
                    row_buffered_bhsd[d] = val;
                    thread_sum_buf[d] += val;
                }
#else
                for (int d = 0; d < dim_size; ++d) {
                    float val = to_float(row_global_bshd[d]);
                    row_buffered_bhsd[d] = val;
                    thread_sum_buf[d] += val;
                }
#endif
            }

            const float inv_seq_len = 1.0f / seq_size;
#if defined(__AVX2__)
            const __m256 inv_len_vec = _mm256_set1_ps(inv_seq_len);
            int d = 0;
            for (; d <= dim_size - 8; d += 8) _mm256_storeu_ps(target_mean + d, _mm256_mul_ps(_mm256_loadu_ps(thread_sum_buf + d), inv_len_vec));
            for (; d < dim_size; ++d) target_mean[d] = thread_sum_buf[d] * inv_seq_len;
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
            const float32x4_t inv_len_vec = vdupq_n_f32(inv_seq_len);
            int d = 0;
            for (; d <= dim_size - 4; d += 4) vst1q_f32(target_mean + d, vmulq_f32(vld1q_f32(thread_sum_buf + d), inv_len_vec));
            for (; d < dim_size; ++d) target_mean[d] = thread_sum_buf[d] * inv_seq_len;
#else
            for (int d = 0; d < dim_size; ++d) target_mean[d] = thread_sum_buf[d] * inv_seq_len;
#endif

            for (int s = 0; s < seq_size; ++s) {
                const float *row_buffered_bhsd = thread_head_buf_bhsd + s * dim_size;
#if defined(__AVX2__)
                int d = 0;
                for (; d <= dim_size - 8; d += 8) _mm256_storeu_ps(thread_smoothed_buf + d, _mm256_sub_ps(_mm256_loadu_ps(row_buffered_bhsd + d), _mm256_loadu_ps(target_mean + d)));
                for (; d < dim_size; ++d) thread_smoothed_buf[d] = row_buffered_bhsd[d] - target_mean[d];
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
                int d = 0;
                for (; d <= dim_size - 4; d += 4) vst1q_f32(thread_smoothed_buf + d, vsubq_f32(vld1q_f32(row_buffered_bhsd + d), vld1q_f32(target_mean + d)));
                for (; d < dim_size; ++d) thread_smoothed_buf[d] = row_buffered_bhsd[d] - target_mean[d];
#else
                for (int d = 0; d < dim_size; ++d) thread_smoothed_buf[d] = row_buffered_bhsd[d] - target_mean[d];
#endif
                quantize_row_per_group_simd(thread_smoothed_buf, target_quant_bhsd + s * dim_size, target_scale_bhsn + s * num_blocks, dim_size, 1.0f, thread_sum_buf);
            }
        }
    }
}

class WorkspaceManager {
public:
    WorkspaceManager() = default;
    ~WorkspaceManager() {
        for (auto &p : workspace_)
            if (p) aligned_free(p);
    }
    void **get_workspace(const std::vector<size_t> &s) {
        if (workspace_.empty()) {
            workspace_.resize(s.size(), nullptr);
            current_sizes_.resize(s.size(), 0);
        }
        for (size_t i = 0; i < s.size(); ++i) {
            if (s[i] > current_sizes_[i]) {
                if (workspace_[i]) aligned_free(workspace_[i]);
                aligned_alloc(&workspace_[i], s[i], 64);
                current_sizes_[i] = s[i];
            }
        }
        return workspace_.data();
    }

private:
    std::vector<void *> workspace_;
    std::vector<size_t> current_sizes_;
};

template <typename KVDtype>
struct SAGE_CPU_IMPL {
    using TQ = float;
    using TKV = KVDtype;
    using TO = float;
    int32_t Br, Bc, Q_Head, KV_Head, threads;
    float *acc_o, *acc_s, *logsum, *scoremax, *scoremax_prev, *score_scale, *score_sum;
    int8_t *q_quant, *k_quant_global, *v_quant_global;
    float *q_scale, *k_scale_global, *v_scale_global, *k_smoothed_buf, *q_scaled_buf;

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
        k_quant_global = static_cast<int8_t *>(ws[8]);
        q_scale = static_cast<float *>(ws[9]);
        k_scale_global = static_cast<float *>(ws[10]);
        k_smoothed_buf = static_cast<float *>(ws[11]);
        q_scaled_buf = static_cast<float *>(ws[12]);
        v_quant_global = static_cast<int8_t *>(ws[15]);
        v_scale_global = static_cast<float *>(ws[16]);

        p_quant = static_cast<int8_t *>(ws[18]);
        p_scale = static_cast<float *>(ws[19]);
    }

    void init_temp(float *l, float *sm, float *o, int Br_f, int D) {
        for (int i = 0; i < Br_f; ++i) {
            l[i] = 0.0f;
            sm[i] = -std::numeric_limits<float>::infinity();
        }
        if (o) memset(o, 0, Br_f * D * sizeof(float));
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

    void sage_attn_prefill(const TQ *Q, TO *O, const float *K_mean, const float *V_mean, int32_t batch_size, int32_t head_size, int32_t seq_size_q, int32_t seq_size_k, int32_t dim_size, bool causal) {
        const int32_t Tr = (seq_size_q + Br - 1) / Br, Tc = (seq_size_k + Bc - 1) / Bc;
        const float local_scale = 1.0f / sqrtf(static_cast<float>(dim_size));
        const int32_t kv_group = (Q_Head > 0 && KV_Head > 0) ? Q_Head / KV_Head : 1;
        const int32_t num_k_blocks = dim_size / QK_K_BLOCK_SIZE;
        for (int32_t b = 0; b < batch_size; ++b) {
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1)
            for (int32_t h = 0; h < head_size; ++h) {
                const int32_t tid = omp_get_thread_num();
                const int32_t kvh = h / kv_group;
                float *po = acc_o + tid * Br * dim_size, *ps = acc_s + tid * Br * Bc;
                float *plog = logsum + tid * Br, *pmax = scoremax + tid * Br, *pmax_p = scoremax_prev + tid * Br;
                float *pscale = score_scale + tid * Br, *psum = score_sum + tid * Br;
                int8_t *p_q_q = q_quant + tid * Br * dim_size;
                const int8_t *p_k_q_g = k_quant_global + (b * KV_Head + kvh) * seq_size_k * dim_size;
                const int8_t *p_v_q_g = v_quant_global + (b * KV_Head + kvh) * seq_size_k * dim_size;
                float *p_q_s = q_scale + tid * Br * num_k_blocks;
                const float *p_k_s_g = k_scale_global + (b * KV_Head + kvh) * seq_size_k * num_k_blocks;
                const float *p_v_s_g = v_scale_global + (b * KV_Head + kvh) * seq_size_k * num_k_blocks;
                float *p_q_scaled = q_scaled_buf + tid * dim_size;
                const float *p_V_m = V_mean + (b * KV_Head + kvh) * dim_size;

                int8_t *p_p_q = p_quant + tid * Br * Bc;
                float *p_p_s = p_scale + tid * Br;

                for (int32_t tr = 0; tr < Tr; ++tr) {
                    int32_t Br_f = std::min(Br, seq_size_q - tr * Br);
                    init_temp(plog, pmax, po, Br_f, dim_size);
                    const TQ *tile_q_bshd = Q + (size_t)b * seq_size_q * head_size * dim_size + (size_t)tr * Br * head_size * dim_size + (size_t)h * dim_size;
                    for (int r = 0; r < Br_f; ++r) quantize_row_per_group_simd(tile_q_bshd + (size_t)r * head_size * dim_size, p_q_q + r * dim_size, p_q_s + r * num_k_blocks, dim_size, local_scale, p_q_scaled);
                    for (int32_t tc = 0; tc < Tc; ++tc) {
                        int32_t Bc_f = std::min(Bc, seq_size_k - tc * Bc);
                        const int kv_offset = seq_size_k - seq_size_q;
                        quantize_and_mma0_sdot(Br_f, Bc_f, p_q_q, p_k_q_g + tc * Bc * dim_size, ps, p_q_s, p_k_s_g + tc * Bc * num_k_blocks, dim_size, tr * Br + kv_offset, tc * Bc, causal);
                        softmax(Br_f, Bc_f, ps, pmax, pmax_p, pscale, psum, plog);
                        rescale(Br_f, po, pscale, dim_size);

                        quantize_p_rows(Br_f, Bc_f, ps, p_p_q, p_p_s);

                        const int8_t *v_q = p_v_q_g + tc * Bc * dim_size;
                        const float *v_s = p_v_s_g + tc * Bc * num_k_blocks;

                        mma1(Br_f, Bc_f, p_p_q, p_p_s, v_q, v_s, po, dim_size);
                    }
                    TO *tile_o_bshd = O + (size_t)b * seq_size_q * head_size * dim_size + (size_t)tr * Br * head_size * dim_size + (size_t)h * dim_size;
                    scale_and_store(Br_f, po, plog, p_V_m, tile_o_bshd, head_size, dim_size);
                }
            }
        }
    }
    void sage_attn_decode(const TQ *Q, TO *O, const float *K_mean, const float *V_mean, int32_t batch_size, int32_t head_size, int32_t seq_size_k, int32_t dim_size, bool causal) {
        const int32_t Tc = (seq_size_k + Bc - 1) / Bc;
        const float local_scale = 1.0f / sqrtf(static_cast<float>(dim_size));
        const int32_t kv_group = (Q_Head > 0 && KV_Head > 0) ? Q_Head / KV_Head : 1;
        const int32_t num_k_blocks = dim_size / QK_K_BLOCK_SIZE;
#pragma omp parallel for num_threads(threads) collapse(2)
        for (int32_t b = 0; b < batch_size; ++b) {
            for (int32_t h = 0; h < head_size; ++h) {
                const int32_t Br_f = 1;
                const int32_t tid = omp_get_thread_num();
                const int32_t kvh = h / kv_group;
                float *po = acc_o + tid * Br_f * dim_size, *ps = acc_s + tid * Br_f * Bc;
                float *plog = logsum + tid * Br_f, *pmax = scoremax + tid * Br_f, *pmax_p = scoremax_prev + tid * Br_f;
                float *pscale = score_scale + tid * Br_f, *psum = score_sum + tid * Br_f;
                int8_t *p_q_q = q_quant + tid * Br_f * dim_size;
                const int8_t *p_k_q_g = k_quant_global + (b * KV_Head + kvh) * seq_size_k * dim_size;
                const int8_t *p_v_q_g = v_quant_global + (b * KV_Head + kvh) * seq_size_k * dim_size;
                float *p_q_s = q_scale + tid * Br_f * num_k_blocks;
                const float *p_k_s_g = k_scale_global + (b * KV_Head + kvh) * seq_size_k * num_k_blocks;
                const float *p_v_s_g = v_scale_global + (b * KV_Head + kvh) * seq_size_k * num_k_blocks;
                float *p_q_scaled = q_scaled_buf + tid * dim_size;
                const float *p_V_m = V_mean + (b * KV_Head + kvh) * dim_size;

                // [新增] 获取量化P矩阵的工作区指针
                int8_t *p_p_q = p_quant + tid * Br_f * Bc;
                float *p_p_s = p_scale + tid * Br_f;

                const TQ *tile_q_bshd = Q + (size_t)b * 1 * head_size * dim_size + (size_t)0 * head_size * dim_size + (size_t)h * dim_size;
                quantize_row_per_group_simd(tile_q_bshd, p_q_q, p_q_s, dim_size, local_scale, p_q_scaled);
                init_temp(plog, pmax, po, Br_f, dim_size);
                for (int32_t tc = 0; tc < Tc; ++tc) {
                    int32_t Bc_f = std::min(Bc, seq_size_k - tc * Bc);
                    quantize_and_mma0_sdot(Br_f, Bc_f, p_q_q, p_k_q_g + tc * Bc * dim_size, ps, p_q_s, p_k_s_g + tc * Bc * num_k_blocks, dim_size, seq_size_k - 1, tc * Bc, causal);
                    softmax(Br_f, Bc_f, ps, pmax, pmax_p, pscale, psum, plog);
                    rescale(Br_f, po, pscale, dim_size);

                    quantize_p_rows(Br_f, Bc_f, ps, p_p_q, p_p_s);

                    const int8_t *v_q = p_v_q_g + tc * Bc * dim_size;
                    const float *v_s = p_v_s_g + tc * Bc * num_k_blocks;

                    mma1(Br_f, Bc_f, p_p_q, p_p_s, v_q, v_s, po, dim_size);
                }
                TO *tile_o_bshd = O + (size_t)b * 1 * head_size * dim_size + (size_t)0 * head_size * dim_size + (size_t)h * dim_size;
                scale_and_store(Br_f, po, plog, p_V_m, tile_o_bshd, head_size, dim_size);
            }
        }
    }
    void quantize_and_mma0_sdot(int Br_f, int Bc_f, const int8_t *q_q, const int8_t *k_q, float *s, const float *q_s, const float *k_s, int D, int grs, int gcs, bool causal) {
        const int num_k_blocks = D / QK_K_BLOCK_SIZE;
        for (int r = 0; r < Br_f; ++r)
            for (int c = 0; c < Bc_f; ++c) {
                if (causal && (gcs + c) > (grs + r)) {
                    s[r * Bc + c] = NEG_INF;
                    continue;
                }
                const int8_t *q_ql = q_q + r * D, *k_ql = k_q + c * D;
                const float *q_sl = q_s + r * num_k_blocks, *k_sl = k_s + c * num_k_blocks;
                float total_f32 = 0.0f;
                for (int g = 0; g < num_k_blocks; ++g) {
                    const int g_start = g * QK_K_BLOCK_SIZE;
                    int32_t g_dot = 0;
#if defined(__AVX2__)
                    __m256i acc_i32_v = _mm256_setzero_si256();
                    int d = 0;
                    for (; d <= QK_K_BLOCK_SIZE - 16; d += 16) {
                        __m128i q_i8_v = _mm_loadu_si128((const __m128i *)(q_ql + g_start + d));
                        __m128i k_i8_v = _mm_loadu_si128((const __m128i *)(k_ql + g_start + d));
                        __m256i q_i16_v = _mm256_cvtepi8_epi16(q_i8_v);
                        __m256i k_i16_v = _mm256_cvtepi8_epi16(k_i8_v);
                        __m256i prod_i32_v = _mm256_madd_epi16(k_i16_v, q_i16_v);
                        acc_i32_v = _mm256_add_epi32(acc_i32_v, prod_i32_v);
                    }
                    g_dot = hsum_i32(acc_i32_v);
                    for (; d < QK_K_BLOCK_SIZE; ++d) g_dot += (q_ql + g_start)[d] * (k_ql + g_start)[d];
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(__ARM_FEATURE_DOTPROD)
                    int32x4_t acc_i32_vec = vdupq_n_s32(0);
                    int d = 0;
                    for (; d <= QK_K_BLOCK_SIZE - 16; d += 16) acc_i32_vec = vdotq_s32(acc_i32_vec, vld1q_s8(q_ql + g_start + d), vld1q_s8(k_ql + g_start + d));
                    g_dot = vaddvq_s32(acc_i32_vec);
                    for (; d < QK_K_BLOCK_SIZE; ++d) g_dot += (q_ql + g_start)[d] * (k_ql + g_start)[d];
#else
                    for (int d = 0; d < QK_K_BLOCK_SIZE; ++d) g_dot += (q_ql + g_start)[d] * (k_ql + g_start)[d];
#endif
                    total_f32 += (float)g_dot * q_sl[g] * k_sl[g];
                }
                s[r * Bc + c] = total_f32;
            }
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

    void mma1(int Br_f, int Bc_f, const int8_t *p_quant_block, const float *p_scale_block, const int8_t *v_quant_block, const float *v_scale_block, float *acc_o, int D) {
        const int num_v_blocks = D / QK_K_BLOCK_SIZE;
        for (int r = 0; r < Br_f; ++r) {
            const float p_row_scale = p_scale_block[r];
            if (fabsf(p_row_scale) < 1e-9) continue;

            const int8_t *p_quant_row = p_quant_block + r * Bc;
            float *o_row = acc_o + r * D;

            for (int c = 0; c < Bc_f; ++c) {
                const int8_t p_quant_scalar = p_quant_row[c];
                if (p_quant_scalar == 0) continue;

                const float p_dequant_val = (float)p_quant_scalar * p_row_scale;

                const int8_t *v_q_row = v_quant_block + c * D;
                const float *v_s_row = v_scale_block + c * num_v_blocks;

#if defined(__AVX2__) && defined(__FMA__)
                const __m256 p_vec = _mm256_set1_ps(p_dequant_val);
                for (int g = 0; g < num_v_blocks; ++g) {
                    const int g_start = g * QK_K_BLOCK_SIZE;
                    const __m256 v_scale_vec = _mm256_set1_ps(v_s_row[g]);
                    for (int d_group = 0; d_group < QK_K_BLOCK_SIZE; d_group += 8) {
                        const int d = g_start + d_group;
                        __m128i v_i8_vec_part = _mm_loadl_epi64((const __m128i *)(v_q_row + d));
                        __m256i v_i32_vec = _mm256_cvtepi8_epi32(v_i8_vec_part);
                        __m256 v_f32_vec = _mm256_cvtepi32_ps(v_i32_vec);
                        __m256 dequant_v_vec = _mm256_mul_ps(v_f32_vec, v_scale_vec);
                        __m256 o_vec = _mm256_loadu_ps(o_row + d);
                        o_vec = _mm256_fmadd_ps(p_vec, dequant_v_vec, o_vec);
                        _mm256_storeu_ps(o_row + d, o_vec);
                    }
                }
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
                const float32x4_t p_vec = vdupq_n_f32(p_dequant_val);
                for (int g = 0; g < num_v_blocks; ++g) {
                    const int g_start = g * QK_K_BLOCK_SIZE;
                    const float32x4_t v_scale_vec = vdupq_n_f32(v_s_row[g]);
                    int d = 0;
                    for (; d <= QK_K_BLOCK_SIZE - 8; d += 8) {
                        int8x8_t v_i8 = vld1_s8(v_q_row + g_start + d);
                        int16x8_t v_i16 = vmovl_s8(v_i8);
                        int32x4_t v_i32_lo = vmovl_s16(vget_low_s16(v_i16));
                        int32x4_t v_i32_hi = vmovl_s16(vget_high_s16(v_i16));
                        float32x4_t v_f32_lo = vcvtq_f32_s32(v_i32_lo);
                        float32x4_t v_f32_hi = vcvtq_f32_s32(v_i32_hi);
                        v_f32_lo = vmulq_f32(v_f32_lo, v_scale_vec);
                        v_f32_hi = vmulq_f32(v_f32_hi, v_scale_vec);
                        float32x4_t o_f32_lo = vld1q_f32(o_row + g_start + d);
                        float32x4_t o_f32_hi = vld1q_f32(o_row + g_start + d + 4);
                        // FMA: O += P * V_dequant
                        o_f32_lo = vfmaq_f32(o_f32_lo, p_vec, v_f32_lo);
                        o_f32_hi = vfmaq_f32(o_f32_hi, p_vec, v_f32_hi);
                        vst1q_f32(o_row + g_start + d, o_f32_lo);
                        vst1q_f32(o_row + g_start + d + 4, o_f32_hi);
                    }
                    for (; d < QK_K_BLOCK_SIZE; ++d) {
                        o_row[g_start + d] += p_dequant_val * ((float)v_q_row[g_start + d] * v_s_row[g]);
                    }
                }
#else
                for (int g = 0; g < num_v_blocks; ++g) {
                    const int g_start = g * QK_K_BLOCK_SIZE;
                    const float v_s = v_s_row[g];
                    for (int d = 0; d < QK_K_BLOCK_SIZE; ++d) {
                        o_row[g_start + d] += p_dequant_val * ((float)v_q_row[g_start + d] * v_s);
                    }
                }
#endif
            }
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
};

template <typename KVDtype>
void sage_attention_forward_cpu_dispatch(
    const float *Q, const void *K_in, const void *V_in, const float *K_mean_ext,
    const float *V_mean_ext, float *O, int32_t batch_size, int32_t q_head,
    int32_t kv_head, int32_t seq_size_q, int32_t seq_size_k, int32_t dim_size,
    bool causal_mask, int32_t threads, int32_t br, int32_t bc,
    int32_t cache_stride_s) {
    if (dim_size % QK_K_BLOCK_SIZE != 0) {
        std::cerr << "Error: dim_size must be divisible by QK_K_BLOCK_SIZE\n";
        return;
    }
    const int32_t num_k_blocks = dim_size / QK_K_BLOCK_SIZE;

    thread_local WorkspaceManager manager;
    SAGE_CPU_IMPL<KVDtype> op;
    op.configure(br, bc, q_head, kv_head, threads);

    const int32_t current_br = (seq_size_q > 1) ? br : 1;

    const std::vector<size_t> ws_sizes = {
        (size_t)threads * current_br * dim_size * sizeof(float),                  // 0: acc_o
        (size_t)threads * current_br * bc * sizeof(float),                        // 1: acc_s
        (size_t)threads * current_br * sizeof(float),                             // 2: logsum
        (size_t)threads * current_br * sizeof(float),                             // 3: scoremax
        (size_t)threads * current_br * sizeof(float),                             // 4: scoremax_prev
        (size_t)threads * current_br * sizeof(float),                             // 5: score_scale
        (size_t)threads * current_br * sizeof(float),                             // 6: score_sum
        (size_t)threads * current_br * dim_size * sizeof(int8_t),                 // 7: q_quant
        (size_t)batch_size * kv_head * seq_size_k * dim_size * sizeof(int8_t),    // 8: k_quant_global
        (size_t)threads * current_br * num_k_blocks * sizeof(float),              // 9: q_scale
        (size_t)batch_size * kv_head * seq_size_k * num_k_blocks * sizeof(float), // 10: k_scale_global
        (size_t)threads * dim_size * sizeof(float),                               // 11: k_smoothed_buf
        (size_t)threads * dim_size * sizeof(float),                               // 12: q_scaled_buf
        (size_t)threads * dim_size * sizeof(float),                               // 13: temp_k_sum
        (size_t)threads * seq_size_k * dim_size * sizeof(float),                  // 14: temp_k_head_buffer
        (size_t)batch_size * kv_head * seq_size_k * dim_size * sizeof(int8_t),    // 15: v_quant_global
        (size_t)batch_size * kv_head * seq_size_k * num_k_blocks * sizeof(float), // 16: v_scale_global
        (size_t)threads * seq_size_k * dim_size * sizeof(float),                  // 17: temp_v_head_buffer
        (size_t)threads * current_br * bc * sizeof(int8_t),                       // 18: p_quant [新增]
        (size_t)threads * current_br * sizeof(float)                              // 19: p_scale [新增]
    };
    void **workspace = manager.get_workspace(ws_sizes);
    op.init_workspace(workspace);

    if constexpr (std::is_same_v<KVDtype, block_q8_0f>) {
        const auto *k_blocks_bhsd = reinterpret_cast<const block_q8_0f *>(K_in);
        const auto *v_blocks_bhsd = reinterpret_cast<const block_q8_0f *>(V_in);

#pragma omp parallel for collapse(3) num_threads(threads)
        for (int b = 0; b < batch_size; ++b) {
            for (int h = 0; h < kv_head; ++h) {
                for (int s = 0; s < seq_size_k; ++s) {
                    size_t sparse_offset = ((size_t)b * kv_head + h) * cache_stride_s + s;
                    const block_q8_0f *k_block_src = k_blocks_bhsd + sparse_offset * num_k_blocks;
                    const block_q8_0f *v_block_src = v_blocks_bhsd + sparse_offset * num_k_blocks;

                    size_t dense_offset = ((size_t)b * kv_head + h) * seq_size_k + s;
                    int8_t *k_quant_dest = op.k_quant_global + dense_offset * dim_size;
                    float *k_scale_dest = op.k_scale_global + dense_offset * num_k_blocks;
                    int8_t *v_quant_dest = op.v_quant_global + dense_offset * dim_size;
                    float *v_scale_dest = op.v_scale_global + dense_offset * num_k_blocks;

                    for (int g = 0; g < num_k_blocks; ++g) {
                        k_scale_dest[g] = k_block_src[g].scale;
                        memcpy(k_quant_dest + g * QK8_0F, k_block_src[g].qs, QK8_0F);
                        v_scale_dest[g] = v_block_src[g].scale;
                        memcpy(v_quant_dest + g * QK8_0F, v_block_src[g].qs, QK8_0F);
                    }
                }
            }
        }
        if (seq_size_q > 1) {
            op.sage_attn_prefill(Q, O, K_mean_ext, V_mean_ext, batch_size, q_head,
                                 seq_size_q, seq_size_k, dim_size, causal_mask);
        } else {
            op.sage_attn_decode(Q, O, K_mean_ext, V_mean_ext, batch_size, q_head,
                                seq_size_k, dim_size, causal_mask);
        }
    } else {
        std::vector<float> K_mean_internal((size_t)batch_size * kv_head * dim_size);
        std::vector<float> V_mean_internal((size_t)batch_size * kv_head * dim_size);
        const auto *K = static_cast<const KVDtype *>(K_in);
        const auto *V = static_cast<const KVDtype *>(V_in);
        compute_mean_and_quantize_tensor<KVDtype>(
            K, K_mean_internal.data(), op.k_quant_global, op.k_scale_global,
            batch_size, kv_head, seq_size_k, dim_size, threads,
            (float *)workspace[13], (float *)workspace[11], (float *)workspace[14]);
        compute_mean_and_quantize_tensor<KVDtype>(
            V, V_mean_internal.data(), op.v_quant_global, op.v_scale_global,
            batch_size, kv_head, seq_size_k, dim_size, threads,
            (float *)workspace[13], (float *)workspace[11], (float *)workspace[17]);

        if (seq_size_q > 1) {
            op.sage_attn_prefill(Q, O, K_mean_internal.data(), V_mean_internal.data(),
                                 batch_size, q_head, seq_size_q, seq_size_k, dim_size,
                                 causal_mask);
        } else {
            op.sage_attn_decode(Q, O, K_mean_internal.data(), V_mean_internal.data(),
                                batch_size, q_head, seq_size_k, dim_size,
                                causal_mask);
        }
    }
}
} // namespace sage_attn_cpu

#else
namespace sage_attn_cpu {
const int QK_K_BLOCK_SIZE = 128;
#define NEG_INF std::numeric_limits<float>::lowest()

template <typename T>
inline float to_float(T val);
template <>
inline float to_float<float>(float val) {
    return val;
}
template <>
inline float to_float<mllm_fp16_t>(mllm_fp16_t val) {
    return MLLM_FP16_TO_FP32(val);
}

// ======================= SIMD HELPERS =======================
#ifdef __AVX2__
inline float _mm256_hmax_ps(__m256 x) {
    __m128 lo = _mm256_castps256_ps128(x);
    __m128 hi = _mm256_extractf128_ps(x, 1);
    __m128 max_val = _mm_max_ps(lo, hi);
    max_val = _mm_max_ps(max_val, _mm_shuffle_ps(max_val, max_val, _MM_SHUFFLE(0, 0, 2, 2)));
    max_val = _mm_max_ps(max_val, _mm_shuffle_ps(max_val, max_val, _MM_SHUFFLE(0, 0, 0, 1)));
    return _mm_cvtss_f32(max_val);
}
inline float hsum_ps(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}
inline int32_t hsum_i32(__m256i v) {
    __m128i vlo = _mm256_castsi256_si128(v);
    __m128i vhi = _mm256_extracti128_si256(v, 1);
    __m128i vsum = _mm_add_epi32(vlo, vhi);
    vsum = _mm_add_epi32(vsum, _mm_shuffle_epi32(vsum, _MM_SHUFFLE(1, 0, 3, 2)));
    vsum = _mm_add_epi32(vsum, _mm_shuffle_epi32(vsum, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(vsum);
}

inline __m256 load_and_convert_to_fp32_vec(const float *ptr) {
    return _mm256_loadu_ps(ptr);
}
#ifdef __F16C__
inline __m256 load_and_convert_to_fp32_vec(const mllm_fp16_t *ptr) {
    return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)ptr));
}
#endif
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
inline float _vmaxvq_f32_hmax(float32x4_t x) {
    return vmaxvq_f32(x);
}
inline void load_and_convert_to_fp32x4x2(const float *ptr, float32x4_t &out_lo, float32x4_t &out_hi) {
    out_lo = vld1q_f32(ptr);
    out_hi = vld1q_f32(ptr + 4);
}
#if defined(__ARM_FP16_FORMAT_IEEE)
inline void load_and_convert_to_fp32x4x2(const mllm_fp16_t *ptr, float32x4_t &out_lo, float32x4_t &out_hi) {
    float16x8_t v_f16 = vld1q_f16(reinterpret_cast<const float16_t *>(ptr));
    out_lo = vcvt_f32_f16(vget_low_f16(v_f16));
    out_hi = vcvt_f32_f16(vget_high_f16(v_f16));
}
#endif
#endif

inline void aligned_alloc(void **ptr, size_t required_bytes, size_t align) {
    if (align % sizeof(void *) != 0 || (align & (align - 1)) != 0) {
        *ptr = nullptr;
        return;
    }
    if (posix_memalign(ptr, align, required_bytes) != 0) { *ptr = nullptr; }
}
inline void aligned_free(void *ptr) {
    free(ptr);
}

inline void quantize_row_per_group_simd(const float *float_row, int8_t *int8_row, float *scales, int dim_size, float sm_scale, float *temp_buf) {
    const int num_groups = dim_size / QK_K_BLOCK_SIZE;

    for (int g = 0; g < num_groups; ++g) {
        const int group_start_idx = g * QK_K_BLOCK_SIZE;
        const float *group_float_row = float_row + group_start_idx;
        float *group_temp_buf = temp_buf; // reuse the same temp buffer

        // Apply softmax scale if needed (only for Q)
        for (int d = 0; d < QK_K_BLOCK_SIZE; ++d) {
            group_temp_buf[d] = group_float_row[d] * sm_scale;
        }

        float max_abs_val = 0.0f;
#if defined(__AVX2__)
        __m256 max_vec = _mm256_setzero_ps();
        const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
        int d = 0;
        for (; d <= QK_K_BLOCK_SIZE - 8; d += 8) {
            max_vec = _mm256_max_ps(max_vec, _mm256_and_ps(_mm256_loadu_ps(group_temp_buf + d), abs_mask));
        }
        max_abs_val = _mm256_hmax_ps(max_vec);
        for (; d < QK_K_BLOCK_SIZE; ++d) max_abs_val = std::max(max_abs_val, fabsf(group_temp_buf[d]));
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
        float32x4_t max_vec = vdupq_n_f32(0.0f);
        int d = 0;
        for (; d <= QK_K_BLOCK_SIZE - 4; d += 4) {
            max_vec = vmaxq_f32(max_vec, vabsq_f32(vld1q_f32(group_temp_buf + d)));
        }
        max_abs_val = vmaxvq_f32(max_vec);
        for (; d < QK_K_BLOCK_SIZE; ++d) max_abs_val = std::max(max_abs_val, fabsf(group_temp_buf[d]));
#else
        for (int d = 0; d < QK_K_BLOCK_SIZE; ++d) max_abs_val = std::max(max_abs_val, fabsf(group_temp_buf[d]));
#endif

        const float scale = (max_abs_val > 1e-9f) ? max_abs_val / 127.0f : 0.0f;
        scales[g] = scale;
        const float inv_scale = (scale > 1e-9f) ? 1.0f / scale : 0.0f;

        int8_t *group_int8_row = int8_row + group_start_idx;

#if defined(__AVX2__)
        __m256 inv_scale_vec = _mm256_set1_ps(inv_scale);
        for (int d = 0; d <= QK_K_BLOCK_SIZE - 8; d += 8) {
            __m256i val_i32 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(group_temp_buf + d), inv_scale_vec));
            __m128i val_i16 = _mm_packs_epi32(_mm256_castsi256_si128(val_i32), _mm256_extracti128_si256(val_i32, 1));
            __m128i val_i8 = _mm_packs_epi16(val_i16, val_i16);
            *(int64_t *)(group_int8_row + d) = _mm_cvtsi128_si64(val_i8);
        }
        for (; d < QK_K_BLOCK_SIZE; ++d) group_int8_row[d] = static_cast<int8_t>(roundf(group_temp_buf[d] * inv_scale));
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
        float32x4_t inv_scale_vec = vdupq_n_f32(inv_scale);
        d = 0;
        for (; d <= QK_K_BLOCK_SIZE - 16; d += 16) {
            int32x4_t i32_0 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(group_temp_buf + d + 0), inv_scale_vec));
            int32x4_t i32_1 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(group_temp_buf + d + 4), inv_scale_vec));
            int32x4_t i32_2 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(group_temp_buf + d + 8), inv_scale_vec));
            int32x4_t i32_3 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(group_temp_buf + d + 12), inv_scale_vec));
            int16x8_t i16_0 = vcombine_s16(vqmovn_s32(i32_0), vqmovn_s32(i32_1));
            int16x8_t i16_1 = vcombine_s16(vqmovn_s32(i32_2), vqmovn_s32(i32_3));
            vst1q_s8(group_int8_row + d, vcombine_s8(vqmovn_s16(i16_0), vqmovn_s16(i16_1)));
        }
        for (; d < QK_K_BLOCK_SIZE; ++d) group_int8_row[d] = static_cast<int8_t>(roundf(group_temp_buf[d] * inv_scale));
#else
        for (int d = 0; d < QK_K_BLOCK_SIZE; ++d) group_int8_row[d] = static_cast<int8_t>(roundf(group_temp_buf[d] * inv_scale));
#endif
    }
}

template <typename KVDtype>
void compute_channel_means(const KVDtype *tensor, float *mean_tensor, int batch_size, int head_size, int seq_len, int dim_size) {
#pragma omp parallel for collapse(3)
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < head_size; ++h) {
            for (int d = 0; d < dim_size; d += 8) { // AVX2 processes 8 floats at a time
#if (defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(__ARM_FP16_FORMAT_IEEE)
                float32x4_t sum_vec = vdupq_n_f32(0.0f);
                for (int s = 0; s < seq_len; ++s) {
                    int idx = b * (seq_len * head_size * dim_size) + s * (head_size * dim_size) + h * (dim_size) + d;
                    float32x4_t val_vec;
                    if constexpr (std::is_same_v<KVDtype, mllm_fp16_t>) {
                        val_vec = vcvt_f32_f16(vld1_f16(reinterpret_cast<const float16_t *>(tensor + idx)));
                    } else {
                        val_vec = vld1q_f32(tensor + idx);
                    }
                    sum_vec = vaddq_f32(sum_vec, val_vec);
                }
                float inv_seq_len = 1.0f / seq_len;
                float32x4_t mean_vec = vmulq_f32(sum_vec, vdupq_n_f32(inv_seq_len));
                // Note: Original code used d+=4, but NEON loop processed only 4 floats.
                // Assuming it should be d+=4 for the NEON part. Let's stick to the original logic.
                vst1q_f32(mean_tensor + b * head_size * dim_size + h * dim_size + d, mean_vec);
#elif defined(__AVX2__)
                // =========== AVX2 IMPLEMENTATION START ===========
                if (d + 8 > dim_size) { // Handle remainder
                    for (int i = 0; i < (dim_size - d); ++i) {
                        double sum = 0.0;
                        for (int s = 0; s < seq_len; ++s) {
                            int idx = b * (seq_len * head_size * dim_size) + s * (head_size * dim_size) + h * (dim_size) + d + i;
                            sum += to_float(tensor[idx]);
                        }
                        mean_tensor[b * head_size * dim_size + h * dim_size + d + i] = static_cast<float>(sum / seq_len);
                    }
                    continue; // Skip to next d in the outer loop
                }

                __m256 sum_vec = _mm256_setzero_ps();
                for (int s = 0; s < seq_len; ++s) {
                    const KVDtype *current_row = tensor + b * (seq_len * head_size * dim_size) + s * (head_size * dim_size) + h * (dim_size) + d;
                    __m256 val_vec = load_and_convert_to_fp32_vec(current_row);
                    sum_vec = _mm256_add_ps(sum_vec, val_vec);
                }
                const float inv_seq_len = 1.0f / seq_len;
                __m256 inv_len_vec = _mm256_set1_ps(inv_seq_len);
                __m256 mean_vec = _mm256_mul_ps(sum_vec, inv_len_vec);
                _mm256_storeu_ps(mean_tensor + b * head_size * dim_size + h * dim_size + d, mean_vec);
                // =========== AVX2 IMPLEMENTATION END ===========
#else
                // Fallback for non-AVX2/NEON
                double sum[8] = {0.0};
                for (int s = 0; s < seq_len; ++s) {
                    for (int i = 0; i < 8; ++i) {
                        if (d + i < dim_size) {
                            int idx = b * (seq_len * head_size * dim_size) + s * (head_size * dim_size) + h * (dim_size) + d + i;
                            sum[i] += to_float(tensor[idx]);
                        }
                    }
                }
                for (int i = 0; i < 8; ++i) {
                    if (d + i < dim_size) {
                        mean_tensor[b * head_size * dim_size + h * dim_size + d + i] = static_cast<float>(sum[i] / seq_len);
                    }
                }
#endif
            }
        }
    }
}

template <typename KVDtype>
void compute_mean_and_quantize_k(
    const KVDtype *K,
    float *mean_tensor,
    int8_t *k_quant_global,
    float *k_scale_global,
    int batch_size, int kv_head_size, int seq_size_k, int dim_size,
    int threads,
    float *temp_k_sum,           // [threads * dim_size]
    float *temp_k_smoothed,      // [threads * dim_size]
    float *temp_k_head_buffer) { // [threads * seq_size_k * dim_size]

#pragma omp parallel for num_threads(threads) collapse(2)
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < kv_head_size; ++h) {
            const int thread_id = omp_get_thread_num();
            float *thread_sum_buf = temp_k_sum + thread_id * dim_size;
            float *thread_smoothed_buf = temp_k_smoothed + thread_id * dim_size;
            float *thread_k_head_buf = temp_k_head_buffer + thread_id * seq_size_k * dim_size;

            float *target_mean = mean_tensor + b * kv_head_size * dim_size + h * dim_size;
            const int num_k_blocks = dim_size / QK_K_BLOCK_SIZE;
            int8_t *target_k_quant = k_quant_global + (b * kv_head_size + h) * seq_size_k * dim_size;
            float *target_k_scale = k_scale_global + (b * kv_head_size + h) * seq_size_k * num_k_blocks;

            const int k_stride = kv_head_size * dim_size;

            memset(thread_sum_buf, 0, dim_size * sizeof(float));

            for (int s = 0; s < seq_size_k; ++s) {
                const KVDtype *k_row_global = K + b * seq_size_k * k_stride + s * k_stride + h * dim_size;
                float *k_row_buffered = thread_k_head_buf + s * dim_size;

#if defined(__AVX2__)
                int d = 0;
                for (; d <= dim_size - 8; d += 8) {
                    const __m256 val_vec = load_and_convert_to_fp32_vec(k_row_global + d);
                    _mm256_storeu_ps(k_row_buffered + d, val_vec);
                    __m256 sum_vec = _mm256_loadu_ps(thread_sum_buf + d);
                    sum_vec = _mm256_add_ps(sum_vec, val_vec);
                    _mm256_storeu_ps(thread_sum_buf + d, sum_vec);
                }
                for (; d < dim_size; ++d) {
                    float val = to_float(k_row_global[d]);
                    k_row_buffered[d] = val;
                    thread_sum_buf[d] += val;
                }
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
                int d = 0;
                for (; d <= dim_size - 8; d += 8) {
                    float32x4_t val_vec_lo, val_vec_hi;
                    load_and_convert_to_fp32x4x2(k_row_global + d, val_vec_lo, val_vec_hi);

                    vst1q_f32(k_row_buffered + d, val_vec_lo);
                    vst1q_f32(k_row_buffered + d + 4, val_vec_hi);

                    float32x4_t sum_vec_lo = vld1q_f32(thread_sum_buf + d);
                    float32x4_t sum_vec_hi = vld1q_f32(thread_sum_buf + d + 4);

                    vst1q_f32(thread_sum_buf + d, vaddq_f32(sum_vec_lo, val_vec_lo));
                    vst1q_f32(thread_sum_buf + d + 4, vaddq_f32(sum_vec_hi, val_vec_hi));
                }
                for (; d < dim_size; ++d) {
                    float val = to_float(k_row_global[d]);
                    k_row_buffered[d] = val;
                    thread_sum_buf[d] += val;
                }
#else
                for (int d = 0; d < dim_size; ++d) {
                    float val = to_float(k_row_global[d]);
                    k_row_buffered[d] = val;
                    thread_sum_buf[d] += val;
                }
#endif
            }

            const float inv_seq_len = 1.0f / seq_size_k;
#if defined(__AVX2__)
            const __m256 inv_len_vec = _mm256_set1_ps(inv_seq_len);
            int d = 0;
            for (; d <= dim_size - 8; d += 8) {
                __m256 sum_vec = _mm256_loadu_ps(thread_sum_buf + d);
                __m256 mean_vec = _mm256_mul_ps(sum_vec, inv_len_vec);
                _mm256_storeu_ps(target_mean + d, mean_vec);
            }
            for (; d < dim_size; ++d) {
                target_mean[d] = thread_sum_buf[d] * inv_seq_len;
            }
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
            const float32x4_t inv_len_vec = vdupq_n_f32(inv_seq_len);
            int d = 0;
            for (; d <= dim_size - 4; d += 4) {
                float32x4_t sum_vec = vld1q_f32(thread_sum_buf + d);
                float32x4_t mean_vec = vmulq_f32(sum_vec, inv_len_vec);
                vst1q_f32(target_mean + d, mean_vec);
            }
            for (; d < dim_size; ++d) {
                target_mean[d] = thread_sum_buf[d] * inv_seq_len;
            }
#else
            for (int d = 0; d < dim_size; ++d) {
                target_mean[d] = thread_sum_buf[d] * inv_seq_len;
            }
#endif

            for (int s = 0; s < seq_size_k; ++s) {
                const float *k_row_buffered = thread_k_head_buf + s * dim_size;

#if defined(__AVX2__)
                int d = 0;
                for (; d <= dim_size - 8; d += 8) {
                    const __m256 k_vec = _mm256_loadu_ps(k_row_buffered + d);
                    const __m256 mean_vec = _mm256_loadu_ps(target_mean + d);
                    const __m256 smoothed_vec = _mm256_sub_ps(k_vec, mean_vec);
                    _mm256_storeu_ps(thread_smoothed_buf + d, smoothed_vec);
                }
                for (; d < dim_size; ++d) {
                    thread_smoothed_buf[d] = k_row_buffered[d] - target_mean[d];
                }
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
                int d = 0;
                for (; d <= dim_size - 4; d += 4) {
                    const float32x4_t k_vec = vld1q_f32(k_row_buffered + d);
                    const float32x4_t mean_vec = vld1q_f32(target_mean + d);
                    const float32x4_t smoothed_vec = vsubq_f32(k_vec, mean_vec);
                    vst1q_f32(thread_smoothed_buf + d, smoothed_vec);
                }
                for (; d < dim_size; ++d) {
                    thread_smoothed_buf[d] = k_row_buffered[d] - target_mean[d];
                }
#else
                for (int d = 0; d < dim_size; ++d) {
                    thread_smoothed_buf[d] = k_row_buffered[d] - target_mean[d];
                }
#endif
                quantize_row_per_group_simd(thread_smoothed_buf,
                                            target_k_quant + s * dim_size,
                                            target_k_scale + s * num_k_blocks,
                                            dim_size, 1.0f, thread_sum_buf);
            }
        }
    }
}

class WorkspaceManager {
public:
    WorkspaceManager() = default;
    ~WorkspaceManager() {
        for (auto &ptr : workspace_) {
            if (ptr) aligned_free(ptr);
        }
    }
    void **get_workspace(const std::vector<size_t> &required_sizes) {
        if (workspace_.empty()) {
            workspace_.resize(required_sizes.size(), nullptr);
            current_sizes_.resize(required_sizes.size(), 0);
        }
        for (size_t i = 0; i < required_sizes.size(); ++i) {
            if (required_sizes[i] > current_sizes_[i]) {
                if (workspace_[i]) aligned_free(workspace_[i]);
                aligned_alloc(&workspace_[i], required_sizes[i], 64);
                current_sizes_[i] = required_sizes[i];
            }
        }
        return workspace_.data();
    }

private:
    std::vector<void *> workspace_;
    std::vector<size_t> current_sizes_;
};

template <typename KVDtype>
struct SAGE_CPU_IMPL {
    using dtype_q_in_t = float;
    using dtype_kv_in_t = KVDtype;
    using dtype_out_t = float;
    int32_t Br, Bc, Q_Head, KV_Head, threads;
    float *acc_o_, *acc_s_, *logsum_, *scoremax_, *scoremax_prev_, *score_scale_, *score_sum_;
    int8_t *q_quant_tile_, *k_quant_tile_;
    float *q_scale_, *k_scale_, *k_smoothed_row_buf_, *q_scaled_row_buf_;

    void configure(int32_t Br_, int32_t Bc_, int32_t Q_Head_, int32_t KV_Head_, int32_t threads_) {
        Br = Br_;
        Bc = Bc_;
        Q_Head = Q_Head_;
        KV_Head = KV_Head_;
        threads = threads_;
    }
    void init_workspace(void **workspace) {
        acc_o_ = static_cast<float *>(workspace[0]);
        acc_s_ = static_cast<float *>(workspace[1]);
        logsum_ = static_cast<float *>(workspace[2]);
        scoremax_ = static_cast<float *>(workspace[3]);
        scoremax_prev_ = static_cast<float *>(workspace[4]);
        score_scale_ = static_cast<float *>(workspace[5]);
        score_sum_ = static_cast<float *>(workspace[6]);
        q_quant_tile_ = static_cast<int8_t *>(workspace[7]);
        k_quant_tile_ = static_cast<int8_t *>(workspace[8]);
        q_scale_ = static_cast<float *>(workspace[9]);
        k_scale_ = static_cast<float *>(workspace[10]);
        k_smoothed_row_buf_ = static_cast<float *>(workspace[11]);
        q_scaled_row_buf_ = static_cast<float *>(workspace[12]);
    }

    void sage_attn_prefill(const dtype_q_in_t *__restrict__ Q, const dtype_kv_in_t *__restrict__ K, const dtype_kv_in_t *__restrict__ V, dtype_out_t *__restrict__ O, const float *K_mean, const float *V_mean, int32_t batch_size, int32_t head_size, int32_t seq_size_q, int32_t seq_size_k, int32_t dim_size, bool causal_mask) {
        const int32_t Tr = (seq_size_q + Br - 1) / Br;
        const int32_t Tc = (seq_size_k + Bc - 1) / Bc;
        const float local_scale = 1.0f / sqrtf(static_cast<float>(dim_size));
        const int32_t kv_group_size = (Q_Head > 0 && KV_Head > 0) ? Q_Head / KV_Head : 1;
        const int32_t num_k_blocks = dim_size / QK_K_BLOCK_SIZE;

        for (int32_t b_idx = 0; b_idx < batch_size; ++b_idx) {
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1)
            for (int32_t h_idx = 0; h_idx < head_size; ++h_idx) {
                const int32_t thread_id = omp_get_thread_num();
                const int32_t this_thread_kv_head = h_idx / kv_group_size;

                float *p_acc_o = acc_o_ + thread_id * Br * dim_size;
                float *p_acc_s = acc_s_ + thread_id * Br * Bc;
                float *p_logsum = logsum_ + thread_id * Br;
                float *p_scoremax = scoremax_ + thread_id * Br;
                float *p_scoremax_prev = scoremax_prev_ + thread_id * Br;
                float *p_score_scale = score_scale_ + thread_id * Br;
                float *p_score_sum = score_sum_ + thread_id * Br;
                int8_t *p_q_quant = q_quant_tile_ + thread_id * Br * dim_size;
                const int8_t *p_k_quant_global = k_quant_tile_ + (b_idx * KV_Head + this_thread_kv_head) * seq_size_k * dim_size;
                float *p_q_scale = q_scale_ + thread_id * Br * num_k_blocks;
                const float *p_k_scale_global = k_scale_ + (b_idx * KV_Head + this_thread_kv_head) * seq_size_k * num_k_blocks;
                float *p_q_scaled = q_scaled_row_buf_ + thread_id * dim_size;

                const float *p_V_mean = V_mean + b_idx * KV_Head * dim_size + this_thread_kv_head * dim_size;
                const int k_stride = KV_Head * dim_size;

                for (int32_t t_r_idx = 0; t_r_idx < Tr; ++t_r_idx) {
                    int32_t Br_fixed = std::min(Br, seq_size_q - t_r_idx * Br);
                    init_temp(p_logsum, p_scoremax, p_acc_o, Br_fixed, dim_size);

                    const dtype_q_in_t *tile_q_main = Q + b_idx * seq_size_q * head_size * dim_size + t_r_idx * Br * head_size * dim_size + h_idx * dim_size;
                    for (int r = 0; r < Br_fixed; ++r) {
                        quantize_row_per_group_simd(tile_q_main + r * (head_size * dim_size), p_q_quant + r * dim_size, p_q_scale + r * num_k_blocks, dim_size, local_scale, p_q_scaled);
                    }

                    for (int32_t t_c_idx = 0; t_c_idx < Tc; ++t_c_idx) {
                        int32_t Bc_fixed = std::min(Bc, seq_size_k - t_c_idx * Bc);
                        const dtype_kv_in_t *tile_v = V + b_idx * seq_size_k * k_stride + t_c_idx * Bc * k_stride + this_thread_kv_head * dim_size;

                        quantize_and_mma0_sdot(Br_fixed, Bc_fixed, p_q_quant, p_k_quant_global + t_c_idx * Bc * dim_size, p_acc_s, p_q_scale, p_k_scale_global + t_c_idx * Bc * num_k_blocks, dim_size, t_r_idx * Br, t_c_idx * Bc, causal_mask);
                        softmax(Br_fixed, Bc_fixed, p_acc_s, p_scoremax, p_scoremax_prev, p_score_scale, p_score_sum, p_logsum);
                        rescale(Br_fixed, p_acc_o, p_score_scale, dim_size);
                        mma1(Br_fixed, Bc_fixed, p_acc_s, tile_v, p_V_mean, p_acc_o, KV_Head, dim_size);
                    }

                    dtype_out_t *tile_o = O + b_idx * seq_size_q * head_size * dim_size + t_r_idx * Br * head_size * dim_size + h_idx * dim_size;
                    scale_and_store(Br_fixed, p_acc_o, p_logsum, p_V_mean, tile_o, head_size, dim_size);
                }
            }
        }
    }

    void sage_attn_decode(const dtype_q_in_t *__restrict__ Q, const dtype_kv_in_t *__restrict__ K, const dtype_kv_in_t *__restrict__ V, dtype_out_t *__restrict__ O, const float *K_mean, const float *V_mean, int32_t batch_size, int32_t head_size, int32_t seq_size_k, int32_t dim_size, bool causal_mask) {
        const int32_t Tc = (seq_size_k + Bc - 1) / Bc;
        const float local_scale = 1.0f / sqrtf(static_cast<float>(dim_size));
        const int32_t kv_group_size = (Q_Head > 0 && KV_Head > 0) ? Q_Head / KV_Head : 1;
        const int32_t num_k_blocks = dim_size / QK_K_BLOCK_SIZE;

        for (int32_t b_idx = 0; b_idx < batch_size; ++b_idx) {
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1)
            for (int32_t h_idx = 0; h_idx < head_size; ++h_idx) {
                const int32_t Br_fixed = 1;
                const int32_t thread_id = omp_get_thread_num();
                const int32_t this_thread_kv_head = h_idx / kv_group_size;

                float *p_acc_o = acc_o_ + thread_id * Br_fixed * dim_size;
                float *p_acc_s = acc_s_ + thread_id * Br_fixed * Bc;
                float *p_logsum = logsum_ + thread_id * Br_fixed;
                float *p_scoremax = scoremax_ + thread_id * Br_fixed;
                float *p_scoremax_prev = scoremax_prev_ + thread_id * Br_fixed;
                float *p_score_scale = score_scale_ + thread_id * Br_fixed;
                float *p_score_sum = score_sum_ + thread_id * Br_fixed;
                int8_t *p_q_quant = q_quant_tile_ + thread_id * Br_fixed * dim_size;
                const int8_t *p_k_quant_global = k_quant_tile_ + (b_idx * KV_Head + this_thread_kv_head) * seq_size_k * dim_size;
                float *p_q_scale = q_scale_ + thread_id * Br_fixed * num_k_blocks;
                const float *p_k_scale_global = k_scale_ + (b_idx * KV_Head + this_thread_kv_head) * seq_size_k * num_k_blocks;
                float *p_q_scaled = q_scaled_row_buf_ + thread_id * dim_size;

                const float *p_V_mean = V_mean + b_idx * KV_Head * dim_size + this_thread_kv_head * dim_size;
                const int k_stride = KV_Head * dim_size;

                const dtype_q_in_t *tile_q_decode = Q + b_idx * head_size * dim_size + h_idx * dim_size;
                quantize_row_per_group_simd(tile_q_decode, p_q_quant, p_q_scale, dim_size, local_scale, p_q_scaled);

                init_temp(p_logsum, p_scoremax, p_acc_o, Br_fixed, dim_size);

                for (int32_t t_c_idx = 0; t_c_idx < Tc; ++t_c_idx) {
                    int32_t Bc_fixed = std::min(Bc, seq_size_k - t_c_idx * Bc);
                    const dtype_kv_in_t *tile_v = V + b_idx * seq_size_k * k_stride + t_c_idx * Bc * k_stride + this_thread_kv_head * dim_size;

                    quantize_and_mma0_sdot(Br_fixed, Bc_fixed, p_q_quant, p_k_quant_global + t_c_idx * Bc * dim_size, p_acc_s, p_q_scale, p_k_scale_global + t_c_idx * Bc * num_k_blocks, dim_size, seq_size_k - 1, t_c_idx * Bc, causal_mask);
                    softmax(Br_fixed, Bc_fixed, p_acc_s, p_scoremax, p_scoremax_prev, p_score_scale, p_score_sum, p_logsum);
                    rescale(Br_fixed, p_acc_o, p_score_scale, dim_size);
                    mma1(Br_fixed, Bc_fixed, p_acc_s, tile_v, p_V_mean, p_acc_o, KV_Head, dim_size);
                }
                dtype_out_t *tile_o = O + b_idx * head_size * dim_size + h_idx * dim_size;
                scale_and_store(Br_fixed, p_acc_o, p_logsum, p_V_mean, tile_o, head_size, dim_size);
            }
        }
    }

    void init_temp(float *logsum, float *scoremax, float *acc_o, int Br_fixed, int dim_size) {
        for (int i = 0; i < Br_fixed; ++i) {
            logsum[i] = 0.0f;
            scoremax[i] = NEG_INF;
        }
        memset(acc_o, 0, Br_fixed * dim_size * sizeof(float));
    }

    void quantize_and_mma0_sdot(int Br_fixed, int Bc_fixed, const int8_t *q_quant_tile, const int8_t *k_quant_tile, float *acc_s, const float *q_scale, const float *k_scale, int dim_size, int global_r_start, int global_c_start, bool causal) {
        const int num_k_blocks = dim_size / QK_K_BLOCK_SIZE;

        for (int r = 0; r < Br_fixed; ++r) {
            for (int c = 0; c < Bc_fixed; ++c) {
                if (causal && (global_c_start + c) > (global_r_start + r)) {
                    acc_s[r * Bc + c] = NEG_INF;
                    continue;
                }

                const int8_t *q_quant_line = q_quant_tile + r * dim_size;
                const int8_t *k_quant_line = k_quant_tile + c * dim_size;
                const float *q_scale_line = q_scale + r * num_k_blocks;
                const float *k_scale_line = k_scale + c * num_k_blocks;

                float total_f32 = 0.0f;

                // Loop over groups/blocks
                for (int g = 0; g < num_k_blocks; ++g) {
                    const int group_start_idx = g * QK_K_BLOCK_SIZE;
                    const int8_t *q_group_ptr = q_quant_line + group_start_idx;
                    const int8_t *k_group_ptr = k_quant_line + group_start_idx;

                    int32_t group_dot_product_i32 = 0;
#if (defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(__ARM_FEATURE_DOTPROD)
                    int32x4_t acc_i32_vec = vdupq_n_s32(0);
                    int d = 0;
                    for (; d <= QK_K_BLOCK_SIZE - 16; d += 16) {
                        acc_i32_vec = vdotq_s32(acc_i32_vec, vld1q_s8(q_group_ptr + d), vld1q_s8(k_group_ptr + d));
                    }
                    group_dot_product_i32 = vaddvq_s32(acc_i32_vec);
                    for (; d < QK_K_BLOCK_SIZE; ++d) {
                        group_dot_product_i32 += q_group_ptr[d] * k_group_ptr[d];
                    }
#elif defined(__AVX2__)
                    __m256i acc_i32_v = _mm256_setzero_si256();
                    int d = 0;
                    // Process 16 bytes (16 pairs of signed int8) at a time
                    for (; d <= QK_K_BLOCK_SIZE - 16; d += 16) {
                        // Load 16 int8 values for q and k
                        __m128i q_i8_v = _mm_loadu_si128((const __m128i *)(q_group_ptr + d));
                        __m128i k_i8_v = _mm_loadu_si128((const __m128i *)(k_group_ptr + d));
                        __m256i q_i16_v = _mm256_cvtepi8_epi16(q_i8_v);
                        __m256i k_i16_v = _mm256_cvtepi8_epi16(k_i8_v);
                        __m256i prod_i32_v = _mm256_madd_epi16(q_i16_v, k_i16_v);
                        acc_i32_v = _mm256_add_epi32(acc_i32_v, prod_i32_v);
                    }

                    group_dot_product_i32 = hsum_i32(acc_i32_v);

                    // Process any remaining elements
                    for (; d < QK_K_BLOCK_SIZE; ++d) {
                        group_dot_product_i32 += q_group_ptr[d] * k_group_ptr[d];
                    }
#else
                    // Fallback for other platforms
                    for (int d = 0; d < QK_K_BLOCK_SIZE; ++d) {
                        group_dot_product_i32 += q_group_ptr[d] * k_group_ptr[d];
                    }
#endif
                    // Accumulate the scaled result of this group
                    total_f32 += (float)group_dot_product_i32 * q_scale_line[g] * k_scale_line[g];
                }

                acc_s[r * Bc + c] = total_f32;
            }
        }
    }

    void softmax(int Br_fixed, int Bc_fixed, float *acc_s, float *scoremax, float *scoremax_prev, float *score_scale, float *score_sum, float *logsum) {
        memcpy(scoremax_prev, scoremax, Br_fixed * sizeof(float));
        for (int r = 0; r < Br_fixed; ++r) {
            float *row = acc_s + r * Bc;
            float current_max = scoremax[r];
            for (int c = 0; c < Bc_fixed; ++c) current_max = std::max(current_max, row[c]);
            scoremax[r] = current_max;
        }
        for (int r = 0; r < Br_fixed; ++r) score_scale[r] = expf(scoremax_prev[r] - scoremax[r]);
        for (int r = 0; r < Br_fixed; ++r) {
            float *row = acc_s + r * Bc;
            float sm = scoremax[r];
            float sum = 0.f;
            for (int c = 0; c < Bc_fixed; ++c) {
                if (row[c] > NEG_INF / 2) {
                    float val = expf(row[c] - sm);
                    row[c] = val;
                    sum += val;
                } else {
                    row[c] = 0.f;
                }
            }
            score_sum[r] = sum;
        }
        for (int r = 0; r < Br_fixed; ++r) logsum[r] = logsum[r] * score_scale[r] + score_sum[r];
    }
    void rescale(int Br_fixed, float *acc_o, const float *score_scale, int dim_size) {
        for (int r = 0; r < Br_fixed; ++r) {
            float scale_val = score_scale[r];
            float *row_ptr = acc_o + r * dim_size;
#if defined(__AVX2__)
            __m256 scale_vec = _mm256_set1_ps(scale_val);
            int d = 0;
            for (; d <= dim_size - 8; d += 8) _mm256_storeu_ps(row_ptr + d, _mm256_mul_ps(_mm256_loadu_ps(row_ptr + d), scale_vec));
            for (; d < dim_size; ++d) row_ptr[d] *= scale_val;
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
            float32x4_t scale_vec = vdupq_n_f32(scale_val);
            int d = 0;
            for (; d <= dim_size - 4; d += 4) vst1q_f32(row_ptr + d, vmulq_f32(vld1q_f32(row_ptr + d), scale_vec));
            for (; d < dim_size; ++d) row_ptr[d] *= scale_val;
#else
            for (int d = 0; d < dim_size; ++d) row_ptr[d] *= scale_val;
#endif
        }
    }

    void mma1(int Br_fixed, int Bc_fixed, const float *p_block, const dtype_kv_in_t *v_block, const float *v_mean, float *acc_o, int kv_head_size, int dim_size) {
        int v_stride = kv_head_size * dim_size;

#if (defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(__ARM_FP16_FORMAT_IEEE)
        for (int r = 0; r < Br_fixed; ++r) {
            float *o_row = acc_o + r * dim_size;
            const float *p_row = p_block + r * Bc;

            for (int d = 0; d < dim_size; d += 4) {
                float32x4_t o_acc_vec = vld1q_f32(o_row + d);
                const float32x4_t vm_vec = vld1q_f32(v_mean + d);

                int c = 0;
                for (; c <= Bc_fixed - 4; c += 4) {
                    __builtin_prefetch(v_block + (c + 8) * v_stride + d, 0, 0);

                    const float p0 = p_row[c + 0];
                    const float p1 = p_row[c + 1];
                    const float p2 = p_row[c + 2];
                    const float p3 = p_row[c + 3];

                    const dtype_kv_in_t *v_row0 = v_block + (c + 0) * v_stride;
                    const dtype_kv_in_t *v_row1 = v_block + (c + 1) * v_stride;
                    const dtype_kv_in_t *v_row2 = v_block + (c + 2) * v_stride;
                    const dtype_kv_in_t *v_row3 = v_block + (c + 3) * v_stride;

                    float32x4_t v_vec0, v_vec1, v_vec2, v_vec3;
                    if constexpr (std::is_same_v<KVDtype, mllm_fp16_t>) {
                        v_vec0 = vcvt_f32_f16(vld1_f16(reinterpret_cast<const float16_t *>(v_row0 + d)));
                        v_vec1 = vcvt_f32_f16(vld1_f16(reinterpret_cast<const float16_t *>(v_row1 + d)));
                        v_vec2 = vcvt_f32_f16(vld1_f16(reinterpret_cast<const float16_t *>(v_row2 + d)));
                        v_vec3 = vcvt_f32_f16(vld1_f16(reinterpret_cast<const float16_t *>(v_row3 + d)));
                    } else { // float
                        v_vec0 = vld1q_f32(v_row0 + d);
                        v_vec1 = vld1q_f32(v_row1 + d);
                        v_vec2 = vld1q_f32(v_row2 + d);
                        v_vec3 = vld1q_f32(v_row3 + d);
                    }
                    o_acc_vec = vfmaq_n_f32(o_acc_vec, vsubq_f32(v_vec0, vm_vec), p0);
                    o_acc_vec = vfmaq_n_f32(o_acc_vec, vsubq_f32(v_vec1, vm_vec), p1);
                    o_acc_vec = vfmaq_n_f32(o_acc_vec, vsubq_f32(v_vec2, vm_vec), p2);
                    o_acc_vec = vfmaq_n_f32(o_acc_vec, vsubq_f32(v_vec3, vm_vec), p3);
                }

                for (; c < Bc_fixed; ++c) {
                    const float p_scalar = p_row[c];
                    const float32x4_t p_vec = vdupq_n_f32(p_scalar);
                    const dtype_kv_in_t *v_row = v_block + c * v_stride;
                    float32x4_t v_vec;
                    if constexpr (std::is_same_v<KVDtype, mllm_fp16_t>) {
                        v_vec = vcvt_f32_f16(vld1_f16(reinterpret_cast<const float16_t *>(v_row + d)));
                    } else {
                        v_vec = vld1q_f32(v_row + d);
                    }
                    float32x4_t v_smoothed = vsubq_f32(v_vec, vm_vec);
                    o_acc_vec = vfmaq_f32(o_acc_vec, p_vec, v_smoothed);
                }

                vst1q_f32(o_row + d, o_acc_vec);
            }
        }
#elif defined(__AVX2__) && defined(__FMA__)
        for (int r = 0; r < Br_fixed; ++r) {
            float *o_row = acc_o + r * dim_size;
            for (int c = 0; c < Bc_fixed; ++c) {
                const float p_scalar = p_block[r * Bc + c];
                if (fabsf(p_scalar) < 1e-9) continue;

                const __m256 p_vec = _mm256_set1_ps(p_scalar);
                const dtype_kv_in_t *v_row = v_block + c * v_stride;

                int d = 0;
                for (; d <= dim_size - 8; d += 8) {
                    const __m256 vm_vec = _mm256_loadu_ps(v_mean + d);
                    const __m256 v_vec = load_and_convert_to_fp32_vec(v_row + d);

                    __m256 o_vec = _mm256_loadu_ps(o_row + d);
                    __m256 v_smoothed = _mm256_sub_ps(v_vec, vm_vec);

                    // Fused Multiply-Add: o_vec = (p_vec * v_smoothed) + o_vec
                    o_vec = _mm256_fmadd_ps(p_vec, v_smoothed, o_vec);

                    _mm256_storeu_ps(o_row + d, o_vec);
                }
                // Remainder loop
                for (; d < dim_size; ++d) {
                    o_row[d] += p_scalar * (to_float(v_row[d]) - v_mean[d]);
                }
            }
        }
#else
        // Fallback for other platforms
        for (int r = 0; r < Br_fixed; ++r) {
            float *o_row = acc_o + r * dim_size;
            for (int c = 0; c < Bc_fixed; ++c) {
                const float p = p_block[r * Bc + c];
                if (fabsf(p) < 1e-9) continue;
                const dtype_kv_in_t *v_row = v_block + c * v_stride;
                for (int d = 0; d < dim_size; ++d) { o_row[d] += p * (to_float(v_row[d]) - v_mean[d]); }
            }
        }
#endif
    }

    void scale_and_store(int Br_fixed, const float *acc_o, const float *logsum, const float *v_mean, float *O, int head_size, int dim_size) {
        int o_stride = head_size * dim_size;
        for (int r = 0; r < Br_fixed; ++r) {
            float inv_logsum = (logsum[r] > 1e-9f) ? 1.f / logsum[r] : 0.f;
            const float *o_row = acc_o + r * dim_size;
            float *O_row = O + r * o_stride;
#if defined(__AVX2__) && defined(__FMA__)
            const __m256 inv_logsum_vec = _mm256_set1_ps(inv_logsum);
            int d = 0;
            for (; d <= dim_size - 8; d += 8) {
                const __m256 o_vec = _mm256_loadu_ps(o_row + d);
                const __m256 vm_vec = _mm256_loadu_ps(v_mean + d);
                _mm256_storeu_ps(O_row + d, _mm256_fmadd_ps(o_vec, inv_logsum_vec, vm_vec));
            }
            for (; d < dim_size; ++d) O_row[d] = o_row[d] * inv_logsum + v_mean[d];
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
            float32x4_t inv_logsum_vec = vdupq_n_f32(inv_logsum);
            int d = 0;
            for (; d <= dim_size - 4; d += 4) {
                float32x4_t o_vec = vld1q_f32(o_row + d);
                float32x4_t vm_vec = vld1q_f32(v_mean + d);
                vst1q_f32(O_row + d, vfmaq_f32(vm_vec, o_vec, inv_logsum_vec));
            }
            for (; d < dim_size; ++d) O_row[d] = o_row[d] * inv_logsum + v_mean[d];
#else
            for (int d = 0; d < dim_size; ++d) O_row[d] = o_row[d] * inv_logsum + v_mean[d];
#endif
        }
    }
};

template <typename KVDtype>
void sage_attention_forward_cpu_dispatch(const float *Q, const KVDtype *K, const KVDtype *V, float *O, int32_t batch_size, int32_t q_head, int32_t kv_head, int32_t seq_size_q, int32_t seq_size_k, int32_t dim_size, bool causal_mask, int32_t threads, int32_t br, int32_t bc) {
    // 确保 dim_size 可以被块大小整除
    if (dim_size % QK_K_BLOCK_SIZE != 0) {
        std::cerr << "Error: dim_size must be divisible by QK_K_BLOCK_SIZE (" << QK_K_BLOCK_SIZE << ")" << std::endl;
        return;
    }
    const int32_t num_k_blocks = dim_size / QK_K_BLOCK_SIZE;

    std::vector<float> V_mean(batch_size * kv_head * dim_size);
    compute_channel_means<KVDtype>(V, V_mean.data(), batch_size, kv_head, seq_size_k, dim_size);

    thread_local WorkspaceManager manager;

    std::vector<float> K_mean(batch_size * kv_head * dim_size);

    if (seq_size_q > 1) { // Prefill 阶段
        const std::vector<size_t> required_sizes = {
            (size_t)threads * br * dim_size * sizeof(float),                          // 0: acc_o_
            (size_t)threads * br * bc * sizeof(float),                                // 1: acc_s_
            (size_t)threads * br * sizeof(float),                                     // 2: logsum_
            (size_t)threads * br * sizeof(float),                                     // 3: scoremax_
            (size_t)threads * br * sizeof(float),                                     // 4: scoremax_prev_
            (size_t)threads * br * sizeof(float),                                     // 5: score_scale_
            (size_t)threads * br * sizeof(float),                                     // 6: score_sum_
            (size_t)threads * br * dim_size * sizeof(int8_t),                         // 7: q_quant_tile_
            (size_t)batch_size * kv_head * seq_size_k * dim_size * sizeof(int8_t),    // 8: k_quant_tile_ (global)
            (size_t)threads * br * num_k_blocks * sizeof(float),                      // 9: q_scale_
            (size_t)batch_size * kv_head * seq_size_k * num_k_blocks * sizeof(float), // 10: k_scale_ (global)
            (size_t)threads * dim_size * sizeof(float),                               // 11: k_smoothed_row_buf_
            (size_t)threads * dim_size * sizeof(float),                               // 12: q_scaled_row_buf_
            (size_t)threads * dim_size * sizeof(float),                               // 13: temp_k_sum
            (size_t)threads * seq_size_k * dim_size * sizeof(float)                   // 14: temp_k_head_buffer
        };
        void **workspace = manager.get_workspace(required_sizes);

        int8_t *k_quant_global_buffer = static_cast<int8_t *>(workspace[8]);
        float *k_scale_global_buffer = static_cast<float *>(workspace[10]);

        compute_mean_and_quantize_k<KVDtype>(K, K_mean.data(), k_quant_global_buffer, k_scale_global_buffer,
                                             batch_size, kv_head, seq_size_k, dim_size, threads,
                                             static_cast<float *>(workspace[13]),
                                             static_cast<float *>(workspace[11]),
                                             static_cast<float *>(workspace[14]));

        SAGE_CPU_IMPL<KVDtype> op;
        op.configure(br, bc, q_head, kv_head, threads);
        op.init_workspace(workspace);
        op.sage_attn_prefill(Q, K, V, O, K_mean.data(), V_mean.data(), batch_size, q_head, seq_size_q, seq_size_k, dim_size, causal_mask);

    } else {
        const int32_t decode_br = 1;
        const std::vector<size_t> required_sizes = {
            (size_t)threads * decode_br * dim_size * sizeof(float),
            (size_t)threads * decode_br * bc * sizeof(float),
            (size_t)threads * decode_br * sizeof(float),
            (size_t)threads * decode_br * sizeof(float),
            (size_t)threads * decode_br * sizeof(float),
            (size_t)threads * decode_br * sizeof(float),
            (size_t)threads * decode_br * sizeof(float),
            (size_t)threads * decode_br * dim_size * sizeof(int8_t),
            (size_t)batch_size * kv_head * seq_size_k * dim_size * sizeof(int8_t),
            (size_t)threads * decode_br * num_k_blocks * sizeof(float),
            (size_t)batch_size * kv_head * seq_size_k * num_k_blocks * sizeof(float),
            (size_t)threads * dim_size * sizeof(float),
            (size_t)threads * dim_size * sizeof(float),
            (size_t)threads * dim_size * sizeof(float),
            (size_t)threads * seq_size_k * dim_size * sizeof(float)};
        void **workspace = manager.get_workspace(required_sizes);

        int8_t *k_quant_global_buffer = static_cast<int8_t *>(workspace[8]);
        float *k_scale_global_buffer = static_cast<float *>(workspace[10]);

        compute_mean_and_quantize_k<KVDtype>(K, K_mean.data(), k_quant_global_buffer, k_scale_global_buffer,
                                             batch_size, kv_head, seq_size_k, dim_size, threads,
                                             static_cast<float *>(workspace[13]),
                                             static_cast<float *>(workspace[11]),
                                             static_cast<float *>(workspace[14]));

        SAGE_CPU_IMPL<KVDtype> op;
        op.configure(br, bc, q_head, kv_head, threads);
        op.init_workspace(workspace);
        op.sage_attn_decode(Q, K, V, O, K_mean.data(), V_mean.data(), batch_size, q_head, seq_size_k, dim_size, causal_mask);
    }
}
} // namespace sage_attn_cpu
#endif // SAGE_ATTENTION_CPU_HPP