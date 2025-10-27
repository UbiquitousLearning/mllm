#ifndef MLLM_FA2_CAL_HPP
#define MLLM_FA2_CAL_HPP

#include <cstdint>
#include <omp.h>
#include <cassert>
#include <limits>
#include <cmath>
#include <cstring>
#include "Types.hpp"
#include "backends/cpu/third_party/ggml/QuantizeFP16.hpp"
#include "backends/cpu/third_party/ggml/ComputeUtils.hpp"

#ifdef __AVX2__
#include <immintrin.h>
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
#include <arm_neon.h>
#endif

namespace mobi_attn {

// ========================================
// 数学函数和工具
// ========================================
#define NEG_INF std::numeric_limits<float>::lowest()

#ifdef __AVX2__
// Horizontal max of a __m256 vector
inline float _mm256_hmax_ps(__m256 x) {
    __m128 lo = _mm256_castps256_ps128(x);
    __m128 hi = _mm256_extractf128_ps(x, 1);
    __m128 max_val = _mm_max_ps(lo, hi);
    max_val = _mm_max_ps(max_val, _mm_shuffle_ps(max_val, max_val, _MM_SHUFFLE(0, 0, 2, 2)));
    max_val = _mm_max_ps(max_val, _mm_shuffle_ps(max_val, max_val, _MM_SHUFFLE(0, 0, 0, 1)));
    return _mm_cvtss_f32(max_val);
}

// Horizontal sum of a __m256 vector
inline float _mm256_hadd_ps(__m256 x) {
    __m128 lo = _mm256_castps256_ps128(x);
    __m128 hi = _mm256_extractf128_ps(x, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
inline float _vmaxvq_f32_hmax(float32x4_t x) {
    return vmaxvq_f32(x);
}

inline float _vaddvq_f32_hadd(float32x4_t x) {
    return vaddvq_f32(x);
}
#endif

void aligned_alloc(void **ptr, size_t required_bytes, size_t align) {
    if (align % sizeof(void *) != 0 || (align & (align - 1)) != 0) {
        *ptr = nullptr;
        return;
    }
    if (posix_memalign(ptr, align, required_bytes) != 0) {
        *ptr = nullptr;
    }
}

void aligned_free(void *ptr) {
    free(ptr);
}

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

// ========================================
// FlashAttention2 核心实现 (FP32版本)
// ========================================
struct FA_2_GQA_QKV_FP32_BSHD_O_FP32_BSHD_ACC_FP32_IMPL {
    using dtype_q_in_t = float;
    using dtype_kv_in_t = dtype_q_in_t;
    using dtype_out_t = dtype_q_in_t;
    using dtype_t = dtype_out_t;
    using acc_dtype_t = float;

    int32_t Br;
    int32_t Bc;
    int32_t Q_Head;
    int32_t KV_Head;
    int32_t threads;
    bool high_precision;

    void configure(int32_t Br_, int32_t Bc_, int32_t Q_Head_, int32_t KV_Head_, int32_t threads_, bool high_precision_) {
        Br = Br_;
        Bc = Bc_;
        Q_Head = Q_Head_;
        KV_Head = KV_Head_;
        threads = threads_;
        high_precision = high_precision_;
    }

    void init_workspace(acc_dtype_t *acc_o, acc_dtype_t *acc_s,
                        acc_dtype_t *logsum, acc_dtype_t *scoremax, acc_dtype_t *scoremax_prev,
                        acc_dtype_t *score_scale, acc_dtype_t *score_sum) {
        acc_o_ = acc_o;
        acc_s_ = acc_s;
        logsum_ = logsum;
        scoremax_ = scoremax;
        scoremax_prev_ = scoremax_prev;
        score_scale_ = score_scale;
        score_sum_ = score_sum;
    }

    void fa2(const dtype_t *__restrict__ Q, const dtype_t *__restrict__ K,
             const dtype_t *__restrict__ V, dtype_t *__restrict__ O, const int32_t batch_size,
             const int32_t head_size, const int32_t seq_size_q, const int32_t seq_size_k,
             const int32_t dim_size, bool causal_mask = true) {
        assert(Br == Bc);
        assert(Q_Head % KV_Head == 0);
        assert(head_size % threads == 0);
#ifdef __AVX2__
        assert(dim_size % 8 == 0); // AVX processes 8 floats at a time
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
        assert(dim_size % 4 == 0); // NEON processes 4 floats at a time
#endif
        if (seq_size_q != 1) {
            __fa2_prefill_append(Q, K, V, O, batch_size, head_size, seq_size_q, seq_size_k, dim_size,
                                 causal_mask);
        } else {
            __fa2_decode(Q, K, V, O, batch_size, head_size, seq_size_q, seq_size_k, dim_size,
                         causal_mask);
        }
    }

private:
    inline void __fa2_prefill_append(const dtype_t *__restrict__ Q, const dtype_t *__restrict__ K,
                                     const dtype_t *__restrict__ V, dtype_t *__restrict__ O,
                                     const int32_t batch_size, const int32_t head_size, // head_size 就是 Q_Head
                                     const int32_t seq_size_q, const int32_t seq_size_k,
                                     const int32_t dim_size, bool causal_mask = true) {
        const int32_t Tr = seq_size_q / Br;
        const int32_t Tr_left = seq_size_q % Br;
        const int32_t Tc = seq_size_k / Bc;
        const int32_t Tc_left = seq_size_k % Bc;

        const float local_scale = 1.0f / sqrtf(static_cast<float>(dim_size));

        const int32_t kv_group_size = Q_Head / KV_Head;

        for (int32_t b_idx = 0; b_idx < batch_size; ++b_idx) {
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1) if (threads > 1)
            for (int32_t h_idx = 0; h_idx < head_size; ++h_idx) {
                const int32_t thread_id = omp_get_thread_num();
                const int32_t this_thread_head = h_idx;

                const int32_t this_thread_kv_head = this_thread_head / kv_group_size;

                for (int t_r_idx = 0; t_r_idx < Tr; ++t_r_idx) {
                    init_temp(logsum_ + thread_id * Br, scoremax_ + thread_id * Br,
                              acc_o_ + thread_id * Br * dim_size, dim_size);
                    for (int t_c_idx = 0; t_c_idx < Tc; ++t_c_idx) {
                        const dtype_t *tile_q = Q + b_idx * seq_size_q * head_size * dim_size + t_r_idx * Br * head_size * dim_size + this_thread_head * dim_size;
                        const dtype_t *tile_k = K + b_idx * seq_size_k * KV_Head * dim_size + t_c_idx * Bc * KV_Head * dim_size + this_thread_kv_head * dim_size;
                        const dtype_t *tile_v = V + b_idx * seq_size_k * KV_Head * dim_size + t_c_idx * Bc * KV_Head * dim_size + this_thread_kv_head * dim_size;

                        acc_dtype_t *tile_acc_s = acc_s_ + thread_id * Br * Bc;
                        acc_dtype_t *acc_o = acc_o_ + thread_id * Br * dim_size;

                        mma0(tile_q, tile_k, tile_acc_s, dim_size, head_size * dim_size, KV_Head * dim_size, t_r_idx, t_c_idx, seq_size_q, seq_size_k, causal_mask);
                        softmax(tile_acc_s, scoremax_ + thread_id * Br, scoremax_prev_ + thread_id * Br, score_scale_ + thread_id * Br, score_sum_ + thread_id * Br, logsum_ + thread_id * Br, local_scale, t_r_idx, t_c_idx, seq_size_q, seq_size_k, causal_mask);
                        rescale(acc_o, score_scale_ + thread_id * Br, dim_size, t_r_idx, t_c_idx, seq_size_q, seq_size_k, causal_mask);
                        mma1(tile_acc_s, tile_v, acc_o, KV_Head, dim_size, t_r_idx, t_c_idx, seq_size_q, seq_size_k, causal_mask);
                    }
                    if (Tc_left) {
                        const dtype_t *tile_q = Q + b_idx * seq_size_q * head_size * dim_size + t_r_idx * Br * head_size * dim_size + this_thread_head * dim_size;
                        const dtype_t *tile_k = K + b_idx * seq_size_k * KV_Head * dim_size + Tc * Bc * KV_Head * dim_size + this_thread_kv_head * dim_size;
                        const dtype_t *tile_v = V + b_idx * seq_size_k * KV_Head * dim_size + Tc * Bc * KV_Head * dim_size + this_thread_kv_head * dim_size;
                        acc_dtype_t *tile_acc_s = acc_s_ + thread_id * Br * Bc;
                        acc_dtype_t *acc_o = acc_o_ + thread_id * Br * dim_size;
                        mma0_pa_n_fixed(Br, Tc_left, tile_q, tile_k, tile_acc_s, dim_size, head_size * dim_size, KV_Head * dim_size, t_r_idx, Tc, seq_size_q, seq_size_k, causal_mask);
                        softmax_pa_n_fixed(Br, Tc_left, tile_acc_s, scoremax_ + thread_id * Br, scoremax_prev_ + thread_id * Br, score_scale_ + thread_id * Br, score_sum_ + thread_id * Br, logsum_ + thread_id * Br, local_scale, t_r_idx, Tc, seq_size_q, seq_size_k, causal_mask);
                        rescale_pa_n_fixed(Br, Tc_left, acc_o, score_scale_ + thread_id * Br, dim_size, t_r_idx, Tc, seq_size_q, seq_size_k, causal_mask);
                        mma1_pa_n_fixed(Br, Tc_left, tile_acc_s, tile_v, acc_o, KV_Head, dim_size, t_r_idx, Tc, seq_size_q, seq_size_k, causal_mask);
                    }
                    scale_and_store(acc_o_ + thread_id * Br * dim_size, logsum_ + thread_id * Br, O + b_idx * seq_size_q * head_size * dim_size + t_r_idx * Br * head_size * dim_size + this_thread_head * dim_size, t_r_idx, head_size, dim_size);
                }
                if (Tr_left) {
                    init_temp(logsum_ + thread_id * Br, scoremax_ + thread_id * Br, acc_o_ + thread_id * Br * dim_size, dim_size);
                    for (int t_c_idx = 0; t_c_idx < Tc; ++t_c_idx) {
                        const dtype_t *tile_q = Q + b_idx * seq_size_q * head_size * dim_size + Tr * Br * head_size * dim_size + this_thread_head * dim_size;
                        const dtype_t *tile_k = K + b_idx * seq_size_k * KV_Head * dim_size + t_c_idx * Bc * KV_Head * dim_size + this_thread_kv_head * dim_size;
                        const dtype_t *tile_v = V + b_idx * seq_size_k * KV_Head * dim_size + t_c_idx * Bc * KV_Head * dim_size + this_thread_kv_head * dim_size;
                        acc_dtype_t *tile_acc_s = acc_s_ + thread_id * Br * Bc;
                        acc_dtype_t *acc_o = acc_o_ + thread_id * Br * dim_size;
                        mma0_pa_n_fixed(Tr_left, Bc, tile_q, tile_k, tile_acc_s, dim_size, head_size * dim_size, KV_Head * dim_size, Tr, t_c_idx, seq_size_q, seq_size_k, causal_mask);
                        softmax_pa_n_fixed(Tr_left, Bc, tile_acc_s, scoremax_ + thread_id * Br, scoremax_prev_ + thread_id * Br, score_scale_ + thread_id * Br, score_sum_ + thread_id * Br, logsum_ + thread_id * Br, local_scale, Tr, t_c_idx, seq_size_q, seq_size_k, causal_mask);
                        rescale_pa_n_fixed(Tr_left, Bc, acc_o, score_scale_ + thread_id * Br, dim_size, Tr, t_c_idx, seq_size_q, seq_size_k, causal_mask);
                        mma1_pa_n_fixed(Tr_left, Bc, tile_acc_s, tile_v, acc_o, KV_Head, dim_size, Tr, t_c_idx, seq_size_q, seq_size_k, causal_mask);
                    }
                    if (Tc_left) {
                        const dtype_t *tile_q = Q + b_idx * seq_size_q * head_size * dim_size + Tr * Br * head_size * dim_size + this_thread_head * dim_size;
                        const dtype_t *tile_k = K + b_idx * seq_size_k * KV_Head * dim_size + Tc * Bc * KV_Head * dim_size + this_thread_kv_head * dim_size;
                        const dtype_t *tile_v = V + b_idx * seq_size_k * KV_Head * dim_size + Tc * Bc * KV_Head * dim_size + this_thread_kv_head * dim_size;
                        acc_dtype_t *tile_acc_s = acc_s_ + thread_id * Br * Bc;
                        acc_dtype_t *acc_o = acc_o_ + thread_id * Br * dim_size;
                        mma0_pa_n_fixed(Tr_left, Tc_left, tile_q, tile_k, tile_acc_s, dim_size, head_size * dim_size, KV_Head * dim_size, Tr, Tc, seq_size_q, seq_size_k, causal_mask);
                        softmax_pa_n_fixed(Tr_left, Tc_left, tile_acc_s, scoremax_ + thread_id * Br, scoremax_prev_ + thread_id * Br, score_scale_ + thread_id * Br, score_sum_ + thread_id * Br, logsum_ + thread_id * Br, local_scale, Tr, Tc, seq_size_q, seq_size_k, causal_mask);
                        rescale_pa_n_fixed(Tr_left, Tc_left, acc_o, score_scale_ + thread_id * Br, dim_size, Tr, Tc, seq_size_q, seq_size_k, causal_mask);
                        mma1_pa_n_fixed(Tr_left, Tc_left, tile_acc_s, tile_v, acc_o, KV_Head, dim_size, Tr, Tc, seq_size_q, seq_size_k, causal_mask);
                    }
                    scale_and_store_pa_n_fixed(Tr_left, acc_o_ + thread_id * Br * dim_size, logsum_ + thread_id * Br, O + b_idx * seq_size_q * head_size * dim_size + Tr * Br * head_size * dim_size + this_thread_head * dim_size, Tr, head_size, dim_size);
                }
            }
        }
    }

    inline void __fa2_decode(const dtype_t *__restrict__ Q, const dtype_t *__restrict__ K,
                             const dtype_t *__restrict__ V, dtype_t *__restrict__ O,
                             const int32_t batch_size, const int32_t head_size,
                             const int32_t seq_size_q, const int32_t seq_size_k,
                             const int32_t dim_size, bool causal_mask = true) {
        const int32_t Tr = 1;
        const int32_t Tc = seq_size_k / Bc;
        const int32_t Tc_left = seq_size_k % Bc;

        const float local_scale = 1.0f / sqrtf(static_cast<float>(dim_size));
        const int32_t kv_group_size = (Q_Head > 0 && KV_Head > 0) ? Q_Head / KV_Head : 1;

        for (int32_t b_idx = 0; b_idx < batch_size; ++b_idx) {
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1) if (threads > 1)
            for (int32_t h_idx = 0; h_idx < head_size; ++h_idx) {
                const int32_t thread_id = omp_get_thread_num();
                const int32_t this_thread_head = h_idx;
                const int32_t this_thread_kv_head = this_thread_head / kv_group_size;

                for (int t_r_idx = 0; t_r_idx < Tr; ++t_r_idx) {
                    init_temp_d(logsum_ + thread_id * Br, scoremax_ + thread_id * Br, acc_o_ + thread_id * Br * dim_size, dim_size);
                    for (int t_c_idx = 0; t_c_idx < Tc; ++t_c_idx) {
                        const dtype_t *tile_q = Q + b_idx * seq_size_q * head_size * dim_size + t_r_idx * 1 * head_size * dim_size + this_thread_head * dim_size;
                        const dtype_t *tile_k = K + b_idx * seq_size_k * KV_Head * dim_size + t_c_idx * Bc * KV_Head * dim_size + this_thread_kv_head * dim_size;
                        const dtype_t *tile_v = V + b_idx * seq_size_k * KV_Head * dim_size + t_c_idx * Bc * KV_Head * dim_size + this_thread_kv_head * dim_size;

                        acc_dtype_t *tile_acc_s = acc_s_ + thread_id * Br * Bc;
                        acc_dtype_t *acc_o = acc_o_ + thread_id * Br * dim_size;

                        mma0_d(tile_q, tile_k, tile_acc_s, dim_size, KV_Head * dim_size, t_r_idx, t_c_idx, seq_size_q, seq_size_k, causal_mask);
                        softmax_d(tile_acc_s, scoremax_ + thread_id * Br, scoremax_prev_ + thread_id * Br, score_scale_ + thread_id * Br, score_sum_ + thread_id * Br, logsum_ + thread_id * Br, local_scale, t_r_idx, t_c_idx, seq_size_q, seq_size_k, causal_mask);
                        rescale_d(acc_o, score_scale_ + thread_id * Br, dim_size, t_r_idx, t_c_idx, seq_size_q, seq_size_k, causal_mask);
                        mma1_d(tile_acc_s, tile_v, acc_o, KV_Head, dim_size, t_r_idx, t_c_idx, seq_size_q, seq_size_k, causal_mask);
                    }
                    if (Tc_left) {
                        const dtype_t *tile_q = Q + b_idx * seq_size_q * head_size * dim_size + t_r_idx * 1 * head_size * dim_size + this_thread_head * dim_size;
                        const dtype_t *tile_k = K + b_idx * seq_size_k * KV_Head * dim_size + Tc * Bc * KV_Head * dim_size + this_thread_kv_head * dim_size;
                        const dtype_t *tile_v = V + b_idx * seq_size_k * KV_Head * dim_size + Tc * Bc * KV_Head * dim_size + this_thread_kv_head * dim_size;
                        acc_dtype_t *tile_acc_s = acc_s_ + thread_id * Br * Bc;
                        acc_dtype_t *acc_o = acc_o_ + thread_id * Br * dim_size;
                        mma0_d_n_fixed(Tc_left, tile_q, tile_k, tile_acc_s, dim_size, KV_Head * dim_size, t_r_idx, Tc, seq_size_q, seq_size_k, causal_mask);
                        softmax_d_n_fixed(Tc_left, tile_acc_s, scoremax_ + thread_id * Br, scoremax_prev_ + thread_id * Br, score_scale_ + thread_id * Br, score_sum_ + thread_id * Br, logsum_ + thread_id * Br, local_scale, t_r_idx, Tc, seq_size_q, seq_size_k, causal_mask);
                        rescale_d_n_fixed(Tc_left, acc_o, score_scale_ + thread_id * Br, dim_size, t_r_idx, Tc, seq_size_q, seq_size_k, causal_mask);
                        mma1_d_n_fixed(Tc_left, tile_acc_s, tile_v, acc_o, KV_Head, dim_size, t_r_idx, Tc, seq_size_q, seq_size_k, causal_mask);
                    }
                    scale_and_store_d(acc_o_ + thread_id * Br * dim_size, logsum_ + thread_id * Br, O + b_idx * seq_size_q * head_size * dim_size + t_r_idx * 1 * head_size * dim_size + this_thread_head * dim_size, t_r_idx, head_size, dim_size);
                }
            }
        }
    }

    inline void init_temp(acc_dtype_t *logsum, acc_dtype_t *scoremax, acc_dtype_t *acc_o, const int32_t dim_size) {
#ifdef __AVX2__
        __m256 zero_vec = _mm256_set1_ps(0.0f);
        __m256 neg_inf_vec = _mm256_set1_ps(NEG_INF);

        int i = 0;
        for (; i <= Br - 8; i += 8) {
            _mm256_storeu_ps(logsum + i, zero_vec);
            _mm256_storeu_ps(scoremax + i, neg_inf_vec);
        }
        for (; i < Br; ++i) {
            logsum[i] = 0.0f;
            scoremax[i] = NEG_INF;
        }

        for (int j = 0; j < Br * dim_size; j += 8) {
            _mm256_storeu_ps(acc_o + j, zero_vec);
        }
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
        float32x4_t zero_vec = vdupq_n_f32(0.0f);
        float32x4_t neg_inf_vec = vdupq_n_f32(NEG_INF);

        int i = 0;
        for (; i <= Br - 4; i += 4) {
            vst1q_f32(logsum + i, zero_vec);
            vst1q_f32(scoremax + i, neg_inf_vec);
        }
        for (; i < Br; ++i) {
            logsum[i] = 0.0f;
            scoremax[i] = NEG_INF;
        }

        for (int j = 0; j < Br * dim_size; j += 4) {
            vst1q_f32(acc_o + j, zero_vec);
        }
#endif
    }

    inline void mma0(const dtype_t *__restrict__ q_block, const dtype_t *__restrict__ k_block,
                     acc_dtype_t *__restrict__ acc_s, const int32_t dim_size,
                     const int32_t q_stride_size, const int32_t kv_stride_size,
                     const int32_t t_r_idx, const int32_t t_c_idx,
                     const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
#ifdef __AVX2__
        // AVX2 implementation remains unchanged.
        const int32_t global_r_start = t_r_idx * Br;
        const int32_t global_r_end = global_r_start + Br;
        const int32_t global_c_start = t_c_idx * Bc;
        int delta_pos = seq_size_k - seq_size_q;

        if (causal_mask && (global_c_start - delta_pos > (global_r_end - 1))) { return; }

#pragma unroll
        for (int32_t b_r_idx = 0; b_r_idx < Br; ++b_r_idx) {
            const dtype_t *q_block_line = q_block + b_r_idx * q_stride_size;
#pragma unroll
            for (int32_t b_c_idx = 0; b_c_idx < Bc; ++b_c_idx) {
                const dtype_t *k_block_line = k_block + b_c_idx * kv_stride_size;

                __m256 sum_vec = _mm256_setzero_ps();
                int i = 0;
                for (; i <= dim_size - 8; i += 8) {
                    __builtin_prefetch(q_block_line + i + 64);
                    __builtin_prefetch(k_block_line + i + 64);
                    __m256 q_vec = _mm256_loadu_ps(q_block_line + i);
                    __m256 k_vec = _mm256_loadu_ps(k_block_line + i);
                    sum_vec = _mm256_fmadd_ps(q_vec, k_vec, sum_vec);
                }
                acc_dtype_t total = _mm256_hadd_ps(sum_vec);
                for (; i < dim_size; ++i) { total += q_block_line[i] * k_block_line[i]; }

                acc_s[b_r_idx * Bc + b_c_idx] = total;
            }
        }

        if (causal_mask && (global_r_end == (t_c_idx * Bc + Bc) - delta_pos)) {
            for (int i = 0; i < Br; ++i) {
                for (int j = 0; j < Bc; ++j) {
                    if (j > i) { acc_s[i * Bc + j] = NEG_INF; }
                }
            }
        }
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
        const int32_t global_r_start = t_r_idx * Br;
        const int32_t global_r_end = global_r_start + Br;
        const int32_t global_c_start = t_c_idx * Bc;
        int delta_pos = seq_size_k - seq_size_q;

        if (causal_mask && (global_c_start - delta_pos > (global_r_end - 1))) { return; }

        for (int32_t b_r_base = 0; b_r_base < Br; b_r_base += 4) {
            for (int32_t b_c_base = 0; b_c_base < Bc; b_c_base += 4) {
                float32x4_t accumulators[16];
                for (int i = 0; i < 16; ++i) {
                    accumulators[i] = vdupq_n_f32(0.0f);
                }

                const dtype_t *q0_ptr = q_block + (b_r_base + 0) * q_stride_size;
                const dtype_t *q1_ptr = q_block + (b_r_base + 1) * q_stride_size;
                const dtype_t *q2_ptr = q_block + (b_r_base + 2) * q_stride_size;
                const dtype_t *q3_ptr = q_block + (b_r_base + 3) * q_stride_size;

                for (int k = 0; k < dim_size; k += 4) {
                    float32x4_t q_vec0 = vld1q_f32(q0_ptr + k);
                    float32x4_t q_vec1 = vld1q_f32(q1_ptr + k);
                    float32x4_t q_vec2 = vld1q_f32(q2_ptr + k);
                    float32x4_t q_vec3 = vld1q_f32(q3_ptr + k);

                    float32x4_t k_vec;

                    k_vec = vld1q_f32(k_block + (b_c_base + 0) * kv_stride_size + k);
                    accumulators[0] = vfmaq_f32(accumulators[0], q_vec0, k_vec);
                    accumulators[4] = vfmaq_f32(accumulators[4], q_vec1, k_vec);
                    accumulators[8] = vfmaq_f32(accumulators[8], q_vec2, k_vec);
                    accumulators[12] = vfmaq_f32(accumulators[12], q_vec3, k_vec);

                    k_vec = vld1q_f32(k_block + (b_c_base + 1) * kv_stride_size + k);
                    accumulators[1] = vfmaq_f32(accumulators[1], q_vec0, k_vec);
                    accumulators[5] = vfmaq_f32(accumulators[5], q_vec1, k_vec);
                    accumulators[9] = vfmaq_f32(accumulators[9], q_vec2, k_vec);
                    accumulators[13] = vfmaq_f32(accumulators[13], q_vec3, k_vec);

                    k_vec = vld1q_f32(k_block + (b_c_base + 2) * kv_stride_size + k);
                    accumulators[2] = vfmaq_f32(accumulators[2], q_vec0, k_vec);
                    accumulators[6] = vfmaq_f32(accumulators[6], q_vec1, k_vec);
                    accumulators[10] = vfmaq_f32(accumulators[10], q_vec2, k_vec);
                    accumulators[14] = vfmaq_f32(accumulators[14], q_vec3, k_vec);

                    k_vec = vld1q_f32(k_block + (b_c_base + 3) * kv_stride_size + k);
                    accumulators[3] = vfmaq_f32(accumulators[3], q_vec0, k_vec);
                    accumulators[7] = vfmaq_f32(accumulators[7], q_vec1, k_vec);
                    accumulators[11] = vfmaq_f32(accumulators[11], q_vec2, k_vec);
                    accumulators[15] = vfmaq_f32(accumulators[15], q_vec3, k_vec);
                }

                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        acc_s[(b_r_base + i) * Bc + (b_c_base + j)] = vaddvq_f32(accumulators[i * 4 + j]);
                    }
                }
            }
        }
        if (causal_mask) {
            for (int i = 0; i < Br; ++i) {
                for (int j = 0; j < Bc; ++j) {
                    if ((global_c_start + j) > (global_r_start + i + delta_pos)) {
                        acc_s[i * Bc + j] = NEG_INF;
                    }
                }
            }
        }
#endif
    }

    inline void softmax(acc_dtype_t *__restrict__ acc_s, acc_dtype_t *scoremax, acc_dtype_t *scoremax_prev,
                        acc_dtype_t *score_scale, acc_dtype_t *score_sum, acc_dtype_t *logsum,
                        const float scale,
                        const int32_t t_r_idx, const int32_t t_c_idx,
                        const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
        if (high_precision) {
#ifdef __AVX2__
            const int32_t global_r_start = t_r_idx * Br;
            const int32_t global_c_start = t_c_idx * Bc;
            int delta_pos = seq_size_k - seq_size_q;
            if (causal_mask && (global_c_start - delta_pos > (global_r_start + Br - 1))) return;
            memcpy(scoremax_prev, scoremax, Br * sizeof(acc_dtype_t));
            for (int br = 0; br < Br; ++br) {
                __m256 max_vec = _mm256_set1_ps(scoremax[br]);
                acc_dtype_t *row = acc_s + br * Bc;
                int bc = 0;
                for (; bc <= Bc - 8; bc += 8) { max_vec = _mm256_max_ps(max_vec, _mm256_loadu_ps(row + bc)); }
                float max_val = _mm256_hmax_ps(max_vec);
                for (; bc < Bc; ++bc) { max_val = fmaxf(max_val, row[bc]); }
                scoremax[br] = max_val;
            }
            for (int br = 0; br < Br; ++br) { score_scale[br] = expf((scoremax_prev[br] - scoremax[br]) * scale); }
            for (int br = 0; br < Br; ++br) {
                const float sm = scoremax[br];
                acc_dtype_t *row = acc_s + br * Bc;
                float sum = 0.0f;
                for (int bc = 0; bc < Bc; ++bc) {
                    float val = expf((row[bc] - sm) * scale);
                    row[bc] = val;
                    sum += val;
                }
                score_sum[br] = sum;
            }
            for (int br = 0; br < Br; ++br) { logsum[br] = logsum[br] * score_scale[br] + score_sum[br]; }
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
            const int32_t global_r_start = t_r_idx * Br;
            const int32_t global_c_start = t_c_idx * Bc;
            int delta_pos = seq_size_k - seq_size_q;
            if (causal_mask && (global_c_start - delta_pos > (global_r_start + Br - 1))) return;
            memcpy(scoremax_prev, scoremax, Br * sizeof(acc_dtype_t));
            for (int br = 0; br < Br; ++br) {
                float32x4_t max_vec = vdupq_n_f32(scoremax[br]);
                acc_dtype_t *row = acc_s + br * Bc;
                int bc = 0;
                for (; bc <= Bc - 4; bc += 4) {
                    max_vec = vmaxq_f32(max_vec, vld1q_f32(row + bc));
                }
                float max_val = _vmaxvq_f32_hmax(max_vec);
                for (; bc < Bc; ++bc) { max_val = fmaxf(max_val, row[bc]); }
                scoremax[br] = max_val;
            }
            for (int br = 0; br < Br; ++br) {
                score_scale[br] = expf((scoremax_prev[br] - scoremax[br]) * scale);
            }
            for (int br = 0; br < Br; ++br) {
                const float sm = scoremax[br];
                acc_dtype_t *row = acc_s + br * Bc;
                float sum = 0.0f;
                for (int bc = 0; bc < Bc; ++bc) {
                    float val = expf((row[bc] - sm) * scale);
                    row[bc] = val;
                    sum += val;
                }
                score_sum[br] = sum;
            }
            for (int br = 0; br < Br; ++br) {
                logsum[br] = logsum[br] * score_scale[br] + score_sum[br];
            }
#endif
        } else {
#ifdef __AVX2__
            const int32_t global_r_start = t_r_idx * Br;
            const int32_t global_c_start = t_c_idx * Bc;
            int delta_pos = seq_size_k - seq_size_q;
            if (causal_mask && (global_c_start - delta_pos > (global_r_start + Br - 1))) return;
            memcpy(scoremax_prev, scoremax, Br * sizeof(acc_dtype_t));
            for (int br = 0; br < Br; ++br) {
                __m256 max_vec = _mm256_set1_ps(scoremax[br]);
                acc_dtype_t *row = acc_s + br * Bc;
                int bc = 0;
                for (; bc <= Bc - 8; bc += 8) { max_vec = _mm256_max_ps(max_vec, _mm256_loadu_ps(row + bc)); }
                float max_val = _mm256_hmax_ps(max_vec);
                for (; bc < Bc; ++bc) { max_val = fmaxf(max_val, row[bc]); }
                scoremax[br] = max_val;
            }
            for (int br = 0; br < Br; ++br) { score_scale[br] = expf((scoremax_prev[br] - scoremax[br]) * scale); }
            for (int br = 0; br < Br; ++br) {
                const float sm = scoremax[br];
                acc_dtype_t *row = acc_s + br * Bc;
                float sum = 0.0f;
                for (int bc = 0; bc < Bc; ++bc) {
                    float val = expf((row[bc] - sm) * scale);
                    row[bc] = val;
                    sum += val;
                }
                score_sum[br] = sum;
            }
            for (int br = 0; br < Br; ++br) { logsum[br] = logsum[br] * score_scale[br] + score_sum[br]; }
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
            const int32_t global_r_start = t_r_idx * Br;
            const int32_t global_c_start = t_c_idx * Bc;
            int delta_pos = seq_size_k - seq_size_q;
            if (causal_mask && (global_c_start - delta_pos > (global_r_start + Br - 1))) return;
            memcpy(scoremax_prev, scoremax, Br * sizeof(acc_dtype_t));
            for (int br = 0; br < Br; ++br) {
                float32x4_t max_vec = vdupq_n_f32(scoremax[br]);
                acc_dtype_t *row = acc_s + br * Bc;
                int bc = 0;
                for (; bc <= Bc - 4; bc += 4) {
                    max_vec = vmaxq_f32(max_vec, vld1q_f32(row + bc));
                }
                float max_val = vmaxvq_f32(max_vec);
                for (; bc < Bc; ++bc) { max_val = fmaxf(max_val, row[bc]); }
                scoremax[br] = max_val;
            }
            for (int br = 0; br < Br; ++br) {
                score_scale[br] = expf((scoremax_prev[br] - scoremax[br]) * scale);
            }
            for (int br = 0; br < Br; ++br) {
                const float sm = scoremax[br];
                acc_dtype_t *row = acc_s + br * Bc;
                float32x4_t sum_vec = vdupq_n_f32(0.0f);
                const float32x4_t sm_vec = vdupq_n_f32(sm);
                const float32x4_t scale_vec = vdupq_n_f32(scale);
                int bc = 0;
                for (; bc <= Bc - 4; bc += 4) {
                    float32x4_t s_vec = vld1q_f32(row + bc);
                    float32x4_t scaled_s_vec = vmulq_f32(vsubq_f32(s_vec, sm_vec), scale_vec);
                    float32x4_t p_vec = exp_ps_f32(scaled_s_vec);
                    vst1q_f32(row + bc, p_vec);
                    sum_vec = vaddq_f32(sum_vec, p_vec);
                }
                float sum = vaddvq_f32(sum_vec);
                for (; bc < Bc; ++bc) {
                    float val = expf((row[bc] - sm) * scale);
                    row[bc] = val;
                    sum += val;
                }
                score_sum[br] = sum;
            }
            for (int br = 0; br < Br; ++br) {
                logsum[br] = logsum[br] * score_scale[br] + score_sum[br];
            }
#endif
        }
    }

    inline void rescale(acc_dtype_t *__restrict__ acc_o, acc_dtype_t *__restrict__ score_scale,
                        const int32_t dim_size, const int32_t t_r_idx, const int32_t t_c_idx,
                        const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
#ifdef __AVX2__
        const int32_t global_r_start = t_r_idx * Br;
        const int32_t global_c_start = t_c_idx * Bc;
        int delta_pos = seq_size_k - seq_size_q;
        if (causal_mask && (global_c_start - delta_pos > (global_r_start + Br - 1))) return;

#pragma unroll
        for (int i = 0; i < Br; ++i) {
            __m256 scale_v = _mm256_set1_ps(score_scale[i]);
            float *row_ptr = acc_o + i * dim_size;
            for (int j = 0; j < dim_size; j += 8) {
                __m256 acc = _mm256_loadu_ps(row_ptr + j);
                acc = _mm256_mul_ps(acc, scale_v);
                _mm256_storeu_ps(row_ptr + j, acc);
            }
        }
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
        const int32_t global_r_start = t_r_idx * Br;
        const int32_t global_c_start = t_c_idx * Bc;
        int delta_pos = seq_size_k - seq_size_q;
        if (causal_mask && (global_c_start - delta_pos > (global_r_start + Br - 1))) return;

        for (int i = 0; i < Br; ++i) {
            float32x4_t scale_v = vdupq_n_f32(score_scale[i]);
            float *row_ptr = acc_o + i * dim_size;
            for (int j = 0; j < dim_size; j += 4) {
                float32x4_t acc = vld1q_f32(row_ptr + j);
                acc = vmulq_f32(acc, scale_v);
                vst1q_f32(row_ptr + j, acc);
            }
        }
#endif
    }

    inline void mma1(const acc_dtype_t *__restrict__ w_block, const dtype_t *__restrict__ v_block,
                     acc_dtype_t *__restrict__ acc_o, const int32_t kv_head_size, const int32_t dim_size,
                     const int32_t t_r_idx, const int32_t t_c_idx, const int32_t seq_size_q,
                     const int32_t seq_size_k, bool causal_mask) {
#ifdef __AVX2__
        // AVX2 implementation remains unchanged.
        const int32_t global_r_start = t_r_idx * Br;
        const int32_t global_c_start = t_c_idx * Bc;
        int delta_pos = seq_size_k - seq_size_q;
        if (causal_mask && (global_c_start - delta_pos > (global_r_start + Br - 1))) return;

        const int32_t v_stride_size = kv_head_size * dim_size;

#pragma unroll
        for (int b_r_idx = 0; b_r_idx < Br; ++b_r_idx) {
            for (int d_base = 0; d_base < dim_size; d_base += 8) {
                __m256 acc = _mm256_loadu_ps(acc_o + b_r_idx * dim_size + d_base);
#pragma unroll
                for (int b_c_idx = 0; b_c_idx < Bc; ++b_c_idx) {
                    __m256 w_vec = _mm256_set1_ps(w_block[b_r_idx * Bc + b_c_idx]);
                    const float *v_ptr = v_block + b_c_idx * v_stride_size + d_base;
                    __m256 v_vec = _mm256_loadu_ps(v_ptr);
                    acc = _mm256_fmadd_ps(w_vec, v_vec, acc);
                }
                _mm256_storeu_ps(acc_o + b_r_idx * dim_size + d_base, acc);
            }
        }
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
        const int32_t global_r_start = t_r_idx * Br;
        const int32_t global_c_start = t_c_idx * Bc;
        int delta_pos = seq_size_k - seq_size_q;
        if (causal_mask && (global_c_start - delta_pos > (global_r_start + Br - 1))) return;

        const int32_t v_stride_size = kv_head_size * dim_size;

        for (int d_base = 0; d_base < dim_size; d_base += 4) {
            float32x4_t acc0 = vld1q_f32(acc_o + 0 * dim_size + d_base);
            float32x4_t acc1 = vld1q_f32(acc_o + 1 * dim_size + d_base);
            float32x4_t acc2 = vld1q_f32(acc_o + 2 * dim_size + d_base);
            float32x4_t acc3 = vld1q_f32(acc_o + 3 * dim_size + d_base);

            for (int b_c_idx = 0; b_c_idx < Bc; ++b_c_idx) {
                const dtype_t *v_ptr = v_block + b_c_idx * v_stride_size + d_base;
                float32x4_t v_vec = vld1q_f32(v_ptr);

                float32x4_t w_vec;

                w_vec = vdupq_n_f32(w_block[0 * Bc + b_c_idx]); // P[0][b_c_idx]
                acc0 = vfmaq_f32(acc0, v_vec, w_vec);

                w_vec = vdupq_n_f32(w_block[1 * Bc + b_c_idx]); // P[1][b_c_idx]
                acc1 = vfmaq_f32(acc1, v_vec, w_vec);

                w_vec = vdupq_n_f32(w_block[2 * Bc + b_c_idx]); // P[2][b_c_idx]
                acc2 = vfmaq_f32(acc2, v_vec, w_vec);

                w_vec = vdupq_n_f32(w_block[3 * Bc + b_c_idx]); // P[3][b_c_idx]
                acc3 = vfmaq_f32(acc3, v_vec, w_vec);
            }

            vst1q_f32(acc_o + 0 * dim_size + d_base, acc0);
            vst1q_f32(acc_o + 1 * dim_size + d_base, acc1);
            vst1q_f32(acc_o + 2 * dim_size + d_base, acc2);
            vst1q_f32(acc_o + 3 * dim_size + d_base, acc3);
        }
#endif
    }

    inline void scale_and_store(const acc_dtype_t *__restrict__ acc_o,
                                const acc_dtype_t *__restrict__ logsum,
                                dtype_t *__restrict__ o_block, const int32_t t_r_idx,
                                const int32_t head_size, const int32_t dim_size) {
#ifdef __AVX2__
#pragma unroll
        for (int i = 0; i < Br; ++i) {
            dtype_t *o_block_line = o_block + i * head_size * dim_size;
            __m256 reciprocal_logsum_vec = _mm256_set1_ps(1.0f / logsum[i]);
            int j = 0;
            for (; j <= dim_size - 8; j += 8) {
                __m256 vec_acc_o = _mm256_loadu_ps(acc_o + i * dim_size + j);
                __m256 result_vec = _mm256_mul_ps(vec_acc_o, reciprocal_logsum_vec);
                _mm256_storeu_ps(o_block_line + j, result_vec);
            }
            float reciprocal_logsum = 1.0f / logsum[i];
            for (; j < dim_size; ++j) {
                o_block_line[j] = acc_o[i * dim_size + j] * reciprocal_logsum;
            }
        }
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
        for (int i = 0; i < Br; ++i) {
            dtype_t *o_block_line = o_block + i * head_size * dim_size;
            float32x4_t reciprocal_logsum_vec = vdupq_n_f32(1.0f / logsum[i]);
            int j = 0;
            for (; j <= dim_size - 4; j += 4) {
                float32x4_t vec_acc_o = vld1q_f32(acc_o + i * dim_size + j);
                float32x4_t result_vec = vmulq_f32(vec_acc_o, reciprocal_logsum_vec);
                vst1q_f32(o_block_line + j, result_vec);
            }
            float reciprocal_logsum = 1.0f / logsum[i];
            for (; j < dim_size; ++j) {
                o_block_line[j] = acc_o[i * dim_size + j] * reciprocal_logsum;
            }
        }
#endif
    }

    inline void mma0_pa_n_fixed(const int32_t Br_n_fixed, const int32_t Bc_n_fixed,
                                const dtype_t *__restrict__ q_block,
                                const dtype_t *__restrict__ k_block, acc_dtype_t *__restrict__ acc_s,
                                const int32_t dim_size, const int32_t q_stride_size, const int32_t kv_stride_size,
                                const int32_t t_r_idx, const int32_t t_c_idx,
                                const int32_t seq_size_q, const int32_t seq_size_k,
                                bool causal_mask) {
#ifdef __AVX2__
        const int32_t global_r_start = t_r_idx * Br;
        const int32_t global_r_end = global_r_start + Br_n_fixed;
        const int32_t global_c_start = t_c_idx * Bc;
        int delta_pos = seq_size_k - seq_size_q;
        if (causal_mask && (global_c_start - delta_pos > (global_r_end - 1))) { return; }

        for (int32_t b_r_idx = 0; b_r_idx < Br_n_fixed; ++b_r_idx) {
            const dtype_t *q_block_line = q_block + b_r_idx * q_stride_size;
            for (int32_t b_c_idx = 0; b_c_idx < Bc_n_fixed; ++b_c_idx) {
                const dtype_t *k_block_line = k_block + b_c_idx * kv_stride_size;
                __m256 sum_vec = _mm256_setzero_ps();
                int i = 0;
                for (; i <= dim_size - 8; i += 8) {
                    sum_vec = _mm256_fmadd_ps(_mm256_loadu_ps(q_block_line + i), _mm256_loadu_ps(k_block_line + i), sum_vec);
                }
                acc_dtype_t total = _mm256_hadd_ps(sum_vec);
                for (; i < dim_size; ++i) { total += q_block_line[i] * k_block_line[i]; }
                acc_s[b_r_idx * Bc + b_c_idx] = total;
            }
        }

        if (causal_mask && (global_r_end == (global_c_start + Bc_n_fixed) - delta_pos)) {
            for (int i = 0; i < Br_n_fixed; ++i) {
                for (int j = 0; j < Bc_n_fixed; ++j) {
                    if (j > i) { acc_s[i * Bc + j] = NEG_INF; }
                }
            }
        }
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
        const int32_t global_r_start = t_r_idx * Br;
        const int32_t global_r_end = global_r_start + Br_n_fixed;
        const int32_t global_c_start = t_c_idx * Bc;
        int delta_pos = seq_size_k - seq_size_q;
        if (causal_mask && (global_c_start - delta_pos > (global_r_end - 1))) { return; }
        for (int32_t b_r_idx = 0; b_r_idx < Br_n_fixed; ++b_r_idx) {
            const dtype_t *q_block_line = q_block + b_r_idx * q_stride_size;
            for (int32_t b_c_idx = 0; b_c_idx < Bc_n_fixed; ++b_c_idx) {
                const dtype_t *k_block_line = k_block + b_c_idx * kv_stride_size;

                float32x4_t sum_vec = vdupq_n_f32(0.0f);
                int k = 0;
                for (; k <= dim_size - 4; k += 4) {
                    float32x4_t q_vec = vld1q_f32(q_block_line + k);
                    float32x4_t k_vec = vld1q_f32(k_block_line + k);
                    sum_vec = vfmaq_f32(sum_vec, q_vec, k_vec);
                }
                acc_dtype_t total = vaddvq_f32(sum_vec);
                for (; k < dim_size; ++k) {
                    total += q_block_line[k] * k_block_line[k];
                }
                acc_s[b_r_idx * Bc + b_c_idx] = total;
            }
        }
        if (causal_mask) {
            for (int i = 0; i < Br_n_fixed; ++i) {
                for (int j = 0; j < Bc_n_fixed; ++j) {
                    if ((global_c_start + j) > (global_r_start + i + delta_pos)) {
                        acc_s[i * Bc + j] = NEG_INF;
                    }
                }
            }
        }
#endif
    }

    inline void softmax_pa_n_fixed(const int32_t Br_n_fixed, const int32_t Bc_n_fixed,
                                   acc_dtype_t *__restrict__ acc_s, acc_dtype_t *scoremax,
                                   acc_dtype_t *scoremax_prev, acc_dtype_t *score_scale,
                                   acc_dtype_t *score_sum, acc_dtype_t *logsum,
                                   const float scale,
                                   const int32_t t_r_idx, const int32_t t_c_idx,
                                   const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
        const int32_t global_r_start = t_r_idx * Br;
        const int32_t global_c_start = t_c_idx * Bc;
        int delta_pos = seq_size_k - seq_size_q;
        if (causal_mask && (global_c_start - delta_pos > (global_r_start + Br_n_fixed - 1))) return;
        memcpy(scoremax_prev, scoremax, Br_n_fixed * sizeof(acc_dtype_t));
        for (int br = 0; br < Br_n_fixed; ++br) {
            acc_dtype_t *row = acc_s + br * Bc;
            float max_val = NEG_INF;
            for (int bc = 0; bc < Bc_n_fixed; ++bc) max_val = fmaxf(max_val, row[bc]);
            scoremax[br] = fmaxf(max_val, scoremax[br]);
        }
        for (int br = 0; br < Br_n_fixed; ++br) {
            score_scale[br] = expf((scoremax_prev[br] - scoremax[br]) * scale);
        }
        for (int br = 0; br < Br_n_fixed; ++br) {
            acc_dtype_t *row = acc_s + br * Bc;
            float current_sum = 0.0f;
            for (int bc = 0; bc < Bc_n_fixed; ++bc) {
                float val = expf((row[bc] - scoremax[br]) * scale);
                row[bc] = val;
                current_sum += val;
            }
            score_sum[br] = current_sum;
        }
        for (int br = 0; br < Br_n_fixed; ++br) { logsum[br] = logsum[br] * score_scale[br] + score_sum[br]; }
    }

    inline void rescale_pa_n_fixed(const int32_t Br_n_fixed, const int32_t Bc_n_fixed,
                                   acc_dtype_t *__restrict__ acc_o,
                                   acc_dtype_t *__restrict__ score_scale, const int32_t dim_size,
                                   const int32_t t_r_idx, const int32_t t_c_idx,
                                   const int32_t seq_size_q, const int32_t seq_size_k,
                                   bool causal_mask) {
#ifdef __AVX2__
        const int32_t global_r_start = t_r_idx * Br;
        const int32_t global_c_start = t_c_idx * Bc;
        int delta_pos = seq_size_k - seq_size_q;
        if (causal_mask && (global_c_start - delta_pos > (global_r_start + Br_n_fixed - 1))) return;

        for (int i = 0; i < Br_n_fixed; ++i) {
            float *row_ptr = acc_o + i * dim_size;
            __m256 scale_v = _mm256_set1_ps(score_scale[i]);
            for (int j = 0; j < dim_size; j += 8) {
                _mm256_storeu_ps(row_ptr + j, _mm256_mul_ps(_mm256_loadu_ps(row_ptr + j), scale_v));
            }
        }
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
        const int32_t global_r_start = t_r_idx * Br;
        const int32_t global_c_start = t_c_idx * Bc;
        int delta_pos = seq_size_k - seq_size_q;
        if (causal_mask && (global_c_start - delta_pos > (global_r_start + Br_n_fixed - 1))) return;

        for (int i = 0; i < Br_n_fixed; ++i) {
            float *row_ptr = acc_o + i * dim_size;
            float32x4_t scale_v = vdupq_n_f32(score_scale[i]);
            for (int j = 0; j < dim_size; j += 4) {
                vst1q_f32(row_ptr + j, vmulq_f32(vld1q_f32(row_ptr + j), scale_v));
            }
        }
#endif
    }

    inline void mma1_pa_n_fixed(const int32_t Br_n_fixed, const int32_t Bc_n_fixed,
                                const acc_dtype_t *__restrict__ w_block,
                                const dtype_t *__restrict__ v_block, acc_dtype_t *__restrict__ acc_o,
                                const int32_t kv_head_size, const int32_t dim_size,
                                const int32_t t_r_idx, const int32_t t_c_idx,
                                const int32_t seq_size_q, const int32_t seq_size_k,
                                bool causal_mask) {
#ifdef __AVX2__
        const int32_t global_r_start = t_r_idx * Br;
        const int32_t global_c_start = t_c_idx * Bc;
        int delta_pos = seq_size_k - seq_size_q;
        if (causal_mask && (global_c_start - delta_pos > (global_r_start + Br_n_fixed - 1))) return;

        const int32_t v_stride_size = kv_head_size * dim_size;

        for (int b_r_idx = 0; b_r_idx < Br_n_fixed; ++b_r_idx) {
            for (int d_base = 0; d_base < dim_size; d_base += 8) {
                __m256 acc = _mm256_loadu_ps(acc_o + b_r_idx * dim_size + d_base);
                for (int b_c_idx = 0; b_c_idx < Bc_n_fixed; ++b_c_idx) {
                    __m256 w_vec = _mm256_set1_ps(w_block[b_r_idx * Bc + b_c_idx]);
                    const float *v_ptr = v_block + b_c_idx * v_stride_size + d_base;
                    acc = _mm256_fmadd_ps(w_vec, _mm256_loadu_ps(v_ptr), acc);
                }
                _mm256_storeu_ps(acc_o + b_r_idx * dim_size + d_base, acc);
            }
        }
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
        const int32_t global_r_start = t_r_idx * Br;
        const int32_t global_c_start = t_c_idx * Bc;
        int delta_pos = seq_size_k - seq_size_q;
        if (causal_mask && (global_c_start - delta_pos > (global_r_start + Br_n_fixed - 1))) return;

        const int32_t v_stride_size = kv_head_size * dim_size;

        for (int b_r_idx = 0; b_r_idx < Br_n_fixed; ++b_r_idx) {
            for (int d_base = 0; d_base < dim_size; d_base += 4) {
                float32x4_t acc_vec = vld1q_f32(acc_o + b_r_idx * dim_size + d_base);
                for (int b_c_idx = 0; b_c_idx < Bc_n_fixed; ++b_c_idx) {
                    float32x4_t w_vec = vdupq_n_f32(w_block[b_r_idx * Bc + b_c_idx]);
                    const float *v_ptr = v_block + b_c_idx * v_stride_size + d_base;
                    acc_vec = vfmaq_f32(acc_vec, vld1q_f32(v_ptr), w_vec);
                }
                vst1q_f32(acc_o + b_r_idx * dim_size + d_base, acc_vec);
            }
        }
#endif
    }

    inline void scale_and_store_pa_n_fixed(const int32_t Br_n_fixed,
                                           const acc_dtype_t *__restrict__ acc_o,
                                           const acc_dtype_t *__restrict__ logsum,
                                           dtype_t *__restrict__ o_block, const int32_t t_r_idx,
                                           const int32_t head_size, const int32_t dim_size) {
#ifdef __AVX2__
        for (int i = 0; i < Br_n_fixed; ++i) {
            dtype_t *o_block_line = o_block + i * head_size * dim_size;
            float reciprocal_logsum = 1.0f / logsum[i];
            __m256 reciprocal_logsum_vec = _mm256_set1_ps(reciprocal_logsum);
            int j = 0;
            for (; j <= dim_size - 8; j += 8) {
                __m256 vec_acc_o = _mm256_loadu_ps(acc_o + i * dim_size + j);
                _mm256_storeu_ps(o_block_line + j, _mm256_mul_ps(vec_acc_o, reciprocal_logsum_vec));
            }
            for (; j < dim_size; ++j) {
                o_block_line[j] = acc_o[i * dim_size + j] * reciprocal_logsum;
            }
        }
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
        for (int i = 0; i < Br_n_fixed; ++i) {
            dtype_t *o_block_line = o_block + i * head_size * dim_size;
            float reciprocal_logsum = 1.0f / logsum[i];
            float32x4_t reciprocal_logsum_vec = vdupq_n_f32(reciprocal_logsum);
            int j = 0;
            for (; j <= dim_size - 4; j += 4) {
                float32x4_t vec_acc_o = vld1q_f32(acc_o + i * dim_size + j);
                vst1q_f32(o_block_line + j, vmulq_f32(vec_acc_o, reciprocal_logsum_vec));
            }
            for (; j < dim_size; ++j) {
                o_block_line[j] = acc_o[i * dim_size + j] * reciprocal_logsum;
            }
        }
#endif
    }

    // Decode mode functions
    inline void init_temp_d(acc_dtype_t *logsum, acc_dtype_t *scoremax, acc_dtype_t *acc_o,
                            const int32_t dim_size) {
#ifdef __AVX2__
        logsum[0] = 0.0f;
        scoremax[0] = NEG_INF;
        __m256 zero_vec = _mm256_setzero_ps();
        for (int i = 0; i < 1 * dim_size; i += 8) { _mm256_storeu_ps(acc_o + i, zero_vec); }
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
        logsum[0] = 0.0f;
        scoremax[0] = NEG_INF;
        float32x4_t zero_vec = vdupq_n_f32(0.0f);
        for (int i = 0; i < 1 * dim_size; i += 4) {
            vst1q_f32(acc_o + i, zero_vec);
        }
#endif
    }

    inline void mma0_d(const dtype_t *__restrict__ q_block, const dtype_t *__restrict__ k_block,
                       acc_dtype_t *__restrict__ acc_s, const int32_t dim_size,
                       const int32_t kv_stride_size, const int32_t t_r_idx, const int32_t t_c_idx,
                       const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
#ifdef __AVX2__
        const dtype_t *q_block_line = q_block;
#pragma unroll
        for (int32_t b_c_idx = 0; b_c_idx < Bc; ++b_c_idx) {
            const dtype_t *k_block_line = k_block + b_c_idx * kv_stride_size;
            __m256 sum_vec = _mm256_setzero_ps();
            int i = 0;
            for (; i <= dim_size - 8; i += 8) {
                sum_vec = _mm256_fmadd_ps(_mm256_loadu_ps(q_block_line + i), _mm256_loadu_ps(k_block_line + i), sum_vec);
            }
            acc_dtype_t total = _mm256_hadd_ps(sum_vec);
            for (; i < dim_size; ++i) { total += q_block_line[i] * k_block_line[i]; }
            acc_s[b_c_idx] = total;
        }
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
        const dtype_t *q_block_line = q_block;
        for (int32_t b_c_idx = 0; b_c_idx < Bc; ++b_c_idx) {
            const dtype_t *k_block_line = k_block + b_c_idx * kv_stride_size;
            float32x4_t sum_vec = vdupq_n_f32(0.0f);
            int i = 0;
            for (; i <= dim_size - 4; i += 4) {
                sum_vec = vfmaq_f32(sum_vec, vld1q_f32(q_block_line + i), vld1q_f32(k_block_line + i));
            }
            acc_dtype_t total = _vaddvq_f32_hadd(sum_vec);
            for (; i < dim_size; ++i) { total += q_block_line[i] * k_block_line[i]; }
            acc_s[b_c_idx] = total;
        }
#endif
    }

    inline void softmax_d(acc_dtype_t *__restrict__ acc_s, acc_dtype_t *scoremax,
                          acc_dtype_t *scoremax_prev, acc_dtype_t *score_scale,
                          acc_dtype_t *score_sum, acc_dtype_t *logsum,
                          const float scale,
                          const int32_t t_r_idx,
                          const int32_t t_c_idx, const int32_t seq_size_q,
                          const int32_t seq_size_k, bool causal_mask) {
#ifdef __AVX2__
        scoremax_prev[0] = scoremax[0];
        float max_val = NEG_INF;
        for (int bc = 0; bc < Bc; ++bc) max_val = fmaxf(max_val, acc_s[bc]);
        scoremax[0] = fmaxf(max_val, scoremax[0]);
        score_scale[0] = expf((scoremax_prev[0] - scoremax[0]) * scale);
        float current_sum = 0.0f;
        for (int bc = 0; bc < Bc; ++bc) {
            float val = expf((acc_s[bc] - scoremax[0]) * scale);
            acc_s[bc] = val;
            current_sum += val;
        }
        score_sum[0] = current_sum;
        logsum[0] = logsum[0] * score_scale[0] + score_sum[0];

#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
        scoremax_prev[0] = scoremax[0];
        float32x4_t max_vec = vdupq_n_f32(scoremax[0]);
        int bc = 0;
        for (; bc <= Bc - 4; bc += 4) {
            max_vec = vmaxq_f32(max_vec, vld1q_f32(acc_s + bc));
        }
        float max_val = vmaxvq_f32(max_vec);
        for (; bc < Bc; ++bc) {
            max_val = fmaxf(max_val, acc_s[bc]);
        }
        scoremax[0] = max_val;
        score_scale[0] = expf((scoremax_prev[0] - scoremax[0]) * scale);
        float32x4_t sum_vec = vdupq_n_f32(0.0f);
        const float32x4_t sm_vec = vdupq_n_f32(scoremax[0]);
        const float32x4_t scale_vec = vdupq_n_f32(scale);
        bc = 0;
        for (; bc <= Bc - 4; bc += 4) {
            float32x4_t s_vec = vld1q_f32(acc_s + bc);
            float32x4_t scaled_s_vec = vmulq_f32(vsubq_f32(s_vec, sm_vec), scale_vec);
            float32x4_t p_vec = exp_ps_f32(scaled_s_vec);
            vst1q_f32(acc_s + bc, p_vec);
            sum_vec = vaddq_f32(sum_vec, p_vec);
        }
        float current_sum = vaddvq_f32(sum_vec);
        for (; bc < Bc; ++bc) {
            float val = expf((acc_s[bc] - scoremax[0]) * scale);
            acc_s[bc] = val;
            current_sum += val;
        }
        score_sum[0] = current_sum;
        logsum[0] = logsum[0] * score_scale[0] + score_sum[0];
#endif
    }

    inline void rescale_d(acc_dtype_t *__restrict__ acc_o, acc_dtype_t *__restrict__ score_scale,
                          const int32_t dim_size, const int32_t t_r_idx, const int32_t t_c_idx,
                          const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
#ifdef __AVX2__
        __m256 scale_v = _mm256_set1_ps(score_scale[0]);
        for (int j = 0; j < dim_size; j += 8) {
            __m256 acc = _mm256_loadu_ps(acc_o + j);
            acc = _mm256_mul_ps(acc, scale_v);
            _mm256_storeu_ps(acc_o + j, acc);
        }
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
        float32x4_t scale_v = vdupq_n_f32(score_scale[0]);
        for (int j = 0; j < dim_size; j += 4) {
            float32x4_t acc = vld1q_f32(acc_o + j);
            acc = vmulq_f32(acc, scale_v);
            vst1q_f32(acc_o + j, acc);
        }
#endif
    }

    inline void mma1_d(const acc_dtype_t *__restrict__ w_block, const dtype_t *__restrict__ v_block,
                       acc_dtype_t *__restrict__ acc_o, const int32_t kv_head_size,
                       const int32_t dim_size, const int32_t t_r_idx, const int32_t t_c_idx,
                       const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
#ifdef __AVX2__
        const int32_t v_stride_size = kv_head_size * dim_size;
        for (int d_base = 0; d_base < dim_size; d_base += 8) {
            __m256 acc = _mm256_loadu_ps(acc_o + d_base);
            for (int b_c_idx = 0; b_c_idx < Bc; ++b_c_idx) {
                __m256 w_vec = _mm256_set1_ps(w_block[b_c_idx]);
                const float *v_ptr = v_block + b_c_idx * v_stride_size + d_base;
                acc = _mm256_fmadd_ps(w_vec, _mm256_loadu_ps(v_ptr), acc);
            }
            _mm256_storeu_ps(acc_o + d_base, acc);
        }
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
        const int32_t v_stride_size = kv_head_size * dim_size;
        int d_base = 0;
        for (; d_base <= dim_size - 4; d_base += 4) {
            float32x4_t acc_vec = vld1q_f32(acc_o + d_base);
            for (int b_c_idx = 0; b_c_idx < Bc; ++b_c_idx) {
                float32x4_t w_vec = vdupq_n_f32(w_block[b_c_idx]);
                const float *v_ptr = v_block + b_c_idx * v_stride_size + d_base;
                acc_vec = vfmaq_f32(acc_vec, vld1q_f32(v_ptr), w_vec);
            }
            vst1q_f32(acc_o + d_base, acc_vec);
        }
        for (; d_base < dim_size; ++d_base) {
            float acc = acc_o[d_base];
            for (int b_c_idx = 0; b_c_idx < Bc; ++b_c_idx) {
                acc += w_block[b_c_idx] * v_block[b_c_idx * v_stride_size + d_base];
            }
            acc_o[d_base] = acc;
        }
#endif
    }

    inline void scale_and_store_d(const acc_dtype_t *__restrict__ acc_o,
                                  const acc_dtype_t *__restrict__ logsum,
                                  dtype_t *__restrict__ o_block, const int32_t t_r_idx,
                                  const int32_t head_size, const int32_t dim_size) {
#ifdef __AVX2__
        float reciprocal_logsum = 1.0f / logsum[0];
        __m256 reciprocal_logsum_vec = _mm256_set1_ps(reciprocal_logsum);
        int j = 0;
        for (; j <= dim_size - 8; j += 8) {
            _mm256_storeu_ps(o_block + j, _mm256_mul_ps(_mm256_loadu_ps(acc_o + j), reciprocal_logsum_vec));
        }
        for (; j < dim_size; ++j) {
            o_block[j] = acc_o[j] * reciprocal_logsum;
        }
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
        float reciprocal_logsum = 1.0f / logsum[0];
        float32x4_t reciprocal_logsum_vec = vdupq_n_f32(reciprocal_logsum);
        int j = 0;
        for (; j <= dim_size - 4; j += 4) {
            vst1q_f32(o_block + j, vmulq_f32(vld1q_f32(acc_o + j), reciprocal_logsum_vec));
        }
        for (; j < dim_size; ++j) {
            o_block[j] = acc_o[j] * reciprocal_logsum;
        }
#endif
    }

    // Decode n-fixed functions
    inline void mma0_d_n_fixed(const int32_t Bc_n_fixed, const dtype_t *__restrict__ q_block,
                               const dtype_t *__restrict__ k_block, acc_dtype_t *__restrict__ acc_s,
                               const int32_t dim_size, const int32_t kv_stride_size,
                               const int32_t t_r_idx, const int32_t t_c_idx, const int32_t seq_size_q,
                               const int32_t seq_size_k, bool causal_mask) {
        const dtype_t *q_block_line = q_block;
        for (int32_t b_c_idx = 0; b_c_idx < Bc_n_fixed; ++b_c_idx) {
            const dtype_t *k_block_line = k_block + b_c_idx * kv_stride_size;
            float total = 0.0f;
            for (int i = 0; i < dim_size; ++i) { total += q_block_line[i] * k_block_line[i]; }
            acc_s[b_c_idx] = total;
        }
    }

    inline void softmax_d_n_fixed(const int32_t Bc_n_fixed, acc_dtype_t *__restrict__ acc_s,
                                  acc_dtype_t *scoremax, acc_dtype_t *scoremax_prev,
                                  acc_dtype_t *score_scale, acc_dtype_t *score_sum,
                                  acc_dtype_t *logsum,
                                  const float scale,
                                  const int32_t t_r_idx, const int32_t t_c_idx,
                                  const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
        scoremax_prev[0] = scoremax[0];
        float max_val = NEG_INF;
        for (int bc = 0; bc < Bc_n_fixed; ++bc) max_val = fmaxf(max_val, acc_s[bc]);
        scoremax[0] = fmaxf(max_val, scoremax[0]);
        score_scale[0] = expf((scoremax_prev[0] - scoremax[0]) * scale);
        float current_sum = 0.0f;
        for (int bc = 0; bc < Bc_n_fixed; ++bc) {
            float val = expf((acc_s[bc] - scoremax[0]) * scale);
            acc_s[bc] = val;
            current_sum += val;
        }
        score_sum[0] = current_sum;
        logsum[0] = logsum[0] * score_scale[0] + score_sum[0];
    }

    inline void rescale_d_n_fixed(const int32_t Bc_n_fixed, acc_dtype_t *__restrict__ acc_o,
                                  acc_dtype_t *__restrict__ score_scale, const int32_t dim_size,
                                  const int32_t t_r_idx, const int32_t t_c_idx,
                                  const int32_t seq_size_q, const int32_t seq_size_k,
                                  bool causal_mask) {
        float scale = score_scale[0];
        for (int j = 0; j < dim_size; ++j) {
            acc_o[j] *= scale;
        }
    }

    inline void mma1_d_n_fixed(const int32_t Bc_n_fixed, const acc_dtype_t *__restrict__ w_block,
                               const dtype_t *__restrict__ v_block, acc_dtype_t *__restrict__ acc_o,
                               const int32_t kv_head_size, const int32_t dim_size, const int32_t t_r_idx,
                               const int32_t t_c_idx, const int32_t seq_size_q,
                               const int32_t seq_size_k, bool causal_mask) {
        const int32_t v_stride_size = kv_head_size * dim_size;
        for (int d_base = 0; d_base < dim_size; ++d_base) {
            float acc = acc_o[d_base];
            for (int b_c_idx = 0; b_c_idx < Bc_n_fixed; ++b_c_idx) {
                acc += w_block[b_c_idx] * v_block[b_c_idx * v_stride_size + d_base];
            }
            acc_o[d_base] = acc;
        }
    }

private:
    acc_dtype_t *acc_o_;
    acc_dtype_t *acc_s_;
    acc_dtype_t *logsum_;
    acc_dtype_t *scoremax_;
    acc_dtype_t *scoremax_prev_;
    acc_dtype_t *score_scale_;
    acc_dtype_t *score_sum_;
};

struct FA_2_GQA_Q_FP32_KV_FP16_BSHD_O_FP32_BSHD_ACC_FP32_IMPL {
    using dtype_q_in_t = float;
    using dtype_kv_in_t = mllm_fp16_t;
    using dtype_out_t = float;
    using acc_dtype_t = float;

    int32_t Br, Bc, Q_Head, KV_Head, threads;
    bool high_precision;

    void configure(int32_t Br_, int32_t Bc_, int32_t Q_Head_, int32_t KV_Head_, int32_t threads_, bool high_precision_) {
        Br = Br_;
        Bc = Bc_;
        Q_Head = Q_Head_;
        KV_Head = KV_Head_;
        threads = threads_;
        high_precision = high_precision_;
    }

    void init_workspace(acc_dtype_t *acc_o, acc_dtype_t *acc_s,
                        acc_dtype_t *logsum, acc_dtype_t *scoremax, acc_dtype_t *scoremax_prev,
                        acc_dtype_t *score_scale, acc_dtype_t *score_sum) {
        acc_o_ = acc_o;
        acc_s_ = acc_s;
        logsum_ = logsum;
        scoremax_ = scoremax;
        scoremax_prev_ = scoremax_prev;
        score_scale_ = score_scale;
        score_sum_ = score_sum;
    }

    void fa2(const dtype_q_in_t *__restrict__ Q, const dtype_kv_in_t *__restrict__ K,
             const dtype_kv_in_t *__restrict__ V, dtype_out_t *__restrict__ O, const int32_t batch_size,
             const int32_t head_size, const int32_t seq_size_q, const int32_t seq_size_k,
             const int32_t dim_size, bool causal_mask = true) {
        assert(Br == Bc);
        assert(head_size % threads == 0);
#ifdef __AVX2__
        assert(dim_size % 8 == 0);
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
        assert(dim_size % 4 == 0);
        assert(Q_Head % KV_Head == 0);
#endif

        if (seq_size_q != 1) {
            __fa2_prefill_append(Q, K, V, O, batch_size, head_size, seq_size_q, seq_size_k, dim_size, causal_mask);
        } else {
            __fa2_decode(Q, K, V, O, batch_size, head_size, seq_size_q, seq_size_k, dim_size, causal_mask);
        }
    }

private:
#if (defined(__ARM_NEON) || defined(__ARM_NEON__))
#define MLLM_NEON_F32x4_FROM_FP16(addr) vcvt_f32_f16(vld1_f16((const __fp16 *)(addr)))
#endif

    inline void __fa2_prefill_append(const dtype_q_in_t *__restrict__ Q, const dtype_kv_in_t *__restrict__ K,
                                     const dtype_kv_in_t *__restrict__ V, dtype_out_t *__restrict__ O,
                                     const int32_t batch_size, const int32_t head_size,
                                     const int32_t seq_size_q, const int32_t seq_size_k,
                                     const int32_t dim_size, bool causal_mask = true) {
        const int32_t Tr = seq_size_q / Br;
        const int32_t Tr_left = seq_size_q % Br;
        const int32_t Tc = seq_size_k / Bc;
        const int32_t Tc_left = seq_size_k % Tc;

        const float local_scale = 1.0f / sqrtf(static_cast<float>(dim_size));
        const int32_t kv_group_size = (Q_Head > 0 && KV_Head > 0) ? Q_Head / KV_Head : 1;

        for (int32_t b_idx = 0; b_idx < batch_size; ++b_idx) {
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1) if (threads > 1)
            for (int32_t h_idx = 0; h_idx < head_size; ++h_idx) {
                const int32_t thread_id = omp_get_thread_num();
                const int32_t this_thread_head = h_idx;
                const int32_t this_thread_kv_head = this_thread_head / kv_group_size;

                for (int t_r_idx = 0; t_r_idx < Tr; ++t_r_idx) {
                    init_temp(logsum_ + thread_id * Br, scoremax_ + thread_id * Br,
                              acc_o_ + thread_id * Br * dim_size, dim_size);
                    for (int t_c_idx = 0; t_c_idx < Tc; ++t_c_idx) {
                        const dtype_q_in_t *tile_q = Q + b_idx * seq_size_q * head_size * dim_size + t_r_idx * Br * head_size * dim_size + this_thread_head * dim_size;
                        const dtype_kv_in_t *tile_k = K + b_idx * seq_size_k * KV_Head * dim_size + t_c_idx * Bc * KV_Head * dim_size + this_thread_kv_head * dim_size;
                        const dtype_kv_in_t *tile_v = V + b_idx * seq_size_k * KV_Head * dim_size + t_c_idx * Bc * KV_Head * dim_size + this_thread_kv_head * dim_size;

                        acc_dtype_t *tile_acc_s = acc_s_ + thread_id * Br * Bc;
                        acc_dtype_t *acc_o = acc_o_ + thread_id * Br * dim_size;

                        mma0(tile_q, tile_k, tile_acc_s, dim_size, head_size * dim_size, KV_Head * dim_size, t_r_idx, t_c_idx, seq_size_q, seq_size_k, causal_mask);
                        softmax(tile_acc_s, scoremax_ + thread_id * Br, scoremax_prev_ + thread_id * Br, score_scale_ + thread_id * Br, score_sum_ + thread_id * Br, logsum_ + thread_id * Br, local_scale, t_r_idx, t_c_idx, seq_size_q, seq_size_k, causal_mask);
                        rescale(acc_o, score_scale_ + thread_id * Br, dim_size, t_r_idx, t_c_idx, seq_size_q, seq_size_k, causal_mask);
                        mma1(tile_acc_s, tile_v, acc_o, KV_Head, dim_size, t_r_idx, t_c_idx, seq_size_q, seq_size_k, causal_mask);
                    }
                    if (Tc_left) {
                        const dtype_q_in_t *tile_q = Q + b_idx * seq_size_q * head_size * dim_size + t_r_idx * Br * head_size * dim_size + this_thread_head * dim_size;
                        const dtype_kv_in_t *tile_k = K + b_idx * seq_size_k * KV_Head * dim_size + Tc * Bc * KV_Head * dim_size + this_thread_kv_head * dim_size;
                        const dtype_kv_in_t *tile_v = V + b_idx * seq_size_k * KV_Head * dim_size + Tc * Bc * KV_Head * dim_size + this_thread_kv_head * dim_size;
                        acc_dtype_t *tile_acc_s = acc_s_ + thread_id * Br * Bc;
                        acc_dtype_t *acc_o = acc_o_ + thread_id * Br * dim_size;
                        mma0_pa_n_fixed(Br, Tc_left, tile_q, tile_k, tile_acc_s, dim_size, head_size * dim_size, KV_Head * dim_size, t_r_idx, Tc, seq_size_q, seq_size_k, causal_mask);
                        softmax_pa_n_fixed(Br, Tc_left, tile_acc_s, scoremax_ + thread_id * Br, scoremax_prev_ + thread_id * Br, score_scale_ + thread_id * Br, score_sum_ + thread_id * Br, logsum_ + thread_id * Br, local_scale, t_r_idx, Tc, seq_size_q, seq_size_k, causal_mask);
                        rescale_pa_n_fixed(Br, Tc_left, acc_o, score_scale_ + thread_id * Br, dim_size, t_r_idx, Tc, seq_size_q, seq_size_k, causal_mask);
                        mma1_pa_n_fixed(Br, Tc_left, tile_acc_s, tile_v, acc_o, KV_Head, dim_size, t_r_idx, Tc, seq_size_q, seq_size_k, causal_mask);
                    }
                    scale_and_store(acc_o_ + thread_id * Br * dim_size, logsum_ + thread_id * Br, O + b_idx * seq_size_q * head_size * dim_size + t_r_idx * Br * head_size * dim_size + this_thread_head * dim_size, t_r_idx, head_size, dim_size);
                }
                if (Tr_left) {
                    init_temp(logsum_ + thread_id * Br, scoremax_ + thread_id * Br, acc_o_ + thread_id * Br * dim_size, dim_size);
                    for (int t_c_idx = 0; t_c_idx < Tc; ++t_c_idx) {
                        const dtype_q_in_t *tile_q = Q + b_idx * seq_size_q * head_size * dim_size + Tr * Br * head_size * dim_size + this_thread_head * dim_size;
                        const dtype_kv_in_t *tile_k = K + b_idx * seq_size_k * KV_Head * dim_size + t_c_idx * Bc * KV_Head * dim_size + this_thread_kv_head * dim_size;
                        const dtype_kv_in_t *tile_v = V + b_idx * seq_size_k * KV_Head * dim_size + t_c_idx * Bc * KV_Head * dim_size + this_thread_kv_head * dim_size;
                        acc_dtype_t *tile_acc_s = acc_s_ + thread_id * Br * Bc;
                        acc_dtype_t *acc_o = acc_o_ + thread_id * Br * dim_size;
                        mma0_pa_n_fixed(Tr_left, Bc, tile_q, tile_k, tile_acc_s, dim_size, head_size * dim_size, KV_Head * dim_size, Tr, t_c_idx, seq_size_q, seq_size_k, causal_mask);
                        softmax_pa_n_fixed(Tr_left, Bc, tile_acc_s, scoremax_ + thread_id * Br, scoremax_prev_ + thread_id * Br, score_scale_ + thread_id * Br, score_sum_ + thread_id * Br, logsum_ + thread_id * Br, local_scale, Tr, t_c_idx, seq_size_q, seq_size_k, causal_mask);
                        rescale_pa_n_fixed(Tr_left, Bc, acc_o, score_scale_ + thread_id * Br, dim_size, Tr, t_c_idx, seq_size_q, seq_size_k, causal_mask);
                        mma1_pa_n_fixed(Tr_left, Bc, tile_acc_s, tile_v, acc_o, KV_Head, dim_size, Tr, t_c_idx, seq_size_q, seq_size_k, causal_mask);
                    }
                    if (Tc_left) {
                        const dtype_q_in_t *tile_q = Q + b_idx * seq_size_q * head_size * dim_size + Tr * Br * head_size * dim_size + this_thread_head * dim_size;
                        const dtype_kv_in_t *tile_k = K + b_idx * seq_size_k * KV_Head * dim_size + Tc * Bc * KV_Head * dim_size + this_thread_kv_head * dim_size;
                        const dtype_kv_in_t *tile_v = V + b_idx * seq_size_k * KV_Head * dim_size + Tc * Bc * KV_Head * dim_size + this_thread_kv_head * dim_size;
                        acc_dtype_t *tile_acc_s = acc_s_ + thread_id * Br * Bc;
                        acc_dtype_t *acc_o = acc_o_ + thread_id * Br * dim_size;
                        mma0_pa_n_fixed(Tr_left, Tc_left, tile_q, tile_k, tile_acc_s, dim_size, head_size * dim_size, KV_Head * dim_size, Tr, Tc, seq_size_q, seq_size_k, causal_mask);
                        softmax_pa_n_fixed(Tr_left, Tc_left, tile_acc_s, scoremax_ + thread_id * Br, scoremax_prev_ + thread_id * Br, score_scale_ + thread_id * Br, score_sum_ + thread_id * Br, logsum_ + thread_id * Br, local_scale, Tr, Tc, seq_size_q, seq_size_k, causal_mask);
                        rescale_pa_n_fixed(Tr_left, Tc_left, acc_o, score_scale_ + thread_id * Br, dim_size, Tr, Tc, seq_size_q, seq_size_k, causal_mask);
                        mma1_pa_n_fixed(Tr_left, Tc_left, tile_acc_s, tile_v, acc_o, KV_Head, dim_size, Tr, Tc, seq_size_q, seq_size_k, causal_mask);
                    }
                    scale_and_store_pa_n_fixed(Tr_left, acc_o_ + thread_id * Br * dim_size, logsum_ + thread_id * Br, O + b_idx * seq_size_q * head_size * dim_size + Tr * Br * head_size * dim_size + this_thread_head * dim_size, Tr, head_size, dim_size);
                }
            }
        }
    }

    inline void __fa2_decode(const dtype_q_in_t *__restrict__ Q, const dtype_kv_in_t *__restrict__ K,
                             const dtype_kv_in_t *__restrict__ V, dtype_out_t *__restrict__ O,
                             const int32_t batch_size, const int32_t head_size,
                             const int32_t seq_size_q, const int32_t seq_size_k,
                             const int32_t dim_size, bool causal_mask = true) {
        const int32_t Tr = 1;
        const int32_t Tc = seq_size_k / Bc;
        const int32_t Tc_left = seq_size_k % Bc;

        const float local_scale = 1.0f / sqrtf(static_cast<float>(dim_size));
        const int32_t kv_group_size = (Q_Head > 0 && KV_Head > 0) ? Q_Head / KV_Head : 1;

        for (int32_t b_idx = 0; b_idx < batch_size; ++b_idx) {
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1) if (threads > 1)
            for (int32_t h_idx = 0; h_idx < head_size; ++h_idx) {
                const int32_t thread_id = omp_get_thread_num();
                const int32_t this_thread_head = h_idx;
                const int32_t this_thread_kv_head = this_thread_head / kv_group_size;

                const int t_r_idx = 0;
                init_temp_d(logsum_ + thread_id * Br, scoremax_ + thread_id * Br, acc_o_ + thread_id * Br * dim_size, dim_size);
                for (int t_c_idx = 0; t_c_idx < Tc; ++t_c_idx) {
                    const dtype_q_in_t *tile_q = Q + b_idx * seq_size_q * head_size * dim_size + t_r_idx * 1 * head_size * dim_size + this_thread_head * dim_size;
                    const dtype_kv_in_t *tile_k = K + b_idx * seq_size_k * KV_Head * dim_size + t_c_idx * Bc * KV_Head * dim_size + this_thread_kv_head * dim_size;
                    const dtype_kv_in_t *tile_v = V + b_idx * seq_size_k * KV_Head * dim_size + t_c_idx * Bc * KV_Head * dim_size + this_thread_kv_head * dim_size;

                    acc_dtype_t *tile_acc_s = acc_s_ + thread_id * Br * Bc;
                    acc_dtype_t *acc_o = acc_o_ + thread_id * Br * dim_size;

                    mma0_d(tile_q, tile_k, tile_acc_s, dim_size, KV_Head * dim_size, t_r_idx, t_c_idx, seq_size_q, seq_size_k, causal_mask);
                    softmax_d(tile_acc_s, scoremax_ + thread_id * Br, scoremax_prev_ + thread_id * Br, score_scale_ + thread_id * Br, score_sum_ + thread_id * Br, logsum_ + thread_id * Br, local_scale, t_r_idx, t_c_idx, seq_size_q, seq_size_k, causal_mask);
                    rescale_d(acc_o, score_scale_ + thread_id * Br, dim_size, t_r_idx, t_c_idx, seq_size_q, seq_size_k, causal_mask);
                    mma1_d(tile_acc_s, tile_v, acc_o, KV_Head, dim_size, t_r_idx, t_c_idx, seq_size_q, seq_size_k, causal_mask);
                }
                if (Tc_left) {
                    const dtype_q_in_t *tile_q = Q + b_idx * seq_size_q * head_size * dim_size + t_r_idx * 1 * head_size * dim_size + this_thread_head * dim_size;
                    const dtype_kv_in_t *tile_k = K + b_idx * seq_size_k * KV_Head * dim_size + Tc * Bc * KV_Head * dim_size + this_thread_kv_head * dim_size;
                    const dtype_kv_in_t *tile_v = V + b_idx * seq_size_k * KV_Head * dim_size + Tc * Bc * KV_Head * dim_size + this_thread_kv_head * dim_size;
                    acc_dtype_t *tile_acc_s = acc_s_ + thread_id * Br * Bc;
                    acc_dtype_t *acc_o = acc_o_ + thread_id * Br * dim_size;
                    mma0_d_n_fixed(Tc_left, tile_q, tile_k, tile_acc_s, dim_size, KV_Head * dim_size, t_r_idx, Tc, seq_size_q, seq_size_k, causal_mask);
                    softmax_d_n_fixed(Tc_left, tile_acc_s, scoremax_ + thread_id * Br, scoremax_prev_ + thread_id * Br, score_scale_ + thread_id * Br, score_sum_ + thread_id * Br, logsum_ + thread_id * Br, local_scale, t_r_idx, Tc, seq_size_q, seq_size_k, causal_mask);
                    rescale_d_n_fixed(Tc_left, acc_o, score_scale_ + thread_id * Br, dim_size, t_r_idx, Tc, seq_size_q, seq_size_k, causal_mask);
                    mma1_d_n_fixed(Tc_left, tile_acc_s, tile_v, acc_o, KV_Head, dim_size, t_r_idx, Tc, seq_size_q, seq_size_k, causal_mask);
                }
                scale_and_store_d(acc_o_ + thread_id * Br * dim_size, logsum_ + thread_id * Br, O + b_idx * seq_size_q * head_size * dim_size + t_r_idx * 1 * head_size * dim_size + this_thread_head * dim_size, t_r_idx, head_size, dim_size);
            }
        }
    }

    inline void init_temp(acc_dtype_t *logsum, acc_dtype_t *scoremax, acc_dtype_t *acc_o, const int32_t dim_size) {
#ifdef __AVX2__
        __m256 zero_vec = _mm256_set1_ps(0.0f);
        __m256 neg_inf_vec = _mm256_set1_ps(NEG_INF);

        int i = 0;
        for (; i <= Br - 8; i += 8) {
            _mm256_storeu_ps(logsum + i, zero_vec);
            _mm256_storeu_ps(scoremax + i, neg_inf_vec);
        }
        for (; i < Br; ++i) {
            logsum[i] = 0.0f;
            scoremax[i] = NEG_INF;
        }

        for (int j = 0; j < Br * dim_size; j += 8) {
            _mm256_storeu_ps(acc_o + j, zero_vec);
        }
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
        float32x4_t zero_vec = vdupq_n_f32(0.0f);
        float32x4_t neg_inf_vec = vdupq_n_f32(NEG_INF);
        int i = 0;
        for (; i <= Br - 4; i += 4) {
            vst1q_f32(logsum + i, zero_vec);
            vst1q_f32(scoremax + i, neg_inf_vec);
        }
        for (; i < Br; ++i) {
            logsum[i] = 0.0f;
            scoremax[i] = NEG_INF;
        }
        for (int j = 0; j < Br * dim_size; j += 4) {
            vst1q_f32(acc_o + j, zero_vec);
        }
#endif
    }

    inline void mma0(const dtype_q_in_t *__restrict__ q_block, const dtype_kv_in_t *__restrict__ k_block,
                     acc_dtype_t *__restrict__ acc_s, const int32_t dim_size,
                     const int32_t q_stride_size, const int32_t kv_stride_size,
                     const int32_t t_r_idx, const int32_t t_c_idx,
                     const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
#ifdef __AVX2__
        const int32_t global_r_start = t_r_idx * Br, global_r_end = global_r_start + Br;
        const int32_t global_c_start = t_c_idx * Bc;
        int delta_pos = seq_size_k - seq_size_q;
        if (causal_mask && (global_c_start - delta_pos > (global_r_end - 1))) { return; }
        for (int32_t b_r_idx = 0; b_r_idx < Br; ++b_r_idx) {
            const dtype_q_in_t *q_block_line = q_block + b_r_idx * q_stride_size;
            for (int32_t b_c_idx = 0; b_c_idx < Bc; ++b_c_idx) {
                const dtype_kv_in_t *k_block_line = k_block + b_c_idx * kv_stride_size;
                __m256 sum_vec = _mm256_setzero_ps();
                int i = 0;
                for (; i <= dim_size - 8; i += 8) {
                    __m256 q_vec = _mm256_loadu_ps(q_block_line + i);
                    __m256 k_vec = MLLM_F32Cx8_LOAD(k_block_line + i);
                    sum_vec = _mm256_fmadd_ps(q_vec, k_vec, sum_vec);
                }
                acc_dtype_t total = _mm256_hadd_ps(sum_vec);
                for (; i < dim_size; ++i) { total += q_block_line[i] * MLLM_FP16_TO_FP32(k_block_line[i]); }
                acc_s[b_r_idx * Bc + b_c_idx] = total;
            }
        }
        if (causal_mask && (global_r_end == (t_c_idx * Bc + Bc) - delta_pos)) {
            for (int i = 0; i < Br; ++i) {
                for (int j = 0; j < Bc; ++j) {
                    if (j > i) { acc_s[i * Bc + j] = NEG_INF; }
                }
            }
        }
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
        const int32_t global_r_start = t_r_idx * Br;
        const int32_t global_r_end = global_r_start + Br;
        const int32_t global_c_start = t_c_idx * Bc;
        int delta_pos = seq_size_k - seq_size_q;

        if (causal_mask && (global_c_start - delta_pos > (global_r_end - 1))) { return; }

        alignas(16) __fp16 q_f16_buf[Br * dim_size];
        for (int32_t i = 0; i < Br; ++i) {
            const float *q_line_f32 = q_block + i * q_stride_size;
            __fp16 *q_line_f16 = q_f16_buf + i * dim_size;
            for (int j = 0; j < dim_size; ++j) {
                q_line_f16[j] = (__fp16)q_line_f32[j];
            }
        }

        for (int32_t b_r_base = 0; b_r_base < Br; b_r_base += 4) {
            for (int32_t b_c_base = 0; b_c_base < Bc; b_c_base += 4) {
                const __fp16 *q_base_ptr = q_f16_buf + b_r_base * dim_size;
                const __fp16 *k_base_ptr = (const __fp16 *)k_block + b_c_base * kv_stride_size;
                float *acc_s_base_ptr = acc_s + b_r_base * Bc + b_c_base;

#pragma unroll
                for (int32_t b_r_offset = 0; b_r_offset < 4; ++b_r_offset) {
                    const __fp16 *q_row = q_base_ptr + b_r_offset * dim_size;

                    const __fp16 *k_row0 = k_base_ptr + 0 * kv_stride_size;
                    const __fp16 *k_row1 = k_base_ptr + 1 * kv_stride_size;
                    const __fp16 *k_row2 = k_base_ptr + 2 * kv_stride_size;
                    const __fp16 *k_row3 = k_base_ptr + 3 * kv_stride_size;

                    float32x4_t sum0 = vdupq_n_f32(0.0f);
                    float32x4_t sum1 = vdupq_n_f32(0.0f);
                    float32x4_t sum2 = vdupq_n_f32(0.0f);
                    float32x4_t sum3 = vdupq_n_f32(0.0f);

                    int32_t k = 0;
                    for (; k <= dim_size - 8; k += 8) {
                        float16x8_t q_vec = vld1q_f16(q_row + k);

                        float16x8_t k_vec0 = vld1q_f16(k_row0 + k);
                        sum0 = vfmlalq_low_f16(sum0, q_vec, k_vec0);
                        sum0 = vfmlalq_high_f16(sum0, q_vec, k_vec0);

                        float16x8_t k_vec1 = vld1q_f16(k_row1 + k);
                        sum1 = vfmlalq_low_f16(sum1, q_vec, k_vec1);
                        sum1 = vfmlalq_high_f16(sum1, q_vec, k_vec1);

                        float16x8_t k_vec2 = vld1q_f16(k_row2 + k);
                        sum2 = vfmlalq_low_f16(sum2, q_vec, k_vec2);
                        sum2 = vfmlalq_high_f16(sum2, q_vec, k_vec2);

                        float16x8_t k_vec3 = vld1q_f16(k_row3 + k);
                        sum3 = vfmlalq_low_f16(sum3, q_vec, k_vec3);
                        sum3 = vfmlalq_high_f16(sum3, q_vec, k_vec3);
                    }

                    float total0 = vaddvq_f32(sum0);
                    float total1 = vaddvq_f32(sum1);
                    float total2 = vaddvq_f32(sum2);
                    float total3 = vaddvq_f32(sum3);

                    for (; k < dim_size; ++k) {
                        total0 += (float)q_row[k] * (float)k_row0[k];
                        total1 += (float)q_row[k] * (float)k_row1[k];
                        total2 += (float)q_row[k] * (float)k_row2[k];
                        total3 += (float)q_row[k] * (float)k_row3[k];
                    }

                    float *acc_s_row = acc_s_base_ptr + b_r_offset * Bc;
                    acc_s_row[0] = total0;
                    acc_s_row[1] = total1;
                    acc_s_row[2] = total2;
                    acc_s_row[3] = total3;
                }
            }
        }

        if (causal_mask) {
            for (int i = 0; i < Br; ++i) {
                for (int j = 0; j < Bc; ++j) {
                    if ((global_c_start + j) > (global_r_start + i + delta_pos)) {
                        acc_s[i * Bc + j] = NEG_INF;
                    }
                }
            }
        }
#endif
    }

    inline void softmax(acc_dtype_t *__restrict__ acc_s, acc_dtype_t *scoremax, acc_dtype_t *scoremax_prev,
                        acc_dtype_t *score_scale, acc_dtype_t *score_sum, acc_dtype_t *logsum,
                        const float scale, const int32_t t_r_idx, const int32_t t_c_idx,
                        const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
        const int32_t global_r_start = t_r_idx * Br;
        const int32_t global_c_start = t_c_idx * Bc;
        int delta_pos = seq_size_k - seq_size_q;
        if (causal_mask && (global_c_start - delta_pos > (global_r_start + Br - 1))) return;
        memcpy(scoremax_prev, scoremax, Br * sizeof(acc_dtype_t));
#ifdef __AVX2__
        for (int br = 0; br < Br; ++br) {
            __m256 max_vec = _mm256_set1_ps(scoremax[br]);
            acc_dtype_t *row = acc_s + br * Bc;
            int bc = 0;
            for (; bc <= Bc - 8; bc += 8) { max_vec = _mm256_max_ps(max_vec, _mm256_loadu_ps(row + bc)); }
            float max_val = _mm256_hmax_ps(max_vec);
            for (; bc < Bc; ++bc) { max_val = fmaxf(max_val, row[bc]); }
            scoremax[br] = max_val;
        }
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
        for (int br = 0; br < Br; ++br) {
            float32x4_t max_vec = vdupq_n_f32(scoremax[br]);
            acc_dtype_t *row = acc_s + br * Bc;
            int bc = 0;
            for (; bc <= Bc - 4; bc += 4) { max_vec = vmaxq_f32(max_vec, vld1q_f32(row + bc)); }
            float max_val = _vmaxvq_f32_hmax(max_vec);
            for (; bc < Bc; ++bc) { max_val = fmaxf(max_val, row[bc]); }
            scoremax[br] = max_val;
        }
#endif
        for (int br = 0; br < Br; ++br) { score_scale[br] = expf((scoremax_prev[br] - scoremax[br]) * scale); }
        for (int br = 0; br < Br; ++br) {
            const float sm = scoremax[br];
            acc_dtype_t *row = acc_s + br * Bc;
            float sum = 0.0f;
            for (int bc = 0; bc < Bc; ++bc) {
                float val = expf((row[bc] - sm) * scale);
                row[bc] = val;
                sum += val;
            }
            score_sum[br] = sum;
        }
        for (int br = 0; br < Br; ++br) { logsum[br] = logsum[br] * score_scale[br] + score_sum[br]; }
    }

    inline void rescale(acc_dtype_t *__restrict__ acc_o, acc_dtype_t *__restrict__ score_scale,
                        const int32_t dim_size, const int32_t t_r_idx, const int32_t t_c_idx,
                        const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
#ifdef __AVX2__
        // (无变化)
        const int32_t global_r_start = t_r_idx * Br;
        const int32_t global_c_start = t_c_idx * Bc;
        int delta_pos = seq_size_k - seq_size_q;
        if (causal_mask && (global_c_start - delta_pos > (global_r_start + Br - 1))) return;
        for (int i = 0; i < Br; ++i) {
            __m256 scale_v = _mm256_set1_ps(score_scale[i]);
            float *row_ptr = acc_o + i * dim_size;
            for (int j = 0; j < dim_size; j += 8) {
                __m256 acc = _mm256_loadu_ps(row_ptr + j);
                acc = _mm256_mul_ps(acc, scale_v);
                _mm256_storeu_ps(row_ptr + j, acc);
            }
        }
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
        const int32_t global_r_start = t_r_idx * Br;
        const int32_t global_c_start = t_c_idx * Bc;
        int delta_pos = seq_size_k - seq_size_q;
        if (causal_mask && (global_c_start - delta_pos > (global_r_start + Br - 1))) return;
        for (int i = 0; i < Br; ++i) {
            float32x4_t scale_v = vdupq_n_f32(score_scale[i]);
            float *row_ptr = acc_o + i * dim_size;
            for (int j = 0; j < dim_size; j += 4) {
                float32x4_t acc = vld1q_f32(row_ptr + j);
                acc = vmulq_f32(acc, scale_v);
                vst1q_f32(row_ptr + j, acc);
            }
        }
#endif
    }

    inline void mma1(const acc_dtype_t *__restrict__ w_block, const dtype_kv_in_t *__restrict__ v_block,
                     acc_dtype_t *__restrict__ acc_o, const int32_t kv_head_size, const int32_t dim_size,
                     const int32_t t_r_idx, const int32_t t_c_idx, const int32_t seq_size_q,
                     const int32_t seq_size_k, bool causal_mask) {
#ifdef __AVX2__
        const int32_t global_r_start = t_r_idx * Br;
        const int32_t global_c_start = t_c_idx * Bc;
        int delta_pos = seq_size_k - seq_size_q;
        if (causal_mask && (global_c_start - delta_pos > (global_r_start + Br - 1))) return;
        const int32_t v_stride_size = kv_head_size * dim_size;
        for (int b_r_idx = 0; b_r_idx < Br; ++b_r_idx) {
            for (int d_base = 0; d_base < dim_size; d_base += 8) {
                __m256 acc = _mm256_loadu_ps(acc_o + b_r_idx * dim_size + d_base);
                for (int b_c_idx = 0; b_c_idx < Bc; ++b_c_idx) {
                    __m256 w_vec = _mm256_set1_ps(w_block[b_r_idx * Bc + b_c_idx]);
                    const dtype_kv_in_t *v_ptr = v_block + b_c_idx * v_stride_size + d_base;
                    __m256 v_vec = MLLM_F32Cx8_LOAD(v_ptr);
                    acc = _mm256_fmadd_ps(w_vec, v_vec, acc);
                }
                _mm256_storeu_ps(acc_o + b_r_idx * dim_size + d_base, acc);
            }
        }
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
        const int32_t global_r_start = t_r_idx * Br;
        const int32_t global_c_start = t_c_idx * Bc;
        int delta_pos = seq_size_k - seq_size_q;
        if (causal_mask && (global_c_start - delta_pos > (global_r_start + Br - 1))) return;

        alignas(16) __fp16 w_f16_buf[Br * Bc];
        for (int i = 0; i < Br * Bc; ++i) {
            w_f16_buf[i] = (__fp16)w_block[i];
        }

        const int32_t v_stride = kv_head_size * dim_size;
        for (int d_base = 0; d_base < dim_size; d_base += 8) {
            float32x4_t acc0[2], acc1[2], acc2[2], acc3[2];

            acc0[0] = vld1q_f32(acc_o + 0 * dim_size + d_base);
            acc0[1] = vld1q_f32(acc_o + 0 * dim_size + d_base + 4);
            acc1[0] = vld1q_f32(acc_o + 1 * dim_size + d_base);
            acc1[1] = vld1q_f32(acc_o + 1 * dim_size + d_base + 4);
            acc2[0] = vld1q_f32(acc_o + 2 * dim_size + d_base);
            acc2[1] = vld1q_f32(acc_o + 2 * dim_size + d_base + 4);
            acc3[0] = vld1q_f32(acc_o + 3 * dim_size + d_base);
            acc3[1] = vld1q_f32(acc_o + 3 * dim_size + d_base + 4);

#pragma unroll
            for (int k_inner = 0; k_inner < Bc; ++k_inner) {
                const float16x8_t v_vec = vld1q_f16((const __fp16 *)v_block + k_inner * v_stride + d_base);

                const float16x8_t w0_vec = vdupq_n_f16(w_f16_buf[0 * Bc + k_inner]);
                acc0[0] = vfmlalq_low_f16(acc0[0], v_vec, w0_vec);
                acc0[1] = vfmlalq_high_f16(acc0[1], v_vec, w0_vec);

                const float16x8_t w1_vec = vdupq_n_f16(w_f16_buf[1 * Bc + k_inner]);
                acc1[0] = vfmlalq_low_f16(acc1[0], v_vec, w1_vec);
                acc1[1] = vfmlalq_high_f16(acc1[1], v_vec, w1_vec);

                const float16x8_t w2_vec = vdupq_n_f16(w_f16_buf[2 * Bc + k_inner]);
                acc2[0] = vfmlalq_low_f16(acc2[0], v_vec, w2_vec);
                acc2[1] = vfmlalq_high_f16(acc2[1], v_vec, w2_vec);

                const float16x8_t w3_vec = vdupq_n_f16(w_f16_buf[3 * Bc + k_inner]);
                acc3[0] = vfmlalq_low_f16(acc3[0], v_vec, w3_vec);
                acc3[1] = vfmlalq_high_f16(acc3[1], v_vec, w3_vec);
            }

            vst1q_f32(acc_o + 0 * dim_size + d_base, acc0[0]);
            vst1q_f32(acc_o + 0 * dim_size + d_base + 4, acc0[1]);
            vst1q_f32(acc_o + 1 * dim_size + d_base, acc1[0]);
            vst1q_f32(acc_o + 1 * dim_size + d_base + 4, acc1[1]);
            vst1q_f32(acc_o + 2 * dim_size + d_base, acc2[0]);
            vst1q_f32(acc_o + 2 * dim_size + d_base + 4, acc2[1]);
            vst1q_f32(acc_o + 3 * dim_size + d_base, acc3[0]);
            vst1q_f32(acc_o + 3 * dim_size + d_base + 4, acc3[1]);
        }
#endif
    }

    inline void scale_and_store(const acc_dtype_t *__restrict__ acc_o, const acc_dtype_t *__restrict__ logsum,
                                dtype_out_t *__restrict__ o_block, const int32_t t_r_idx,
                                const int32_t head_size, const int32_t dim_size) {
        for (int i = 0; i < Br; ++i) {
            dtype_out_t *o_block_line = o_block + i * head_size * dim_size; // << 保持 BSHD 的行步长
#ifdef __AVX2__
            __m256 reciprocal_logsum_vec = _mm256_set1_ps(1.0f / logsum[i]);
            int j = 0;
            for (; j <= dim_size - 8; j += 8) {
                __m256 vec_acc_o = _mm256_loadu_ps(acc_o + i * dim_size + j);
                __m256 result_vec = _mm256_mul_ps(vec_acc_o, reciprocal_logsum_vec);
                _mm256_storeu_ps(o_block_line + j, result_vec);
            }
            float reciprocal_logsum = 1.0f / logsum[i];
            for (; j < dim_size; ++j) { o_block_line[j] = acc_o[i * dim_size + j] * reciprocal_logsum; }
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
            for (int i = 0; i < Br; ++i) {
                dtype_out_t *o_block_line = o_block + i * head_size * dim_size;
                float reciprocal_logsum = 1.0f / logsum[i];
                float32x4_t reciprocal_logsum_vec = vdupq_n_f32(reciprocal_logsum);
                int j = 0;
                for (; j <= dim_size - 4; j += 4) {
                    float32x4_t vec_acc_o = vld1q_f32(acc_o + i * dim_size + j);
                    vst1q_f32(o_block_line + j, vmulq_f32(vec_acc_o, reciprocal_logsum_vec));
                }
                for (; j < dim_size; ++j) {
                    o_block_line[j] = acc_o[i * dim_size + j] * reciprocal_logsum;
                }
            }
#endif
        }
    }

    inline void mma0_pa_n_fixed(const int32_t Br_n_fixed, const int32_t Bc_n_fixed,
                                const dtype_q_in_t *__restrict__ q_block, const dtype_kv_in_t *__restrict__ k_block,
                                acc_dtype_t *__restrict__ acc_s, const int32_t dim_size,
                                const int32_t q_stride_size, const int32_t kv_stride_size,
                                const int32_t t_r_idx, const int32_t t_c_idx, const int32_t seq_size_q,
                                const int32_t seq_size_k, bool causal_mask) {
#ifdef __AVX2__
        const int32_t global_r_start = t_r_idx * Br;
        const int32_t global_r_end = global_r_start + Br_n_fixed;
        const int32_t global_c_start = t_c_idx * Bc;
        int delta_pos = seq_size_k - seq_size_q;
        if (causal_mask && (global_c_start - delta_pos > (global_r_end - 1))) { return; }
        for (int32_t b_r_idx = 0; b_r_idx < Br_n_fixed; ++b_r_idx) {
            const dtype_q_in_t *q_block_line = q_block + b_r_idx * q_stride_size;
            for (int32_t b_c_idx = 0; b_c_idx < Bc_n_fixed; ++b_c_idx) {
                const dtype_kv_in_t *k_block_line = k_block + b_c_idx * kv_stride_size;
                __m256 sum_vec = _mm256_setzero_ps();
                int i = 0;
                for (; i <= dim_size - 8; i += 8) {
                    sum_vec = _mm256_fmadd_ps(_mm256_loadu_ps(q_block_line + i), MLLM_F32Cx8_LOAD(k_block_line + i), sum_vec);
                }
                acc_dtype_t total = _mm256_hadd_ps(sum_vec);
                for (; i < dim_size; ++i) { total += q_block_line[i] * MLLM_FP16_TO_FP32(k_block_line[i]); }
                acc_s[b_r_idx * Bc + b_c_idx] = total;
            }
        }
        if (causal_mask && (global_r_end == (global_c_start + Bc_n_fixed) - delta_pos)) {
            for (int i = 0; i < Br_n_fixed; ++i) {
                for (int j = 0; j < Bc_n_fixed; ++j) {
                    if (j > i) { acc_s[i * Bc + j] = NEG_INF; }
                }
            }
        }
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
        const int32_t global_r_start = t_r_idx * Br;
        const int32_t global_r_end = global_r_start + Br_n_fixed;
        const int32_t global_c_start = t_c_idx * Bc;
        int delta_pos = seq_size_k - seq_size_q;
        if (causal_mask && (global_c_start - delta_pos > (global_r_end - 1))) { return; }
        for (int32_t b_r_idx = 0; b_r_idx < Br_n_fixed; ++b_r_idx) {
            const dtype_q_in_t *q_block_line = q_block + b_r_idx * q_stride_size;
            for (int32_t b_c_idx = 0; b_c_idx < Bc_n_fixed; ++b_c_idx) {
                const dtype_kv_in_t *k_block_line = k_block + b_c_idx * kv_stride_size;
                float32x4_t sum_vec = vdupq_n_f32(0.0f);
                int i = 0;
                for (; i <= dim_size - 4; i += 4) {
                    float32x4_t q_vec = vld1q_f32(q_block_line + i);
                    float32x4_t k_vec = vcvt_f32_f16(vld1_f16((const __fp16 *)k_block_line + i));
                    sum_vec = vfmaq_f32(sum_vec, q_vec, k_vec);
                }
                acc_dtype_t total = vaddvq_f32(sum_vec);
                for (; i < dim_size; ++i) {
                    total += q_block_line[i] * (float)k_block_line[i];
                }
                acc_s[b_r_idx * Bc + b_c_idx] = total;
            }
        }
        if (causal_mask) {
            for (int i = 0; i < Br_n_fixed; ++i) {
                for (int j = 0; j < Bc_n_fixed; ++j) {
                    if ((global_c_start + j) > (global_r_start + i + delta_pos)) {
                        acc_s[i * Bc + j] = NEG_INF;
                    }
                }
            }
        }
#endif
    }

    inline void softmax_pa_n_fixed(const int32_t Br_n_fixed, const int32_t Bc_n_fixed,
                                   acc_dtype_t *__restrict__ acc_s, acc_dtype_t *scoremax,
                                   acc_dtype_t *scoremax_prev, acc_dtype_t *score_scale,
                                   acc_dtype_t *score_sum, acc_dtype_t *logsum,
                                   const float scale, const int32_t t_r_idx, const int32_t t_c_idx,
                                   const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
        const int32_t global_r_start = t_r_idx * Br;
        const int32_t global_c_start = t_c_idx * Bc;
        int delta_pos = seq_size_k - seq_size_q;
        if (causal_mask && (global_c_start - delta_pos > (global_r_start + Br_n_fixed - 1))) return;
        memcpy(scoremax_prev, scoremax, Br_n_fixed * sizeof(acc_dtype_t));
        for (int br = 0; br < Br_n_fixed; ++br) {
            float max_val = scoremax[br];
            acc_dtype_t *row = acc_s + br * Bc;
            for (int bc = 0; bc < Bc_n_fixed; ++bc) {
                max_val = fmaxf(max_val, row[bc]);
            }
            scoremax[br] = max_val;
        }
        for (int br = 0; br < Br_n_fixed; ++br) {
            score_scale[br] = expf((scoremax_prev[br] - scoremax[br]) * scale);
        }
        for (int br = 0; br < Br_n_fixed; ++br) {
            const float sm = scoremax[br];
            acc_dtype_t *row = acc_s + br * Bc;
            float current_sum = 0.0f;
            for (int bc = 0; bc < Bc_n_fixed; ++bc) {
                float val = expf((row[bc] - sm) * scale);
                row[bc] = val;
                current_sum += val;
            }
            score_sum[br] = current_sum;
        }
        for (int br = 0; br < Br_n_fixed; ++br) {
            logsum[br] = logsum[br] * score_scale[br] + score_sum[br];
        }
    }

    inline void rescale_pa_n_fixed(const int32_t Br_n_fixed, const int32_t Bc_n_fixed,
                                   acc_dtype_t *__restrict__ acc_o, acc_dtype_t *__restrict__ score_scale,
                                   const int32_t dim_size, const int32_t t_r_idx, const int32_t t_c_idx,
                                   const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
#ifdef __AVX2__
        const int32_t global_r_start = t_r_idx * Br;
        const int32_t global_c_start = t_c_idx * Bc;
        int delta_pos = seq_size_k - seq_size_q;
        if (causal_mask && (global_c_start - delta_pos > (global_r_start + Br_n_fixed - 1))) return;
        for (int i = 0; i < Br_n_fixed; ++i) {
            float *row_ptr = acc_o + i * dim_size;
            __m256 scale_v = _mm256_set1_ps(score_scale[i]);
            for (int j = 0; j < dim_size; j += 8) {
                _mm256_storeu_ps(row_ptr + j, _mm256_mul_ps(_mm256_loadu_ps(row_ptr + j), scale_v));
            }
        }
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
        const int32_t global_r_start = t_r_idx * Br;
        const int32_t global_c_start = t_c_idx * Bc;
        int delta_pos = seq_size_k - seq_size_q;
        if (causal_mask && (global_c_start - delta_pos > (global_r_start + Br_n_fixed - 1))) return;
        for (int i = 0; i < Br_n_fixed; ++i) {
            float *row_ptr = acc_o + i * dim_size;
            float32x4_t scale_v = vdupq_n_f32(score_scale[i]);
            for (int j = 0; j < dim_size; j += 4) {
                vst1q_f32(row_ptr + j, vmulq_f32(vld1q_f32(row_ptr + j), scale_v));
            }
        }
#endif
    }

    inline void mma1_pa_n_fixed(const int32_t Br_n_fixed, const int32_t Bc_n_fixed,
                                const acc_dtype_t *__restrict__ w_block, const dtype_kv_in_t *__restrict__ v_block,
                                acc_dtype_t *__restrict__ acc_o, const int32_t kv_head_size, const int32_t dim_size,
                                const int32_t t_r_idx, const int32_t t_c_idx, const int32_t seq_size_q,
                                const int32_t seq_size_k, bool causal_mask) {
#ifdef __AVX2__
        const int32_t global_r_start = t_r_idx * Br;
        const int32_t global_c_start = t_c_idx * Bc;
        int delta_pos = seq_size_k - seq_size_q;
        if (causal_mask && (global_c_start - delta_pos > (global_r_start + Br_n_fixed - 1))) return;
        const int32_t v_stride_size = kv_head_size * dim_size;
        for (int b_r_idx = 0; b_r_idx < Br_n_fixed; ++b_r_idx) {
            for (int d_base = 0; d_base < dim_size; d_base += 8) {
                __m256 acc = _mm256_loadu_ps(acc_o + b_r_idx * dim_size + d_base);
                for (int b_c_idx = 0; b_c_idx < Bc_n_fixed; ++b_c_idx) {
                    __m256 w_vec = _mm256_set1_ps(w_block[b_r_idx * Bc + b_c_idx]);
                    const dtype_kv_in_t *v_ptr = v_block + b_c_idx * v_stride_size + d_base;
                    acc = _mm256_fmadd_ps(w_vec, MLLM_F32Cx8_LOAD(v_ptr), acc);
                }
                _mm256_storeu_ps(acc_o + b_r_idx * dim_size + d_base, acc);
            }
        }
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
        const int32_t global_r_start = t_r_idx * Br;
        const int32_t global_c_start = t_c_idx * Bc;
        int delta_pos = seq_size_k - seq_size_q;
        if (causal_mask && (global_c_start - delta_pos > (global_r_start + Br_n_fixed - 1))) return;
        const int32_t v_stride_size = kv_head_size * dim_size;
        for (int b_r_idx = 0; b_r_idx < Br_n_fixed; ++b_r_idx) {
            for (int d_base = 0; d_base < dim_size; d_base += 4) {
                float32x4_t acc_vec = vld1q_f32(acc_o + b_r_idx * dim_size + d_base);
                for (int b_c_idx = 0; b_c_idx < Bc_n_fixed; ++b_c_idx) {
                    float32x4_t w_vec = vdupq_n_f32(w_block[b_r_idx * Bc + b_c_idx]);
                    const dtype_kv_in_t *v_ptr = v_block + b_c_idx * v_stride_size + d_base;
                    float32x4_t v_vec = vcvt_f32_f16(vld1_f16((const __fp16 *)v_ptr));
                    acc_vec = vfmaq_f32(acc_vec, v_vec, w_vec);
                }
                vst1q_f32(acc_o + b_r_idx * dim_size + d_base, acc_vec);
            }
        }
#endif
    }

    inline void scale_and_store_pa_n_fixed(const int32_t Br_n_fixed, const acc_dtype_t *__restrict__ acc_o,
                                           const acc_dtype_t *__restrict__ logsum, dtype_out_t *__restrict__ o_block,
                                           const int32_t t_r_idx, const int32_t head_size, const int32_t dim_size) {
        for (int i = 0; i < Br_n_fixed; ++i) {
            dtype_out_t *o_block_line = o_block + i * head_size * dim_size;
#ifdef __AVX2__
            float reciprocal_logsum = 1.0f / logsum[i];
            __m256 reciprocal_logsum_vec = _mm256_set1_ps(reciprocal_logsum);
            int j = 0;
            for (; j <= dim_size - 8; j += 8) {
                __m256 vec_acc_o = _mm256_loadu_ps(acc_o + i * dim_size + j);
                _mm256_storeu_ps(o_block_line + j, _mm256_mul_ps(vec_acc_o, reciprocal_logsum_vec));
            }
            for (; j < dim_size; ++j) { o_block_line[j] = acc_o[i * dim_size + j] * reciprocal_logsum; }
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
            float reciprocal_logsum = 1.0f / logsum[i];
            float32x4_t reciprocal_logsum_vec = vdupq_n_f32(reciprocal_logsum);
            int j = 0;
            for (; j <= dim_size - 4; j += 4) {
                float32x4_t vec_acc_o = vld1q_f32(acc_o + i * dim_size + j);
                vst1q_f32(o_block_line + j, vmulq_f32(vec_acc_o, reciprocal_logsum_vec));
            }
            for (; j < dim_size; ++j) { o_block_line[j] = acc_o[i * dim_size + j] * reciprocal_logsum; }
#endif
        }
    }

    inline void init_temp_d(acc_dtype_t *logsum, acc_dtype_t *scoremax, acc_dtype_t *acc_o, const int32_t dim_size) {
        logsum[0] = 0.0f;
        scoremax[0] = NEG_INF;
#ifdef __AVX2__
        __m256 zero_vec = _mm256_setzero_ps();
        for (int i = 0; i < 1 * dim_size; i += 8) { _mm256_storeu_ps(acc_o + i, zero_vec); }
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
        float32x4_t zero_vec = vdupq_n_f32(0.0f);
        for (int i = 0; i < 1 * dim_size; i += 4) { vst1q_f32(acc_o + i, zero_vec); }
#endif
    }

    inline void mma0_d(const dtype_q_in_t *__restrict__ q_block, const dtype_kv_in_t *__restrict__ k_block,
                       acc_dtype_t *__restrict__ acc_s, const int32_t dim_size,
                       const int32_t kv_stride_size, const int32_t t_r_idx, const int32_t t_c_idx,
                       const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
#ifdef __AVX2__
        const dtype_q_in_t *q_block_line = q_block;
        for (int32_t b_c_idx = 0; b_c_idx < Bc; ++b_c_idx) {
            const dtype_kv_in_t *k_block_line = k_block + b_c_idx * kv_stride_size;
            __m256 sum_vec = _mm256_setzero_ps();
            int i = 0;
            for (; i <= dim_size - 8; i += 8) {
                sum_vec = _mm256_fmadd_ps(_mm256_loadu_ps(q_block_line + i), MLLM_F32Cx8_LOAD(k_block_line + i), sum_vec);
            }
            acc_dtype_t total = _mm256_hadd_ps(sum_vec);
            for (; i < dim_size; ++i) { total += q_block_line[i] * MLLM_FP16_TO_FP32(k_block_line[i]); }
            acc_s[b_c_idx] = total;
        }
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
        for (int32_t b_c_idx = 0; b_c_idx < Bc; ++b_c_idx) {
            const dtype_kv_in_t *k_block_line = k_block + b_c_idx * kv_stride_size;
            float32x4_t sum_vec = vdupq_n_f32(0.0f);
            int i = 0;
            for (; i <= dim_size - 4; i += 4) {
                sum_vec = vfmaq_f32(sum_vec, vld1q_f32(q_block + i), vcvt_f32_f16(vld1_f16((const __fp16 *)k_block_line + i)));
            }
            acc_dtype_t total = vaddvq_f32(sum_vec);
            for (; i < dim_size; ++i) { total += q_block[i] * (float)k_block_line[i]; }
            acc_s[b_c_idx] = total;
        }
#endif
    }

    inline void softmax_d(acc_dtype_t *__restrict__ acc_s,
                          acc_dtype_t *scoremax, acc_dtype_t *scoremax_prev, acc_dtype_t *score_scale,
                          acc_dtype_t *score_sum, acc_dtype_t *logsum, const float scale,
                          const int32_t t_r_idx, const int32_t t_c_idx,
                          const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
        scoremax_prev[0] = scoremax[0];
        float max_val = NEG_INF;
        for (int bc = 0; bc < Bc; ++bc) max_val = fmaxf(max_val, acc_s[bc]);
        scoremax[0] = fmaxf(max_val, scoremax[0]);
        score_scale[0] = expf((scoremax_prev[0] - scoremax[0]) * scale);
        float current_sum = 0.0f;
        for (int bc = 0; bc < Bc; ++bc) {
            float val = expf((acc_s[bc] - scoremax[0]) * scale);
            acc_s[bc] = val;
            current_sum += val;
        }
        score_sum[0] = current_sum;
        logsum[0] = logsum[0] * score_scale[0] + score_sum[0];
    }

    inline void rescale_d(acc_dtype_t *__restrict__ acc_o, acc_dtype_t *__restrict__ score_scale,
                          const int32_t dim_size, const int32_t t_r_idx, const int32_t t_c_idx,
                          const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
#ifdef __AVX2__
        __m256 scale_v = _mm256_set1_ps(score_scale[0]);
        for (int j = 0; j < dim_size; j += 8) {
            __m256 acc = _mm256_loadu_ps(acc_o + j);
            acc = _mm256_mul_ps(acc, scale_v);
            _mm256_storeu_ps(acc_o + j, acc);
        }
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
        float32x4_t scale_v = vdupq_n_f32(score_scale[0]);
        for (int j = 0; j < dim_size; j += 4) {
            vst1q_f32(acc_o + j, vmulq_f32(vld1q_f32(acc_o + j), scale_v));
        }
#endif
    }

    inline void mma1_d(const acc_dtype_t *__restrict__ w_block, const dtype_kv_in_t *__restrict__ v_block,
                       acc_dtype_t *__restrict__ acc_o, const int32_t kv_head_size, const int32_t dim_size,
                       const int32_t t_r_idx, const int32_t t_c_idx, const int32_t seq_size_q,
                       const int32_t seq_size_k, bool causal_mask) {
#ifdef __AVX2__
        const int32_t v_stride_size = kv_head_size * dim_size;
        for (int d_base = 0; d_base < dim_size; d_base += 8) {
            __m256 acc = _mm256_loadu_ps(acc_o + d_base);
            for (int b_c_idx = 0; b_c_idx < Bc; ++b_c_idx) {
                __m256 w_vec = _mm256_set1_ps(w_block[b_c_idx]);
                const dtype_kv_in_t *v_ptr = v_block + b_c_idx * v_stride_size + d_base;
                __m256 v_vec = MLLM_F32Cx8_LOAD(v_ptr);
                acc = _mm256_fmadd_ps(w_vec, v_vec, acc);
            }
            _mm256_storeu_ps(acc_o + d_base, acc);
        }
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
        const int32_t v_stride_size = kv_head_size * dim_size;
        int d_base = 0;
        for (; d_base <= dim_size - 4; d_base += 4) {
            float32x4_t acc_vec = vld1q_f32(acc_o + d_base);
            for (int b_c_idx = 0; b_c_idx < Bc; ++b_c_idx) {
                float32x4_t w_vec = vdupq_n_f32(w_block[b_c_idx]);
                const __fp16 *v_ptr = (const __fp16 *)v_block + b_c_idx * v_stride_size + d_base;
                acc_vec = vfmaq_f32(acc_vec, vcvt_f32_f16(vld1_f16(v_ptr)), w_vec);
            }
            vst1q_f32(acc_o + d_base, acc_vec);
        }
        for (; d_base < dim_size; ++d_base) {
            float acc = acc_o[d_base];
            for (int b_c_idx = 0; b_c_idx < Bc; ++b_c_idx) {
                acc += w_block[b_c_idx] * (float)v_block[b_c_idx * v_stride_size + d_base];
            }
            acc_o[d_base] = acc;
        }
#endif
    }

    inline void scale_and_store_d(const acc_dtype_t *__restrict__ acc_o,
                                  const acc_dtype_t *__restrict__ logsum,
                                  dtype_out_t *__restrict__ o_block, const int32_t t_r_idx,
                                  const int32_t head_size, const int32_t dim_size) {
        float reciprocal_logsum = 1.0f / logsum[0];
        int j = 0;
#ifdef __AVX2__
        __m256 reciprocal_logsum_vec = _mm256_set1_ps(reciprocal_logsum);
        for (; j <= dim_size - 8; j += 8) {
            _mm256_storeu_ps(o_block + j, _mm256_mul_ps(_mm256_loadu_ps(acc_o + j), reciprocal_logsum_vec));
        }
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
        float32x4_t reciprocal_logsum_vec = vdupq_n_f32(reciprocal_logsum);
        for (; j <= dim_size - 4; j += 4) {
            vst1q_f32(o_block + j, vmulq_f32(vld1q_f32(acc_o + j), reciprocal_logsum_vec));
        }
#endif
        for (; j < dim_size; ++j) {
            o_block[j] = acc_o[j] * reciprocal_logsum;
        }
    }

    inline void mma0_d_n_fixed(const int32_t Bc_n_fixed, const dtype_q_in_t *__restrict__ q_block,
                               const dtype_kv_in_t *__restrict__ k_block, acc_dtype_t *__restrict__ acc_s,
                               const int32_t dim_size, const int32_t kv_stride_size,
                               const int32_t t_r_idx, const int32_t t_c_idx, const int32_t seq_size_q,
                               const int32_t seq_size_k, bool causal_mask) {
        const dtype_q_in_t *q_block_line = q_block;
        for (int32_t b_c_idx = 0; b_c_idx < Bc_n_fixed; ++b_c_idx) {
            const dtype_kv_in_t *k_block_line = k_block + b_c_idx * kv_stride_size;
            float total = 0.0f;
            for (int i = 0; i < dim_size; ++i) { total += q_block_line[i] * MLLM_FP16_TO_FP32(k_block_line[i]); }
            acc_s[b_c_idx] = total;
        }
    }

    inline void softmax_d_n_fixed(const int32_t Bc_n_fixed, acc_dtype_t *__restrict__ acc_s,
                                  acc_dtype_t *scoremax, acc_dtype_t *scoremax_prev,
                                  acc_dtype_t *score_scale, acc_dtype_t *score_sum,
                                  acc_dtype_t *logsum, const float scale,
                                  const int32_t t_r_idx, const int32_t t_c_idx,
                                  const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
        scoremax_prev[0] = scoremax[0];
        float max_val = NEG_INF;
        for (int bc = 0; bc < Bc_n_fixed; ++bc) max_val = fmaxf(max_val, acc_s[bc]);
        scoremax[0] = fmaxf(max_val, scoremax[0]);
        score_scale[0] = expf((scoremax_prev[0] - scoremax[0]) * scale);
        float current_sum = 0.0f;
        for (int bc = 0; bc < Bc_n_fixed; ++bc) {
            float val = expf((acc_s[bc] - scoremax[0]) * scale);
            acc_s[bc] = val;
            current_sum += val;
        }
        score_sum[0] = current_sum;
        logsum[0] = logsum[0] * score_scale[0] + score_sum[0];
    }

    inline void rescale_d_n_fixed(const int32_t Bc_n_fixed, acc_dtype_t *__restrict__ acc_o,
                                  acc_dtype_t *__restrict__ score_scale, const int32_t dim_size,
                                  const int32_t t_r_idx, const int32_t t_c_idx,
                                  const int32_t seq_size_q, const int32_t seq_size_k,
                                  bool causal_mask) {
        float scale = score_scale[0];
        for (int j = 0; j < dim_size; ++j) { acc_o[j] *= scale; }
    }

    inline void mma1_d_n_fixed(const int32_t Bc_n_fixed, const acc_dtype_t *__restrict__ w_block,
                               const dtype_kv_in_t *__restrict__ v_block, acc_dtype_t *__restrict__ acc_o,
                               const int32_t kv_head_size, const int32_t dim_size, const int32_t t_r_idx,
                               const int32_t t_c_idx, const int32_t seq_size_q,
                               const int32_t seq_size_k, bool causal_mask) {
        const int32_t v_stride_size = kv_head_size * dim_size;
        for (int d_base = 0; d_base < dim_size; ++d_base) {
            float acc = acc_o[d_base];
            for (int b_c_idx = 0; b_c_idx < Bc_n_fixed; ++b_c_idx) {
                acc += w_block[b_c_idx] * MLLM_FP16_TO_FP32(v_block[b_c_idx * v_stride_size + d_base]);
            }
            acc_o[d_base] = acc;
        }
    }

private:
    // float scale_;
    acc_dtype_t *acc_o_;
    acc_dtype_t *acc_s_;
    acc_dtype_t *logsum_;
    acc_dtype_t *scoremax_;
    acc_dtype_t *scoremax_prev_;
    acc_dtype_t *score_scale_;
    acc_dtype_t *score_sum_;
};

template <typename Impl>
struct FlashAttn2T {
public:
    using dtype_q_in_t = typename Impl::dtype_q_in_t;
    using dtype_kv_in_t = typename Impl::dtype_kv_in_t;
    using dtype_out_t = typename Impl::dtype_out_t;
    using acc_dtype_t = typename Impl::acc_dtype_t;

    void configure(int32_t Br, int32_t Bc, int32_t Q_Head, int32_t KV_Head, int32_t threads, bool high_precision) {
        impl_.configure(Br, Bc, Q_Head, KV_Head, threads, high_precision);
    }

    void init_workspace(acc_dtype_t *acc_o, acc_dtype_t *acc_s,
                        acc_dtype_t *logsum, acc_dtype_t *scoremax, acc_dtype_t *scoremax_prev,
                        acc_dtype_t *score_scale, acc_dtype_t *score_sum) {
        // Note: workspace pointers are always float, acc_s_cast is removed
        impl_.init_workspace(acc_o, acc_s, logsum, scoremax, scoremax_prev, score_scale, score_sum);
    }

    void operator()(const dtype_q_in_t *__restrict__ Q, const dtype_kv_in_t *__restrict__ K,
                    const dtype_kv_in_t *__restrict__ V, dtype_out_t *__restrict__ O,
                    const int32_t batch_size, const int32_t head_size,
                    const int32_t seq_size_q, const int32_t seq_size_k,
                    const int32_t dim_size, bool causal_mask = true) {
        impl_.fa2(Q, K, V, O, batch_size, head_size, seq_size_q, seq_size_k, dim_size, causal_mask);
    }

private:
    Impl impl_;
};

class WorkspaceManager {
public:
    WorkspaceManager() :
        workspace_{}, current_sizes_{} {
    }

    ~WorkspaceManager() {
        for (int i = 0; i < 7; ++i) {
            if (workspace_[i]) {
                aligned_free(workspace_[i]);
            }
        }
    }

    void **get_workspace(const size_t *required_sizes) {
        for (int i = 0; i < 7; ++i) {
            if (required_sizes[i] > current_sizes_[i]) {
                if (workspace_[i]) {
                    aligned_free(workspace_[i]);
                }
                aligned_alloc(&workspace_[i], required_sizes[i], 32);
                current_sizes_[i] = required_sizes[i];
            }
        }
        return workspace_;
    }

private:
    WorkspaceManager(const WorkspaceManager &) = delete;
    WorkspaceManager &operator=(const WorkspaceManager &) = delete;

    void *workspace_[7];
    size_t current_sizes_[7];
};

} // namespace mobi_attn

void flash_attention_2_forward(
    const void *Q, const void *K, const void *V, void *O,
    int32_t batch_size, int32_t head_size, int32_t seq_size_q, int32_t seq_size_k, int32_t dim_size,
    bool causal_mask, bool use_fp32, int32_t threads, int32_t br, int32_t bc,
    int32_t q_head, int32_t kv_head, bool high_precision_exp) {
    thread_local mobi_attn::WorkspaceManager manager;

    const size_t acc_o_size = threads * br * dim_size * sizeof(float);
    const size_t acc_s_size = threads * br * bc * sizeof(float);
    const size_t logsum_size = threads * br * sizeof(float);
    const size_t scoremax_size = threads * br * sizeof(float);
    const size_t scoremax_prev_size = threads * br * sizeof(float);
    const size_t score_scale_size = threads * br * sizeof(float);
    const size_t score_sum_size = threads * br * sizeof(float);

    const size_t required_sizes[7] = {
        acc_o_size, acc_s_size, logsum_size, scoremax_size,
        scoremax_prev_size, score_scale_size, score_sum_size};

    void **workspace = manager.get_workspace(required_sizes);

    if (use_fp32) {
        mobi_attn::FlashAttn2T<mobi_attn::FA_2_GQA_QKV_FP32_BSHD_O_FP32_BSHD_ACC_FP32_IMPL> op;
        op.configure(br, bc, q_head, kv_head, threads, high_precision_exp);

        op.init_workspace(
            static_cast<float *>(workspace[0]), static_cast<float *>(workspace[1]),
            static_cast<float *>(workspace[2]), static_cast<float *>(workspace[3]),
            static_cast<float *>(workspace[4]), static_cast<float *>(workspace[5]),
            static_cast<float *>(workspace[6]));

        op(static_cast<const float *>(Q), static_cast<const float *>(K), static_cast<const float *>(V),
           static_cast<float *>(O),
           batch_size, head_size, seq_size_q, seq_size_k, dim_size, causal_mask);
    } else {
        mobi_attn::FlashAttn2T<mobi_attn::FA_2_GQA_Q_FP32_KV_FP16_BSHD_O_FP32_BSHD_ACC_FP32_IMPL> op;
        op.configure(br, bc, q_head, kv_head, threads, high_precision_exp);

        op.init_workspace(
            static_cast<float *>(workspace[0]), static_cast<float *>(workspace[1]),
            static_cast<float *>(workspace[2]), static_cast<float *>(workspace[3]),
            static_cast<float *>(workspace[4]), static_cast<float *>(workspace[5]),
            static_cast<float *>(workspace[6]));

        op(static_cast<const float *>(Q), static_cast<const mllm_fp16_t *>(K), static_cast<const mllm_fp16_t *>(V),
           static_cast<float *>(O),
           batch_size, head_size, seq_size_q, seq_size_k, dim_size, causal_mask);
    }
}
#endif // MLLM_FA2_CAL_HPP