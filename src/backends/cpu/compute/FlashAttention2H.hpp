#ifndef MLLM_FA2H_CAL_HPP
#define MLLM_FA2H_CAL_HPP

#include <cstdint>
#include <omp.h>
#include <cassert>
#include <limits>
#include <cmath>
#include <cstring>
#include "Types.hpp"
#include "backends/cpu/third_party/ggml/QuantizeFP16.hpp"
#include "backends/cpu/third_party/ggml/ComputeUtils.hpp"

// 平台相关的头文件和宏定义
#ifdef __AVX2__
#include <immintrin.h>
#define NEG_INF_F32 (-std::numeric_limits<float>::infinity())

// Horizontal max of a __m256 vector
inline float hmax_ps_avx(__m256 x) {
    __m128 lo = _mm256_castps256_ps128(x);
    __m128 hi = _mm256_extractf128_ps(x, 1);
    __m128 max_val = _mm_max_ps(lo, hi);
    max_val = _mm_max_ps(max_val, _mm_shuffle_ps(max_val, max_val, _MM_SHUFFLE(0, 0, 2, 2)));
    max_val = _mm_max_ps(max_val, _mm_shuffle_ps(max_val, max_val, _MM_SHUFFLE(0, 0, 0, 1)));
    return _mm_cvtss_f32(max_val);
}

// Horizontal sum of a __m256 vector
inline float hadd_ps_avx(__m256 x) {
    __m128 lo = _mm256_castps256_ps128(x);
    __m128 hi = _mm256_extractf128_ps(x, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

#elif __ARM_NEON
#include <arm_neon.h>
#define NEG_INF_F32 (-std::numeric_limits<float>::infinity())

// NEON版本：水平最大值 (Horizontal max of a float32x4_t vector)
inline float hmax_ps_neon(float32x4_t x) {
    return vmaxvq_f32(x);
}

// NEON版本：水平求和 (Horizontal sum of a float32x4_t vector)
inline float hadd_ps_neon(float32x4_t x) {
    return vaddvq_f32(x);
}

#else
#error "Unsupported architecture. Please define __AVX2__ or __ARM_NEON."
#endif

// Common aligned allocation/free functions
inline void platform_aligned_alloc(void **ptr, size_t required_bytes, size_t align) {
    if (align % sizeof(void *) != 0 || (align & (align - 1)) != 0) {
        *ptr = nullptr;
        return;
    }
    if (posix_memalign(ptr, align, required_bytes) != 0) {
        *ptr = nullptr;
    }
}

inline void platform_aligned_free(void *ptr) {
    free(ptr);
}

namespace mobi_attn {

// ========================================
// FlashAttention2 核心实现 (FP32版本) - BHSD Layout
// ========================================
struct FA_2_GQA_QKV_FP32_BHSD_O_FP32_BHSD_ACC_FP32_IMPL {
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

    // Workspace pointers
    acc_dtype_t *acc_o_;
    acc_dtype_t *acc_s_;
    acc_dtype_t *logsum_;
    acc_dtype_t *scoremax_;
    acc_dtype_t *scoremax_prev_;
    acc_dtype_t *score_scale_;
    acc_dtype_t *score_sum_;

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
             const int32_t dim_size, bool causal_mask,
             int32_t q_head_skp, int32_t k_head_skp, int32_t v_head_skp) {
        assert(Br == Bc);
        assert(Q_Head % KV_Head == 0);
        assert(head_size % threads == 0);
#ifdef __AVX2__
        assert(dim_size % 8 == 0);
#elif __ARM_NEON
        assert(dim_size % 4 == 0);
#endif

        if (seq_size_q != 1) {
            __fa2_prefill_append(Q, K, V, O, batch_size, head_size, seq_size_q, seq_size_k, dim_size, causal_mask, q_head_skp, k_head_skp, v_head_skp);
        } else {
            __fa2_decode(Q, K, V, O, batch_size, head_size, seq_size_q, seq_size_k, dim_size, causal_mask, q_head_skp, k_head_skp, v_head_skp);
        }
    }

private:
    inline void __fa2_prefill_append(const dtype_t *__restrict__ Q, const dtype_t *__restrict__ K,
                                     const dtype_t *__restrict__ V, dtype_t *__restrict__ O,
                                     const int32_t batch_size, const int32_t head_size,
                                     const int32_t seq_size_q, const int32_t seq_size_k,
                                     const int32_t dim_size, bool causal_mask,
                                     int32_t q_head_skp, int32_t k_head_skp, int32_t v_head_skp) {
        const int32_t Tr = (seq_size_q + Br - 1) / Br;
        const int32_t Tc = (seq_size_k + Bc - 1) / Bc;

        const float local_scale = 1.0f / sqrtf(static_cast<float>(dim_size));
        const int32_t kv_group_size = Q_Head / KV_Head;

        for (int32_t b_idx = 0; b_idx < batch_size; ++b_idx) {
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1) if (threads > 1)
            for (int32_t h_idx = 0; h_idx < head_size; ++h_idx) {
                const int32_t thread_id = omp_get_thread_num();
                const int32_t this_thread_head = h_idx;
                const int32_t this_thread_kv_head = this_thread_head / kv_group_size;

                const dtype_t *q_batch_base = Q + b_idx * head_size * q_head_skp;
                const dtype_t *k_batch_base = K + b_idx * KV_Head * k_head_skp;
                const dtype_t *v_batch_base = V + b_idx * KV_Head * v_head_skp;
                dtype_t *o_batch_base = O + b_idx * head_size * q_head_skp;

                for (int t_r_idx = 0; t_r_idx < Tr; ++t_r_idx) {
                    const int32_t Br_n_fixed = (t_r_idx == Tr - 1) ? (seq_size_q - t_r_idx * Br) : Br;
                    init_temp(logsum_ + thread_id * Br, scoremax_ + thread_id * Br,
                              acc_o_ + thread_id * Br * dim_size, Br_n_fixed, dim_size);

                    for (int t_c_idx = 0; t_c_idx < Tc; ++t_c_idx) {
                        const int32_t Bc_n_fixed = (t_c_idx == Tc - 1) ? (seq_size_k - t_c_idx * Bc) : Bc;

                        const dtype_t *tile_q = q_batch_base + this_thread_head * q_head_skp + t_r_idx * Br * dim_size;
                        const dtype_t *tile_k = k_batch_base + this_thread_kv_head * k_head_skp + t_c_idx * Bc * dim_size;
                        const dtype_t *tile_v = v_batch_base + this_thread_kv_head * v_head_skp + t_c_idx * Bc * dim_size;

                        acc_dtype_t *tile_acc_s = acc_s_ + thread_id * Br * Bc;
                        acc_dtype_t *acc_o = acc_o_ + thread_id * Br * dim_size;

                        mma0(Br_n_fixed, Bc_n_fixed, tile_q, tile_k, tile_acc_s, dim_size, t_r_idx, t_c_idx, seq_size_q, seq_size_k, causal_mask);
                        softmax(Br_n_fixed, Bc_n_fixed, tile_acc_s, scoremax_ + thread_id * Br, scoremax_prev_ + thread_id * Br, score_scale_ + thread_id * Br, score_sum_ + thread_id * Br, logsum_ + thread_id * Br, local_scale);
                        rescale(Br_n_fixed, acc_o, score_scale_ + thread_id * Br, dim_size);
                        mma1(Br_n_fixed, Bc_n_fixed, tile_acc_s, tile_v, acc_o, dim_size);
                    }

                    dtype_t *o_block_ptr = o_batch_base + this_thread_head * q_head_skp + t_r_idx * Br * dim_size;
                    scale_and_store(Br_n_fixed, acc_o_ + thread_id * Br * dim_size, logsum_ + thread_id * Br, o_block_ptr, dim_size);
                }
            }
        }
    }

    inline void __fa2_decode(const dtype_t *__restrict__ Q, const dtype_t *__restrict__ K,
                             const dtype_t *__restrict__ V, dtype_t *__restrict__ O,
                             const int32_t batch_size, const int32_t head_size,
                             const int32_t seq_size_q, const int32_t seq_size_k,
                             const int32_t dim_size, bool causal_mask,
                             int32_t q_head_skp, int32_t k_head_skp, int32_t v_head_skp) {
        const int32_t Tc = (seq_size_k + Bc - 1) / Bc;

        const float local_scale = 1.0f / sqrtf(static_cast<float>(dim_size));
        const int32_t kv_group_size = (Q_Head > 0 && KV_Head > 0) ? Q_Head / KV_Head : 1;

        for (int32_t b_idx = 0; b_idx < batch_size; ++b_idx) {
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1) if (threads > 1)
            for (int32_t h_idx = 0; h_idx < head_size; ++h_idx) {
                const int32_t thread_id = omp_get_thread_num();
                const int32_t this_thread_head = h_idx;
                const int32_t this_thread_kv_head = this_thread_head / kv_group_size;

                init_temp_d(logsum_ + thread_id, scoremax_ + thread_id, acc_o_ + thread_id * dim_size, dim_size);

                const dtype_t *q_batch_base = Q + b_idx * head_size * q_head_skp;
                const dtype_t *k_batch_base = K + b_idx * KV_Head * k_head_skp;
                const dtype_t *v_batch_base = V + b_idx * KV_Head * v_head_skp;
                dtype_t *o_batch_base = O + b_idx * head_size * q_head_skp;

                for (int t_c_idx = 0; t_c_idx < Tc; ++t_c_idx) {
                    const int32_t Bc_n_fixed = (t_c_idx == Tc - 1) ? (seq_size_k - t_c_idx * Bc) : Bc;

                    const dtype_t *tile_q = q_batch_base + this_thread_head * q_head_skp;
                    const dtype_t *tile_k = k_batch_base + this_thread_kv_head * k_head_skp + t_c_idx * Bc * dim_size;
                    const dtype_t *tile_v = v_batch_base + this_thread_kv_head * v_head_skp + t_c_idx * Bc * dim_size;

                    acc_dtype_t *tile_acc_s = acc_s_ + thread_id * Bc;
                    acc_dtype_t *acc_o = acc_o_ + thread_id * dim_size;

                    mma0_d(Bc_n_fixed, tile_q, tile_k, tile_acc_s, dim_size, t_c_idx, seq_size_k, causal_mask);
                    softmax_d(Bc_n_fixed, tile_acc_s, scoremax_ + thread_id, scoremax_prev_ + thread_id, score_scale_ + thread_id, score_sum_ + thread_id, logsum_ + thread_id, local_scale);
                    rescale_d(acc_o, score_scale_ + thread_id, dim_size);
                    mma1_d(Bc_n_fixed, tile_acc_s, tile_v, acc_o, dim_size);
                }

                dtype_t *o_block_ptr = o_batch_base + this_thread_head * q_head_skp;
                scale_and_store_d(acc_o_ + thread_id * dim_size, logsum_ + thread_id, o_block_ptr, dim_size);
            }
        }
    }

    inline void init_temp(acc_dtype_t *logsum, acc_dtype_t *scoremax, acc_dtype_t *acc_o, const int32_t Br_n_fixed, const int32_t dim_size) {
        for (int i = 0; i < Br_n_fixed; ++i) {
            logsum[i] = 0.0f;
            scoremax[i] = NEG_INF_F32;
        }
#ifdef __AVX2__
        __m256 zero_vec = _mm256_setzero_ps();
        for (int j = 0; j < Br_n_fixed * dim_size; j += 8) {
            _mm256_storeu_ps(acc_o + j, zero_vec);
        }
#elif __ARM_NEON
        float32x4_t zero_vec = vdupq_n_f32(0.0f);
        for (int j = 0; j < Br_n_fixed * dim_size; j += 4) {
            vst1q_f32(acc_o + j, zero_vec);
        }
#endif
    }

    inline void mma0(const int32_t Br_n_fixed, const int32_t Bc_n_fixed,
                     const dtype_t *__restrict__ q_block, const dtype_t *__restrict__ k_block,
                     acc_dtype_t *__restrict__ acc_s, const int32_t dim_size,
                     const int32_t t_r_idx, const int32_t t_c_idx,
                     const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
        const int32_t global_r_start = t_r_idx * Br;
        const int32_t global_c_start = t_c_idx * Bc;
        const int32_t delta_pos = seq_size_k - seq_size_q;

        for (int32_t b_r_idx = 0; b_r_idx < Br_n_fixed; ++b_r_idx) {
            const dtype_t *q_block_line = q_block + b_r_idx * dim_size;
            for (int32_t b_c_idx = 0; b_c_idx < Bc_n_fixed; ++b_c_idx) {
                const int32_t global_r_idx = global_r_start + b_r_idx;
                const int32_t global_c_idx = global_c_start + b_c_idx;
                if (causal_mask && (global_c_idx > global_r_idx + delta_pos)) {
                    acc_s[b_r_idx * Bc + b_c_idx] = NEG_INF_F32;
                    continue;
                }
                const dtype_t *k_block_line = k_block + b_c_idx * dim_size;
#ifdef __AVX2__
                __m256 sum_vec = _mm256_setzero_ps();
                int i = 0;
                for (; i <= dim_size - 8; i += 8) {
                    __m256 q_vec = _mm256_loadu_ps(q_block_line + i);
                    __m256 k_vec = _mm256_loadu_ps(k_block_line + i);
                    sum_vec = _mm256_fmadd_ps(q_vec, k_vec, sum_vec);
                }
                acc_dtype_t total = hadd_ps_avx(sum_vec);
                for (; i < dim_size; ++i) { total += q_block_line[i] * k_block_line[i]; }
#elif __ARM_NEON
                float32x4_t sum_vec = vdupq_n_f32(0.0f);
                int i = 0;
                for (; i <= dim_size - 4; i += 4) {
                    sum_vec = vfmaq_f32(sum_vec, vld1q_f32(q_block_line + i), vld1q_f32(k_block_line + i));
                }
                acc_dtype_t total = hadd_ps_neon(sum_vec);
                for (; i < dim_size; ++i) { total += q_block_line[i] * k_block_line[i]; }
#endif
                acc_s[b_r_idx * Bc + b_c_idx] = total;
            }
        }
    }

    inline void softmax(const int32_t Br_n_fixed, const int32_t Bc_n_fixed,
                        acc_dtype_t *__restrict__ acc_s, acc_dtype_t *scoremax, acc_dtype_t *scoremax_prev,
                        acc_dtype_t *score_scale, acc_dtype_t *score_sum, acc_dtype_t *logsum,
                        const float scale) {
        memcpy(scoremax_prev, scoremax, Br_n_fixed * sizeof(acc_dtype_t));

        for (int br = 0; br < Br_n_fixed; ++br) {
            acc_dtype_t *row = acc_s + br * Bc;
            float block_max = NEG_INF_F32;
            for (int bc = 0; bc < Bc_n_fixed; ++bc) {
                block_max = fmaxf(block_max, row[bc]);
            }
            scoremax[br] = fmaxf(scoremax[br], block_max);
        }

        for (int br = 0; br < Br_n_fixed; ++br) {
            score_scale[br] = expf((scoremax_prev[br] - scoremax[br]) * scale);
        }

        for (int br = 0; br < Br_n_fixed; ++br) {
            const float current_max = scoremax[br];
            acc_dtype_t *row = acc_s + br * Bc;
            float sum = 0.0f;
            for (int bc = 0; bc < Bc_n_fixed; ++bc) {
                if (row[bc] == NEG_INF_F32) {
                    row[bc] = 0.0f;
                    continue;
                }
                float val = expf((row[bc] - current_max) * scale);
                row[bc] = val;
                sum += val;
            }
            score_sum[br] = sum;
        }

        for (int br = 0; br < Br_n_fixed; ++br) {
            logsum[br] = logsum[br] * score_scale[br] + score_sum[br];
        }
    }

    inline void rescale(const int32_t Br_n_fixed, acc_dtype_t *__restrict__ acc_o, acc_dtype_t *__restrict__ score_scale, const int32_t dim_size) {
        for (int i = 0; i < Br_n_fixed; ++i) {
#ifdef __AVX2__
            __m256 scale_v = _mm256_set1_ps(score_scale[i]);
            float *row_ptr = acc_o + i * dim_size;
            for (int j = 0; j < dim_size; j += 8) {
                __m256 acc = _mm256_loadu_ps(row_ptr + j);
                _mm256_storeu_ps(row_ptr + j, _mm256_mul_ps(acc, scale_v));
            }
#elif __ARM_NEON
            float32x4_t scale_v = vdupq_n_f32(score_scale[i]);
            float *row_ptr = acc_o + i * dim_size;
            for (int j = 0; j < dim_size; j += 4) {
                float32x4_t acc = vld1q_f32(row_ptr + j);
                vst1q_f32(row_ptr + j, vmulq_f32(acc, scale_v));
            }
#endif
        }
    }

    inline void mma1(const int32_t Br_n_fixed, const int32_t Bc_n_fixed,
                     const acc_dtype_t *__restrict__ w_block, const dtype_t *__restrict__ v_block,
                     acc_dtype_t *__restrict__ acc_o, const int32_t dim_size) {
        const int32_t v_stride_size = dim_size;

        for (int b_r_idx = 0; b_r_idx < Br_n_fixed; ++b_r_idx) {
#ifdef __AVX2__
            for (int d_base = 0; d_base < dim_size; d_base += 8) {
                __m256 acc = _mm256_loadu_ps(acc_o + b_r_idx * dim_size + d_base);
                for (int b_c_idx = 0; b_c_idx < Bc_n_fixed; ++b_c_idx) {
                    acc = _mm256_fmadd_ps(_mm256_set1_ps(w_block[b_r_idx * Bc + b_c_idx]), _mm256_loadu_ps(v_block + b_c_idx * v_stride_size + d_base), acc);
                }
                _mm256_storeu_ps(acc_o + b_r_idx * dim_size + d_base, acc);
            }
#elif __ARM_NEON
            for (int d_base = 0; d_base < dim_size; d_base += 4) {
                float32x4_t acc = vld1q_f32(acc_o + b_r_idx * dim_size + d_base);
                for (int b_c_idx = 0; b_c_idx < Bc_n_fixed; ++b_c_idx) {
                    acc = vfmaq_f32(acc, vdupq_n_f32(w_block[b_r_idx * Bc + b_c_idx]), vld1q_f32(v_block + b_c_idx * v_stride_size + d_base));
                }
                vst1q_f32(acc_o + b_r_idx * dim_size + d_base, acc);
            }
#endif
        }
    }

    inline void scale_and_store(const int32_t Br_n_fixed, const acc_dtype_t *__restrict__ acc_o,
                                const acc_dtype_t *__restrict__ logsum,
                                dtype_t *__restrict__ o_block, const int32_t dim_size) {
        for (int i = 0; i < Br_n_fixed; ++i) {
            dtype_t *o_block_line = o_block + i * dim_size;
            float reciprocal_logsum = (logsum[i] == 0.0f) ? 0.0f : 1.0f / logsum[i];
#ifdef __AVX2__
            __m256 reciprocal_logsum_vec = _mm256_set1_ps(reciprocal_logsum);
            for (int j = 0; j <= dim_size - 8; j += 8) {
                __m256 vec_acc_o = _mm256_loadu_ps(acc_o + i * dim_size + j);
                _mm256_storeu_ps(o_block_line + j, _mm256_mul_ps(vec_acc_o, reciprocal_logsum_vec));
            }
            for (int j = dim_size - (dim_size % 8); j < dim_size; ++j) { o_block_line[j] = acc_o[i * dim_size + j] * reciprocal_logsum; }
#elif __ARM_NEON
            float32x4_t reciprocal_logsum_vec = vdupq_n_f32(reciprocal_logsum);
            for (int j = 0; j <= dim_size - 4; j += 4) {
                float32x4_t vec_acc_o = vld1q_f32(acc_o + i * dim_size + j);
                vst1q_f32(o_block_line + j, vmulq_f32(vec_acc_o, reciprocal_logsum_vec));
            }
            for (int j = dim_size - (dim_size % 4); j < dim_size; ++j) { o_block_line[j] = acc_o[i * dim_size + j] * reciprocal_logsum; }
#endif
        }
    }

    inline void init_temp_d(acc_dtype_t *logsum, acc_dtype_t *scoremax, acc_dtype_t *acc_o, const int32_t dim_size) {
        logsum[0] = 0.0f;
        scoremax[0] = NEG_INF_F32;
#ifdef __AVX2__
        __m256 zero_vec = _mm256_setzero_ps();
        for (int i = 0; i < dim_size; i += 8) { _mm256_storeu_ps(acc_o + i, zero_vec); }
#elif __ARM_NEON
        float32x4_t zero_vec = vdupq_n_f32(0.0f);
        for (int i = 0; i < dim_size; i += 4) { vst1q_f32(acc_o + i, zero_vec); }
#endif
    }

    inline void mma0_d(const int32_t Bc_n_fixed, const dtype_t *__restrict__ q_block,
                       const dtype_t *__restrict__ k_block, acc_dtype_t *__restrict__ acc_s,
                       const int32_t dim_size, const int32_t t_c_idx,
                       const int32_t seq_size_k, bool causal_mask) {
        const int32_t global_c_start = t_c_idx * Bc;
        const int32_t global_r_idx = seq_size_k - 1;

        for (int32_t b_c_idx = 0; b_c_idx < Bc_n_fixed; ++b_c_idx) {
            const int32_t global_c_idx = global_c_start + b_c_idx;
            if (causal_mask && global_c_idx > global_r_idx) {
                acc_s[b_c_idx] = NEG_INF_F32;
                continue;
            }
            const dtype_t *k_block_line = k_block + b_c_idx * dim_size;
#ifdef __AVX2__
            __m256 sum_vec = _mm256_setzero_ps();
            int i = 0;
            for (; i <= dim_size - 8; i += 8) {
                sum_vec = _mm256_fmadd_ps(_mm256_loadu_ps(q_block + i), _mm256_loadu_ps(k_block_line + i), sum_vec);
            }
            acc_dtype_t total = hadd_ps_avx(sum_vec);
            for (int i = dim_size - (dim_size % 8); i < dim_size; ++i) { total += q_block[i] * k_block_line[i]; }
#elif __ARM_NEON
            float32x4_t sum_vec = vdupq_n_f32(0.0f);
            int i = 0;
            for (; i <= dim_size - 4; i += 4) {
                sum_vec = vfmaq_f32(sum_vec, vld1q_f32(q_block + i), vld1q_f32(k_block_line + i));
            }
            acc_dtype_t total = hadd_ps_neon(sum_vec);
            for (int i = dim_size - (dim_size % 4); i < dim_size; ++i) { total += q_block[i] * k_block_line[i]; }
#endif
            acc_s[b_c_idx] = total;
        }
    }

    inline void softmax_d(const int32_t Bc_n_fixed, acc_dtype_t *__restrict__ acc_s,
                          acc_dtype_t *scoremax, acc_dtype_t *scoremax_prev,
                          acc_dtype_t *score_scale, acc_dtype_t *score_sum, acc_dtype_t *logsum,
                          const float scale) {
        scoremax_prev[0] = scoremax[0];

        float block_max = NEG_INF_F32;
        for (int bc = 0; bc < Bc_n_fixed; ++bc) block_max = fmaxf(block_max, acc_s[bc]);
        scoremax[0] = fmaxf(scoremax[0], block_max);

        score_scale[0] = expf((scoremax_prev[0] - scoremax[0]) * scale);

        float current_sum = 0.0f;
        for (int bc = 0; bc < Bc_n_fixed; ++bc) {
            if (acc_s[bc] == NEG_INF_F32) {
                acc_s[bc] = 0.0f;
                continue;
            }
            float val = expf((acc_s[bc] - scoremax[0]) * scale);
            acc_s[bc] = val;
            current_sum += val;
        }
        score_sum[0] = current_sum;
        logsum[0] = logsum[0] * score_scale[0] + score_sum[0];
    }

    inline void rescale_d(acc_dtype_t *__restrict__ acc_o, acc_dtype_t *__restrict__ score_scale, const int32_t dim_size) {
#ifdef __AVX2__
        __m256 scale_v = _mm256_set1_ps(score_scale[0]);
        for (int j = 0; j < dim_size; j += 8) {
            _mm256_storeu_ps(acc_o + j, _mm256_mul_ps(_mm256_loadu_ps(acc_o + j), scale_v));
        }
#elif __ARM_NEON
        float32x4_t scale_v = vdupq_n_f32(score_scale[0]);
        for (int j = 0; j < dim_size; j += 4) {
            vst1q_f32(acc_o + j, vmulq_f32(vld1q_f32(acc_o + j), scale_v));
        }
#endif
    }

    inline void mma1_d(const int32_t Bc_n_fixed, const acc_dtype_t *__restrict__ w_block,
                       const dtype_t *__restrict__ v_block, acc_dtype_t *__restrict__ acc_o, const int32_t dim_size) {
        const int32_t v_stride_size = dim_size;
#ifdef __AVX2__
        for (int d_base = 0; d_base < dim_size; d_base += 8) {
            __m256 acc = _mm256_loadu_ps(acc_o + d_base);
            for (int b_c_idx = 0; b_c_idx < Bc_n_fixed; ++b_c_idx) {
                acc = _mm256_fmadd_ps(_mm256_set1_ps(w_block[b_c_idx]), _mm256_loadu_ps(v_block + b_c_idx * v_stride_size + d_base), acc);
            }
            _mm256_storeu_ps(acc_o + d_base, acc);
        }
#elif __ARM_NEON
        for (int d_base = 0; d_base < dim_size; d_base += 4) {
            float32x4_t acc = vld1q_f32(acc_o + d_base);
            for (int b_c_idx = 0; b_c_idx < Bc_n_fixed; ++b_c_idx) {
                acc = vfmaq_f32(acc, vdupq_n_f32(w_block[b_c_idx]), vld1q_f32(v_block + b_c_idx * v_stride_size + d_base));
            }
            vst1q_f32(acc_o + d_base, acc);
        }
#endif
    }

    inline void scale_and_store_d(const acc_dtype_t *__restrict__ acc_o, const acc_dtype_t *__restrict__ logsum,
                                  dtype_t *__restrict__ o_block, const int32_t dim_size) {
        float reciprocal_logsum = (logsum[0] == 0.0f) ? 0.0f : 1.0f / logsum[0];
#ifdef __AVX2__
        __m256 reciprocal_logsum_vec = _mm256_set1_ps(reciprocal_logsum);
        for (int j = 0; j <= dim_size - 8; j += 8) {
            _mm256_storeu_ps(o_block + j, _mm256_mul_ps(_mm256_loadu_ps(acc_o + j), reciprocal_logsum_vec));
        }
        for (int j = dim_size - (dim_size % 8); j < dim_size; ++j) { o_block[j] = acc_o[j] * reciprocal_logsum; }
#elif __ARM_NEON
        float32x4_t reciprocal_logsum_vec = vdupq_n_f32(reciprocal_logsum);
        for (int j = 0; j <= dim_size - 4; j += 4) {
            vst1q_f32(o_block + j, vmulq_f32(vld1q_f32(acc_o + j), reciprocal_logsum_vec));
        }
        for (int j = dim_size - (dim_size % 4); j < dim_size; ++j) { o_block[j] = acc_o[j] * reciprocal_logsum; }
#endif
    }
};

// ========================================
// FlashAttention2 核心实现 ( Q FP32/KV FP16 输入,FP32 输出版本) - BHSD Layout
// ========================================
struct FA_2_GQA_Q_FP32_KV_FP16_BHSD_O_FP32_BHSD_ACC_FP32_IMPL {
    using dtype_q_in_t = float;
    using dtype_kv_in_t = mllm_fp16_t;
    using dtype_out_t = float;
    using acc_dtype_t = float;

    int32_t Br, Bc, Q_Head, KV_Head, threads;
    bool high_precision;

    acc_dtype_t *acc_o_;
    acc_dtype_t *acc_s_;
    acc_dtype_t *logsum_;
    acc_dtype_t *scoremax_;
    acc_dtype_t *scoremax_prev_;
    acc_dtype_t *score_scale_;
    acc_dtype_t *score_sum_;

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
             const int32_t dim_size, bool causal_mask,
             int32_t q_head_skp, int32_t k_head_skp, int32_t v_head_skp) {
        assert(Br == Bc);
        assert(head_size % threads == 0);
        assert(Q_Head % KV_Head == 0);

#ifdef __AVX2__
        assert(dim_size % 8 == 0);
#elif __ARM_NEON
        assert(dim_size % 4 == 0);
#endif

        if (seq_size_q != 1) {
            __fa2_prefill_append(Q, K, V, O, batch_size, head_size, seq_size_q, seq_size_k, dim_size, causal_mask, q_head_skp, k_head_skp, v_head_skp);
        } else {
            __fa2_decode(Q, K, V, O, batch_size, head_size, seq_size_q, seq_size_k, dim_size, causal_mask, q_head_skp, k_head_skp, v_head_skp);
        }
    }

private:
    inline void __fa2_prefill_append(const dtype_q_in_t *__restrict__ Q, const dtype_kv_in_t *__restrict__ K,
                                     const dtype_kv_in_t *__restrict__ V, dtype_out_t *__restrict__ O,
                                     const int32_t batch_size, const int32_t head_size,
                                     const int32_t seq_size_q, const int32_t seq_size_k,
                                     const int32_t dim_size, bool causal_mask,
                                     int32_t q_head_skp, int32_t k_head_skp, int32_t v_head_skp) {
        const int32_t Tr = (seq_size_q + Br - 1) / Br;
        const int32_t Tc = (seq_size_k + Bc - 1) / Bc;

        const float local_scale = 1.0f / sqrtf(static_cast<float>(dim_size));
        const int32_t kv_group_size = (Q_Head > 0 && KV_Head > 0) ? Q_Head / KV_Head : 1;

        for (int32_t b_idx = 0; b_idx < batch_size; ++b_idx) {
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1) if (threads > 1)
            for (int32_t h_idx = 0; h_idx < head_size; ++h_idx) {
                const int32_t thread_id = omp_get_thread_num();
                const int32_t this_thread_head = h_idx;
                const int32_t this_thread_kv_head = this_thread_head / kv_group_size;

                const dtype_q_in_t *q_batch_base = Q + b_idx * head_size * q_head_skp;
                const dtype_kv_in_t *k_batch_base = K + b_idx * KV_Head * k_head_skp;
                const dtype_kv_in_t *v_batch_base = V + b_idx * KV_Head * v_head_skp;
                dtype_out_t *o_batch_base = O + b_idx * head_size * q_head_skp;

                for (int t_r_idx = 0; t_r_idx < Tr; ++t_r_idx) {
                    const int32_t Br_n_fixed = (t_r_idx == Tr - 1) ? (seq_size_q - t_r_idx * Br) : Br;
                    init_temp(logsum_ + thread_id * Br, scoremax_ + thread_id * Br,
                              acc_o_ + thread_id * Br * dim_size, Br_n_fixed, dim_size);
                    for (int t_c_idx = 0; t_c_idx < Tc; ++t_c_idx) {
                        const int32_t Bc_n_fixed = (t_c_idx == Tc - 1) ? (seq_size_k - t_c_idx * Bc) : Bc;

                        const dtype_q_in_t *tile_q = q_batch_base + this_thread_head * q_head_skp + t_r_idx * Br * dim_size;
                        const dtype_kv_in_t *tile_k = k_batch_base + this_thread_kv_head * k_head_skp + t_c_idx * Bc * dim_size;
                        const dtype_kv_in_t *tile_v = v_batch_base + this_thread_kv_head * v_head_skp + t_c_idx * Bc * dim_size;

                        acc_dtype_t *tile_acc_s = acc_s_ + thread_id * Br * Bc;
                        acc_dtype_t *acc_o = acc_o_ + thread_id * Br * dim_size;

                        mma0(Br_n_fixed, Bc_n_fixed, tile_q, tile_k, tile_acc_s, dim_size, t_r_idx, t_c_idx, seq_size_q, seq_size_k, causal_mask);
                        softmax(Br_n_fixed, Bc_n_fixed, tile_acc_s, scoremax_ + thread_id * Br, scoremax_prev_ + thread_id * Br, score_scale_ + thread_id * Br, score_sum_ + thread_id * Br, logsum_ + thread_id * Br, local_scale);
                        rescale(Br_n_fixed, acc_o, score_scale_ + thread_id * Br, dim_size);
                        mma1(Br_n_fixed, Bc_n_fixed, tile_acc_s, tile_v, acc_o, dim_size);
                    }
                    dtype_out_t *o_block_ptr = o_batch_base + this_thread_head * q_head_skp + t_r_idx * Br * dim_size;
                    scale_and_store(Br_n_fixed, acc_o_ + thread_id * Br * dim_size, logsum_ + thread_id * Br, o_block_ptr, dim_size);
                }
            }
        }
    }

    inline void __fa2_decode(const dtype_q_in_t *__restrict__ Q, const dtype_kv_in_t *__restrict__ K,
                             const dtype_kv_in_t *__restrict__ V, dtype_out_t *__restrict__ O,
                             const int32_t batch_size, const int32_t head_size,
                             const int32_t seq_size_q, const int32_t seq_size_k,
                             const int32_t dim_size, bool causal_mask,
                             int32_t q_head_skp, int32_t k_head_skp, int32_t v_head_skp) {
        const int32_t Tc = (seq_size_k + Bc - 1) / Bc;

        const float local_scale = 1.0f / sqrtf(static_cast<float>(dim_size));
        const int32_t kv_group_size = (Q_Head > 0 && KV_Head > 0) ? Q_Head / KV_Head : 1;

        for (int32_t b_idx = 0; b_idx < batch_size; ++b_idx) {
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1) if (threads > 1)
            for (int32_t h_idx = 0; h_idx < head_size; ++h_idx) {
                const int32_t thread_id = omp_get_thread_num();
                const int32_t this_thread_head = h_idx;
                const int32_t this_thread_kv_head = this_thread_head / kv_group_size;

                init_temp_d(logsum_ + thread_id, scoremax_ + thread_id, acc_o_ + thread_id * dim_size, dim_size);

                const dtype_q_in_t *q_batch_base = Q + b_idx * head_size * q_head_skp;
                const dtype_kv_in_t *k_batch_base = K + b_idx * KV_Head * k_head_skp;
                const dtype_kv_in_t *v_batch_base = V + b_idx * KV_Head * v_head_skp;
                dtype_out_t *o_batch_base = O + b_idx * head_size * q_head_skp;

                for (int t_c_idx = 0; t_c_idx < Tc; ++t_c_idx) {
                    const int32_t Bc_n_fixed = (t_c_idx == Tc - 1) ? (seq_size_k - t_c_idx * Bc) : Bc;

                    const dtype_q_in_t *tile_q = q_batch_base + this_thread_head * q_head_skp;
                    const dtype_kv_in_t *tile_k = k_batch_base + this_thread_kv_head * k_head_skp + t_c_idx * Bc * dim_size;
                    const dtype_kv_in_t *tile_v = v_batch_base + this_thread_kv_head * v_head_skp + t_c_idx * Bc * dim_size;

                    acc_dtype_t *tile_acc_s = acc_s_ + thread_id * Bc;
                    acc_dtype_t *acc_o = acc_o_ + thread_id * dim_size;

                    mma0_d(Bc_n_fixed, tile_q, tile_k, tile_acc_s, dim_size, t_c_idx, seq_size_k, causal_mask);
                    softmax_d(Bc_n_fixed, tile_acc_s, scoremax_ + thread_id, scoremax_prev_ + thread_id, score_scale_ + thread_id, score_sum_ + thread_id, logsum_ + thread_id, local_scale);
                    rescale_d(acc_o, score_scale_ + thread_id, dim_size);
                    mma1_d(Bc_n_fixed, tile_acc_s, tile_v, acc_o, dim_size);
                }
                dtype_out_t *o_block_ptr = o_batch_base + this_thread_head * q_head_skp;
                scale_and_store_d(acc_o_ + thread_id * dim_size, logsum_ + thread_id, o_block_ptr, dim_size);
            }
        }
    }

    inline void init_temp(acc_dtype_t *logsum, acc_dtype_t *scoremax, acc_dtype_t *acc_o, const int32_t Br_n_fixed, const int32_t dim_size) {
        for (int i = 0; i < Br_n_fixed; ++i) {
            logsum[i] = 0.0f;
            scoremax[i] = NEG_INF_F32;
        }
#ifdef __AVX2__
        __m256 zero_vec = _mm256_setzero_ps();
        for (int j = 0; j < Br_n_fixed * dim_size; j += 8) {
            _mm256_storeu_ps(acc_o + j, zero_vec);
        }
#elif __ARM_NEON
        float32x4_t zero_vec = vdupq_n_f32(0.0f);
        for (int j = 0; j < Br_n_fixed * dim_size; j += 4) {
            vst1q_f32(acc_o + j, zero_vec);
        }
#endif
    }

    inline void mma0(const int32_t Br_n_fixed, const int32_t Bc_n_fixed,
                     const dtype_q_in_t *__restrict__ q_block, const dtype_kv_in_t *__restrict__ k_block,
                     acc_dtype_t *__restrict__ acc_s, const int32_t dim_size,
                     const int32_t t_r_idx, const int32_t t_c_idx,
                     const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
        const int32_t global_r_start = t_r_idx * Br;
        const int32_t global_c_start = t_c_idx * Bc;
        const int32_t delta_pos = seq_size_k - seq_size_q;

        for (int32_t b_r_idx = 0; b_r_idx < Br_n_fixed; ++b_r_idx) {
            const dtype_q_in_t *q_block_line = q_block + b_r_idx * dim_size;
            for (int32_t b_c_idx = 0; b_c_idx < Bc_n_fixed; ++b_c_idx) {
                const int32_t global_r_idx = global_r_start + b_r_idx;
                const int32_t global_c_idx = global_c_start + b_c_idx;
                if (causal_mask && (global_c_idx > global_r_idx + delta_pos)) {
                    acc_s[b_r_idx * Bc + b_c_idx] = NEG_INF_F32;
                    continue;
                }
                const dtype_kv_in_t *k_block_line = k_block + b_c_idx * dim_size;
#ifdef __AVX2__
                __m256 sum_vec = _mm256_setzero_ps();
                int i = 0;
                for (; i <= dim_size - 8; i += 8) {
                    __m256 q_vec = _mm256_loadu_ps(q_block_line + i);
                    __m256 k_vec = MLLM_F32Cx8_LOAD(k_block_line + i);
                    sum_vec = _mm256_fmadd_ps(q_vec, k_vec, sum_vec);
                }
                acc_dtype_t total = hadd_ps_avx(sum_vec);
                for (; i < dim_size; ++i) { total += q_block_line[i] * MLLM_FP16_TO_FP32(k_block_line[i]); }
#elif __ARM_NEON
                float32x4_t sum_vec = vdupq_n_f32(0.0f);
                int i = 0;
                for (; i <= dim_size - 4; i += 4) {
                    float32x4_t q_vec = vld1q_f32(q_block_line + i);
                    float32x4_t k_vec = vcvt_f32_f16(vld1_f16((const __fp16 *)(k_block_line + i)));
                    sum_vec = vfmaq_f32(sum_vec, q_vec, k_vec);
                }
                acc_dtype_t total = hadd_ps_neon(sum_vec);
                for (; i < dim_size; ++i) { total += q_block_line[i] * MLLM_FP16_TO_FP32(k_block_line[i]); }
#endif
                acc_s[b_r_idx * Bc + b_c_idx] = total;
            }
        }
    }

    inline void softmax(const int32_t Br_n_fixed, const int32_t Bc_n_fixed,
                        acc_dtype_t *__restrict__ acc_s, acc_dtype_t *scoremax, acc_dtype_t *scoremax_prev,
                        acc_dtype_t *score_scale, acc_dtype_t *score_sum, acc_dtype_t *logsum,
                        const float scale) {
        memcpy(scoremax_prev, scoremax, Br_n_fixed * sizeof(acc_dtype_t));
        for (int br = 0; br < Br_n_fixed; ++br) {
            acc_dtype_t *row = acc_s + br * Bc;
            float block_max = NEG_INF_F32;
            for (int bc = 0; bc < Bc_n_fixed; ++bc) block_max = fmaxf(block_max, row[bc]);
            scoremax[br] = fmaxf(scoremax[br], block_max);
        }
        for (int br = 0; br < Br_n_fixed; ++br) score_scale[br] = expf((scoremax_prev[br] - scoremax[br]) * scale);
        for (int br = 0; br < Br_n_fixed; ++br) {
            const float current_max = scoremax[br];
            acc_dtype_t *row = acc_s + br * Bc;
            float sum = 0.0f;
            for (int bc = 0; bc < Bc_n_fixed; ++bc) {
                if (row[bc] == NEG_INF_F32) {
                    row[bc] = 0.0f;
                    continue;
                }
                float val = expf((row[bc] - current_max) * scale);
                row[bc] = val;
                sum += val;
            }
            score_sum[br] = sum;
        }
        for (int br = 0; br < Br_n_fixed; ++br) logsum[br] = logsum[br] * score_scale[br] + score_sum[br];
    }

    inline void rescale(const int32_t Br_n_fixed, acc_dtype_t *__restrict__ acc_o, acc_dtype_t *__restrict__ score_scale, const int32_t dim_size) {
        for (int i = 0; i < Br_n_fixed; ++i) {
#ifdef __AVX2__
            __m256 scale_v = _mm256_set1_ps(score_scale[i]);
            float *row_ptr = acc_o + i * dim_size;
            for (int j = 0; j < dim_size; j += 8) _mm256_storeu_ps(row_ptr + j, _mm256_mul_ps(_mm256_loadu_ps(row_ptr + j), scale_v));
#elif __ARM_NEON
            float32x4_t scale_v = vdupq_n_f32(score_scale[i]);
            float *row_ptr = acc_o + i * dim_size;
            for (int j = 0; j < dim_size; j += 4) vst1q_f32(row_ptr + j, vmulq_f32(vld1q_f32(row_ptr + j), scale_v));
#endif
        }
    }

    inline void mma1(const int32_t Br_n_fixed, const int32_t Bc_n_fixed,
                     const acc_dtype_t *__restrict__ w_block, const dtype_kv_in_t *__restrict__ v_block,
                     acc_dtype_t *__restrict__ acc_o, const int32_t dim_size) {
        const int32_t v_stride_size = dim_size;
        for (int b_r_idx = 0; b_r_idx < Br_n_fixed; ++b_r_idx) {
#ifdef __AVX2__
            for (int d_base = 0; d_base < dim_size; d_base += 8) {
                __m256 acc = _mm256_loadu_ps(acc_o + b_r_idx * dim_size + d_base);
                for (int b_c_idx = 0; b_c_idx < Bc_n_fixed; ++b_c_idx) {
                    acc = _mm256_fmadd_ps(_mm256_set1_ps(w_block[b_r_idx * Bc + b_c_idx]), MLLM_F32Cx8_LOAD(v_block + b_c_idx * v_stride_size + d_base), acc);
                }
                _mm256_storeu_ps(acc_o + b_r_idx * dim_size + d_base, acc);
            }
#elif __ARM_NEON
            for (int d_base = 0; d_base < dim_size; d_base += 4) {
                float32x4_t acc = vld1q_f32(acc_o + b_r_idx * dim_size + d_base);
                for (int b_c_idx = 0; b_c_idx < Bc_n_fixed; ++b_c_idx) {
                    float32x4_t w_vec = vdupq_n_f32(w_block[b_r_idx * Bc + b_c_idx]);
                    float32x4_t v_vec = vcvt_f32_f16(vld1_f16((const __fp16 *)(v_block + b_c_idx * v_stride_size + d_base)));
                    acc = vfmaq_f32(acc, w_vec, v_vec);
                }
                vst1q_f32(acc_o + b_r_idx * dim_size + d_base, acc);
            }
#endif
        }
    }

    inline void scale_and_store(const int32_t Br_n_fixed, const acc_dtype_t *__restrict__ acc_o, const acc_dtype_t *__restrict__ logsum,
                                dtype_out_t *__restrict__ o_block, const int32_t dim_size) {
        for (int i = 0; i < Br_n_fixed; ++i) {
            dtype_out_t *o_block_line = o_block + i * dim_size;
            float reciprocal_logsum = (logsum[i] == 0.0f) ? 0.0f : 1.0f / logsum[i];
#ifdef __AVX2__
            __m256 reciprocal_logsum_vec = _mm256_set1_ps(reciprocal_logsum);
            for (int j = 0; j <= dim_size - 8; j += 8) _mm256_storeu_ps(o_block_line + j, _mm256_mul_ps(_mm256_loadu_ps(acc_o + i * dim_size + j), reciprocal_logsum_vec));
            for (int j = dim_size - (dim_size % 8); j < dim_size; ++j) o_block_line[j] = acc_o[i * dim_size + j] * reciprocal_logsum;
#elif __ARM_NEON
            float32x4_t reciprocal_logsum_vec = vdupq_n_f32(reciprocal_logsum);
            for (int j = 0; j <= dim_size - 4; j += 4) vst1q_f32(o_block_line + j, vmulq_f32(vld1q_f32(acc_o + i * dim_size + j), reciprocal_logsum_vec));
            for (int j = dim_size - (dim_size % 4); j < dim_size; ++j) o_block_line[j] = acc_o[i * dim_size + j] * reciprocal_logsum;
#endif
        }
    }

    inline void init_temp_d(acc_dtype_t *logsum, acc_dtype_t *scoremax, acc_dtype_t *acc_o, const int32_t dim_size) {
        logsum[0] = 0.0f;
        scoremax[0] = NEG_INF_F32;
#ifdef __AVX2__
        __m256 zero_vec = _mm256_setzero_ps();
        for (int i = 0; i < dim_size; i += 8) { _mm256_storeu_ps(acc_o + i, zero_vec); }
#elif __ARM_NEON
        float32x4_t zero_vec = vdupq_n_f32(0.0f);
        for (int i = 0; i < dim_size; i += 4) { vst1q_f32(acc_o + i, zero_vec); }
#endif
    }

    inline void mma0_d(const int32_t Bc_n_fixed, const dtype_q_in_t *__restrict__ q_block,
                       const dtype_kv_in_t *__restrict__ k_block, acc_dtype_t *__restrict__ acc_s,
                       const int32_t dim_size, const int32_t t_c_idx,
                       const int32_t seq_size_k, bool causal_mask) {
        const int32_t global_c_start = t_c_idx * Bc;
        const int32_t global_r_idx = seq_size_k - 1;
        for (int32_t b_c_idx = 0; b_c_idx < Bc_n_fixed; ++b_c_idx) {
            const int32_t global_c_idx = global_c_start + b_c_idx;
            if (causal_mask && global_c_idx > global_r_idx) {
                acc_s[b_c_idx] = NEG_INF_F32;
                continue;
            }
            const dtype_kv_in_t *k_block_line = k_block + b_c_idx * dim_size;
#ifdef __AVX2__
            __m256 sum_vec = _mm256_setzero_ps();
            int i = 0;
            for (; i <= dim_size - 8; i += 8) sum_vec = _mm256_fmadd_ps(_mm256_loadu_ps(q_block + i), MLLM_F32Cx8_LOAD(k_block_line + i), sum_vec);
            acc_dtype_t total = hadd_ps_avx(sum_vec);
            for (; i < dim_size; ++i) total += q_block[i] * MLLM_FP16_TO_FP32(k_block_line[i]);
#elif __ARM_NEON
            float32x4_t sum_vec = vdupq_n_f32(0.0f);
            int i = 0;
            for (; i <= dim_size - 4; i += 4) {
                float32x4_t q_vec = vld1q_f32(q_block + i);
                float32x4_t k_vec = vcvt_f32_f16(vld1_f16((const __fp16 *)(k_block_line + i)));
                sum_vec = vfmaq_f32(sum_vec, q_vec, k_vec);
            }
            acc_dtype_t total = hadd_ps_neon(sum_vec);
            for (; i < dim_size; ++i) total += q_block[i] * MLLM_FP16_TO_FP32(k_block_line[i]);
#endif
            acc_s[b_c_idx] = total;
        }
    }

    inline void softmax_d(const int32_t Bc_n_fixed, acc_dtype_t *__restrict__ acc_s,
                          acc_dtype_t *scoremax, acc_dtype_t *scoremax_prev,
                          acc_dtype_t *score_scale, acc_dtype_t *score_sum, acc_dtype_t *logsum,
                          const float scale) {
        scoremax_prev[0] = scoremax[0];
        float block_max = NEG_INF_F32;
        for (int bc = 0; bc < Bc_n_fixed; ++bc) block_max = fmaxf(block_max, acc_s[bc]);
        scoremax[0] = fmaxf(scoremax[0], block_max);
        score_scale[0] = expf((scoremax_prev[0] - scoremax[0]) * scale);
        float current_sum = 0.0f;
        for (int bc = 0; bc < Bc_n_fixed; ++bc) {
            if (acc_s[bc] == NEG_INF_F32) {
                acc_s[bc] = 0.0f;
                continue;
            }
            float val = expf((acc_s[bc] - scoremax[0]) * scale);
            acc_s[bc] = val;
            current_sum += val;
        }
        score_sum[0] = current_sum;
        logsum[0] = logsum[0] * score_scale[0] + score_sum[0];
    }

    inline void rescale_d(acc_dtype_t *__restrict__ acc_o, acc_dtype_t *__restrict__ score_scale, const int32_t dim_size) {
#ifdef __AVX2__
        __m256 scale_v = _mm256_set1_ps(score_scale[0]);
        for (int j = 0; j < dim_size; j += 8) _mm256_storeu_ps(acc_o + j, _mm256_mul_ps(_mm256_loadu_ps(acc_o + j), scale_v));
#elif __ARM_NEON
        float32x4_t scale_v = vdupq_n_f32(score_scale[0]);
        for (int j = 0; j < dim_size; j += 4) vst1q_f32(acc_o + j, vmulq_f32(vld1q_f32(acc_o + j), scale_v));
#endif
    }

    inline void mma1_d(const int32_t Bc_n_fixed, const acc_dtype_t *__restrict__ w_block,
                       const dtype_kv_in_t *__restrict__ v_block, acc_dtype_t *__restrict__ acc_o, const int32_t dim_size) {
        const int32_t v_stride_size = dim_size;
#ifdef __AVX2__
        for (int d_base = 0; d_base < dim_size; d_base += 8) {
            __m256 acc = _mm256_loadu_ps(acc_o + d_base);
            for (int b_c_idx = 0; b_c_idx < Bc_n_fixed; ++b_c_idx) {
                acc = _mm256_fmadd_ps(_mm256_set1_ps(w_block[b_c_idx]), MLLM_F32Cx8_LOAD(v_block + b_c_idx * v_stride_size + d_base), acc);
            }
            _mm256_storeu_ps(acc_o + d_base, acc);
        }
#elif __ARM_NEON
        for (int d_base = 0; d_base < dim_size; d_base += 4) {
            float32x4_t acc = vld1q_f32(acc_o + d_base);
            for (int b_c_idx = 0; b_c_idx < Bc_n_fixed; ++b_c_idx) {
                float32x4_t w_vec = vdupq_n_f32(w_block[b_c_idx]);
                float32x4_t v_vec = vcvt_f32_f16(vld1_f16((const __fp16 *)(v_block + b_c_idx * v_stride_size + d_base)));
                acc = vfmaq_f32(acc, w_vec, v_vec);
            }
            vst1q_f32(acc_o + d_base, acc);
        }
#endif
    }

    inline void scale_and_store_d(const acc_dtype_t *__restrict__ acc_o, const acc_dtype_t *__restrict__ logsum,
                                  dtype_out_t *__restrict__ o_block, const int32_t dim_size) {
        float reciprocal_logsum = (logsum[0] == 0.0f) ? 0.0f : 1.0f / logsum[0];
#ifdef __AVX2__
        __m256 reciprocal_logsum_vec = _mm256_set1_ps(reciprocal_logsum);
        for (int j = 0; j <= dim_size - 8; j += 8) _mm256_storeu_ps(o_block + j, _mm256_mul_ps(_mm256_loadu_ps(acc_o + j), reciprocal_logsum_vec));
        for (int j = dim_size - (dim_size % 8); j < dim_size; ++j) o_block[j] = acc_o[j] * reciprocal_logsum;
#elif __ARM_NEON
        float32x4_t reciprocal_logsum_vec = vdupq_n_f32(reciprocal_logsum);
        for (int j = 0; j <= dim_size - 4; j += 4) vst1q_f32(o_block + j, vmulq_f32(vld1q_f32(acc_o + j), reciprocal_logsum_vec));
        for (int j = dim_size - (dim_size % 4); j < dim_size; ++j) o_block[j] = acc_o[j] * reciprocal_logsum;
#endif
    }
};

template <typename Impl>
struct FlashAttn2HeadFirstT {
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
        impl_.init_workspace(acc_o, acc_s, logsum, scoremax, scoremax_prev, score_scale, score_sum);
    }

    void operator()(const dtype_q_in_t *__restrict__ Q, const dtype_kv_in_t *__restrict__ K,
                    const dtype_kv_in_t *__restrict__ V, dtype_out_t *__restrict__ O,
                    const int32_t batch_size, const int32_t head_size,
                    const int32_t seq_size_q, const int32_t seq_size_k,
                    const int32_t dim_size, bool causal_mask,
                    int32_t q_head_skp, int32_t k_head_skp, int32_t v_head_skp) {
        impl_.fa2(Q, K, V, O, batch_size, head_size, seq_size_q, seq_size_k, dim_size, causal_mask, q_head_skp, k_head_skp, v_head_skp);
    }

private:
    Impl impl_;
};

} // namespace mobi_attn

inline void flash_attention_2_forward_h(
    const void *Q, const void *K, const void *V, void *O,
    int32_t batch_size, int32_t head_size, int32_t seq_size_q, int32_t seq_size_k, int32_t dim_size,
    bool causal_mask, bool use_fp32, int32_t threads, int32_t br, int32_t bc,
    int32_t q_head, int32_t kv_head, bool high_precision_exp,
    int32_t q_head_skp, int32_t k_head_skp, int32_t v_head_skp) {
    const size_t align = 32;
    const size_t acc_o_size = threads * br * dim_size * sizeof(float);
    const size_t acc_s_size = threads * br * bc * sizeof(float);
    const size_t logsum_size = threads * br * sizeof(float);
    const size_t scoremax_size = threads * br * sizeof(float);
    const size_t scoremax_prev_size = threads * br * sizeof(float);
    const size_t score_scale_size = threads * br * sizeof(float);
    const size_t score_sum_size = threads * br * sizeof(float);

    void *workspace_ptr = nullptr;
    size_t total_workspace_size = acc_o_size + acc_s_size + logsum_size + scoremax_size + scoremax_prev_size + score_scale_size + score_sum_size;

    platform_aligned_alloc(&workspace_ptr, total_workspace_size, align);
    if (workspace_ptr == nullptr) {
        return;
    }

    float *acc_o = static_cast<float *>(workspace_ptr);
    float *acc_s = acc_o + threads * br * dim_size;
    float *logsum = acc_s + threads * br * bc;
    float *scoremax = logsum + threads * br;
    float *scoremax_prev = scoremax + threads * br;
    float *score_scale = scoremax_prev + threads * br;
    float *score_sum = score_scale + threads * br;

    if (use_fp32) {
        mobi_attn::FlashAttn2HeadFirstT<mobi_attn::FA_2_GQA_QKV_FP32_BHSD_O_FP32_BHSD_ACC_FP32_IMPL> op;
        op.configure(br, bc, q_head, kv_head, threads, high_precision_exp);
        op.init_workspace(acc_o, acc_s, logsum, scoremax, scoremax_prev, score_scale, score_sum);
        op(static_cast<const float *>(Q), static_cast<const float *>(K), static_cast<const float *>(V),
           static_cast<float *>(O),
           batch_size, head_size, seq_size_q, seq_size_k, dim_size, causal_mask,
           q_head_skp, k_head_skp, v_head_skp);
    } else {
        mobi_attn::FlashAttn2HeadFirstT<mobi_attn::FA_2_GQA_Q_FP32_KV_FP16_BHSD_O_FP32_BHSD_ACC_FP32_IMPL> op;
        op.configure(br, bc, q_head, kv_head, threads, high_precision_exp);
        op.init_workspace(acc_o, acc_s, logsum, scoremax, scoremax_prev, score_scale, score_sum);
        op(static_cast<const float *>(Q),
           static_cast<const mllm_fp16_t *>(K),
           static_cast<const mllm_fp16_t *>(V),
           static_cast<float *>(O),
           batch_size, head_size, seq_size_q, seq_size_k, dim_size, causal_mask,
           q_head_skp, k_head_skp, v_head_skp);
    }

    if (workspace_ptr) {
        platform_aligned_free(workspace_ptr);
    }
}

#endif // MLLM_FA2H_CAL_HPP