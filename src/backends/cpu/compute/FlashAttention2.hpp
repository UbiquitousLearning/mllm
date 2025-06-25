#ifndef MLLM_FA2_CAL_HPP
#define MLLM_FA2_CAL_HPP

#include <cstdint>
#include <omp.h>
#include <cassert>
#include <limits>
#include <cmath>
#include <cstring>
#include "Types.hpp"
#include "VecDot.hpp"

#ifdef __AVX2__
#include <immintrin.h>
#elif __ARM_NEON__
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
#elif __ARM_NEON__
// NEON版本：水平最大值 (Horizontal max of a float32x4_t vector)
// float32x4_t 中包含4个float, vmaxvq_f32可以直接找到这4个float中的最大值
inline float _vmaxvq_f32_hmax(float32x4_t x) {
    return vmaxvq_f32(x);
}

// NEON版本：水平求和 (Horizontal sum of a float32x4_t vector)
// float32x4_t 中包含4个float, vaddvq_f32可以直接将这4个float相加
inline float _vaddvq_f32_hadd(float32x4_t x) {
    return vaddvq_f32(x);
}
#endif

// ========================================
// 内存对齐分配函数
// ========================================
// 使用 posix_memalign 进行分配
void aligned_alloc(void **ptr, size_t required_bytes, size_t align) {
    // posix_memalign 要求 alignment 必须是 void* 大小的整数倍，并且是 2 的幂
    if (align % sizeof(void *) != 0 || (align & (align - 1)) != 0) {
        *ptr = nullptr;
        return;
    }

    // posix_memalign 返回 0 表示成功，否则返回错误码
    if (posix_memalign(ptr, align, required_bytes) != 0) {
        *ptr = nullptr;
    }
}

// 直接使用标准 free 进行释放
void aligned_free(void *ptr) {
    free(ptr);
}

// ========================================
// FlashAttention2 核心实现 (FP32版本)
// ========================================
struct FA_2_GQA_QKV_FP32_BSHD_O_FP32_BSHD_ACC_FP32_IMPL {
    using dtype_q_in_t = float;
    using dtype_kv_in_t = dtype_q_in_t;
    using dtype_out_t = dtype_q_in_t;
    using dtype_t = dtype_out_t;
    using acc_dtype_t = float;
    // 添加配置参数作为成员变量
    int32_t Br;
    int32_t Bc;
    int32_t Q_Head;
    int32_t KV_Head;
    int32_t threads;
    bool high_precision;
    // 配置参数初始化
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
#elif __ARM_NEON__
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
    // 核心计算函数
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

        // 【关键修改 1】计算Q头与KV头的分组对应关系
        const int32_t kv_group_size = Q_Head / KV_Head;

        for (int32_t b_idx = 0; b_idx < batch_size; ++b_idx) {
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1) if (threads > 1)
            for (int32_t h_idx = 0; h_idx < head_size; ++h_idx) { // h_idx 是当前Q头的索引
                const int32_t thread_id = omp_get_thread_num();
                const int32_t this_thread_head = h_idx;

                // 【关键修改 1】计算当前Q头 (h_idx) 对应的KV头索引
                const int32_t this_thread_kv_head = this_thread_head / kv_group_size;

                // --- 主循环 (Tr) ---
                for (int t_r_idx = 0; t_r_idx < Tr; ++t_r_idx) {
                    init_temp(logsum_ + thread_id * Br, scoremax_ + thread_id * Br,
                              acc_o_ + thread_id * Br * dim_size, dim_size);
                    for (int t_c_idx = 0; t_c_idx < Tc; ++t_c_idx) {
                        // Q 的指针计算保持不变，因为它有 Q_Head 个头
                        const dtype_t *tile_q = Q + b_idx * seq_size_q * head_size * dim_size + t_r_idx * Br * head_size * dim_size + this_thread_head * dim_size;

                        // 【关键修改 2】K 和 V 的指针计算，必须使用 KV_Head 和映射后的 this_thread_kv_head
                        const dtype_t *tile_k = K + b_idx * seq_size_k * KV_Head * dim_size + t_c_idx * Bc * KV_Head * dim_size + this_thread_kv_head * dim_size;
                        const dtype_t *tile_v = V + b_idx * seq_size_k * KV_Head * dim_size + t_c_idx * Bc * KV_Head * dim_size + this_thread_kv_head * dim_size;

                        acc_dtype_t *tile_acc_s = acc_s_ + thread_id * Br * Bc;
                        acc_dtype_t *acc_o = acc_o_ + thread_id * Br * dim_size;

                        // 【关键修改 3】为 mma0 传入Q和K各自的正确步长
                        mma0(tile_q, tile_k, tile_acc_s, dim_size, head_size * dim_size, KV_Head * dim_size, t_r_idx, t_c_idx, seq_size_q, seq_size_k, causal_mask);
                        softmax(tile_acc_s, scoremax_ + thread_id * Br, scoremax_prev_ + thread_id * Br, score_scale_ + thread_id * Br, score_sum_ + thread_id * Br, logsum_ + thread_id * Br, local_scale, t_r_idx, t_c_idx, seq_size_q, seq_size_k, causal_mask);
                        rescale(acc_o, score_scale_ + thread_id * Br, dim_size, t_r_idx, t_c_idx, seq_size_q, seq_size_k, causal_mask);
                        // 【关键修改 3】为 mma1 传入 KV_Head 作为V的头数量，用于计算其内部步长
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
        // FIX: Calculate the ratio of Q_Head to KV_Head to handle GQA/MHA correctly.
        const int32_t kv_group_size = (Q_Head > 0 && KV_Head > 0) ? Q_Head / KV_Head : 1;

        for (int32_t b_idx = 0; b_idx < batch_size; ++b_idx) {
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1) if (threads > 1)
            for (int32_t h_idx = 0; h_idx < head_size; ++h_idx) {
                const int32_t thread_id = omp_get_thread_num();
                const int32_t this_thread_head = h_idx;
                // FIX: Map the current query head 'h_idx' to its corresponding KV head.
                const int32_t this_thread_kv_head = this_thread_head / kv_group_size;

                for (int t_r_idx = 0; t_r_idx < Tr; ++t_r_idx) {
                    init_temp_d(logsum_ + thread_id * Br, scoremax_ + thread_id * Br, acc_o_ + thread_id * Br * dim_size, dim_size);
                    for (int t_c_idx = 0; t_c_idx < Tc; ++t_c_idx) {
                        const dtype_t *tile_q = Q + b_idx * seq_size_q * head_size * dim_size + t_r_idx * 1 * head_size * dim_size + this_thread_head * dim_size;
                        // FIX: Corrected pointer arithmetic for K and V.
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

        // 【最终修正】使用安全的循环来确保完全初始化，不再依赖Br是8的倍数
        int i = 0;
        for (; i <= Br - 8; i += 8) {
            _mm256_storeu_ps(logsum + i, zero_vec);
            _mm256_storeu_ps(scoremax + i, neg_inf_vec);
        }
        // 处理剩余的元素（如果Br不是8的倍数）
        for (; i < Br; ++i) {
            logsum[i] = 0.0f;
            scoremax[i] = NEG_INF;
        }

        // acc_o 的初始化是安全的，因为调用者保证了 dim_size % 8 == 0
        for (int j = 0; j < Br * dim_size; j += 8) {
            _mm256_storeu_ps(acc_o + j, zero_vec);
        }
#elif __ARM_NEON__
        float32x4_t zero_vec = vdupq_n_f32(0.0f);
        float32x4_t neg_inf_vec = vdupq_n_f32(NEG_INF);

        int i = 0;
        // NEON 一次处理4个
        for (; i <= Br - 4; i += 4) {
            vst1q_f32(logsum + i, zero_vec);
            vst1q_f32(scoremax + i, neg_inf_vec);
        }
        // 处理剩余的元素（如果Br不是4的倍数）
        for (; i < Br; ++i) {
            logsum[i] = 0.0f;
            scoremax[i] = NEG_INF;
        }

        // acc_o 的初始化, 调用者保证了 dim_size % 4 == 0
        for (int j = 0; j < Br * dim_size; j += 4) {
            vst1q_f32(acc_o + j, zero_vec);
        }
#endif
    }

    // 【关键修改】函数签名增加 kv_stride_size 参数，用于区分Q和K的步长
    inline void mma0(const dtype_t *__restrict__ q_block, const dtype_t *__restrict__ k_block,
                     acc_dtype_t *__restrict__ acc_s, const int32_t dim_size,
                     const int32_t q_stride_size, const int32_t kv_stride_size,
                     const int32_t t_r_idx, const int32_t t_c_idx,
                     const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
#ifdef __AVX2__
        const int32_t global_r_start = t_r_idx * Br;
        const int32_t global_r_end = global_r_start + Br;
        const int32_t global_c_start = t_c_idx * Bc;
        int delta_pos = seq_size_k - seq_size_q;

        if (causal_mask && (global_c_start - delta_pos > (global_r_end - 1))) { return; }

#pragma unroll
        for (int32_t b_r_idx = 0; b_r_idx < Br; ++b_r_idx) {
            // 【关键修改】使用传入的 q_stride_size
            const dtype_t *q_block_line = q_block + b_r_idx * q_stride_size;
#pragma unroll
            for (int32_t b_c_idx = 0; b_c_idx < Bc; ++b_c_idx) {
                // 【关键修改】使用传入的 kv_stride_size
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
#elif __ARM_NEON__
        const int32_t global_r_start = t_r_idx * Br;
        const int32_t global_r_end = global_r_start + Br;
        const int32_t global_c_start = t_c_idx * Bc;
        int delta_pos = seq_size_k - seq_size_q;

        if (causal_mask && (global_c_start - delta_pos > (global_r_end - 1))) { return; }

        for (int32_t b_r_idx = 0; b_r_idx < Br; ++b_r_idx) {
            const dtype_t *q_block_line = q_block + b_r_idx * q_stride_size;
            for (int32_t b_c_idx = 0; b_c_idx < Bc; ++b_c_idx) {
                const dtype_t *k_block_line = k_block + b_c_idx * kv_stride_size;

                float32x4_t sum_vec = vdupq_n_f32(0.0f);
                int i = 0;
                // 为了性能，可以一次处理8个或更多元素，这里保持与AVX版本类似的逻辑，但向量宽度减半
                for (; i <= dim_size - 8; i += 8) {
                    // 预取数据到缓存
                    __builtin_prefetch(q_block_line + i + 64);
                    __builtin_prefetch(k_block_line + i + 64);
                    // 加载两组4个float数据
                    float32x4_t q_vec0 = vld1q_f32(q_block_line + i);
                    float32x4_t k_vec0 = vld1q_f32(k_block_line + i);
                    float32x4_t q_vec1 = vld1q_f32(q_block_line + i + 4);
                    float32x4_t k_vec1 = vld1q_f32(k_block_line + i + 4);
                    // 融合乘加
                    sum_vec = vfmaq_f32(sum_vec, q_vec0, k_vec0);
                    sum_vec = vfmaq_f32(sum_vec, q_vec1, k_vec1);
                }
                // 处理 dim_size % 8 剩下的部分
                for (; i <= dim_size - 4; i += 4) {
                    float32x4_t q_vec = vld1q_f32(q_block_line + i);
                    float32x4_t k_vec = vld1q_f32(k_block_line + i);
                    sum_vec = vfmaq_f32(sum_vec, q_vec, k_vec);
                }

                acc_dtype_t total = _vaddvq_f32_hadd(sum_vec);
                // 处理最后不足4个的元素
                for (; i < dim_size; ++i) { total += q_block_line[i] * k_block_line[i]; }

                acc_s[b_r_idx * Bc + b_c_idx] = total;
            }
        }
        // 应用因果掩码
        if (causal_mask && (global_r_end == (t_c_idx * Bc + Bc) - delta_pos)) {
            for (int i = 0; i < Br; ++i) {
                for (int j = 0; j < Bc; ++j) {
                    if (j > i) { acc_s[i * Bc + j] = NEG_INF; }
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
#elif __ARM_NEON__
        const int32_t global_r_start = t_r_idx * Br;
        const int32_t global_c_start = t_c_idx * Bc;
        int delta_pos = seq_size_k - seq_size_q;
        if (causal_mask && (global_c_start - delta_pos > (global_r_start + Br - 1))) return;

        memcpy(scoremax_prev, scoremax, Br * sizeof(acc_dtype_t));

        // 1. 找到每行的最大值 m_i
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

        // 2. 计算缩放因子 s_i = exp((m_i_prev - m_i) * scale)
        for (int br = 0; br < Br; ++br) {
            score_scale[br] = expf((scoremax_prev[br] - scoremax[br]) * scale);
        }

        // 3. 计算 P_ij = exp((S_ij - m_i) * scale) 和 l_i = sum(P_ij)
        for (int br = 0; br < Br; ++br) {
            const float sm = scoremax[br];
            acc_dtype_t *row = acc_s + br * Bc;
            float sum = 0.0f;
            // 这里可以进一步用NEON优化expf, 但expf的SIMD实现复杂，暂用标量
            for (int bc = 0; bc < Bc; ++bc) {
                float val = expf((row[bc] - sm) * scale);
                row[bc] = val; // 更新 acc_s 为 P_ij
                sum += val;
            }
            score_sum[br] = sum;
        }

        // 4. 更新 logsum: l_i_new = l_i_prev * s_i + l_i
        for (int br = 0; br < Br; ++br) {
            logsum[br] = logsum[br] * score_scale[br] + score_sum[br];
        }
#endif
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
#elif __ARM_NEON__
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

    // 【关键修改】函数签名增加 kv_head_size 参数，用于计算V的内部步长
    inline void mma1(const acc_dtype_t *__restrict__ w_block, const dtype_t *__restrict__ v_block,
                     acc_dtype_t *__restrict__ acc_o, const int32_t kv_head_size, const int32_t dim_size,
                     const int32_t t_r_idx, const int32_t t_c_idx, const int32_t seq_size_q,
                     const int32_t seq_size_k, bool causal_mask) {
#ifdef __AVX2__
        const int32_t global_r_start = t_r_idx * Br;
        const int32_t global_c_start = t_c_idx * Bc;
        int delta_pos = seq_size_k - seq_size_q;
        if (causal_mask && (global_c_start - delta_pos > (global_r_start + Br - 1))) return;

        // 【关键修改】使用传入的 kv_head_size 来计算 V 的步长
        const int32_t v_stride_size = kv_head_size * dim_size;

#pragma unroll
        for (int b_r_idx = 0; b_r_idx < Br; ++b_r_idx) {
            for (int d_base = 0; d_base < dim_size; d_base += 8) {
                __m256 acc = _mm256_loadu_ps(acc_o + b_r_idx * dim_size + d_base);
#pragma unroll
                for (int b_c_idx = 0; b_c_idx < Bc; ++b_c_idx) {
                    __m256 w_vec = _mm256_set1_ps(w_block[b_r_idx * Bc + b_c_idx]);
                    // 【关键修改】使用计算出的 v_stride_size
                    const float *v_ptr = v_block + b_c_idx * v_stride_size + d_base;
                    __m256 v_vec = _mm256_loadu_ps(v_ptr);
                    acc = _mm256_fmadd_ps(w_vec, v_vec, acc);
                }
                _mm256_storeu_ps(acc_o + b_r_idx * dim_size + d_base, acc);
            }
        }
#elif __ARM_NEON__
        const int32_t global_r_start = t_r_idx * Br;
        const int32_t global_c_start = t_c_idx * Bc;
        int delta_pos = seq_size_k - seq_size_q;
        if (causal_mask && (global_c_start - delta_pos > (global_r_start + Br - 1))) return;

        const int32_t v_stride_size = kv_head_size * dim_size;

        for (int b_r_idx = 0; b_r_idx < Br; ++b_r_idx) {
            for (int d_base = 0; d_base < dim_size; d_base += 4) {
                float32x4_t acc = vld1q_f32(acc_o + b_r_idx * dim_size + d_base);
                for (int b_c_idx = 0; b_c_idx < Bc; ++b_c_idx) {
                    float32x4_t w_vec = vdupq_n_f32(w_block[b_r_idx * Bc + b_c_idx]);
                    const float *v_ptr = v_block + b_c_idx * v_stride_size + d_base;
                    float32x4_t v_vec = vld1q_f32(v_ptr);
                    acc = vfmaq_f32(acc, w_vec, v_vec);
                }
                vst1q_f32(acc_o + b_r_idx * dim_size + d_base, acc);
            }
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
#elif __ARM_NEON__
        for (int i = 0; i < Br; ++i) {
            dtype_t *o_block_line = o_block + i * head_size * dim_size;
            float reciprocal_logsum = 1.0f / logsum[i];
            float32x4_t reciprocal_logsum_vec = vdupq_n_f32(reciprocal_logsum);
            int j = 0;
            for (; j <= dim_size - 4; j += 4) {
                float32x4_t vec_acc_o = vld1q_f32(acc_o + i * dim_size + j);
                float32x4_t result_vec = vmulq_f32(vec_acc_o, reciprocal_logsum_vec);
                vst1q_f32(o_block_line + j, result_vec);
            }
            for (; j < dim_size; ++j) {
                o_block_line[j] = acc_o[i * dim_size + j] * reciprocal_logsum;
            }
        }
#endif
    }

    // N-fixed functions for handling leftovers
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
#elif __ARM_NEON__
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
                int i = 0;
                for (; i <= dim_size - 4; i += 4) {
                    sum_vec = vfmaq_f32(sum_vec, vld1q_f32(q_block_line + i), vld1q_f32(k_block_line + i));
                }
                acc_dtype_t total = _vaddvq_f32_hadd(sum_vec);
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
#elif __ARM_NEON__
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
#elif __ARM_NEON__
        const int32_t global_r_start = t_r_idx * Br;
        const int32_t global_c_start = t_c_idx * Bc;
        int delta_pos = seq_size_k - seq_size_q;
        if (causal_mask && (global_c_start - delta_pos > (global_r_start + Br_n_fixed - 1))) return;

        const int32_t v_stride_size = kv_head_size * dim_size;

        for (int b_r_idx = 0; b_r_idx < Br_n_fixed; ++b_r_idx) {
            for (int d_base = 0; d_base < dim_size; d_base += 4) {
                float32x4_t acc = vld1q_f32(acc_o + b_r_idx * dim_size + d_base);
                for (int b_c_idx = 0; b_c_idx < Bc_n_fixed; ++b_c_idx) {
                    float32x4_t w_vec = vdupq_n_f32(w_block[b_r_idx * Bc + b_c_idx]);
                    const float *v_ptr = v_block + b_c_idx * v_stride_size + d_base;
                    acc = vfmaq_f32(acc, w_vec, vld1q_f32(v_ptr));
                }
                vst1q_f32(acc_o + b_r_idx * dim_size + d_base, acc);
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
#elif __ARM_NEON__
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
#elif __ARM_NEON__
        logsum[0] = 0.0f;
        scoremax[0] = NEG_INF;
        float32x4_t zero_vec = vdupq_n_f32(0.0f);
        // Br 在 decode 模式下为 1
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
#elif __ARM_NEON__
        const dtype_t *q_block_line = q_block; // q 只有一个向量
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
#elif __ARM_NEON__
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
#elif __ARM_NEON__
        const int32_t v_stride_size = kv_head_size * dim_size;
        for (int d_base = 0; d_base < dim_size; d_base += 4) {
            float32x4_t acc = vld1q_f32(acc_o + d_base);
            for (int b_c_idx = 0; b_c_idx < Bc; ++b_c_idx) {
                float32x4_t w_vec = vdupq_n_f32(w_block[b_c_idx]);
                const float *v_ptr = v_block + b_c_idx * v_stride_size + d_base;
                acc = vfmaq_f32(acc, w_vec, vld1q_f32(v_ptr));
            }
            vst1q_f32(acc_o + d_base, acc);
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
#elif __ARM_NEON__
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

// ========================================
// FlashAttention2 核心实现 ( Q FP32/KV FP16 输入,FP32 输出版本)
// ========================================
struct FA_2_GQA_Q_FP32_KV_FP16_BSHD_O_FP32_BSHD_ACC_FP32_IMPL {
    // 【修改】定义多种输入类型
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
#elif __ARM_NEON__
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
#if __ARM_NEON__
    // 定义一个宏，用于从内存加载4个fp16, 并转换为一个fp32向量
    // 这需要 ARMv8.2-A FP16 指令支持
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
// Note: OpenMP is not applied to the head_size loop in the prefill reference
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
        const int32_t Tr = 1; // In decode, seq_size_q is always 1
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

                // In decode mode, t_r_idx is always 0 as we process one token
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

        // 【最终修正】使用安全的循环来确保完全初始化，不再依赖Br是8的倍数
        int i = 0;
        for (; i <= Br - 8; i += 8) {
            _mm256_storeu_ps(logsum + i, zero_vec);
            _mm256_storeu_ps(scoremax + i, neg_inf_vec);
        }
        // 处理剩余的元素（如果Br不是8的倍数）
        for (; i < Br; ++i) {
            logsum[i] = 0.0f;
            scoremax[i] = NEG_INF;
        }

        // acc_o 的初始化是安全的，因为调用者保证了 dim_size % 8 == 0
        for (int j = 0; j < Br * dim_size; j += 8) {
            _mm256_storeu_ps(acc_o + j, zero_vec);
        }
#elif __ARM_NEON__
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
#elif __ARM_NEON__
        const int32_t global_r_start = t_r_idx * Br;
        const int32_t global_r_end = global_r_start + Br;
        const int32_t global_c_start = t_c_idx * Bc;
        int delta_pos = seq_size_k - seq_size_q;
        if (causal_mask && (global_c_start - delta_pos > (global_r_end - 1))) { return; }

        // --- 优化核心 ---
        // 策略: 采用4x4微内核，结合深度循环展开，最大化利用NEON指令。
        //
        // 步骤 1: 将当前需要的Q块从FP32转换为FP16。
        // 这是为了能够使用最高效的FP16 FMA指令 (vfmlalq_f16)。
        // 使用alignas确保缓冲区内存对齐，有利于SIMD加载。
        alignas(16) __fp16 q_f16_buf[Br * dim_size];
        for (int32_t b_r_idx = 0; b_r_idx < Br; ++b_r_idx) {
            const dtype_q_in_t *q_line_f32 = q_block + b_r_idx * q_stride_size;
            __fp16 *q_line_f16 = q_f16_buf + b_r_idx * dim_size;
            int d = 0;
            // 一次转换8个float
            for (; d <= dim_size - 8; d += 8) {
                // 加载两组FP32数据
                float32x4_t f32_vec_0 = vld1q_f32(q_line_f32 + d);
                float32x4_t f32_vec_1 = vld1q_f32(q_line_f32 + d + 4);
                // 转换为一个FP16x8向量并存储
                float16x8_t f16_vec = vcombine_f16(vcvt_f16_f32(f32_vec_0), vcvt_f16_f32(f32_vec_1));
                vst1q_f16(q_line_f16 + d, f16_vec);
            }
            // 处理末尾不足8个的元素
            for (; d < dim_size; ++d) {
                q_line_f16[d] = (__fp16)q_line_f32[d];
            }
        }

        const int32_t br_4_end = (Br / 4) * 4;
        const int32_t bc_4_end = (Bc / 4) * 4;

        // 步骤 2: 4x4 微内核主循环。处理Br和Bc是4的倍数的部分。
        int32_t b_r_idx = 0;
        for (; b_r_idx < br_4_end; b_r_idx += 4) {
            int32_t b_c_idx = 0;
            for (; b_c_idx < bc_4_end; b_c_idx += 4) {
                // 定义 4x4=16 个FP32累加器向量，用于存储中间结果
                float32x4_t acc00, acc01, acc02, acc03;
                float32x4_t acc10, acc11, acc12, acc13;
                float32x4_t acc20, acc21, acc22, acc23;
                float32x4_t acc30, acc31, acc32, acc33;

                // 初始化累加器为0
                acc00 = vdupq_n_f32(0.0f);
                acc01 = vdupq_n_f32(0.0f);
                acc02 = vdupq_n_f32(0.0f);
                acc03 = vdupq_n_f32(0.0f);
                acc10 = vdupq_n_f32(0.0f);
                acc11 = vdupq_n_f32(0.0f);
                acc12 = vdupq_n_f32(0.0f);
                acc13 = vdupq_n_f32(0.0f);
                acc20 = vdupq_n_f32(0.0f);
                acc21 = vdupq_n_f32(0.0f);
                acc22 = vdupq_n_f32(0.0f);
                acc23 = vdupq_n_f32(0.0f);
                acc30 = vdupq_n_f32(0.0f);
                acc31 = vdupq_n_f32(0.0f);
                acc32 = vdupq_n_f32(0.0f);
                acc33 = vdupq_n_f32(0.0f);

                // 获取当前4x4 tile对应的Q和K的行指针
                const __fp16 *q0_ptr = q_f16_buf + (b_r_idx + 0) * dim_size;
                const __fp16 *q1_ptr = q_f16_buf + (b_r_idx + 1) * dim_size;
                const __fp16 *q2_ptr = q_f16_buf + (b_r_idx + 2) * dim_size;
                const __fp16 *q3_ptr = q_f16_buf + (b_r_idx + 3) * dim_size;

                const __fp16 *k0_ptr = (const __fp16 *)k_block + (b_c_idx + 0) * kv_stride_size;
                const __fp16 *k1_ptr = (const __fp16 *)k_block + (b_c_idx + 1) * kv_stride_size;
                const __fp16 *k2_ptr = (const __fp16 *)k_block + (b_c_idx + 2) * kv_stride_size;
                const __fp16 *k3_ptr = (const __fp16 *)k_block + (b_c_idx + 3) * kv_stride_size;

                // 沿向量维度 (dim_size) 进行循环计算
                int k = 0;
                // 深度展开，一次处理8个元素 (float16x8_t)
                for (; k <= dim_size - 8; k += 8) {
                    // 数据预取，减少内存访问延迟
                    __builtin_prefetch(q0_ptr + k + 64);
                    __builtin_prefetch(q1_ptr + k + 64);
                    __builtin_prefetch(k0_ptr + k + 64);
                    __builtin_prefetch(k1_ptr + k + 64);

                    // 加载4行Q的数据
                    float16x8_t q0_vec = vld1q_f16(q0_ptr + k);
                    float16x8_t q1_vec = vld1q_f16(q1_ptr + k);
                    float16x8_t q2_vec = vld1q_f16(q2_ptr + k);
                    float16x8_t q3_vec = vld1q_f16(q3_ptr + k);

                    // 加载K并进行外积计算
                    float16x8_t k0_vec = vld1q_f16(k0_ptr + k);
                    acc00 = vfmlalq_low_f16(acc00, q0_vec, k0_vec);
                    acc00 = vfmlalq_high_f16(acc00, q0_vec, k0_vec);
                    acc10 = vfmlalq_low_f16(acc10, q1_vec, k0_vec);
                    acc10 = vfmlalq_high_f16(acc10, q1_vec, k0_vec);
                    acc20 = vfmlalq_low_f16(acc20, q2_vec, k0_vec);
                    acc20 = vfmlalq_high_f16(acc20, q2_vec, k0_vec);
                    acc30 = vfmlalq_low_f16(acc30, q3_vec, k0_vec);
                    acc30 = vfmlalq_high_f16(acc30, q3_vec, k0_vec);

                    float16x8_t k1_vec = vld1q_f16(k1_ptr + k);
                    acc01 = vfmlalq_low_f16(acc01, q0_vec, k1_vec);
                    acc01 = vfmlalq_high_f16(acc01, q0_vec, k1_vec);
                    acc11 = vfmlalq_low_f16(acc11, q1_vec, k1_vec);
                    acc11 = vfmlalq_high_f16(acc11, q1_vec, k1_vec);
                    acc21 = vfmlalq_low_f16(acc21, q2_vec, k1_vec);
                    acc21 = vfmlalq_high_f16(acc21, q2_vec, k1_vec);
                    acc31 = vfmlalq_low_f16(acc31, q3_vec, k1_vec);
                    acc31 = vfmlalq_high_f16(acc31, q3_vec, k1_vec);

                    float16x8_t k2_vec = vld1q_f16(k2_ptr + k);
                    acc02 = vfmlalq_low_f16(acc02, q0_vec, k2_vec);
                    acc02 = vfmlalq_high_f16(acc02, q0_vec, k2_vec);
                    acc12 = vfmlalq_low_f16(acc12, q1_vec, k2_vec);
                    acc12 = vfmlalq_high_f16(acc12, q1_vec, k2_vec);
                    acc22 = vfmlalq_low_f16(acc22, q2_vec, k2_vec);
                    acc22 = vfmlalq_high_f16(acc22, q2_vec, k2_vec);
                    acc32 = vfmlalq_low_f16(acc32, q3_vec, k2_vec);
                    acc32 = vfmlalq_high_f16(acc32, q3_vec, k2_vec);

                    float16x8_t k3_vec = vld1q_f16(k3_ptr + k);
                    acc03 = vfmlalq_low_f16(acc03, q0_vec, k3_vec);
                    acc03 = vfmlalq_high_f16(acc03, q0_vec, k3_vec);
                    acc13 = vfmlalq_low_f16(acc13, q1_vec, k3_vec);
                    acc13 = vfmlalq_high_f16(acc13, q1_vec, k3_vec);
                    acc23 = vfmlalq_low_f16(acc23, q2_vec, k3_vec);
                    acc23 = vfmlalq_high_f16(acc23, q2_vec, k3_vec);
                    acc33 = vfmlalq_low_f16(acc33, q3_vec, k3_vec);
                    acc33 = vfmlalq_high_f16(acc33, q3_vec, k3_vec);
                }

                // 水平求和，将向量累加器中的4个float相加，得到最终的16个标量结果
                float *c_ptr;
                c_ptr = acc_s + (b_r_idx + 0) * Bc + b_c_idx;
                c_ptr[0] = vaddvq_f32(acc00);
                c_ptr[1] = vaddvq_f32(acc01);
                c_ptr[2] = vaddvq_f32(acc02);
                c_ptr[3] = vaddvq_f32(acc03);
                c_ptr = acc_s + (b_r_idx + 1) * Bc + b_c_idx;
                c_ptr[0] = vaddvq_f32(acc10);
                c_ptr[1] = vaddvq_f32(acc11);
                c_ptr[2] = vaddvq_f32(acc12);
                c_ptr[3] = vaddvq_f32(acc13);
                c_ptr = acc_s + (b_r_idx + 2) * Bc + b_c_idx;
                c_ptr[0] = vaddvq_f32(acc20);
                c_ptr[1] = vaddvq_f32(acc21);
                c_ptr[2] = vaddvq_f32(acc22);
                c_ptr[3] = vaddvq_f32(acc23);
                c_ptr = acc_s + (b_r_idx + 3) * Bc + b_c_idx;
                c_ptr[0] = vaddvq_f32(acc30);
                c_ptr[1] = vaddvq_f32(acc31);
                c_ptr[2] = vaddvq_f32(acc32);
                c_ptr[3] = vaddvq_f32(acc33);

                // 处理末尾不足8个的元素
                for (; k < dim_size; ++k) {
                    acc_s[(b_r_idx + 0) * Bc + b_c_idx + 0] += (float)q0_ptr[k] * (float)k0_ptr[k];
                    acc_s[(b_r_idx + 0) * Bc + b_c_idx + 1] += (float)q0_ptr[k] * (float)k1_ptr[k];
                    acc_s[(b_r_idx + 0) * Bc + b_c_idx + 2] += (float)q0_ptr[k] * (float)k2_ptr[k];
                    acc_s[(b_r_idx + 0) * Bc + b_c_idx + 3] += (float)q0_ptr[k] * (float)k3_ptr[k];

                    acc_s[(b_r_idx + 1) * Bc + b_c_idx + 0] += (float)q1_ptr[k] * (float)k0_ptr[k];
                    acc_s[(b_r_idx + 1) * Bc + b_c_idx + 1] += (float)q1_ptr[k] * (float)k1_ptr[k];
                    acc_s[(b_r_idx + 1) * Bc + b_c_idx + 2] += (float)q1_ptr[k] * (float)k2_ptr[k];
                    acc_s[(b_r_idx + 1) * Bc + b_c_idx + 3] += (float)q1_ptr[k] * (float)k3_ptr[k];

                    acc_s[(b_r_idx + 2) * Bc + b_c_idx + 0] += (float)q2_ptr[k] * (float)k0_ptr[k];
                    acc_s[(b_r_idx + 2) * Bc + b_c_idx + 1] += (float)q2_ptr[k] * (float)k1_ptr[k];
                    acc_s[(b_r_idx + 2) * Bc + b_c_idx + 2] += (float)q2_ptr[k] * (float)k2_ptr[k];
                    acc_s[(b_r_idx + 2) * Bc + b_c_idx + 3] += (float)q2_ptr[k] * (float)k3_ptr[k];

                    acc_s[(b_r_idx + 3) * Bc + b_c_idx + 0] += (float)q3_ptr[k] * (float)k0_ptr[k];
                    acc_s[(b_r_idx + 3) * Bc + b_c_idx + 1] += (float)q3_ptr[k] * (float)k1_ptr[k];
                    acc_s[(b_r_idx + 3) * Bc + b_c_idx + 2] += (float)q3_ptr[k] * (float)k2_ptr[k];
                    acc_s[(b_r_idx + 3) * Bc + b_c_idx + 3] += (float)q3_ptr[k] * (float)k3_ptr[k];
                }
            }
        }

        // 步骤 3: Fallback - 处理所有剩余的行和列。
        // 这部分确保了即使Br, Bc不是4的倍数，计算依然正确。
        for (b_r_idx = 0; b_r_idx < Br; ++b_r_idx) {
            // 如果行已经被4x4内核处理过，则只处理剩余的列
            if (b_r_idx < br_4_end) {
                for (int32_t b_c_idx = bc_4_end; b_c_idx < Bc; ++b_c_idx) {
                    const __fp16 *q_f16_line = q_f16_buf + b_r_idx * dim_size;
                    const dtype_kv_in_t *k_block_line = k_block + b_c_idx * kv_stride_size;
                    float32x4_t sum_vec = vdupq_n_f32(0.0f);
                    int i = 0;
                    for (; i <= dim_size - 8; i += 8) {
                        float16x8_t q_vec = vld1q_f16(q_f16_line + i);
                        float16x8_t k_vec = vld1q_f16((const __fp16 *)k_block_line + i);
                        sum_vec = vfmlalq_low_f16(sum_vec, q_vec, k_vec);
                        sum_vec = vfmlalq_high_f16(sum_vec, q_vec, k_vec);
                    }
                    acc_dtype_t total = vaddvq_f32(sum_vec);
                    for (; i < dim_size; ++i) { total += (float)q_f16_line[i] * (float)k_block_line[i]; }
                    acc_s[b_r_idx * Bc + b_c_idx] = total;
                }
            } else { // 否则，处理整行
                for (int32_t b_c_idx = 0; b_c_idx < Bc; ++b_c_idx) {
                    const __fp16 *q_f16_line = q_f16_buf + b_r_idx * dim_size;
                    const dtype_kv_in_t *k_block_line = k_block + b_c_idx * kv_stride_size;
                    float32x4_t sum_vec = vdupq_n_f32(0.0f);
                    int i = 0;
                    for (; i <= dim_size - 8; i += 8) {
                        float16x8_t q_vec = vld1q_f16(q_f16_line + i);
                        float16x8_t k_vec = vld1q_f16((const __fp16 *)k_block_line + i);
                        sum_vec = vfmlalq_low_f16(sum_vec, q_vec, k_vec);
                        sum_vec = vfmlalq_high_f16(sum_vec, q_vec, k_vec);
                    }
                    acc_dtype_t total = vaddvq_f32(sum_vec);
                    for (; i < dim_size; ++i) { total += (float)q_f16_line[i] * (float)k_block_line[i]; }
                    acc_s[b_r_idx * Bc + b_c_idx] = total;
                }
            }
        }

        // 步骤 4: 应用因果掩码 (逻辑保持不变)
        if (causal_mask && (global_r_end == (t_c_idx * Bc + Bc) - delta_pos)) {
            for (int i = 0; i < Br; ++i) {
                for (int j = 0; j < Bc; ++j) {
                    if (j > i) { acc_s[i * Bc + j] = NEG_INF; }
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
#elif __ARM_NEON__
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
#elif __ARM_NEON__
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
#elif __ARM_NEON__
        const int32_t global_r_start = t_r_idx * Br;
        const int32_t global_c_start = t_c_idx * Bc;
        int delta_pos = seq_size_k - seq_size_q;
        if (causal_mask && (global_c_start - delta_pos > (global_r_start + Br - 1))) return;
        const int32_t v_stride_size = kv_head_size * dim_size;
        // 循环结构 (Br, dim_size, Bc) 对 acc_o 的读写更友好
        for (int b_r_idx = 0; b_r_idx < Br; ++b_r_idx) {
            for (int d_base = 0; d_base < dim_size; d_base += 4) {
                // 加载128位的FP32累加器
                float32x4_t acc_vec = vld1q_f32(acc_o + b_r_idx * dim_size + d_base);
                for (int b_c_idx = 0; b_c_idx < Bc; ++b_c_idx) {
                    // 从w_block直接读取FP32的P标量
                    const float p_scalar_f32 = w_block[b_r_idx * Bc + b_c_idx];
                    // 将FP32标量广播为一个128位向量
                    const float32x4_t p_vec_f32 = vdupq_n_f32(p_scalar_f32);
                    // 加载FP16的V向量
                    const __fp16 *v_ptr = (const __fp16 *)v_block + b_c_idx * v_stride_size + d_base;
                    const float16x4_t v_vec_f16 = vld1_f16(v_ptr);
                    // 将V向量从FP16转换为FP32
                    const float32x4_t v_vec_f32 = vcvt_f32_f16(v_vec_f16);
                    // 执行FP32的融合乘加
                    acc_vec = vfmaq_f32(acc_vec, p_vec_f32, v_vec_f32);
                }
                // 将结果写回内存
                vst1q_f32(acc_o + b_r_idx * dim_size + d_base, acc_vec);
            }
        }
#endif
    }

    inline void scale_and_store(const acc_dtype_t *__restrict__ acc_o, const acc_dtype_t *__restrict__ logsum,
                                dtype_out_t *__restrict__ o_block, const int32_t t_r_idx,
                                const int32_t head_size, const int32_t dim_size) {
        // (无变化，输出O是fp32)
        for (int i = 0; i < Br; ++i) {
            // 【修正】这里的 o_block_line 计算是错误的，它没有正确处理 BSHD 布局下的行步进
            //  正确的行步进已经由外层循环的 o_block 指针计算好了
            //  我们只需要在此基础上按行写入即可
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
#elif __ARM_NEON__
            float32x4_t reciprocal_logsum_vec = vdupq_n_f32(1.0f / logsum[i]);
            int j = 0;
            for (; j <= dim_size - 4; j += 4) {
                vst1q_f32(o_block_line + j, vmulq_f32(vld1q_f32(acc_o + i * dim_size + j), reciprocal_logsum_vec));
            }
            float reciprocal_logsum = 1.0f / logsum[i];
            for (; j < dim_size; ++j) { o_block_line[j] = acc_o[i * dim_size + j] * reciprocal_logsum; }
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
#elif __ARM_NEON__
        const int32_t global_r_start = t_r_idx * Br, global_r_end = global_r_start + Br_n_fixed;
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
                    sum_vec = vfmaq_f32(sum_vec, vld1q_f32(q_block_line + i), MLLM_NEON_F32x4_FROM_FP16(k_block_line + i));
                }
                acc_dtype_t total = _vaddvq_f32_hadd(sum_vec);
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
            acc_dtype_t *row = acc_s + br * Bc;
            float max_val = NEG_INF;
            for (int bc = 0; bc < Bc_n_fixed; ++bc) max_val = fmaxf(max_val, row[bc]);
            scoremax[br] = fmaxf(max_val, scoremax[br]);
        }
        for (int br = 0; br < Br_n_fixed; ++br) { score_scale[br] = expf((scoremax_prev[br] - scoremax[br]) * scale); }
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

    // (无变化)
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
#elif __ARM_NEON__
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
#elif __ARM_NEON__
        const int32_t global_r_start = t_r_idx * Br;
        const int32_t global_c_start = t_c_idx * Bc;
        int delta_pos = seq_size_k - seq_size_q;
        if (causal_mask && (global_c_start - delta_pos > (global_r_start + Br_n_fixed - 1))) return;
        const int32_t v_stride_size = kv_head_size * dim_size;
        for (int b_r_idx = 0; b_r_idx < Br_n_fixed; ++b_r_idx) {
            for (int d_base = 0; d_base < dim_size; d_base += 4) {
                float32x4_t acc = vld1q_f32(acc_o + b_r_idx * dim_size + d_base);
                for (int b_c_idx = 0; b_c_idx < Bc_n_fixed; ++b_c_idx) {
                    float32x4_t w_vec = vdupq_n_f32(w_block[b_r_idx * Bc + b_c_idx]);
                    const dtype_kv_in_t *v_ptr = v_block + b_c_idx * v_stride_size + d_base;
                    acc = vfmaq_f32(acc, w_vec, MLLM_NEON_F32x4_FROM_FP16(v_ptr));
                }
                vst1q_f32(acc_o + b_r_idx * dim_size + d_base, acc);
            }
        }
#endif
    }

    inline void scale_and_store_pa_n_fixed(const int32_t Br_n_fixed, const acc_dtype_t *__restrict__ acc_o,
                                           const acc_dtype_t *__restrict__ logsum, dtype_out_t *__restrict__ o_block,
                                           const int32_t t_r_idx, const int32_t head_size, const int32_t dim_size) {
        for (int i = 0; i < Br_n_fixed; ++i) {
            // 【修正】同上，这里的行步进计算也是错误的
            dtype_out_t *o_block_line = o_block + i * head_size * dim_size; // << 保持 BSHD 的行步长
#ifdef __AVX2__
            float reciprocal_logsum = 1.0f / logsum[i];
            __m256 reciprocal_logsum_vec = _mm256_set1_ps(reciprocal_logsum);
            int j = 0;
            for (; j <= dim_size - 8; j += 8) {
                __m256 vec_acc_o = _mm256_loadu_ps(acc_o + i * dim_size + j);
                _mm256_storeu_ps(o_block_line + j, _mm256_mul_ps(vec_acc_o, reciprocal_logsum_vec));
            }
            for (; j < dim_size; ++j) { o_block_line[j] = acc_o[i * dim_size + j] * reciprocal_logsum; }
#elif __ARM_NEON__
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

    // (此函数无变化，但为完整性一并提供)
    inline void init_temp_d(acc_dtype_t *logsum, acc_dtype_t *scoremax, acc_dtype_t *acc_o, const int32_t dim_size) {
        logsum[0] = 0.0f;
        scoremax[0] = NEG_INF;
#ifdef __AVX2__
        __m256 zero_vec = _mm256_setzero_ps();
        for (int i = 0; i < 1 * dim_size; i += 8) { _mm256_storeu_ps(acc_o + i, zero_vec); }
#elif __ARM_NEON__
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
#elif __ARM_NEON__
        const dtype_q_in_t *q_block_line = q_block;
        for (int32_t b_c_idx = 0; b_c_idx < Bc; ++b_c_idx) {
            const dtype_kv_in_t *k_block_line = k_block + b_c_idx * kv_stride_size;
            float32x4_t sum_vec = vdupq_n_f32(0.0f);
            int i = 0;
            for (; i <= dim_size - 4; i += 4) {
                sum_vec = vfmaq_f32(sum_vec, vld1q_f32(q_block_line + i), MLLM_NEON_F32x4_FROM_FP16(k_block_line + i));
            }
            acc_dtype_t total = _vaddvq_f32_hadd(sum_vec);
            for (; i < dim_size; ++i) { total += q_block_line[i] * MLLM_FP16_TO_FP32(k_block_line[i]); }
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

    // (此函数无变化，但为完整性一并提供)
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
#elif __ARM_NEON__
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
#elif __ARM_NEON__
        const int32_t v_stride_size = kv_head_size * dim_size;
        for (int d_base = 0; d_base < dim_size; d_base += 4) {
            float32x4_t acc = vld1q_f32(acc_o + d_base);
            for (int b_c_idx = 0; b_c_idx < Bc; ++b_c_idx) {
                float32x4_t w_vec = vdupq_n_f32(w_block[b_c_idx]);
                const dtype_kv_in_t *v_ptr = v_block + b_c_idx * v_stride_size + d_base;
                acc = vfmaq_f32(acc, w_vec, MLLM_NEON_F32x4_FROM_FP16(v_ptr));
            }
            vst1q_f32(acc_o + d_base, acc);
        }
#endif
    }
    // (此函数无变化，但为完整性一并提供)
    inline void scale_and_store_d(const acc_dtype_t *__restrict__ acc_o,
                                  const acc_dtype_t *__restrict__ logsum,
                                  dtype_out_t *__restrict__ o_block, const int32_t t_r_idx,
                                  const int32_t head_size, const int32_t dim_size) {
        float reciprocal_logsum = 1.0f / logsum[0];
#ifdef __AVX2__
        __m256 reciprocal_logsum_vec = _mm256_set1_ps(reciprocal_logsum);
        int j = 0;
        for (; j <= dim_size - 8; j += 8) {
            _mm256_storeu_ps(o_block + j, _mm256_mul_ps(_mm256_loadu_ps(acc_o + j), reciprocal_logsum_vec));
        }
#elif __ARM_NEON__
        float32x4_t reciprocal_logsum_vec = vdupq_n_f32(reciprocal_logsum);
        int j = 0;
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

    // (此函数无变化，但为完整性一并提供)
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
                // Scalar fallback for leftover dimensions
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

// ========================================
// 【修改】统一的FlashAttention2接口，改为模板以支持不同实现
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
    // 构造函数：初始化所有指针和大小
    WorkspaceManager() :
        workspace_{}, current_sizes_{} {
    }

    // 析构函数：在线程退出时自动释放所有内存，防止内存泄漏
    ~WorkspaceManager() {
        for (int i = 0; i < 7; ++i) {
            if (workspace_[i]) {
                aligned_free(workspace_[i]);
            }
        }
    }

    // 获取工作空间的核心函数
    void **get_workspace(const size_t *required_sizes) {
        for (int i = 0; i < 7; ++i) {
            // 如果需要的尺寸大于当前分配的尺寸，则重新分配
            if (required_sizes[i] > current_sizes_[i]) {
                // 释放旧的（如果存在）
                if (workspace_[i]) {
                    aligned_free(workspace_[i]);
                }
                // 分配新的
                aligned_alloc(&workspace_[i], required_sizes[i], 32);
                // 更新当前尺寸
                current_sizes_[i] = required_sizes[i];
            }
        }
        return workspace_;
    }

private:
    // 禁止拷贝和赋值，确保每个实例的唯一性
    WorkspaceManager(const WorkspaceManager &) = delete;
    WorkspaceManager &operator=(const WorkspaceManager &) = delete;

    void *workspace_[7];
    size_t current_sizes_[7];
};

} // namespace mobi_attn

/*
void flash_attention_2_forward(
    const void *Q, const void *K, const void *V, void *O,
    int32_t batch_size, int32_t head_size, int32_t seq_size_q, int32_t seq_size_k, int32_t dim_size,
    bool causal_mask, bool use_fp32, int32_t threads, int32_t br, int32_t bc,
    int32_t q_head, int32_t kv_head, bool high_precision_exp) {
    // 工作空间大小与输入数据类型无关，因为内部累加器总是float32
    // acc_s_cast is no longer needed as the intermediate softmax result is kept in float32
    const size_t acc_o_size = threads * br * dim_size * sizeof(float);
    const size_t acc_s_size = threads * br * bc * sizeof(float);
    const size_t logsum_size = threads * br * sizeof(float);
    const size_t scoremax_size = threads * br * sizeof(float);
    const size_t scoremax_prev_size = threads * br * sizeof(float);
    const size_t score_scale_size = threads * br * sizeof(float);
    const size_t score_sum_size = threads * br * sizeof(float);

    // 分配对齐的工作空间
    void *workspace[7];
    mobi_attn::aligned_alloc(&workspace[0], acc_o_size, 32);
    mobi_attn::aligned_alloc(&workspace[1], acc_s_size, 32);
    mobi_attn::aligned_alloc(&workspace[2], logsum_size, 32);
    mobi_attn::aligned_alloc(&workspace[3], scoremax_size, 32);
    mobi_attn::aligned_alloc(&workspace[4], scoremax_prev_size, 32);
    mobi_attn::aligned_alloc(&workspace[5], score_scale_size, 32);
    mobi_attn::aligned_alloc(&workspace[6], score_sum_size, 32);

    if (use_fp32) {
        // 使用纯FP32实现
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
        // 使用FP16输入，FP32输出的实现
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

    for (void *ptr : workspace) {
        if (ptr) mobi_attn::aligned_free(ptr);
    }
}
*/
// 文件位置: FlashAttention2.hpp
// 这是文件最末尾的函数

// 【替换此函数】
void flash_attention_2_forward(
    const void *Q, const void *K, const void *V, void *O,
    int32_t batch_size, int32_t head_size, int32_t seq_size_q, int32_t seq_size_k, int32_t dim_size,
    bool causal_mask, bool use_fp32, int32_t threads, int32_t br, int32_t bc,
    int32_t q_head, int32_t kv_head, bool high_precision_exp) {
    thread_local mobi_attn::WorkspaceManager manager;

    // 计算当前调用所需的各个工作空间大小
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

    // 【修改2】通过 manager 获取工作空间指针
    // manager 会自动处理内存的复用和按需重新分配
    void **workspace = manager.get_workspace(required_sizes);

    if (use_fp32) {
        // 使用纯FP32实现
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
        // 使用FP16输入，FP32输出的实现
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