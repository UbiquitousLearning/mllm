
#ifndef MLLM_FA2_CAL_HPP
#define MLLM_FA2_CAL_HPP

// #include <iostream>
// #include <random>
// #include <chrono>
#ifdef __AVX2__
#include <cstdint>
#include <immintrin.h>
#include <omp.h>
#include <cassert>
#include <limits>
#include <cmath>
#include <cstring>
#include "Types.hpp"
#include "VecDot.hpp"

namespace mobi_attn {

// ========================================
// 数学函数和工具
// ========================================
#define NEG_INF std::numeric_limits<float>::lowest()
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

// ========================================
// 内存对齐分配函数
// ========================================
// 使用 posix_memalign 进行分配
void x86_align_alloc(void **ptr, size_t required_bytes, size_t align) {
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
void x86_align_free(void *ptr) {
    free(ptr);
}

// ========================================
// FlashAttention2 核心实现 (FP32版本)
// ========================================
struct AVX_FA_2_GQA_QKV_FP32_BSHD_O_FP32_BSHD_ACC_FP32_IMPL {
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

    void fa2(const dtype_t *__restrict__ Q, const dtype_t *__restrict__ K,
             const dtype_t *__restrict__ V, dtype_t *__restrict__ O, const int32_t batch_size,
             const int32_t head_size, const int32_t seq_size_q, const int32_t seq_size_k,
             const int32_t dim_size, bool causal_mask = true) {
        assert(Br == Bc);
        // assert(Br % 4 == 0);
        // FIX: Assert that Q_Head is a multiple of KV_Head for valid GQA/MHA.
        assert(Q_Head % KV_Head == 0);
        assert(head_size % threads == 0);
        assert(dim_size % 8 == 0); // AVX processes 8 floats at a time

        if (seq_size_q != 1) {
            __fa2_prefill_append(Q, K, V, O, batch_size, head_size, seq_size_q, seq_size_k, dim_size,
                                 causal_mask);
        } else {
            __fa2_decode(Q, K, V, O, batch_size, head_size, seq_size_q, seq_size_k, dim_size,
                         causal_mask);
        }
    }

private:
    // inline void init_temp(acc_dtype_t *logsum, acc_dtype_t *scoremax, acc_dtype_t *acc_o,
    //                       const int32_t dim_size) {
    //     __m256 zero_vec = _mm256_set1_ps(0.0f);
    //     __m256 neg_inf_vec = _mm256_set1_ps(NEG_INF);

    //     for (int i = 0; i < Br; i += 8) { _mm256_storeu_ps(logsum + i, zero_vec); }
    //     for (int i = 0; i < Br; i += 8) { _mm256_storeu_ps(scoremax + i, neg_inf_vec); }
    //     for (int i = 0; i < Br * dim_size; i += 8) { _mm256_storeu_ps(acc_o + i, zero_vec); }
    // }

    inline void init_temp(acc_dtype_t *logsum, acc_dtype_t *scoremax, acc_dtype_t *acc_o, const int32_t dim_size) {
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
    }

    // 【关键修改】函数签名增加 kv_stride_size 参数，用于区分Q和K的步长
    inline void mma0(const dtype_t *__restrict__ q_block, const dtype_t *__restrict__ k_block,
                     acc_dtype_t *__restrict__ acc_s, const int32_t dim_size,
                     const int32_t q_stride_size, const int32_t kv_stride_size,
                     const int32_t t_r_idx, const int32_t t_c_idx,
                     const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
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
    }

    inline void softmax(acc_dtype_t *__restrict__ acc_s, acc_dtype_t *scoremax, acc_dtype_t *scoremax_prev,
                        acc_dtype_t *score_scale, acc_dtype_t *score_sum, acc_dtype_t *logsum,
                        const float scale,
                        const int32_t t_r_idx, const int32_t t_c_idx,
                        const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
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
    }

    inline void rescale(acc_dtype_t *__restrict__ acc_o, acc_dtype_t *__restrict__ score_scale,
                        const int32_t dim_size, const int32_t t_r_idx, const int32_t t_c_idx,
                        const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
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
    }

    // 【关键修改】函数签名增加 kv_head_size 参数，用于计算V的内部步长
    inline void mma1(const acc_dtype_t *__restrict__ w_block, const dtype_t *__restrict__ v_block,
                     acc_dtype_t *__restrict__ acc_o, const int32_t kv_head_size, const int32_t dim_size,
                     const int32_t t_r_idx, const int32_t t_c_idx, const int32_t seq_size_q,
                     const int32_t seq_size_k, bool causal_mask) {
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
    }

    inline void scale_and_store(const acc_dtype_t *__restrict__ acc_o,
                                const acc_dtype_t *__restrict__ logsum,
                                dtype_t *__restrict__ o_block, const int32_t t_r_idx,
                                const int32_t head_size, const int32_t dim_size) {
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
    }

    // N-fixed functions for handling leftovers
    // FIX: Modified mma0_pa_n_fixed to accept separate strides.
    inline void mma0_pa_n_fixed(const int32_t Br_n_fixed, const int32_t Bc_n_fixed,
                                const dtype_t *__restrict__ q_block,
                                const dtype_t *__restrict__ k_block, acc_dtype_t *__restrict__ acc_s,
                                const int32_t dim_size, const int32_t q_stride_size, const int32_t kv_stride_size,
                                const int32_t t_r_idx, const int32_t t_c_idx,
                                const int32_t seq_size_q, const int32_t seq_size_k,
                                bool causal_mask) {
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
    }

    // FIX: Modified mma1_pa_n_fixed to accept kv_head_size.
    inline void mma1_pa_n_fixed(const int32_t Br_n_fixed, const int32_t Bc_n_fixed,
                                const acc_dtype_t *__restrict__ w_block,
                                const dtype_t *__restrict__ v_block, acc_dtype_t *__restrict__ acc_o,
                                const int32_t kv_head_size, const int32_t dim_size,
                                const int32_t t_r_idx, const int32_t t_c_idx,
                                const int32_t seq_size_q, const int32_t seq_size_k,
                                bool causal_mask) {
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
    }

    inline void scale_and_store_pa_n_fixed(const int32_t Br_n_fixed,
                                           const acc_dtype_t *__restrict__ acc_o,
                                           const acc_dtype_t *__restrict__ logsum,
                                           dtype_t *__restrict__ o_block, const int32_t t_r_idx,
                                           const int32_t head_size, const int32_t dim_size) {
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
    }

    // Decode mode functions
    inline void init_temp_d(acc_dtype_t *logsum, acc_dtype_t *scoremax, acc_dtype_t *acc_o,
                            const int32_t dim_size) {
        logsum[0] = 0.0f;
        scoremax[0] = NEG_INF;
        __m256 zero_vec = _mm256_setzero_ps();
        for (int i = 0; i < 1 * dim_size; i += 8) { _mm256_storeu_ps(acc_o + i, zero_vec); }
    }

    // FIX: Modified mma0_d to accept kv_stride_size.
    inline void mma0_d(const dtype_t *__restrict__ q_block, const dtype_t *__restrict__ k_block,
                       acc_dtype_t *__restrict__ acc_s, const int32_t dim_size,
                       const int32_t kv_stride_size, const int32_t t_r_idx, const int32_t t_c_idx,
                       const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
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
        __m256 scale_v = _mm256_set1_ps(score_scale[0]);
        for (int j = 0; j < dim_size; j += 8) {
            __m256 acc = _mm256_loadu_ps(acc_o + j);
            acc = _mm256_mul_ps(acc, scale_v);
            _mm256_storeu_ps(acc_o + j, acc);
        }
    }

    // FIX: Modified mma1_d to accept kv_head_size.
    inline void mma1_d(const acc_dtype_t *__restrict__ w_block, const dtype_t *__restrict__ v_block,
                       acc_dtype_t *__restrict__ acc_o, const int32_t kv_head_size,
                       const int32_t dim_size, const int32_t t_r_idx, const int32_t t_c_idx,
                       const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
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
    }

    inline void scale_and_store_d(const acc_dtype_t *__restrict__ acc_o,
                                  const acc_dtype_t *__restrict__ logsum,
                                  dtype_t *__restrict__ o_block, const int32_t t_r_idx,
                                  const int32_t head_size, const int32_t dim_size) {
        float reciprocal_logsum = 1.0f / logsum[0];
        __m256 reciprocal_logsum_vec = _mm256_set1_ps(reciprocal_logsum);
        int j = 0;
        for (; j <= dim_size - 8; j += 8) {
            _mm256_storeu_ps(o_block + j, _mm256_mul_ps(_mm256_loadu_ps(acc_o + j), reciprocal_logsum_vec));
        }
        for (; j < dim_size; ++j) {
            o_block[j] = acc_o[j] * reciprocal_logsum;
        }
    }

    // Decode n-fixed functions
    // FIX: Modified mma0_d_n_fixed to accept kv_stride_size.
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

    // FIX: Modified mma1_d_n_fixed to accept kv_head_size.
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
struct AVX_FA_2_GQA_Q_FP32_KV_FP16_BSHD_O_FP32_BSHD_ACC_FP32_IMPL {
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
        // assert(Br % 4 == 0);
        assert(head_size % threads == 0);
        assert(dim_size % 8 == 0);

        if (seq_size_q != 1) {
            __fa2_prefill_append(Q, K, V, O, batch_size, head_size, seq_size_q, seq_size_k, dim_size, causal_mask);
        } else {
            __fa2_decode(Q, K, V, O, batch_size, head_size, seq_size_q, seq_size_k, dim_size, causal_mask);
        }
    }

private:
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
    }

    inline void mma0(const dtype_q_in_t *__restrict__ q_block, const dtype_kv_in_t *__restrict__ k_block,
                     acc_dtype_t *__restrict__ acc_s, const int32_t dim_size,
                     const int32_t q_stride_size, const int32_t kv_stride_size,
                     const int32_t t_r_idx, const int32_t t_c_idx,
                     const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
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
    }

    inline void rescale(acc_dtype_t *__restrict__ acc_o, acc_dtype_t *__restrict__ score_scale,
                        const int32_t dim_size, const int32_t t_r_idx, const int32_t t_c_idx,
                        const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
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
    }

    inline void mma1(const acc_dtype_t *__restrict__ w_block, const dtype_kv_in_t *__restrict__ v_block,
                     acc_dtype_t *__restrict__ acc_o, const int32_t kv_head_size, const int32_t dim_size,
                     const int32_t t_r_idx, const int32_t t_c_idx, const int32_t seq_size_q,
                     const int32_t seq_size_k, bool causal_mask) {
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

            __m256 reciprocal_logsum_vec = _mm256_set1_ps(1.0f / logsum[i]);
            int j = 0;
            for (; j <= dim_size - 8; j += 8) {
                __m256 vec_acc_o = _mm256_loadu_ps(acc_o + i * dim_size + j);
                __m256 result_vec = _mm256_mul_ps(vec_acc_o, reciprocal_logsum_vec);
                _mm256_storeu_ps(o_block_line + j, result_vec);
            }
            float reciprocal_logsum = 1.0f / logsum[i];
            for (; j < dim_size; ++j) { o_block_line[j] = acc_o[i * dim_size + j] * reciprocal_logsum; }
        }
    }

    inline void mma0_pa_n_fixed(const int32_t Br_n_fixed, const int32_t Bc_n_fixed,
                                const dtype_q_in_t *__restrict__ q_block, const dtype_kv_in_t *__restrict__ k_block,
                                acc_dtype_t *__restrict__ acc_s, const int32_t dim_size,
                                const int32_t q_stride_size, const int32_t kv_stride_size,
                                const int32_t t_r_idx, const int32_t t_c_idx, const int32_t seq_size_q,
                                const int32_t seq_size_k, bool causal_mask) {
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
    }

    inline void mma1_pa_n_fixed(const int32_t Br_n_fixed, const int32_t Bc_n_fixed,
                                const acc_dtype_t *__restrict__ w_block, const dtype_kv_in_t *__restrict__ v_block,
                                acc_dtype_t *__restrict__ acc_o, const int32_t kv_head_size, const int32_t dim_size,
                                const int32_t t_r_idx, const int32_t t_c_idx, const int32_t seq_size_q,
                                const int32_t seq_size_k, bool causal_mask) {
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
    }

    inline void scale_and_store_pa_n_fixed(const int32_t Br_n_fixed, const acc_dtype_t *__restrict__ acc_o,
                                           const acc_dtype_t *__restrict__ logsum, dtype_out_t *__restrict__ o_block,
                                           const int32_t t_r_idx, const int32_t head_size, const int32_t dim_size) {
        for (int i = 0; i < Br_n_fixed; ++i) {
            // 【修正】同上，这里的行步进计算也是错误的
            dtype_out_t *o_block_line = o_block + i * head_size * dim_size; // << 保持 BSHD 的行步长

            float reciprocal_logsum = 1.0f / logsum[i];
            __m256 reciprocal_logsum_vec = _mm256_set1_ps(reciprocal_logsum);
            int j = 0;
            for (; j <= dim_size - 8; j += 8) {
                __m256 vec_acc_o = _mm256_loadu_ps(acc_o + i * dim_size + j);
                _mm256_storeu_ps(o_block_line + j, _mm256_mul_ps(vec_acc_o, reciprocal_logsum_vec));
            }
            for (; j < dim_size; ++j) { o_block_line[j] = acc_o[i * dim_size + j] * reciprocal_logsum; }
        }
    }

    // (此函数无变化，但为完整性一并提供)
    inline void init_temp_d(acc_dtype_t *logsum, acc_dtype_t *scoremax, acc_dtype_t *acc_o, const int32_t dim_size) {
        logsum[0] = 0.0f;
        scoremax[0] = NEG_INF;
        __m256 zero_vec = _mm256_setzero_ps();
        for (int i = 0; i < 1 * dim_size; i += 8) { _mm256_storeu_ps(acc_o + i, zero_vec); }
    }

    inline void mma0_d(const dtype_q_in_t *__restrict__ q_block, const dtype_kv_in_t *__restrict__ k_block,
                       acc_dtype_t *__restrict__ acc_s, const int32_t dim_size,
                       const int32_t kv_stride_size, const int32_t t_r_idx, const int32_t t_c_idx,
                       const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
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
        __m256 scale_v = _mm256_set1_ps(score_scale[0]);
        for (int j = 0; j < dim_size; j += 8) {
            __m256 acc = _mm256_loadu_ps(acc_o + j);
            acc = _mm256_mul_ps(acc, scale_v);
            _mm256_storeu_ps(acc_o + j, acc);
        }
    }

    inline void mma1_d(const acc_dtype_t *__restrict__ w_block, const dtype_kv_in_t *__restrict__ v_block,
                       acc_dtype_t *__restrict__ acc_o, const int32_t kv_head_size, const int32_t dim_size,
                       const int32_t t_r_idx, const int32_t t_c_idx, const int32_t seq_size_q,
                       const int32_t seq_size_k, bool causal_mask) {
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
    }
    // (此函数无变化，但为完整性一并提供)
    inline void scale_and_store_d(const acc_dtype_t *__restrict__ acc_o,
                                  const acc_dtype_t *__restrict__ logsum,
                                  dtype_out_t *__restrict__ o_block, const int32_t t_r_idx,
                                  const int32_t head_size, const int32_t dim_size) {
        float reciprocal_logsum = 1.0f / logsum[0];
        __m256 reciprocal_logsum_vec = _mm256_set1_ps(reciprocal_logsum);
        int j = 0;
        for (; j <= dim_size - 8; j += 8) {
            _mm256_storeu_ps(o_block + j, _mm256_mul_ps(_mm256_loadu_ps(acc_o + j), reciprocal_logsum_vec));
        }
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

} // namespace mobi_attn

// ========================================
// 用户接口函数
// ========================================

// 【修改】用户接口函数，增加FP16输入分支
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
    mobi_attn::x86_align_alloc(&workspace[0], acc_o_size, 32);
    mobi_attn::x86_align_alloc(&workspace[1], acc_s_size, 32);
    mobi_attn::x86_align_alloc(&workspace[2], logsum_size, 32);
    mobi_attn::x86_align_alloc(&workspace[3], scoremax_size, 32);
    mobi_attn::x86_align_alloc(&workspace[4], scoremax_prev_size, 32);
    mobi_attn::x86_align_alloc(&workspace[5], score_scale_size, 32);
    mobi_attn::x86_align_alloc(&workspace[6], score_sum_size, 32);

    if (use_fp32) {
        // 使用纯FP32实现
        mobi_attn::FlashAttn2T<mobi_attn::AVX_FA_2_GQA_QKV_FP32_BSHD_O_FP32_BSHD_ACC_FP32_IMPL> op;
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
        mobi_attn::FlashAttn2T<mobi_attn::AVX_FA_2_GQA_Q_FP32_KV_FP16_BSHD_O_FP32_BSHD_ACC_FP32_IMPL> op;
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

    // 释放工作空间
    for (void *ptr : workspace) {
        if (ptr) mobi_attn::x86_align_free(ptr);
    }
}
#elif __ARM_NEON
#include <cstdint>
// 核心修改：将 x86 的 immintrin.h 替换为 ARM 的 arm_neon.h
#include <arm_neon.h>
#include <omp.h>
#include <cassert>
#include <limits>
#include <cmath>
#include <cstring>
#include "Types.hpp"
#include "VecDot.hpp"

namespace mobi_attn {

// ========================================
// 数学函数和工具 (NEON版本)
// ========================================
#define NEG_INF std::numeric_limits<float>::lowest()

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

// ========================================
// 高性能NEON数学函数 (新增)
// ========================================

// 基于多项式逼近的快速exp实现 (NEON版本)
inline float32x4_t vexpq_fast_f32(float32x4_t x) {
    // 定义常量
    const float32x4_t c0 = vdupq_n_f32(1.0f);
    const float32x4_t c1 = vdupq_n_f32(0.0416598990559578f);
    const float32x4_t c2 = vdupq_n_f32(0.166664719581604f);
    const float32x4_t c3 = vdupq_n_f32(0.5000005960464478f);
    const float32x4_t log2e = vdupq_n_f32(1.4426950408889634f);
    const float32x4_t ln2_hi = vdupq_n_f32(0.693145751953125f);
    const float32x4_t ln2_lo = vdupq_n_f32(1.428606765330187e-06f);
    const int32x4_t M_126 = vdupq_n_s32(126);

    // 计算 y = x * log2(e)
    float32x4_t y = vmulq_f32(x, log2e);

    // 对y取整, n = round(y)
    int32x4_t n = vcvtaq_s32_f32(y);

    // 计算 z = x - n * ln2
    float32x4_t n_f = vcvtq_f32_s32(n);
    float32x4_t z = vmlsq_f32(x, n_f, ln2_hi);
    z = vmlsq_f32(z, n_f, ln2_lo);

    // 多项式逼近 exp(z) ~= 1 + z + z^2/2! + ...
    float32x4_t poly = c1;
    poly = vmlaq_f32(c2, poly, z);
    poly = vmlaq_f32(c3, poly, z);
    poly = vmlaq_f32(c0, poly, z);
    poly = vmlaq_f32(poly, vmulq_f32(z, z), poly);

    // 组合结果: poly * 2^n
    int32x4_t m = vaddq_s32(n, M_126);
    m = vshlq_n_s32(m, 23);

    return vmulq_f32(poly, vreinterpretq_f32_s32(m));
}

// ========================================
// 内存对齐分配函数 (重命名以去除x86特定性)
// ========================================
// 使用 posix_memalign 进行分配，此函数在支持POSIX的系统（如Linux）上是通用的
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
// FlashAttention2 核心实现 (FP32版本, NEON)
// ========================================
struct NEON_FA_2_GQA_QKV_FP32_BSHD_O_FP32_BSHD_ACC_FP32_IMPL {
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

    // 初始化工作空间指针
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

    // fa2 主函数，根据Q的序列长度分发到 prefill/append 或 decode 模式
    void fa2(const dtype_t *__restrict__ Q, const dtype_t *__restrict__ K,
             const dtype_t *__restrict__ V, dtype_t *__restrict__ O, const int32_t batch_size,
             const int32_t head_size, const int32_t seq_size_q, const int32_t seq_size_k,
             const int32_t dim_size, bool causal_mask = true) {
        assert(Br == Bc);
        // NEON 一次处理 4 个 float
        assert(dim_size % 4 == 0);
        // 确保Q头是KV头的整数倍，这是GQA/MHA的有效性要求
        assert(Q_Head % KV_Head == 0);
        assert(head_size % threads == 0);

        if (seq_size_q != 1) {
            __fa2_prefill_append(Q, K, V, O, batch_size, head_size, seq_size_q, seq_size_k, dim_size,
                                 causal_mask);
        } else {
            __fa2_decode(Q, K, V, O, batch_size, head_size, seq_size_q, seq_size_k, dim_size,
                         causal_mask);
        }
    }

    // =========================================================================================
    // 以下是 NEON_FA_2_GQA_QKV_FP32_BSHD_O_FP32_BSHD_ACC_FP32_IMPL 结构体内部的私有函数实现
    // 承接第一部分的代码
    // =========================================================================================

private:
    // 核心计算函数 (prefill/append 模式, 适用于 seq_size_q > 1)
    inline void __fa2_prefill_append(const dtype_t *__restrict__ Q, const dtype_t *__restrict__ K,
                                     const dtype_t *__restrict__ V, dtype_t *__restrict__ O,
                                     const int32_t batch_size, const int32_t head_size, // head_size 就是 Q_Head
                                     const int32_t seq_size_q, const int32_t seq_size_k,
                                     const int32_t dim_size, bool causal_mask = true) {
        // Tr, Tc 分别是 Q 和 K/V 在序列长度维度上被切分成的块数
        const int32_t Tr = seq_size_q / Br;
        const int32_t Tr_left = seq_size_q % Br;
        const int32_t Tc = seq_size_k / Bc;
        const int32_t Tc_left = seq_size_k % Bc;

        const float local_scale = 1.0f / sqrtf(static_cast<float>(dim_size));

        // 计算Q头与KV头的分组对应关系 (GQA)
        const int32_t kv_group_size = Q_Head / KV_Head;

        for (int32_t b_idx = 0; b_idx < batch_size; ++b_idx) {
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1) if (threads > 1)
            for (int32_t h_idx = 0; h_idx < head_size; ++h_idx) { // h_idx 是当前Q头的索引
                const int32_t thread_id = omp_get_thread_num();
                const int32_t this_thread_head = h_idx;

                // 计算当前Q头 (h_idx) 对应的KV头索引
                const int32_t this_thread_kv_head = this_thread_head / kv_group_size;

                // --- 主循环 (处理完整的块) ---
                for (int t_r_idx = 0; t_r_idx < Tr; ++t_r_idx) {
                    // 初始化该线程的临时工作空间
                    init_temp(logsum_ + thread_id * Br, scoremax_ + thread_id * Br,
                              acc_o_ + thread_id * Br * dim_size, dim_size);

                    for (int t_c_idx = 0; t_c_idx < Tc; ++t_c_idx) {
                        // Q 的指针计算，它有 Q_Head (==head_size) 个头
                        const dtype_t *tile_q = Q + b_idx * seq_size_q * head_size * dim_size + t_r_idx * Br * head_size * dim_size + this_thread_head * dim_size;

                        // K 和 V 的指针计算，必须使用 KV_Head 和映射后的 this_thread_kv_head
                        const dtype_t *tile_k = K + b_idx * seq_size_k * KV_Head * dim_size + t_c_idx * Bc * KV_Head * dim_size + this_thread_kv_head * dim_size;
                        const dtype_t *tile_v = V + b_idx * seq_size_k * KV_Head * dim_size + t_c_idx * Bc * KV_Head * dim_size + this_thread_kv_head * dim_size;

                        acc_dtype_t *tile_acc_s = acc_s_ + thread_id * Br * Bc;
                        acc_dtype_t *acc_o = acc_o_ + thread_id * Br * dim_size;

                        // Step 1: Q * K^T
                        mma0(tile_q, tile_k, tile_acc_s, dim_size, head_size * dim_size, KV_Head * dim_size, t_r_idx, t_c_idx, seq_size_q, seq_size_k, causal_mask);
                        // Step 2: Softmax
                        softmax(tile_acc_s, scoremax_ + thread_id * Br, scoremax_prev_ + thread_id * Br, score_scale_ + thread_id * Br, score_sum_ + thread_id * Br, logsum_ + thread_id * Br, local_scale, t_r_idx, t_c_idx, seq_size_q, seq_size_k, causal_mask);
                        // Step 3: Rescale O
                        rescale(acc_o, score_scale_ + thread_id * Br, dim_size, t_r_idx, t_c_idx, seq_size_q, seq_size_k, causal_mask);
                        // Step 4: P * V
                        mma1(tile_acc_s, tile_v, acc_o, KV_Head, dim_size, t_r_idx, t_c_idx, seq_size_q, seq_size_k, causal_mask);
                    }
                    // --- 处理 K/V 序列的剩余部分 (Tc_left) ---
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
                    // Step 5: 将最终结果缩放并存回输出 O
                    scale_and_store(acc_o_ + thread_id * Br * dim_size, logsum_ + thread_id * Br, O + b_idx * seq_size_q * head_size * dim_size + t_r_idx * Br * head_size * dim_size + this_thread_head * dim_size, t_r_idx, head_size, dim_size);
                }
                // --- 处理 Q 序列的剩余部分 (Tr_left) ---
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
        // 在 decode 模式下, Q 的序列长度固定为 1, 因此 Tr = 1, Br = 1
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

    // 初始化临时工作区 (NEON 版本)
    inline void init_temp(acc_dtype_t *logsum, acc_dtype_t *scoremax, acc_dtype_t *acc_o, const int32_t dim_size) {
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
    }

    // Q * K^T 计算 (NEON 版本)
    inline void mma0(const dtype_t *__restrict__ q_block, const dtype_t *__restrict__ k_block,
                     acc_dtype_t *__restrict__ acc_s, const int32_t dim_size,
                     const int32_t q_stride_size, const int32_t kv_stride_size,
                     const int32_t t_r_idx, const int32_t t_c_idx,
                     const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
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
    }

    // Softmax (NEON 版本)
    inline void softmax(acc_dtype_t *__restrict__ acc_s, acc_dtype_t *scoremax, acc_dtype_t *scoremax_prev,
                        acc_dtype_t *score_scale, acc_dtype_t *score_sum, acc_dtype_t *logsum,
                        const float scale,
                        const int32_t t_r_idx, const int32_t t_c_idx,
                        const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
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
    }

    // 重缩放累加的输出 O (NEON 版本)
    inline void rescale(acc_dtype_t *__restrict__ acc_o, acc_dtype_t *__restrict__ score_scale,
                        const int32_t dim_size, const int32_t t_r_idx, const int32_t t_c_idx,
                        const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
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
    }

    // P * V 计算 (NEON 版本)
    inline void mma1(const acc_dtype_t *__restrict__ w_block, const dtype_t *__restrict__ v_block,
                     acc_dtype_t *__restrict__ acc_o, const int32_t kv_head_size, const int32_t dim_size,
                     const int32_t t_r_idx, const int32_t t_c_idx, const int32_t seq_size_q,
                     const int32_t seq_size_k, bool causal_mask) {
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
    }

    // 缩放并存储最终结果 (NEON 版本)
    inline void scale_and_store(const acc_dtype_t *__restrict__ acc_o,
                                const acc_dtype_t *__restrict__ logsum,
                                dtype_t *__restrict__ o_block, const int32_t t_r_idx,
                                const int32_t head_size, const int32_t dim_size) {
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
    }

    // --- 处理剩余块的 N-fixed 函数 (NEON 版本) ---
    inline void mma0_pa_n_fixed(const int32_t Br_n_fixed, const int32_t Bc_n_fixed,
                                const dtype_t *__restrict__ q_block,
                                const dtype_t *__restrict__ k_block, acc_dtype_t *__restrict__ acc_s,
                                const int32_t dim_size, const int32_t q_stride_size, const int32_t kv_stride_size,
                                const int32_t t_r_idx, const int32_t t_c_idx,
                                const int32_t seq_size_q, const int32_t seq_size_k,
                                bool causal_mask) {
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
    }

    // (这里的逻辑主要是标量，和AVX版本基本一致)
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
    }

    inline void mma1_pa_n_fixed(const int32_t Br_n_fixed, const int32_t Bc_n_fixed,
                                const acc_dtype_t *__restrict__ w_block,
                                const dtype_t *__restrict__ v_block, acc_dtype_t *__restrict__ acc_o,
                                const int32_t kv_head_size, const int32_t dim_size,
                                const int32_t t_r_idx, const int32_t t_c_idx,
                                const int32_t seq_size_q, const int32_t seq_size_k,
                                bool causal_mask) {
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
    }

    inline void scale_and_store_pa_n_fixed(const int32_t Br_n_fixed,
                                           const acc_dtype_t *__restrict__ acc_o,
                                           const acc_dtype_t *__restrict__ logsum,
                                           dtype_t *__restrict__ o_block, const int32_t t_r_idx,
                                           const int32_t head_size, const int32_t dim_size) {
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
    }

    // --- Decode 模式的辅助函数 (NEON 版本) ---

    inline void init_temp_d(acc_dtype_t *logsum, acc_dtype_t *scoremax, acc_dtype_t *acc_o,
                            const int32_t dim_size) {
        logsum[0] = 0.0f;
        scoremax[0] = NEG_INF;
        float32x4_t zero_vec = vdupq_n_f32(0.0f);
        // Br 在 decode 模式下为 1
        for (int i = 0; i < 1 * dim_size; i += 4) {
            vst1q_f32(acc_o + i, zero_vec);
        }
    }

    inline void mma0_d(const dtype_t *__restrict__ q_block, const dtype_t *__restrict__ k_block,
                       acc_dtype_t *__restrict__ acc_s, const int32_t dim_size,
                       const int32_t kv_stride_size, const int32_t t_r_idx, const int32_t t_c_idx,
                       const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
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
    }

    inline void softmax_d(acc_dtype_t *__restrict__ acc_s, acc_dtype_t *scoremax,
                          acc_dtype_t *scoremax_prev, acc_dtype_t *score_scale,
                          acc_dtype_t *score_sum, acc_dtype_t *logsum,
                          const float scale,
                          const int32_t t_r_idx, const int32_t t_c_idx,
                          const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
        scoremax_prev[0] = scoremax[0];

        float max_val = scoremax[0];
        for (int bc = 0; bc < Bc; ++bc) {
            max_val = fmaxf(max_val, acc_s[bc]);
        }
        scoremax[0] = max_val;

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
        float32x4_t scale_v = vdupq_n_f32(score_scale[0]);
        for (int j = 0; j < dim_size; j += 4) {
            float32x4_t acc = vld1q_f32(acc_o + j);
            acc = vmulq_f32(acc, scale_v);
            vst1q_f32(acc_o + j, acc);
        }
    }

    inline void mma1_d(const acc_dtype_t *__restrict__ w_block, const dtype_t *__restrict__ v_block,
                       acc_dtype_t *__restrict__ acc_o, const int32_t kv_head_size,
                       const int32_t dim_size, const int32_t t_r_idx, const int32_t t_c_idx,
                       const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
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
    }

    inline void scale_and_store_d(const acc_dtype_t *__restrict__ acc_o,
                                  const acc_dtype_t *__restrict__ logsum,
                                  dtype_t *__restrict__ o_block, const int32_t t_r_idx,
                                  const int32_t head_size, const int32_t dim_size) {
        float reciprocal_logsum = 1.0f / logsum[0];
        float32x4_t reciprocal_logsum_vec = vdupq_n_f32(reciprocal_logsum);
        int j = 0;
        for (; j <= dim_size - 4; j += 4) {
            vst1q_f32(o_block + j, vmulq_f32(vld1q_f32(acc_o + j), reciprocal_logsum_vec));
        }
        for (; j < dim_size; ++j) {
            o_block[j] = acc_o[j] * reciprocal_logsum;
        }
    }

    // --- Decode N-fixed 函数 (NEON 版本, 逻辑多为标量) ---
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
        float max_val = scoremax[0];
        for (int bc = 0; bc < Bc_n_fixed; ++bc) max_val = fmaxf(max_val, acc_s[bc]);
        scoremax[0] = max_val;

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
    // 私有成员，用于存储工作空间的指针
    acc_dtype_t *acc_o_;
    acc_dtype_t *acc_s_;
    acc_dtype_t *logsum_;
    acc_dtype_t *scoremax_;
    acc_dtype_t *scoremax_prev_;
    acc_dtype_t *score_scale_;
    acc_dtype_t *score_sum_;
};
// ========================================
// FlashAttention2 核心实现 (Q FP32/KV FP16 输入, FP32 输出, NEON 版本)
// 【注意：本版本为完整、未省略代码的版本】
// ========================================
struct NEON_FA_2_GQA_Q_FP32_KV_FP16_BSHD_O_FP32_BSHD_ACC_FP32_IMPL {
    // 定义不同的输入数据类型
    using dtype_q_in_t = float;
    using dtype_kv_in_t = mllm_fp16_t; // K和V是FP16
    using dtype_out_t = float;
    using acc_dtype_t = float;

    // 配置参数
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

    // 主函数 fa2, 注意 K 和 V 的类型是 dtype_kv_in_t
    void fa2(const dtype_q_in_t *__restrict__ Q, const dtype_kv_in_t *__restrict__ K,
             const dtype_kv_in_t *__restrict__ V, dtype_out_t *__restrict__ O, const int32_t batch_size,
             const int32_t head_size, const int32_t seq_size_q, const int32_t seq_size_k,
             const int32_t dim_size, bool causal_mask = true) {
        assert(Br == Bc);
        assert(dim_size % 4 == 0);
        assert(Q_Head % KV_Head == 0);
        assert(head_size % threads == 0);

        if (seq_size_q != 1) {
            __fa2_prefill_append(Q, K, V, O, batch_size, head_size, seq_size_q, seq_size_k, dim_size, causal_mask);
        } else {
            __fa2_decode(Q, K, V, O, batch_size, head_size, seq_size_q, seq_size_k, dim_size, causal_mask);
        }
    }

private:
    // 定义一个宏，用于从内存加载4个fp16, 并转换为一个fp32向量
    // 这需要 ARMv8.2-A FP16 指令支持
#define MLLM_NEON_F32x4_FROM_FP16(addr) vcvt_f32_f16(vld1_f16((const __fp16 *)(addr)))

    // Prefill/Append 主循环 (混合精度) - 完整版
    inline void __fa2_prefill_append(const dtype_q_in_t *__restrict__ Q, const dtype_kv_in_t *__restrict__ K,
                                     const dtype_kv_in_t *__restrict__ V, dtype_out_t *__restrict__ O,
                                     const int32_t batch_size, const int32_t head_size,
                                     const int32_t seq_size_q, const int32_t seq_size_k,
                                     const int32_t dim_size, bool causal_mask = true) {
        const int32_t Tr = seq_size_q / Br;
        const int32_t Tr_left = seq_size_q % Br;
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

                // --- 主循环 (Tr) ---
                for (int t_r_idx = 0; t_r_idx < Tr; ++t_r_idx) {
                    init_temp(logsum_ + thread_id * Br, scoremax_ + thread_id * Br, acc_o_ + thread_id * Br * dim_size, dim_size);
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
                // --- 处理 Q 序列的剩余部分 (Tr_left) - 完整版 ---
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

    // Decode 主循环 (混合精度) - 完整版
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

    // --- 完整版辅助函数 ---

    // (与FP32版本相同)
    inline void init_temp(acc_dtype_t *logsum, acc_dtype_t *scoremax, acc_dtype_t *acc_o, const int32_t dim_size) {
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
    }

    /*
    // 输入Q为FP32，但在函数内动态转为FP16，以使用最高效的 vfmlalq_f16 指令进行计算
    inline void mma0(const dtype_q_in_t *__restrict__ q_block, const dtype_kv_in_t *__restrict__ k_block,
                     acc_dtype_t *__restrict__ acc_s, const int32_t dim_size,
                     const int32_t q_stride_size, const int32_t kv_stride_size,
                     const int32_t t_r_idx, const int32_t t_c_idx,
                     const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
        // 因果掩码的前置检查
        const int32_t global_r_start = t_r_idx * Br;
        const int32_t global_r_end = global_r_start + Br;
        const int32_t global_c_start = t_c_idx * Bc;
        int delta_pos = seq_size_k - seq_size_q;

        if (causal_mask && (global_c_start - delta_pos > (global_r_end - 1))) { return; }

        // 遍历Br x Bc的块
        for (int32_t b_r_idx = 0; b_r_idx < Br; ++b_r_idx) {
            const dtype_q_in_t *q_block_line = q_block + b_r_idx * q_stride_size;
            for (int32_t b_c_idx = 0; b_c_idx < Bc; ++b_c_idx) {
                const dtype_kv_in_t *k_block_line = k_block + b_c_idx * kv_stride_size;

                // 使用多个FP32累加器，借鉴自您提供的参考文件
                float32x4_t sum0 = vdupq_n_f32(0.0f);
                float32x4_t sum1 = vdupq_n_f32(0.0f);

                int i = 0;
                // 主循环，一次处理16个元素 (两个float16x8_t向量)
                for (; i <= dim_size - 16; i += 16) {
                    __builtin_prefetch(q_block_line + i + 64);
                    __builtin_prefetch(k_block_line + i + 64);

                    // 1. 加载16个 FP32 的 Q
                    float32x4_t q_f32_0 = vld1q_f32(q_block_line + i);
                    float32x4_t q_f32_1 = vld1q_f32(q_block_line + i + 4);
                    float32x4_t q_f32_2 = vld1q_f32(q_block_line + i + 8);
                    float32x4_t q_f32_3 = vld1q_f32(q_block_line + i + 12);

                    // 2. 【核心修改】将加载的 FP32 Q 动态转换为 FP16
                    float16x8_t q_f16_0 = vcombine_f16(vcvt_f16_f32(q_f32_0), vcvt_f16_f32(q_f32_1));
                    float16x8_t q_f16_1 = vcombine_f16(vcvt_f16_f32(q_f32_2), vcvt_f16_f32(q_f32_3));

                    // 3. 加载16个 FP16 的 K
                    float16x8_t k_f16_0 = vld1q_f16((const __fp16 *)k_block_line + i);
                    float16x8_t k_f16_1 = vld1q_f16((const __fp16 *)k_block_line + i + 8);

                    // 4. 【核心修改】直接对两个FP16向量进行乘加，结果累加到FP32寄存器
                    sum0 = vfmlalq_low_f16(sum0, q_f16_0, k_f16_0);
                    sum0 = vfmlalq_high_f16(sum0, q_f16_0, k_f16_0);
                    sum1 = vfmlalq_low_f16(sum1, q_f16_1, k_f16_1);
                    sum1 = vfmlalq_high_f16(sum1, q_f16_1, k_f16_1);
                }

                sum0 = vaddq_f32(sum0, sum1); // 合并累加器

                // 处理剩余的8个元素
                if (i <= dim_size - 8) {
                    float32x4_t q_f32_0 = vld1q_f32(q_block_line + i);
                    float32x4_t q_f32_1 = vld1q_f32(q_block_line + i + 4);
                    float16x8_t q_f16 = vcombine_f16(vcvt_f16_f32(q_f32_0), vcvt_f16_f32(q_f32_1));
                    float16x8_t k_f16 = vld1q_f16((const __fp16 *)k_block_line + i);

                    sum0 = vfmlalq_low_f16(sum0, q_f16, k_f16);
                    sum0 = vfmlalq_high_f16(sum0, q_f16, k_f16);
                    i += 8;
                }

                // 水平求和，得到最终的点积结果
                acc_dtype_t total = vaddvq_f32(sum0);

                // 用标量方式处理最后不足8个的元素
                for (; i < dim_size; ++i) {
                    total += q_block_line[i] * MLLM_FP16_TO_FP32(k_block_line[i]);
                }
                acc_s[b_r_idx * Bc + b_c_idx] = total;
            }
        }

        // 应用因果掩码的后处理
        if (causal_mask && (global_r_end == (t_c_idx * Bc + Bc) - delta_pos)) {
            for (int i = 0; i < Br; ++i) {
                for (int j = 0; j < Bc; ++j) {
                    if (j > i) { acc_s[i * Bc + j] = NEG_INF; }
                }
            }
        }
    }
    */
    // 【最终正确优化版】
    // 核心思路:
    // 1. 在处理每一行Q (b_r_idx) 时，先将其从 FP32 一次性完整转换为 FP16，并存入一个临时缓冲区 q_f16_buf。
    // 2. 在内层循环 (b_c_idx) 中，重复使用这个转换好的 q_f16_buf。
    // 3. 这样就将转换开销均摊，使得内层循环可以零开销地、最高效地执行 FP16 * FP16 -> FP32 的乘加运算。
    inline void mma0(const dtype_q_in_t *__restrict__ q_block, const dtype_kv_in_t *__restrict__ k_block,
                     acc_dtype_t *__restrict__ acc_s, const int32_t dim_size,
                     const int32_t q_stride_size, const int32_t kv_stride_size,
                     const int32_t t_r_idx, const int32_t t_c_idx,
                     const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
        const int32_t global_r_start = t_r_idx * Br;
        const int32_t global_r_end = global_r_start + Br;
        const int32_t global_c_start = t_c_idx * Bc;
        int delta_pos = seq_size_k - seq_size_q;

        if (causal_mask && (global_c_start - delta_pos > (global_r_end - 1))) { return; }

        __fp16 q_f16_buf[dim_size];


        for (int32_t b_r_idx = 0; b_r_idx < Br; ++b_r_idx) {
            const dtype_q_in_t *q_block_line = q_block + b_r_idx * q_stride_size;

            int k = 0;
            for (; k <= dim_size - 8; k += 8) {
                float32x4_t q_f32_0 = vld1q_f32(q_block_line + k);
                float32x4_t q_f32_1 = vld1q_f32(q_block_line + k + 4);
                float16x8_t q_f16 = vcombine_f16(vcvt_f16_f32(q_f32_0), vcvt_f16_f32(q_f32_1));
                vst1q_f16(q_f16_buf + k, q_f16);
            }
            for (; k < dim_size; ++k) {
                q_f16_buf[k] = (__fp16)q_block_line[k];
            }
            for (int32_t b_c_idx = 0; b_c_idx < Bc; ++b_c_idx) {
                const dtype_kv_in_t *k_block_line = k_block + b_c_idx * kv_stride_size;
                float32x4_t sum0 = vdupq_n_f32(0.0f);
                float32x4_t sum1 = vdupq_n_f32(0.0f);
                int i = 0;
                for (; i <= dim_size - 16; i += 16) {
                    float16x8_t q_f16_0 = vld1q_f16(q_f16_buf + i);
                    float16x8_t q_f16_1 = vld1q_f16(q_f16_buf + i + 8);
                    float16x8_t k_f16_0 = vld1q_f16((const __fp16 *)k_block_line + i);
                    float16x8_t k_f16_1 = vld1q_f16((const __fp16 *)k_block_line + i + 8);
                    sum0 = vfmlalq_low_f16(sum0, q_f16_0, k_f16_0);
                    sum0 = vfmlalq_high_f16(sum0, q_f16_0, k_f16_0);
                    sum1 = vfmlalq_low_f16(sum1, q_f16_1, k_f16_1);
                    sum1 = vfmlalq_high_f16(sum1, q_f16_1, k_f16_1);
                }
                sum0 = vaddq_f32(sum0, sum1);
                if (i <= dim_size - 8) {
                    float16x8_t q_f16 = vld1q_f16(q_f16_buf + i);
                    float16x8_t k_f16 = vld1q_f16((const __fp16 *)k_block_line + i);
                    sum0 = vfmlalq_low_f16(sum0, q_f16, k_f16);
                    sum0 = vfmlalq_high_f16(sum0, q_f16, k_f16);
                    i += 8;
                }
                acc_dtype_t total = vaddvq_f32(sum0);
                for (; i < dim_size; ++i) {
                    total += (float)q_f16_buf[i] * (float)k_block_line[i];
                }
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
    }
    // (与FP32版本相同)
    inline void softmax(acc_dtype_t *__restrict__ acc_s, acc_dtype_t *scoremax, acc_dtype_t *scoremax_prev,
                        acc_dtype_t *score_scale, acc_dtype_t *score_sum, acc_dtype_t *logsum,
                        const float scale, const int32_t t_r_idx, const int32_t t_c_idx,
                        const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
        const int32_t global_r_start = t_r_idx * Br;
        const int32_t global_c_start = t_c_idx * Bc;
        int delta_pos = seq_size_k - seq_size_q;
        if (causal_mask && (global_c_start - delta_pos > (global_r_start + Br - 1))) return;
        memcpy(scoremax_prev, scoremax, Br * sizeof(acc_dtype_t));
        for (int br = 0; br < Br; ++br) {
            float32x4_t max_vec = vdupq_n_f32(scoremax[br]);
            acc_dtype_t *row = acc_s + br * Bc;
            int bc = 0;
            for (; bc <= Bc - 4; bc += 4) { max_vec = vmaxq_f32(max_vec, vld1q_f32(row + bc)); }
            float max_val = _vmaxvq_f32_hmax(max_vec);
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
    }

    // (与FP32版本相同)
    inline void rescale(acc_dtype_t *__restrict__ acc_o, acc_dtype_t *__restrict__ score_scale,
                        const int32_t dim_size, const int32_t t_r_idx, const int32_t t_c_idx,
                        const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
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
    }

    /*
    inline void mma1(const acc_dtype_t *__restrict__ w_block, const dtype_kv_in_t *__restrict__ v_block,
                     acc_dtype_t *__restrict__ acc_o, const int32_t kv_head_size, const int32_t dim_size,
                     const int32_t t_r_idx, const int32_t t_c_idx, const int32_t seq_size_q,
                     const int32_t seq_size_k, bool causal_mask) {
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
                    const dtype_kv_in_t *v_ptr = v_block + b_c_idx * v_stride_size + d_base;
                    acc = vfmaq_f32(acc, w_vec, MLLM_NEON_F32x4_FROM_FP16(v_ptr));
                }
                vst1q_f32(acc_o + b_r_idx * dim_size + d_base, acc);
            }
        }
    }
        */
    inline void mma1(const acc_dtype_t *__restrict__ w_block, const dtype_kv_in_t *__restrict__ v_block,
                     acc_dtype_t *__restrict__ acc_o, const int32_t kv_head_size, const int32_t dim_size,
                     const int32_t t_r_idx, const int32_t t_c_idx, const int32_t seq_size_q,
                     const int32_t seq_size_k, bool causal_mask) {
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
    }
    // (与FP32版本相同)
    inline void scale_and_store(const acc_dtype_t *__restrict__ acc_o, const acc_dtype_t *__restrict__ logsum,
                                dtype_out_t *__restrict__ o_block, const int32_t t_r_idx,
                                const int32_t head_size, const int32_t dim_size) {
        for (int i = 0; i < Br; ++i) {
            dtype_out_t *o_block_line = o_block + i * head_size * dim_size;
            float reciprocal_logsum = 1.0f / logsum[i];
            float32x4_t reciprocal_logsum_vec = vdupq_n_f32(reciprocal_logsum);
            int j = 0;
            for (; j <= dim_size - 4; j += 4) {
                vst1q_f32(o_block_line + j, vmulq_f32(vld1q_f32(acc_o + i * dim_size + j), reciprocal_logsum_vec));
            }
            for (; j < dim_size; ++j) { o_block_line[j] = acc_o[i * dim_size + j] * reciprocal_logsum; }
        }
    }

    // (混合精度修改版)
    inline void mma0_pa_n_fixed(const int32_t Br_n_fixed, const int32_t Bc_n_fixed,
                                const dtype_q_in_t *__restrict__ q_block, const dtype_kv_in_t *__restrict__ k_block,
                                acc_dtype_t *__restrict__ acc_s, const int32_t dim_size,
                                const int32_t q_stride_size, const int32_t kv_stride_size,
                                const int32_t t_r_idx, const int32_t t_c_idx, const int32_t seq_size_q,
                                const int32_t seq_size_k, bool causal_mask) {
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
    }

    // (与FP32版本相同)
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

    // (与FP32版本相同)
    inline void rescale_pa_n_fixed(const int32_t Br_n_fixed, const int32_t Bc_n_fixed,
                                   acc_dtype_t *__restrict__ acc_o, acc_dtype_t *__restrict__ score_scale,
                                   const int32_t dim_size, const int32_t t_r_idx, const int32_t t_c_idx,
                                   const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
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
    }

    // (混合精度修改版)
    inline void mma1_pa_n_fixed(const int32_t Br_n_fixed, const int32_t Bc_n_fixed,
                                const acc_dtype_t *__restrict__ w_block, const dtype_kv_in_t *__restrict__ v_block,
                                acc_dtype_t *__restrict__ acc_o, const int32_t kv_head_size, const int32_t dim_size,
                                const int32_t t_r_idx, const int32_t t_c_idx, const int32_t seq_size_q,
                                const int32_t seq_size_k, bool causal_mask) {
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
    }

    // (与FP32版本相同)
    inline void scale_and_store_pa_n_fixed(const int32_t Br_n_fixed, const acc_dtype_t *__restrict__ acc_o,
                                           const acc_dtype_t *__restrict__ logsum, dtype_out_t *__restrict__ o_block,
                                           const int32_t t_r_idx, const int32_t head_size, const int32_t dim_size) {
        for (int i = 0; i < Br_n_fixed; ++i) {
            dtype_out_t *o_block_line = o_block + i * head_size * dim_size;
            float reciprocal_logsum = 1.0f / logsum[i];
            float32x4_t reciprocal_logsum_vec = vdupq_n_f32(reciprocal_logsum);
            int j = 0;
            for (; j <= dim_size - 4; j += 4) {
                vst1q_f32(o_block_line + j, vmulq_f32(vld1q_f32(acc_o + i * dim_size + j), reciprocal_logsum_vec));
            }
            for (; j < dim_size; ++j) { o_block_line[j] = acc_o[i * dim_size + j] * reciprocal_logsum; }
        }
    }

    // --- Decode 模式函数 (完整版) ---

    // (与FP32版本相同)
    inline void init_temp_d(acc_dtype_t *logsum, acc_dtype_t *scoremax, acc_dtype_t *acc_o, const int32_t dim_size) {
        logsum[0] = 0.0f;
        scoremax[0] = NEG_INF;
        float32x4_t zero_vec = vdupq_n_f32(0.0f);
        for (int i = 0; i < 1 * dim_size; i += 4) { vst1q_f32(acc_o + i, zero_vec); }
    }

    // (混合精度修改版)
    inline void mma0_d(const dtype_q_in_t *__restrict__ q_block, const dtype_kv_in_t *__restrict__ k_block,
                       acc_dtype_t *__restrict__ acc_s, const int32_t dim_size,
                       const int32_t kv_stride_size, const int32_t t_r_idx, const int32_t t_c_idx,
                       const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
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
    }

    // (与FP32版本相同)
    inline void softmax_d(acc_dtype_t *__restrict__ acc_s, acc_dtype_t *scoremax, acc_dtype_t *scoremax_prev,
                          acc_dtype_t *score_scale, acc_dtype_t *score_sum, acc_dtype_t *logsum,
                          const float scale, const int32_t t_r_idx, const int32_t t_c_idx,
                          const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
        scoremax_prev[0] = scoremax[0];
        float max_val = scoremax[0];
        for (int bc = 0; bc < Bc; ++bc) max_val = fmaxf(max_val, acc_s[bc]);
        scoremax[0] = max_val;
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

    // (与FP32版本相同)
    inline void rescale_d(acc_dtype_t *__restrict__ acc_o, acc_dtype_t *__restrict__ score_scale,
                          const int32_t dim_size, const int32_t t_r_idx, const int32_t t_c_idx,
                          const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
        float32x4_t scale_v = vdupq_n_f32(score_scale[0]);
        for (int j = 0; j < dim_size; j += 4) {
            vst1q_f32(acc_o + j, vmulq_f32(vld1q_f32(acc_o + j), scale_v));
        }
    }

    // (混合精度修改版)
    inline void mma1_d(const acc_dtype_t *__restrict__ w_block, const dtype_kv_in_t *__restrict__ v_block,
                       acc_dtype_t *__restrict__ acc_o, const int32_t kv_head_size, const int32_t dim_size,
                       const int32_t t_r_idx, const int32_t t_c_idx, const int32_t seq_size_q,
                       const int32_t seq_size_k, bool causal_mask) {
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
    }

    // (与FP32版本相同)
    inline void scale_and_store_d(const acc_dtype_t *__restrict__ acc_o, const acc_dtype_t *__restrict__ logsum,
                                  dtype_out_t *__restrict__ o_block, const int32_t t_r_idx,
                                  const int32_t head_size, const int32_t dim_size) {
        float reciprocal_logsum = 1.0f / logsum[0];
        float32x4_t reciprocal_logsum_vec = vdupq_n_f32(reciprocal_logsum);
        int j = 0;
        for (; j <= dim_size - 4; j += 4) {
            vst1q_f32(o_block + j, vmulq_f32(vld1q_f32(acc_o + j), reciprocal_logsum_vec));
        }
        for (; j < dim_size; ++j) { o_block[j] = acc_o[j] * reciprocal_logsum; }
    }

    // (混合精度修改版)
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

    // (与FP32版本相同)
    inline void softmax_d_n_fixed(const int32_t Bc_n_fixed, acc_dtype_t *__restrict__ acc_s,
                                  acc_dtype_t *scoremax, acc_dtype_t *scoremax_prev,
                                  acc_dtype_t *score_scale, acc_dtype_t *score_sum,
                                  acc_dtype_t *logsum, const float scale,
                                  const int32_t t_r_idx, const int32_t t_c_idx,
                                  const int32_t seq_size_q, const int32_t seq_size_k, bool causal_mask) {
        scoremax_prev[0] = scoremax[0];
        float max_val = scoremax[0];
        for (int bc = 0; bc < Bc_n_fixed; ++bc) max_val = fmaxf(max_val, acc_s[bc]);
        scoremax[0] = max_val;
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

    // (与FP32版本相同)
    inline void rescale_d_n_fixed(const int32_t Bc_n_fixed, acc_dtype_t *__restrict__ acc_o,
                                  acc_dtype_t *__restrict__ score_scale, const int32_t dim_size,
                                  const int32_t t_r_idx, const int32_t t_c_idx,
                                  const int32_t seq_size_q, const int32_t seq_size_k,
                                  bool causal_mask) {
        float scale = score_scale[0];
        for (int j = 0; j < dim_size; ++j) { acc_o[j] *= scale; }
    }

    // (混合精度修改版)
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

} // namespace mobi_attn

void flash_attention_2_forward(
    const void *Q, const void *K, const void *V, void *O,
    int32_t batch_size, int32_t head_size, int32_t seq_size_q, int32_t seq_size_k, int32_t dim_size,
    bool causal_mask, bool use_fp32, int32_t threads, int32_t br, int32_t bc,
    int32_t q_head, int32_t kv_head, bool high_precision_exp) {
    const size_t acc_o_size = threads * br * dim_size * sizeof(float);
    const size_t acc_s_size = threads * br * bc * sizeof(float);
    const size_t logsum_size = threads * br * sizeof(float);
    const size_t scoremax_size = threads * br * sizeof(float);
    const size_t scoremax_prev_size = threads * br * sizeof(float);
    const size_t score_scale_size = threads * br * sizeof(float);
    const size_t score_sum_size = threads * br * sizeof(float);

    // TODO 改为只分配一次
    void *workspace[7];
    mobi_attn::aligned_alloc(&workspace[0], acc_o_size, 32);
    mobi_attn::aligned_alloc(&workspace[1], acc_s_size, 32);
    mobi_attn::aligned_alloc(&workspace[2], logsum_size, 32);
    mobi_attn::aligned_alloc(&workspace[3], scoremax_size, 32);
    mobi_attn::aligned_alloc(&workspace[4], scoremax_prev_size, 32);
    mobi_attn::aligned_alloc(&workspace[5], score_scale_size, 32);
    mobi_attn::aligned_alloc(&workspace[6], score_sum_size, 32);

    if (use_fp32) {
        // 使用纯FP32 NEON实现
        mobi_attn::FlashAttn2T<mobi_attn::NEON_FA_2_GQA_QKV_FP32_BSHD_O_FP32_BSHD_ACC_FP32_IMPL> op;
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
        // 使用FP16输入，FP32输出的NEON实现
        mobi_attn::FlashAttn2T<mobi_attn::NEON_FA_2_GQA_Q_FP32_KV_FP16_BSHD_O_FP32_BSHD_ACC_FP32_IMPL> op;
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

#endif // AVX2

#endif // MLLM_FA2_CAL_HPP