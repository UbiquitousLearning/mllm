#pragma once

#include <vector>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <cstring> // 用于 memcpy
#include "DataType.hpp"
// 为不同平台引入对应的 SIMD 指令集头文件
#if defined(__AVX__) || defined(__AVX2__)
#include <immintrin.h> // Intel/AMD AVX & AVX2 指令集
#elif defined(__aarch64__)
#include <arm_neon.h> // ARM NEON 指令集
#endif

// 引入 OpenMP 头文件以支持多线程并行
#include <omp.h>

#if defined(__AVX__) || defined(__AVX2__)
static inline void transpose_block_8x8_avx(const float *src, float *dst, const int src_stride, const int dst_stride) {
    __m256 row0 = _mm256_loadu_ps(src + 0 * src_stride);
    __m256 row1 = _mm256_loadu_ps(src + 1 * src_stride);
    __m256 row2 = _mm256_loadu_ps(src + 2 * src_stride);
    __m256 row3 = _mm256_loadu_ps(src + 3 * src_stride);
    __m256 row4 = _mm256_loadu_ps(src + 4 * src_stride);
    __m256 row5 = _mm256_loadu_ps(src + 5 * src_stride);
    __m256 row6 = _mm256_loadu_ps(src + 6 * src_stride);
    __m256 row7 = _mm256_loadu_ps(src + 7 * src_stride);
    __m256 t0, t1, t2, t3, t4, t5, t6, t7;
    t0 = _mm256_unpacklo_ps(row0, row1);
    t1 = _mm256_unpackhi_ps(row0, row1);
    t2 = _mm256_unpacklo_ps(row2, row3);
    t3 = _mm256_unpackhi_ps(row2, row3);
    t4 = _mm256_unpacklo_ps(row4, row5);
    t5 = _mm256_unpackhi_ps(row4, row5);
    t6 = _mm256_unpacklo_ps(row6, row7);
    t7 = _mm256_unpackhi_ps(row6, row7);
    __m256 tt0, tt1, tt2, tt3, tt4, tt5, tt6, tt7;
    tt0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
    tt1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
    tt2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
    tt3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
    tt4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
    tt5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
    tt6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
    tt7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));
    row0 = _mm256_permute2f128_ps(tt0, tt4, 0x20);
    row1 = _mm256_permute2f128_ps(tt1, tt5, 0x20);
    row2 = _mm256_permute2f128_ps(tt2, tt6, 0x20);
    row3 = _mm256_permute2f128_ps(tt3, tt7, 0x20);
    row4 = _mm256_permute2f128_ps(tt0, tt4, 0x31);
    row5 = _mm256_permute2f128_ps(tt1, tt5, 0x31);
    row6 = _mm256_permute2f128_ps(tt2, tt6, 0x31);
    row7 = _mm256_permute2f128_ps(tt3, tt7, 0x31);
    _mm256_storeu_ps(dst + 0 * dst_stride, row0);
    _mm256_storeu_ps(dst + 1 * dst_stride, row1);
    _mm256_storeu_ps(dst + 2 * dst_stride, row2);
    _mm256_storeu_ps(dst + 3 * dst_stride, row3);
    _mm256_storeu_ps(dst + 4 * dst_stride, row4);
    _mm256_storeu_ps(dst + 5 * dst_stride, row5);
    _mm256_storeu_ps(dst + 6 * dst_stride, row6);
    _mm256_storeu_ps(dst + 7 * dst_stride, row7);
}
#endif
#if defined(__aarch64__)
static inline void transpose_block_4x4_neon(const float *src, float *dst, const int src_stride, const int dst_stride) {
    float32x4_t row0 = vld1q_f32(src + 0 * src_stride);
    float32x4_t row1 = vld1q_f32(src + 1 * src_stride);
    float32x4_t row2 = vld1q_f32(src + 2 * src_stride);
    float32x4_t row3 = vld1q_f32(src + 3 * src_stride);
    float32x4x2_t p01 = vtrnq_f32(row0, row1);
    float32x4x2_t p23 = vtrnq_f32(row2, row3);
    float32x4_t res0 = vcombine_f32(vget_low_f32(p01.val[0]), vget_low_f32(p23.val[0]));
    float32x4_t res1 = vcombine_f32(vget_low_f32(p01.val[1]), vget_low_f32(p23.val[1]));
    float32x4_t res2 = vcombine_f32(vget_high_f32(p01.val[0]), vget_high_f32(p23.val[0]));
    float32x4_t res3 = vcombine_f32(vget_high_f32(p01.val[1]), vget_high_f32(p23.val[1]));
    vst1q_f32(dst + 0 * dst_stride, res0);
    vst1q_f32(dst + 1 * dst_stride, res1);
    vst1q_f32(dst + 2 * dst_stride, res2);
    vst1q_f32(dst + 3 * dst_stride, res3);
}
#endif
static inline void transpose_matrix_2d_efficient(const float *src, float *dst, const int N, const int M) {
#if defined(__AVX__) || defined(__AVX2__)
    const int BLOCK_DIM = 8;
    for (int i = 0; i < N - (N % BLOCK_DIM); i += BLOCK_DIM) {
        for (int j = 0; j < M - (M % BLOCK_DIM); j += BLOCK_DIM) {
            transpose_block_8x8_avx(src + i * M + j, dst + j * N + i, M, N);
        }
    }
    for (int i = 0; i < N; ++i) {
        for (int j = M - (M % BLOCK_DIM); j < M; ++j) { dst[j * N + i] = src[i * M + j]; }
    }
    for (int i = N - (N % BLOCK_DIM); i < N; ++i) {
        for (int j = 0; j < M - (M % BLOCK_DIM); ++j) { dst[j * N + i] = src[i * M + j]; }
    }
#elif defined(__aarch64__)
    const int BLOCK_DIM = 4;
    for (int i = 0; i < N - (N % BLOCK_DIM); i += BLOCK_DIM) {
        for (int j = 0; j < M - (M % BLOCK_DIM); j += BLOCK_DIM) {
            transpose_block_4x4_neon(src + i * M + j, dst + j * N + i, M, N);
        }
    }
    for (int i = 0; i < N; ++i) {
        for (int j = M - (M % BLOCK_DIM); j < M; ++j) { dst[j * N + i] = src[i * M + j]; }
    }
    for (int i = N - (N % BLOCK_DIM); i < N; ++i) {
        for (int j = 0; j < M - (M % BLOCK_DIM); ++j) { dst[j * N + i] = src[i * M + j]; }
    }
#else
    const int BLOCK_DIM = 16;
    for (int i = 0; i < N; i += BLOCK_DIM) {
        for (int j = 0; j < M; j += BLOCK_DIM) {
            for (int bi = i; bi < i + BLOCK_DIM && bi < N; ++bi) {
                for (int bj = j; bj < j + BLOCK_DIM && bj < M; ++bj) {
                    dst[bj * N + bi] = src[bi * M + bj];
                }
            }
        }
    }
#endif
}

/**
 * @brief 对一个三维浮点数张量进行高效转置
 * 该函数根据指定的维度置换 (permutation) 来重新排列数据。
 *
 * @param src 指向源张量数据的指针。数据布局为 (D1, D2, D3) 的稠密行主序。
 * @param dst 指向目标张量数据的指针。其维度根据 perm 计算得出。
 * @param d1 源张量的第 1 维大小
 * @param d2 源张量的第 2 维大小
 * @param d3 源张量的第 3 维大小
 * @param perm 一个包含 {0, 1, 2} 的置换向量，定义了转置方式。
 */
void transpose3d_efficient(const float *src, float *dst, int d1, int d2, int d3, const std::vector<int> &perm) {
    // --- 1. 输入验证 ---
    if (perm.size() != 3) {
        throw std::invalid_argument("Permutation vector must contain 3 elements.");
    }
    std::vector<int> sorted_perm = perm;
    std::sort(sorted_perm.begin(), sorted_perm.end());
    if (sorted_perm[0] != 0 || sorted_perm[1] != 1 || sorted_perm[2] != 2) {
        throw std::invalid_argument("Permutation vector must be a permutation of {0, 1, 2}.");
    }

    const int src_dims[3] = {d1, d2, d3};

    // --- 2. 处理特殊情况：无需转置 ---
    if (perm[0] == 0 && perm[1] == 1 && perm[2] == 2) {
        const size_t total_elements = static_cast<size_t>(d1) * d2 * d3;
        if (src != dst) {
            memcpy(dst, src, total_elements * sizeof(float));
        }
        return;
    }

    // --- 3. 性能最优路径：只交换最后两个维度 (e.g., NHW -> NWH) ---
    if (perm[0] == 0 && perm[1] == 2 && perm[2] == 1) {
        const int N = d2;
        const int M = d3;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < d1; ++i) {
            const float *src_slice = src + i * (N * M);
            float *dst_slice = dst + i * (M * N);
            transpose_matrix_2d_efficient(src_slice, dst_slice, N, M);
        }
        return;
    }

    // --- 4. 通用路径：处理所有其他维度置换 ---
    const int dst_dims[3] = {src_dims[perm[0]], src_dims[perm[1]], src_dims[perm[2]]};
    const int BLOCK_DIM = 16;

    long src_strides[3] = {(long)d2 * d3, d3, 1};
    long dst_strides[3] = {(long)dst_dims[1] * dst_dims[2], dst_dims[2], 1};

    int p_inv[3];
    p_inv[perm[0]] = 0;
    p_inv[perm[1]] = 1;
    p_inv[perm[2]] = 2;

#pragma omp parallel for schedule(static)
    for (int i0 = 0; i0 < dst_dims[0]; i0 += BLOCK_DIM) {
        for (int j0 = 0; j0 < dst_dims[1]; j0 += BLOCK_DIM) {
            for (int k0 = 0; k0 < dst_dims[2]; k0 += BLOCK_DIM) {
                for (int i = i0; i < i0 + BLOCK_DIM && i < dst_dims[0]; ++i) {
                    for (int j = j0; j < j0 + BLOCK_DIM && j < dst_dims[1]; ++j) {
                        for (int k = k0; k < k0 + BLOCK_DIM && k < dst_dims[2]; ++k) {
                            long dst_idx = (long)i * dst_strides[0] + (long)j * dst_strides[1] + k;
                            int dst_coords[3] = {i, j, k};
                            int src_coords[3];
                            src_coords[p_inv[0]] = dst_coords[0];
                            src_coords[p_inv[1]] = dst_coords[1];
                            src_coords[p_inv[2]] = dst_coords[2];
                            long src_idx = (long)src_coords[0] * src_strides[0] + (long)src_coords[1] * src_strides[1] + src_coords[2];
                            dst[dst_idx] = src[src_idx];
                        }
                    }
                }
            }
        }
    }
}

#if defined(__aarch64__)
// NEON 平台使用 8x8 的 __fp16 块转置
static inline void transpose_block_8x8_neon_fp16(const mllm_fp16_t *src, mllm_fp16_t *dst, const int src_stride, const int dst_stride) {
    // 在 aarch64 上, mllm_fp16_t 就是 __fp16
    float16x8_t r0 = vld1q_f16(src + 0 * src_stride);
    float16x8_t r1 = vld1q_f16(src + 1 * src_stride);
    float16x8_t r2 = vld1q_f16(src + 2 * src_stride);
    float16x8_t r3 = vld1q_f16(src + 3 * src_stride);
    float16x8_t r4 = vld1q_f16(src + 4 * src_stride);
    float16x8_t r5 = vld1q_f16(src + 5 * src_stride);
    float16x8_t r6 = vld1q_f16(src + 6 * src_stride);
    float16x8_t r7 = vld1q_f16(src + 7 * src_stride);
    float16x8x2_t t01 = vtrnq_f16(r0, r1);
    float16x8x2_t t23 = vtrnq_f16(r2, r3);
    float16x8x2_t t45 = vtrnq_f16(r4, r5);
    float16x8x2_t t67 = vtrnq_f16(r6, r7);
    float32x4x2_t z02 = vzipq_f32(vreinterpretq_f32_f16(t01.val[0]), vreinterpretq_f32_f16(t23.val[0]));
    float32x4x2_t z13 = vzipq_f32(vreinterpretq_f32_f16(t01.val[1]), vreinterpretq_f32_f16(t23.val[1]));
    float32x4x2_t z46 = vzipq_f32(vreinterpretq_f32_f16(t45.val[0]), vreinterpretq_f32_f16(t67.val[0]));
    float32x4x2_t z57 = vzipq_f32(vreinterpretq_f32_f16(t45.val[1]), vreinterpretq_f32_f16(t67.val[1]));
    vst1q_f16(dst + 0 * dst_stride, vreinterpretq_f16_f32(z02.val[0]));
    vst1q_f16(dst + 1 * dst_stride, vreinterpretq_f16_f32(z13.val[0]));
    vst1q_f16(dst + 2 * dst_stride, vreinterpretq_f16_f32(z02.val[1]));
    vst1q_f16(dst + 3 * dst_stride, vreinterpretq_f16_f32(z13.val[1]));
    vst1q_f16(dst + 4 * dst_stride, vreinterpretq_f16_f32(z46.val[0]));
    vst1q_f16(dst + 5 * dst_stride, vreinterpretq_f16_f32(z57.val[0]));
    vst1q_f16(dst + 6 * dst_stride, vreinterpretq_f16_f32(z46.val[1]));
    vst1q_f16(dst + 7 * dst_stride, vreinterpretq_f16_f32(z57.val[1]));
}
#endif

// 高效的2D FP16矩阵转置
static inline void transpose_matrix_2d_efficient_fp16(const mllm_fp16_t *src, mllm_fp16_t *dst, const int N, const int M) {
#if defined(__aarch64__)
    const int BLOCK_DIM = 8;
    for (int i = 0; i < N - (N % BLOCK_DIM); i += BLOCK_DIM) {
        for (int j = 0; j < M - (M % BLOCK_DIM); j += BLOCK_DIM) {
            transpose_block_8x8_neon_fp16(src + i * M + j, dst + j * N + i, M, N);
        }
    }
    // 处理边缘情况
    for (int i = 0; i < N; ++i) {
        for (int j = M - (M % BLOCK_DIM); j < M; ++j) { dst[j * N + i] = src[i * M + j]; }
    }
    for (int i = N - (N % BLOCK_DIM); i < N; ++i) {
        for (int j = 0; j < M - (M % BLOCK_DIM); ++j) { dst[j * N + i] = src[i * M + j]; }
    }
#else
    // 在非NEON平台 (如AVX)，使用通用的缓存分块方法
    const int BLOCK_DIM = 16;
    for (int i = 0; i < N; i += BLOCK_DIM) {
        for (int j = 0; j < M; j += BLOCK_DIM) {
            for (int bi = i; bi < i + BLOCK_DIM && bi < N; ++bi) {
                for (int bj = j; bj < j + BLOCK_DIM && bj < M; ++bj) {
                    dst[bj * N + bi] = src[bi * M + bj];
                }
            }
        }
    }
#endif
}

/**
 * @brief 对一个三维 mllm_fp16_t 张量进行高效转置。
 * @param src 指向源张量数据的指针。
 * @param dst 指向目标张量数据的指针。
 * @param d1, d2, d3 源张量的维度。
 * @param perm 维度置换向量, e.g., {0, 2, 1}。
 */
void transpose3d_efficient_fp16(const mllm_fp16_t *src, mllm_fp16_t *dst, int d1, int d2, int d3, const std::vector<int> &perm) {
    // --- 1. 输入验证 ---
    if (perm.size() != 3) { throw std::invalid_argument("Permutation vector must contain 3 elements."); }
    std::vector<int> sorted_perm = perm;
    std::sort(sorted_perm.begin(), sorted_perm.end());
    if (sorted_perm[0] != 0 || sorted_perm[1] != 1 || sorted_perm[2] != 2) {
        throw std::invalid_argument("Permutation vector must be a permutation of {0, 1, 2}.");
    }

    const int src_dims[3] = {d1, d2, d3};

    // --- 2. 处理特殊情况：无需转置 ---
    if (perm[0] == 0 && perm[1] == 1 && perm[2] == 2) {
        const size_t total_elements = static_cast<size_t>(d1) * d2 * d3;
        if (src != dst) {
            memcpy(dst, src, total_elements * sizeof(mllm_fp16_t));
        }
        return;
    }

    // --- 3. 性能最优路径：只交换最后两个维度 (e.g., HSD -> HDS) ---
    if (perm[0] == 0 && perm[1] == 2 && perm[2] == 1) {
        const int N = d2;
        const int M = d3;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < d1; ++i) {
            const mllm_fp16_t *src_slice = src + i * (N * M);
            mllm_fp16_t *dst_slice = dst + i * (M * N);
            transpose_matrix_2d_efficient_fp16(src_slice, dst_slice, N, M);
        }
        return;
    }

    // --- 4. 通用路径：处理所有其他维度置换 ---
    const int dst_dims[3] = {src_dims[perm[0]], src_dims[perm[1]], src_dims[perm[2]]};
    const int BLOCK_DIM = 16;
    long src_strides[3] = {(long)d2 * d3, d3, 1};
    long dst_strides[3] = {(long)dst_dims[1] * dst_dims[2], dst_dims[2], 1};
    int p_inv[3];
    p_inv[perm[0]] = 0;
    p_inv[perm[1]] = 1;
    p_inv[perm[2]] = 2;

#pragma omp parallel for schedule(static)
    for (int i0 = 0; i0 < dst_dims[0]; i0 += BLOCK_DIM) {
        for (int j0 = 0; j0 < dst_dims[1]; j0 += BLOCK_DIM) {
            for (int k0 = 0; k0 < dst_dims[2]; k0 += BLOCK_DIM) {
                for (int i = i0; i < i0 + BLOCK_DIM && i < dst_dims[0]; ++i) {
                    for (int j = j0; j < j0 + BLOCK_DIM && j < dst_dims[1]; ++j) {
                        for (int k = k0; k < k0 + BLOCK_DIM && k < dst_dims[2]; ++k) {
                            long dst_idx = (long)i * dst_strides[0] + (long)j * dst_strides[1] + k;
                            int dst_coords[3] = {i, j, k};
                            int src_coords[3];
                            src_coords[p_inv[0]] = dst_coords[0];
                            src_coords[p_inv[1]] = dst_coords[1];
                            src_coords[p_inv[2]] = dst_coords[2];
                            long src_idx = (long)src_coords[0] * src_strides[0] + (long)src_coords[1] * src_strides[1] + src_coords[2];
                            dst[dst_idx] = src[src_idx];
                        }
                    }
                }
            }
        }
    }
}