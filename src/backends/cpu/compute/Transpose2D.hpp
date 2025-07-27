#pragma once
#include <iostream>
#include "Types.hpp"

// 为不同平台引入对应的 SIMD 指令集头文件
#if defined(__AVX__) || defined(__AVX2__)
#include <immintrin.h> // Intel/AMD AVX 指令集
#elif defined(__aarch64__)
#include <arm_neon.h> // ARM NEON 指令集
#endif

/**
 * @brief 使用 SIMD 指令 (AVX) 转置一个 8x8 的浮点数矩阵块。
 * @param src 指向源数据块左上角的指针
 * @param dst 指向目标数据块左上角的指针
 * @param src_stride 源矩阵的行步长 (即列数)
 * @param dst_stride 目标矩阵的行步长 (即转置前的行数)
 */
#if defined(__AVX__) || defined(__AVX2__)
static inline void transpose_block_8x8_avx(const float *src, float *dst, const int src_stride, const int dst_stride) {
    // 1. 从源矩阵加载8行数据到8个AVX寄存器
    __m256 row0 = _mm256_loadu_ps(src + 0 * src_stride);
    __m256 row1 = _mm256_loadu_ps(src + 1 * src_stride);
    __m256 row2 = _mm256_loadu_ps(src + 2 * src_stride);
    __m256 row3 = _mm256_loadu_ps(src + 3 * src_stride);
    __m256 row4 = _mm256_loadu_ps(src + 4 * src_stride);
    __m256 row5 = _mm256_loadu_ps(src + 5 * src_stride);
    __m256 row6 = _mm256_loadu_ps(src + 6 * src_stride);
    __m256 row7 = _mm256_loadu_ps(src + 7 * src_stride);

    // 2. 在寄存器内进行 8x8 矩阵的转置 (这是一个标准的多步shuffle操作)
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

    // 3. 将转置后的8个寄存器（现在是8列）写回到目标矩阵
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

/**
 * @brief 使用 SIMD 指令 (NEON) 转置一个 4x4 的浮点数矩阵块。
 * @param src 指向源数据块左上角的指针
 * @param dst 指向目标数据块左上角的指针
 * @param src_stride 源矩阵的行步长 (即列数)
 * @param dst_stride 目标矩阵的行步长 (即转置前的行数)
 */
#if defined(__aarch64__)
static inline void transpose_block_4x4_neon(const float *src, float *dst, const int src_stride, const int dst_stride) {
    // 1. 从源矩阵加载4行数据，每行4个float
    float32x4_t row0 = vld1q_f32(src + 0 * src_stride);
    float32x4_t row1 = vld1q_f32(src + 1 * src_stride);
    float32x4_t row2 = vld1q_f32(src + 2 * src_stride);
    float32x4_t row3 = vld1q_f32(src + 3 * src_stride);

    // 2. 使用 VTRN 指令对 4x4 矩阵进行转置
    // 第一次 VTRN, 两两交换元素
    float32x4x2_t p01 = vtrnq_f32(row0, row1); // p01.val[0] = {r0[0], r1[0], r0[2], r1[2]}, p01.val[1] = {r0[1], r1[1], r0[3], r1[3]}
    float32x4x2_t p23 = vtrnq_f32(row2, row3); // p23.val[0] = {r2[0], r3[0], r2[2], r3[2]}, p23.val[1] = {r2[1], r3[1], r2[3], r3[3]}

    // 3. 提取并组合成最终的转置结果
    // 从 p01 和 p23 中提取低位的 64bit (2个float) 并组合
    // res0 = {r0[0], r1[0], r2[0], r3[0]}
    float32x4_t res0 = vcombine_f32(vget_low_f32(p01.val[0]), vget_low_f32(p23.val[0]));
    // 从 p01 和 p23 中提取高位的 64bit (2个float) 并组合
    // res1 = {r0[1], r1[1], r2[1], r3[1]}
    float32x4_t res1 = vcombine_f32(vget_low_f32(p01.val[1]), vget_low_f32(p23.val[1]));
    float32x4_t res2 = vcombine_f32(vget_high_f32(p01.val[0]), vget_high_f32(p23.val[0]));
    float32x4_t res3 = vcombine_f32(vget_high_f32(p01.val[1]), vget_high_f32(p23.val[1]));

    // 4. 将转置后的4个寄存器（现在是4列）写回到目标矩阵
    vst1q_f32(dst + 0 * dst_stride, res0);
    vst1q_f32(dst + 1 * dst_stride, res1);
    vst1q_f32(dst + 2 * dst_stride, res2);
    vst1q_f32(dst + 3 * dst_stride, res3);
}
#endif

/**
 * @brief 对一个二维浮点数矩阵进行高效转置 (dst = src^T)。
 * 自动检测平台并使用 AVX 或 NEON 指令进行加速。
 * 如果平台不支持，则回退到缓存优化的C++实现。
 * @param src 指向源矩阵 (N x M) 的指针
 * @param dst 指向目标矩阵 (M x N) 的指针
 * @param N 源矩阵的行数
 * @param M 源矩阵的列数
 */
inline void transpose_matrix_efficient(const float *src, float *dst, const int N, const int M) {
#if defined(__AVX__) || defined(__AVX2__)
    const int BLOCK_DIM = 8;
    // 使用8x8分块处理大部分矩阵
    for (int i = 0; i < N / BLOCK_DIM * BLOCK_DIM; i += BLOCK_DIM) {
        for (int j = 0; j < M / BLOCK_DIM * BLOCK_DIM; j += BLOCK_DIM) {
            transpose_block_8x8_avx(src + i * M + j, dst + j * N + i, M, N);
        }
    }
    // 处理右侧和下方的剩余部分
    for (int i = 0; i < N; ++i) {
        for (int j = M / BLOCK_DIM * BLOCK_DIM; j < M; ++j) {
            dst[j * N + i] = src[i * M + j];
        }
    }
    for (int i = N / BLOCK_DIM * BLOCK_DIM; i < N; ++i) {
        for (int j = 0; j < M / BLOCK_DIM * BLOCK_DIM; ++j) {
            dst[j * N + i] = src[i * M + j];
        }
    }

#elif defined(__aarch64__)
    const int BLOCK_DIM = 4;
    // 使用4x4分块处理大部分矩阵
    for (int i = 0; i < N / BLOCK_DIM * BLOCK_DIM; i += BLOCK_DIM) {
        for (int j = 0; j < M / BLOCK_DIM * BLOCK_DIM; j += BLOCK_DIM) {
            transpose_block_4x4_neon(src + i * M + j, dst + j * N + i, M, N);
        }
    }
    // 处理剩余部分
    for (int i = 0; i < N; ++i) {
        for (int j = M / BLOCK_DIM * BLOCK_DIM; j < M; ++j) {
            dst[j * N + i] = src[i * M + j];
        }
    }
    for (int i = N / BLOCK_DIM * BLOCK_DIM; i < N; ++i) {
        for (int j = 0; j < M / BLOCK_DIM * BLOCK_DIM; ++j) {
            dst[j * N + i] = src[i * M + j];
        }
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

// --- BEGIN: High-Performance FP16 Transpose Function for ARM NEON ---

#if defined(__aarch64__)

/**
 * @brief 使用ARM NEON指令高效转置一个 8x8 的 __fp16 矩阵块。
 * 【已修正】使用了正确的 VTRN/VZIP 指令序列，避免了 vtrnq_f64 错误。
 * @param src 指向源数据块左上角的指针 (__fp16)
 * @param dst 指向目标数据块左上角的指针 (__fp16)
 * @param src_stride 源矩阵的行步长 (即列数)
 * @param dst_stride 目标矩阵的行步长 (即转置前的行数)
 */
static inline void transpose_block_8x8_neon_fp16(const __fp16 *src, __fp16 *dst, const int src_stride, const int dst_stride) {
    // 1. Load 8 rows from source matrix into 8 NEON registers
    float16x8_t r0 = vld1q_f16(src + 0 * src_stride);
    float16x8_t r1 = vld1q_f16(src + 1 * src_stride);
    float16x8_t r2 = vld1q_f16(src + 2 * src_stride);
    float16x8_t r3 = vld1q_f16(src + 3 * src_stride);
    float16x8_t r4 = vld1q_f16(src + 4 * src_stride);
    float16x8_t r5 = vld1q_f16(src + 5 * src_stride);
    float16x8_t r6 = vld1q_f16(src + 6 * src_stride);
    float16x8_t r7 = vld1q_f16(src + 7 * src_stride);

    // 2. Perform in-register transpose using VTRN and VZIP
    // Stage 1: Transpose 2x2 blocks of __fp16 elements
    float16x8x2_t t01 = vtrnq_f16(r0, r1);
    float16x8x2_t t23 = vtrnq_f16(r2, r3);
    float16x8x2_t t45 = vtrnq_f16(r4, r5);
    float16x8x2_t t67 = vtrnq_f16(r6, r7);

    // Stage 2: Transpose 4x4 blocks by zipping 32-bit (2x fp16) chunks
    float32x4x2_t z02 = vzipq_f32(vreinterpretq_f32_f16(t01.val[0]), vreinterpretq_f32_f16(t23.val[0]));
    float32x4x2_t z13 = vzipq_f32(vreinterpretq_f32_f16(t01.val[1]), vreinterpretq_f32_f16(t23.val[1]));
    float32x4x2_t z46 = vzipq_f32(vreinterpretq_f32_f16(t45.val[0]), vreinterpretq_f32_f16(t67.val[0]));
    float32x4x2_t z57 = vzipq_f32(vreinterpretq_f32_f16(t45.val[1]), vreinterpretq_f32_f16(t67.val[1]));

    // 3. Store the transposed 8x8 block to the destination matrix
    vst1q_f16(dst + 0 * dst_stride, vreinterpretq_f16_f32(z02.val[0]));
    vst1q_f16(dst + 1 * dst_stride, vreinterpretq_f16_f32(z13.val[0]));
    vst1q_f16(dst + 2 * dst_stride, vreinterpretq_f16_f32(z02.val[1]));
    vst1q_f16(dst + 3 * dst_stride, vreinterpretq_f16_f32(z13.val[1]));
    vst1q_f16(dst + 4 * dst_stride, vreinterpretq_f16_f32(z46.val[0]));
    vst1q_f16(dst + 5 * dst_stride, vreinterpretq_f16_f32(z57.val[0]));
    vst1q_f16(dst + 6 * dst_stride, vreinterpretq_f16_f32(z46.val[1]));
    vst1q_f16(dst + 7 * dst_stride, vreinterpretq_f16_f32(z57.val[1]));
}
/**
 * @brief 对一个 mllm_fp16_t 矩阵进行高效转置 (dst = src^T)。
 * 在 aarch64 平台上，此函数使用 NEON SIMD 指令进行极致加速。
 * 在其他平台，回退到缓存优化的C++实现。
 * @param src 指向源矩阵 (N x M) 的指针
 * @param dst 指向目标矩阵 (M x N) 的指针
 * @param N 源矩阵的行数
 * @param M 源矩阵的列数
 */
inline void transpose_matrix_efficient_fp16(const mllm_fp16_t *src, mllm_fp16_t *dst, const int N, const int M) {
    const int BLOCK_DIM = 8;
    // Use an 8x8 block loop to process the majority of the matrix
    for (int i = 0; i < N / BLOCK_DIM * BLOCK_DIM; i += BLOCK_DIM) {
        for (int j = 0; j < M / BLOCK_DIM * BLOCK_DIM; j += BLOCK_DIM) {
            // On ARM, mllm_fp16_t is __fp16, so we can call the NEON helper directly
            transpose_block_8x8_neon_fp16(
                (const __fp16 *)(src + i * M + j),
                (__fp16 *)(dst + j * N + i),
                M, N);
        }
    }

    // Process the remaining rows and columns on the edges using standard C++
    for (int i = 0; i < N; ++i) {
        for (int j = M / BLOCK_DIM * BLOCK_DIM; j < M; ++j) {
            dst[j * N + i] = src[i * M + j];
        }
    }
    for (int i = N / BLOCK_DIM * BLOCK_DIM; i < N; ++i) {
        for (int j = 0; j < M / BLOCK_DIM * BLOCK_DIM; ++j) {
            dst[j * N + i] = src[i * M + j];
        }
    }
}

#endif // __aarch64__

// --- END: High-Performance FP16 Transpose Function for ARM NEON ---