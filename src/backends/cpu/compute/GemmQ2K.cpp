#include "GemmQ2K.hpp"

#include "backends/cpu/third_party/ggml/QuantizeQ8.hpp"

#include <vector>
#include <cassert>
#include <cstring>
#include <omp.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#define M_BLOCK_SIZE 8
#define N_BLOCK_SIZE 16

/**
 * @brief 为 GEMM 操作对 float 矩阵 B(KxN) 进行 Q8_K 量化和打包。
 *
 * 原始的打包方式导致在 GEMM 内核中访问数据时是跨步的 (strided access)，
 * 这严重破坏了缓存局部性。新的打包方式将微内核所需的数据块连续存储。
 *
 * 原始布局 (逻辑上): [N 块][列偏移][K 块]
 * 访问模式: `base + n_idx * K_blocks` -> 步长巨大!
 *
 * 优化后布局 (逻辑上): [N 块][K 块][列偏移]
 * 访问模式: `base + n_idx` -> 完美连续访问!
 *
 * @param B_packed      输出，打包好的 Q8_K 矩阵。
 * @param B_float       输入，行主序的 float 矩阵 (KxN)。
 * @param K             矩阵 B 的行数。
 * @param N             矩阵 B 的列数。
 */
void quantize_and_pack_q8_k_for_gemm(
    block_q8_K *B_packed,
    const float *B_float,
    int K,
    int N) {
    assert(K % QK_K == 0);
    assert(N % N_BLOCK_SIZE == 0);
    const int K_blocks = K / QK_K;
    const int N_chunks = N / N_BLOCK_SIZE;

#pragma omp parallel for num_threads(4)
    for (int j_chunk = 0; j_chunk < N_chunks; ++j_chunk) {
        std::vector<float> temp_col(QK_K);
        for (int k_block = 0; k_block < K_blocks; ++k_block) {
            for (int col_offset = 0; col_offset < N_BLOCK_SIZE; ++col_offset) {
                const int j = j_chunk * N_BLOCK_SIZE;
                for (int k_inner = 0; k_inner < QK_K; ++k_inner) {
                    const int row_idx = (k_block * QK_K) + k_inner;
                    const int col_idx = j + col_offset;
                    temp_col[k_inner] = B_float[(row_idx * N) + col_idx];
                }
                // 核心改动：调整写入位置，确保 N_BLOCK_SIZE 这个维度的数据是连续的
                block_q8_K *dest_block = B_packed + (j_chunk * (K_blocks * N_BLOCK_SIZE)) + (k_block * N_BLOCK_SIZE) + col_offset;
                quantize_row_q8_K(temp_col.data(), dest_block, QK_K);
            }
        }
    }
}

#if defined(__ARM_NEON)
/**
 * @brief 8x16 NEON 微内核
 *
 * 1.  一次性处理一个 8x16 的输出块，共 128 个 float 累加器。
 * 2.  使用 32 个 NEON 向量寄存器中的大部分来保存这些累加器。
 * 3.  对 A 矩阵的数据（scales, quants）在 K 维度内循环前进行预加载和预处理，
 * 在 B 矩阵的数据（quants, bsums）在 N 维度上进行向量化加载。
 * 4.  通过精心设计的指令序列，最大化浮点乘加 (FMLA) 指令的吞吐量。
 * 5.  循环展开和指令重排以减少依赖和流水线停顿。
 */
static inline void micro_kernel_8x16_neon(
    float acc[M_BLOCK_SIZE][N_BLOCK_SIZE],
    const block_q2_K *a_blocks[M_BLOCK_SIZE],
    const block_q8_K *b_block_base) {
    // 8x16 = 128 个累加器，用 32 个 float32x4_t 向量寄存器表示
    float32x4_t acc_vecs[8][4];
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 4; ++j) {
            acc_vecs[i][j] = vld1q_f32(&acc[i][j * 4]);
        }
    }

    // 预加载 B 矩阵的 d
    float32x4_t d_b_vecs[4];
    float d_b_vals[16];
    for (int n = 0; n < 16; ++n) d_b_vals[n] = b_block_base[n].d;
    for (int j = 0; j < 4; ++j) d_b_vecs[j] = vld1q_f32(&d_b_vals[j * 4]);

    // 预加载 A 矩阵的 d 和 dmin
    float d_a_vals[8], dmin_a_vals[8];
    for (int m = 0; m < 8; ++m) {
        d_a_vals[m] = MLLM_FP16_TO_FP32(a_blocks[m]->d);
        dmin_a_vals[m] = MLLM_FP16_TO_FP32(a_blocks[m]->dmin);
    }

    // 内部循环处理一个 QK_K 大小的块
    for (int sub_block_idx = 0; sub_block_idx < QK_K / 16; ++sub_block_idx) {
        // --- 1. 处理 dmin * bsums 部分 ---
        int16_t bsums_vals[16];
        for (int n = 0; n < 16; ++n) bsums_vals[n] = b_block_base[n].bsums[sub_block_idx];

        const int16x8_t bsums_vec_lo = vld1q_s16(bsums_vals);
        const int16x8_t bsums_vec_hi = vld1q_s16(bsums_vals + 8);

        for (int m = 0; m < 8; ++m) {
            const int16_t m_val = (int16_t)(a_blocks[m]->scales[sub_block_idx] >> 4);

            float32x4_t summs_f[4];
            summs_f[0] = vcvtq_f32_s32(vmull_n_s16(vget_low_s16(bsums_vec_lo), m_val));
            summs_f[1] = vcvtq_f32_s32(vmull_n_s16(vget_high_s16(bsums_vec_lo), m_val));
            summs_f[2] = vcvtq_f32_s32(vmull_n_s16(vget_low_s16(bsums_vec_hi), m_val));
            summs_f[3] = vcvtq_f32_s32(vmull_n_s16(vget_high_s16(bsums_vec_hi), m_val));

            for (int j = 0; j < 4; ++j) {
                float32x4_t dmin_a_d_b = vmulq_n_f32(d_b_vecs[j], -dmin_a_vals[m]);
                acc_vecs[m][j] = vmlaq_f32(acc_vecs[m][j], dmin_a_d_b, summs_f[j]);
            }
        }

        // --- 2. 处理主点积部分 ---
        // 预加载 B 矩阵的 quants
        int8x16_t q8_vecs[16];
        for (int n = 0; n < 16; ++n) {
            q8_vecs[n] = vld1q_s8(b_block_base[n].qs + sub_block_idx * 16);
        }

        // 对 A 矩阵的每一行
        for (int m = 0; m < 8; ++m) {
            const int s_val = a_blocks[m]->scales[sub_block_idx] & 0x0F;
            if (s_val == 0) continue;

            uint8_t l_bytes[16];
            for (int k = 0; k < 16; ++k) {
                int k_inner = sub_block_idx * 16 + k;
                const int ib = k_inner / 128, iib = k_inner % 128, iic = iib % 32, cic = iib / 32;
                l_bytes[k] = (a_blocks[m]->qs[ib * 32 + iic] >> (cic * 2)) & 3;
            }
            const int8x16_t l_vec = vreinterpretq_s8_u8(vld1q_u8(l_bytes));

            // 计算一行 A 和 16 列 B 的点积
            int32_t s_dot_vals[16];
            for (int n = 0; n < 16; ++n) {
                const int16x8_t p_lo = vmull_s8(vget_low_s8(l_vec), vget_low_s8(q8_vecs[n]));
                const int16x8_t p_hi = vmull_s8(vget_high_s8(l_vec), vget_high_s8(q8_vecs[n]));
                s_dot_vals[n] = vaddvq_s32(vpaddlq_s16(vaddq_s16(p_lo, p_hi)));
            }

            // 向量化乘法和累加
            const float32x4_t scale_factor = vdupq_n_f32((float)s_val * d_a_vals[m]);
            for (int j = 0; j < 4; ++j) {
                float32x4_t isum_f32 = vcvtq_f32_s32(vld1q_s32(&s_dot_vals[j * 4]));
                float32x4_t term = vmulq_f32(scale_factor, d_b_vecs[j]);
                acc_vecs[m][j] = vmlaq_f32(acc_vecs[m][j], term, isum_f32);
            }
        }
    }

    // 写回累加器
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 4; ++j) {
            vst1q_f32(&acc[i][j * 4], acc_vecs[i][j]);
        }
    }
}
#else
// C++ and AVX2 Fallback
static inline void micro_kernel_8x16_reference(
    float acc[M_BLOCK_SIZE][N_BLOCK_SIZE],
    const block_q2_K *a_blocks[M_BLOCK_SIZE],
    const block_q8_K *b_block_base) {
    float d_all[M_BLOCK_SIZE][N_BLOCK_SIZE], d_min[M_BLOCK_SIZE][N_BLOCK_SIZE];
    for (int m = 0; m < M_BLOCK_SIZE; ++m) {
        for (int n = 0; n < N_BLOCK_SIZE; ++n) {
            d_all[m][n] = MLLM_FP16_TO_FP32(a_blocks[m]->d) * b_block_base[n].d;
            d_min[m][n] = MLLM_FP16_TO_FP32(a_blocks[m]->dmin) * b_block_base[n].d;
        }
    }

    for (int sub_block_idx = 0; sub_block_idx < QK_K / 16; ++sub_block_idx) {
        int32_t isum_tile[M_BLOCK_SIZE][N_BLOCK_SIZE] = {{0}};

        for (int k_in_subblock = 0; k_in_subblock < 16; ++k_in_subblock) {
            const int k_inner = sub_block_idx * 16 + k_in_subblock;

            int s_vals[M_BLOCK_SIZE], l_vals[M_BLOCK_SIZE];
            for (int m = 0; m < M_BLOCK_SIZE; ++m) {
                s_vals[m] = a_blocks[m]->scales[sub_block_idx] & 0x0F;
                const int ib = k_inner / 128, iib = k_inner % 128, iic = iib % 32, cic = iib / 32;
                l_vals[m] = (a_blocks[m]->qs[ib * 32 + iic] >> (cic * 2)) & 3;
            }

            int8_t q_vals[N_BLOCK_SIZE];
            for (int n = 0; n < N_BLOCK_SIZE; ++n) {
                q_vals[n] = b_block_base[n].qs[k_inner];
            }

            for (int m = 0; m < M_BLOCK_SIZE; ++m) {
                for (int n = 0; n < N_BLOCK_SIZE; ++n) {
                    isum_tile[m][n] += s_vals[m] * l_vals[m] * q_vals[n];
                }
            }
        }

        for (int m = 0; m < M_BLOCK_SIZE; ++m) {
            const int m_val = a_blocks[m]->scales[sub_block_idx] >> 4;
            for (int n = 0; n < N_BLOCK_SIZE; ++n) {
                const int32_t summs = m_val * b_block_base[n].bsums[sub_block_idx];
                acc[m][n] += d_all[m][n] * isum_tile[m][n] - d_min[m][n] * summs;
            }
        }
    }
}
#endif

void gemv_q2_k_q8_k(
    float *y,
    const block_q2_K *A,
    const block_q8_K *x,
    int M,
    int K) {
    assert(K % QK_K == 0);
    const int K_blocks = K / QK_K;
#pragma omp parallel for num_threads(4)
    for (int i = 0; i < M; ++i) {
        float row_sum = 0.0f;
        const block_q2_K *A_row = A + i * K_blocks;
        for (int k_block = 0; k_block < K_blocks; ++k_block) {
            const block_q2_K *a_block = A_row + k_block;
            const block_q8_K *x_block = x + k_block;
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
            // ... (gemv NEON code remains the same as it was already efficient)
            const float d = x_block->d * MLLM_FP16_TO_FP32(a_block->d);
            const float dmin = -x_block->d * MLLM_FP16_TO_FP32(a_block->dmin);

            int32_t summs32_total = 0;
            {
                const uint8x16_t mins_and_scales = vld1q_u8(a_block->scales);
                const uint8x16_t mins = vshrq_n_u8(mins_and_scales, 4);
                const int16x8_t q8sums_lo = vld1q_s16(x_block->bsums);
                const int16x8_t q8sums_hi = vld1q_s16(x_block->bsums + 8);
                const int16x8_t mins_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mins)));
                const int16x8_t mins_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(mins)));
                int32x4_t summs32 = vmull_s16(vget_low_s16(mins_lo), vget_low_s16(q8sums_lo));
                summs32 = vmlal_s16(summs32, vget_high_s16(mins_lo), vget_high_s16(q8sums_lo));
                summs32 = vmlal_s16(summs32, vget_low_s16(mins_hi), vget_low_s16(q8sums_hi));
                summs32 = vmlal_s16(summs32, vget_high_s16(mins_hi), vget_high_s16(q8sums_hi));
                summs32_total = vaddvq_s32(summs32);
            }
            row_sum += dmin * summs32_total;

            int32_t isum = 0;
            {
                uint8_t scales_s[16];
                vst1q_u8(scales_s, vandq_u8(vld1q_u8(a_block->scales), vdupq_n_u8(0xF)));
                for (int sub_block_idx = 0; sub_block_idx < QK_K / 16; ++sub_block_idx) {
                    uint8_t l_bytes[16];
                    for (int k = 0; k < 16; ++k) {
                        int k_inner = sub_block_idx * 16 + k;
                        const int ib = k_inner / 128, iib = k_inner % 128, iic = iib % 32, cic = iib / 32;
                        l_bytes[k] = (a_block->qs[ib * 32 + iic] >> (cic * 2)) & 3;
                    }
                    const int8x16_t l_vec = vreinterpretq_s8_u8(vld1q_u8(l_bytes));
                    const int8x16_t q8_vec = vld1q_s8(x_block->qs + sub_block_idx * 16);
                    const int16x8_t p_lo = vmull_s8(vget_low_s8(l_vec), vget_low_s8(q8_vec));
                    const int16x8_t p_hi = vmull_s8(vget_high_s8(l_vec), vget_high_s8(q8_vec));
                    isum += vaddvq_s32(vpaddlq_s16(vaddq_s16(p_lo, p_hi))) * scales_s[sub_block_idx];
                }
            }
            row_sum += d * isum;
#else
            for (int k_inner = 0; k_inner < QK_K; ++k_inner) {
                const float d = MLLM_FP16_TO_FP32(a_block->d);
                const float dmin = MLLM_FP16_TO_FP32(a_block->dmin);
                const int sub_block_idx = k_inner / 16;
                const uint8_t sm_byte = a_block->scales[sub_block_idx];
                const float s = (float)(sm_byte & 0x0F);
                const float m = (float)(sm_byte >> 4);
                const int ib = k_inner / 128, iib = k_inner % 128, iic = iib % 32, c_idx = iib / 32;
                const uint8_t p_byte = a_block->qs[ib * 32 + iic];
                const int L = (p_byte >> (c_idx * 2)) & 3;
                const float a_val = (d * s) * L - (dmin * m);
                const float x_val = x_block->d * (float)x_block->qs[k_inner];
                row_sum += a_val * x_val;
            }
#endif
        }
        y[i] = row_sum;
    }
}

/**
 * @brief矩阵-矩阵乘法 (GEMM): C = A * B
 *
 * 1.  采用三层循环分块策略 (i, j, k)，并使用 OpenMP 对最外层 j 循环进行并行化，
 * 这是 GEMM 优化的经典且高效的并行策略。
 * 2.  主循环以更大的 M_BLOCK_SIZE x N_BLOCK_SIZE (8x16) 的步长进行，
 * 处理更大的数据块，提高了计算密度。
 * 3.  在 K 维度上，一次处理一个 QK_K 大小的块，这与数据的量化方式天然契合。
 * 4.  调用上面优化过的 `micro_kernel_8x16` 来执行核心计算。
 * 5.  对 A 和 B 矩阵中参与计算的块地址进行预计算和传递，使微内核的调用更简洁高效。
 */
void gemm_q2_k_q8_k(
    float *C,
    const block_q2_K *A,
    const block_q8_K *B_packed,
    int M,
    int N,
    int K) {
    assert(M % M_BLOCK_SIZE == 0);
    assert(N % N_BLOCK_SIZE == 0);
    assert(K % QK_K == 0);

    const int K_blocks = K / QK_K;

#pragma omp parallel for num_threads(4)
    for (int j = 0; j < N; j += N_BLOCK_SIZE) {
        for (int i = 0; i < M; i += M_BLOCK_SIZE) {
            float acc[M_BLOCK_SIZE][N_BLOCK_SIZE] = {{0.0f}};

            for (int k_block = 0; k_block < K_blocks; ++k_block) {
                // --- 优化点 3: 干净利落地传递数据指针 ---
                const block_q2_K *a_blocks[M_BLOCK_SIZE];
                for (int m_idx = 0; m_idx < M_BLOCK_SIZE; ++m_idx) {
                    a_blocks[m_idx] = A + (i + m_idx) * K_blocks + k_block;
                }

                // 关键改动：得益于新的 packing 方式，B 矩阵块的地址是连续的
                const block_q8_K *b_block_base = B_packed + (j / N_BLOCK_SIZE) * (K_blocks * N_BLOCK_SIZE) + k_block * N_BLOCK_SIZE;

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
                micro_kernel_8x16_neon(acc, a_blocks, b_block_base);
#else
                micro_kernel_8x16_reference(acc, a_blocks, b_block_base);
#endif
            }

            // --- 将累加结果写回 C 矩阵 ---
            for (int m_idx = 0; m_idx < M_BLOCK_SIZE; ++m_idx) {
                for (int n_idx = 0; n_idx < N_BLOCK_SIZE; ++n_idx) {
                    C[(i + m_idx) * N + (j + n_idx)] = acc[m_idx][n_idx];
                }
            }
        }
    }
}