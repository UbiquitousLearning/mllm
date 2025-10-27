#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "DataType.hpp"
#include "backends/cpu/third_party/ggml/QuantizeFP16.hpp"

// --- 平台检测和微内核尺寸定义 ---
#if defined(__aarch64__)
#include <arm_neon.h>
#define TARGET_ARCH "aarch64 (NEON)"
#define MR_micro 4 // NEON 使用 4x4 微内核
#define NR_micro 4

#elif defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h> // for AVX, FMA
#define TARGET_ARCH "x86_64 (AVX)"
#define MR_micro 8 // AVX 使用 8x8 微内核
#define NR_micro 8

#else
#define TARGET_ARCH "Generic C"
#define MR_micro 4
#define NR_micro 4
#endif

#define KC_BLOCK 256

static inline int min(int a, int b) {
    return a < b ? a : b;
}

// --- 平台特定的打包和微内核函数 ---

#if defined(__aarch64__)
// NEON: 打包B矩阵的一个 4xkc 的 Panel
static void pack_b_for_neon_4x4(float *packed_b, const float *b_ptr, int N, int k_start, int k_end, int n_start) {
    int kc = k_end - k_start;
    const int nc = NR_micro;

    for (int k_local = 0; k_local < kc; ++k_local) {
        const float *b_src_ptr = b_ptr + (k_start + k_local) * N + n_start;
        float *b_dest_ptr = packed_b + k_local * nc;
        memcpy(b_dest_ptr, b_src_ptr, nc * sizeof(float));
    }
}

// NEON: 4x4 微内核
static void gemm_micro_kernel_neon_4x4(float *c_ptr, const float *a_ptr, const float *packed_b_ptr, int kc, int lda, int ldc) {
    float32x4_t c_reg0, c_reg1, c_reg2, c_reg3;
    c_reg0 = vld1q_f32(c_ptr + 0 * ldc);
    c_reg1 = vld1q_f32(c_ptr + 1 * ldc);
    c_reg2 = vld1q_f32(c_ptr + 2 * ldc);
    c_reg3 = vld1q_f32(c_ptr + 3 * ldc);
    for (int k = 0; k < kc; ++k) {
        float32x4_t b_reg = vld1q_f32(packed_b_ptr + k * NR_micro);
        c_reg0 = vfmaq_f32(c_reg0, vld1q_dup_f32(a_ptr + 0 * lda + k), b_reg);
        c_reg1 = vfmaq_f32(c_reg1, vld1q_dup_f32(a_ptr + 1 * lda + k), b_reg);
        c_reg2 = vfmaq_f32(c_reg2, vld1q_dup_f32(a_ptr + 2 * lda + k), b_reg);
        c_reg3 = vfmaq_f32(c_reg3, vld1q_dup_f32(a_ptr + 3 * lda + k), b_reg);
    }
    vst1q_f32(c_ptr + 0 * ldc, c_reg0);
    vst1q_f32(c_ptr + 1 * ldc, c_reg1);
    vst1q_f32(c_ptr + 2 * ldc, c_reg2);
    vst1q_f32(c_ptr + 3 * ldc, c_reg3);
}

#elif defined(__x86_64__) || defined(_M_X64)
// AVX: 打包B矩阵的一个 8xkc 的 Panel
static void pack_b_for_avx_8x8(float *packed_b, const float *b_ptr, int N, int k_start, int k_end, int n_start) {
    int kc = k_end - k_start;
    const int nc = NR_micro;

    for (int k_local = 0; k_local < kc; ++k_local) {
        const float *b_src_ptr = b_ptr + (k_start + k_local) * N + n_start;
        float *b_dest_ptr = packed_b + k_local * nc;
        memcpy(b_dest_ptr, b_src_ptr, nc * sizeof(float));
    }
}

// AVX: 8x8 微内核
static void gemm_micro_kernel_avx_8x8(float *c_ptr, const float *a_ptr, const float *packed_b_ptr, int kc, int lda, int ldc) {
    __m256 c_reg[MR_micro];
    for (int i = 0; i < MR_micro; ++i) {
        c_reg[i] = _mm256_loadu_ps(c_ptr + i * ldc);
    }

    for (int k = 0; k < kc; ++k) {
        __m256 b_reg = _mm256_load_ps(packed_b_ptr + k * NR_micro);
        for (int i = 0; i < MR_micro; ++i) {
            __m256 a_broadcast = _mm256_set1_ps(a_ptr[i * lda + k]);
            c_reg[i] = _mm256_fmadd_ps(a_broadcast, b_reg, c_reg[i]);
        }
    }

    for (int i = 0; i < MR_micro; ++i) {
        _mm256_storeu_ps(c_ptr + i * ldc, c_reg[i]);
    }
}
#endif

// 主 GEMM 函数
void gemm_fp32(float *c_ptr, const float *a_ptr, const float *b_ptr, int M, int N, int K) {
#if defined(__aarch64__) || defined(__x86_64__) || defined(_M_X64)
    // 仅在NEON或AVX路径下，我们才为打包分配内存
    float *packed_b = (float *)malloc(KC_BLOCK * NR_micro * sizeof(float));
    if (!packed_b) return;
#endif

    for (int k_col = 0; k_col < K; k_col += KC_BLOCK) {
        int kc = min(KC_BLOCK, K - k_col);

        for (int i_row = 0; i_row < M; i_row += MR_micro) {
            int mc = min(MR_micro, M - i_row);

            for (int j_col = 0; j_col < N; j_col += NR_micro) {
                int nc = min(NR_micro, N - j_col);

#if defined(__aarch64__)
                if (mc == MR_micro && nc == NR_micro) {
                    pack_b_for_neon_4x4(packed_b, b_ptr, N, k_col, k_col + kc, j_col);
                    gemm_micro_kernel_neon_4x4(c_ptr + i_row * N + j_col, a_ptr + i_row * K + k_col, packed_b, kc, K, N);
                    continue;
                }
#elif defined(__x86_64__) || defined(_M_X64)
                if (mc == MR_micro && nc == NR_micro) {
                    pack_b_for_avx_8x8(packed_b, b_ptr, N, k_col, k_col + kc, j_col);
                    gemm_micro_kernel_avx_8x8(c_ptr + i_row * N + j_col, a_ptr + i_row * K + k_col, packed_b, kc, K, N);
                    continue;
                }
#endif

                // --- 通用C语言路径 (也用于NEON/AVX的边缘情况) ---
                // 直接使用原始A, B矩阵进行计算，确保正确性
                for (int i = 0; i < mc; ++i) {
                    for (int j = 0; j < nc; ++j) {
                        float sum = 0.0f;
                        for (int k = 0; k < kc; ++k) {
                            sum += a_ptr[(i_row + i) * K + (k_col + k)] * b_ptr[(k_col + k) * N + (j_col + j)];
                        }
                        c_ptr[(i_row + i) * N + (j_col + j)] += sum;
                    }
                }
            }
        }
    }

#if defined(__aarch64__) || defined(__x86_64__) || defined(_M_X64)
    free(packed_b);
#endif
}

#if defined(__aarch64__)
// NEON: 打包B矩阵(fp16)的一个 4xkc 的 Panel
static void pack_b_fp16_for_neon_4x4(mllm_fp16_t *packed_b, const mllm_fp16_t *b_ptr, int N, int k_start, int k_end, int n_start) {
    int kc = k_end - k_start;
    const int nc = NR_micro;

    for (int k_local = 0; k_local < kc; ++k_local) {
        const mllm_fp16_t *b_src_ptr = b_ptr + (k_start + k_local) * N + n_start;
        mllm_fp16_t *b_dest_ptr = packed_b + k_local * nc;
        memcpy(b_dest_ptr, b_src_ptr, nc * sizeof(mllm_fp16_t));
    }
}

// NEON: 4x4 微内核 (fp32 * fp16)
static void gemm_micro_kernel_fp32_fp16_neon_4x4(float *c_ptr, const float *a_ptr, const mllm_fp16_t *packed_b_ptr, int kc, int lda, int ldc) {
    float32x4_t c_reg0, c_reg1, c_reg2, c_reg3;
    c_reg0 = vld1q_f32(c_ptr + 0 * ldc);
    c_reg1 = vld1q_f32(c_ptr + 1 * ldc);
    c_reg2 = vld1q_f32(c_ptr + 2 * ldc);
    c_reg3 = vld1q_f32(c_ptr + 3 * ldc);

    for (int k = 0; k < kc; ++k) {
        // 1. 从打包好的B中加载一行fp16
        float16x4_t b_reg_f16 = vld1_f16(packed_b_ptr + k * NR_micro);
        // 2. 将fp16向量转换为fp32向量
        float32x4_t b_reg_f32 = vcvt_f32_f16(b_reg_f16);

        // 3. 执行乘加操作 (与之前相同)
        c_reg0 = vfmaq_f32(c_reg0, vld1q_dup_f32(a_ptr + 0 * lda + k), b_reg_f32);
        c_reg1 = vfmaq_f32(c_reg1, vld1q_dup_f32(a_ptr + 1 * lda + k), b_reg_f32);
        c_reg2 = vfmaq_f32(c_reg2, vld1q_dup_f32(a_ptr + 2 * lda + k), b_reg_f32);
        c_reg3 = vfmaq_f32(c_reg3, vld1q_dup_f32(a_ptr + 3 * lda + k), b_reg_f32);
    }
    vst1q_f32(c_ptr + 0 * ldc, c_reg0);
    vst1q_f32(c_ptr + 1 * ldc, c_reg1);
    vst1q_f32(c_ptr + 2 * ldc, c_reg2);
    vst1q_f32(c_ptr + 3 * ldc, c_reg3);
}

#elif defined(__x86_64__) || defined(_M_X64)
// AVX: 打包B矩阵(fp16)的一个 8xkc 的 Panel
static void pack_b_fp16_for_avx_8x8(mllm_fp16_t *packed_b, const mllm_fp16_t *b_ptr, int N, int k_start, int k_end, int n_start) {
    int kc = k_end - k_start;
    const int nc = NR_micro;

    for (int k_local = 0; k_local < kc; ++k_local) {
        const mllm_fp16_t *b_src_ptr = b_ptr + (k_start + k_local) * N + n_start;
        mllm_fp16_t *b_dest_ptr = packed_b + k_local * nc;
        memcpy(b_dest_ptr, b_src_ptr, nc * sizeof(mllm_fp16_t));
    }
}

// AVX: 8x8 微内核 (fp32 * fp16)
static void gemm_micro_kernel_fp32_fp16_avx_8x8(float *c_ptr, const float *a_ptr, const mllm_fp16_t *packed_b_ptr, int kc, int lda, int ldc) {
    __m256 c_reg[MR_micro];
    for (int i = 0; i < MR_micro; ++i) {
        c_reg[i] = _mm256_loadu_ps(c_ptr + i * ldc);
    }

    for (int k = 0; k < kc; ++k) {
        // 1. 从打包好的B中加载一行fp16 (8个uint16_t) 到一个128位的XMM寄存器
        __m128i b_reg_f16 = _mm_loadu_si128((__m128i const *)(packed_b_ptr + k * NR_micro));
        // 2. 将128位的fp16向量转换为256位的fp32向量
        __m256 b_reg_f32 = _mm256_cvtph_ps(b_reg_f16);

        // 3. 执行乘加操作 (与之前相同)
        for (int i = 0; i < MR_micro; ++i) {
            __m256 a_broadcast = _mm256_set1_ps(a_ptr[i * lda + k]);
            c_reg[i] = _mm256_fmadd_ps(a_broadcast, b_reg_f32, c_reg[i]);
        }
    }

    for (int i = 0; i < MR_micro; ++i) {
        _mm256_storeu_ps(c_ptr + i * ldc, c_reg[i]);
    }
}
#endif

// 新增的 GEMM 函数
void gemm_fp32_fp16(float *c_ptr, const float *a_ptr, const mllm_fp16_t *b_ptr, int M, int N, int K) {
#if defined(__aarch64__) || defined(__x86_64__) || defined(_M_X64)
    // 仅在NEON或AVX路径下，我们才为打包分配内存
    mllm_fp16_t *packed_b = (mllm_fp16_t *)malloc(KC_BLOCK * NR_micro * sizeof(mllm_fp16_t));
    if (!packed_b) return;
#endif

    for (int k_col = 0; k_col < K; k_col += KC_BLOCK) {
        int kc = min(KC_BLOCK, K - k_col);

        for (int i_row = 0; i_row < M; i_row += MR_micro) {
            int mc = min(MR_micro, M - i_row);

            for (int j_col = 0; j_col < N; j_col += NR_micro) {
                int nc = min(NR_micro, N - j_col);

#if defined(__aarch64__)
                if (mc == MR_micro && nc == NR_micro) {
                    pack_b_fp16_for_neon_4x4(packed_b, b_ptr, N, k_col, k_col + kc, j_col);
                    gemm_micro_kernel_fp32_fp16_neon_4x4(c_ptr + i_row * N + j_col, a_ptr + i_row * K + k_col, packed_b, kc, K, N);
                    continue;
                }
#elif defined(__x86_64__) || defined(_M_X64)
                if (mc == MR_micro && nc == NR_micro) {
                    pack_b_fp16_for_avx_8x8(packed_b, b_ptr, N, k_col, k_col + kc, j_col);
                    gemm_micro_kernel_fp32_fp16_avx_8x8(c_ptr + i_row * N + j_col, a_ptr + i_row * K + k_col, packed_b, kc, K, N);
                    continue;
                }
#endif

                // ---- 通用C语言路径 (也用于NEON/AVX的边缘情况) ----
                // 直接使用原始A(fp32)和B(fp16)矩阵进行计算
                for (int i = 0; i < mc; ++i) {
                    for (int j = 0; j < nc; ++j) {
                        float sum = 0.0f;
                        for (int k = 0; k < kc; ++k) {
                            // C[i_row+i][j_col+j] += A[i_row+i][k_col+k] * B[k_col+k][j_col+j]
                            // 关键：使用宏将B的fp16值转换为fp32
                            sum += a_ptr[(i_row + i) * K + (k_col + k)] * MLLM_FP16_TO_FP32(b_ptr[(k_col + k) * N + (j_col + j)]);
                        }
                        c_ptr[(i_row + i) * N + (j_col + j)] += sum;
                    }
                }
            }
        }
    }

#if defined(__aarch64__) || defined(__x86_64__) || defined(_M_X64)
    free(packed_b);
#endif
}
