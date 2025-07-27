// opencl/kernel/matmul.cl

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// ==================================================================
// 1. 宏定义和数据结构
// ==================================================================

#define TILE_SIZE 16
#define QK4_0 32
#define QK8_0 32

typedef struct {
    half d;
    uchar qs[QK4_0 / 2];
} block_q4_0;

// ==================================================================
// 2. FP32 BSHD GEMM 内核
// ==================================================================

/**
 * @brief 高性能浮点矩阵乘法 (FP32 * FP32)，支持 BSHD 布局
 */
__kernel void gemm_fp32(
    __global const float *A,
    __global const float *B,
    __global float *C,
    const int M, const int K, const int N,
    const int H, const int K_b) {
    const int s = get_global_id(1);
    const int n = get_global_id(0);
    const int bh_idx = get_global_id(2);
    const int b = bh_idx / H;
    const int h = bh_idx % H;

    const int local_row = get_local_id(1);
    const int local_col = get_local_id(0);

    __local float a_tile[TILE_SIZE][TILE_SIZE];
    __local float b_tile[TILE_SIZE][TILE_SIZE];

    float acc = 0.0f;
    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < num_tiles; ++t) {
        const int k_start = t * TILE_SIZE;
        const int a_k_idx = k_start + local_col;
        const int b_k_idx = k_start + local_row;
        if (s < M && a_k_idx < K) {
            a_tile[local_row][local_col] = A[(long)b * M * H * K + (long)s * H * K + (long)h * K + a_k_idx];
        } else {
            a_tile[local_row][local_col] = 0.0f;
        }
        if (n < N && b_k_idx < K_b) {
            b_tile[local_row][local_col] = B[(long)b * K_b * H * N + (long)b_k_idx * H * N + (long)h * N + n];
        } else {
            b_tile[local_row][local_col] = 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k_tile = 0; k_tile < TILE_SIZE; ++k_tile) {
            acc += a_tile[local_row][k_tile] * b_tile[k_tile][local_col];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (s < M && n < N) {
        C[(long)b * M * H * N + (long)s * H * N + (long)h * N + n] = acc;
    }
}

// ==================================================================
// 3. FP32 BHSD GEMM 内核 (New)
// ==================================================================
/**
 * @brief 高性能浮点矩阵乘法 (FP32 * FP32)，支持 BHSD 布局
 */
__kernel void gemm_fp32_bhsd(
    __global const float *A,
    __global const float *B,
    __global float *C,
    const int M, const int K, const int N,
    const int H, const int K_b) {
    const int s = get_global_id(1);
    const int n = get_global_id(0);
    const int bh_idx = get_global_id(2);
    const int b = bh_idx / H;
    const int h = bh_idx % H;

    const int local_row = get_local_id(1);
    const int local_col = get_local_id(0);

    __local float a_tile[TILE_SIZE][TILE_SIZE];
    __local float b_tile[TILE_SIZE][TILE_SIZE];

    float acc = 0.0f;
    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < num_tiles; ++t) {
        const int k_start = t * TILE_SIZE;
        const int a_k_idx = k_start + local_col;
        const int b_k_idx = k_start + local_row;
        if (s < M && a_k_idx < K) {
            a_tile[local_row][local_col] = A[(long)b * H * M * K + (long)h * M * K + (long)s * K + a_k_idx];
        } else {
            a_tile[local_row][local_col] = 0.0f;
        }
        if (n < N && b_k_idx < K_b) {
            b_tile[local_row][local_col] = B[(long)b * H * K_b * N + (long)h * K_b * N + (long)b_k_idx * N + n];
        } else {
            b_tile[local_row][local_col] = 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k_tile = 0; k_tile < TILE_SIZE; ++k_tile) {
            acc += a_tile[local_row][k_tile] * b_tile[k_tile][local_col];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (s < M && n < N) {
        C[(long)b * H * M * N + (long)h * M * N + (long)s * N + n] = acc;
    }
}

#if !defined(SUPPORTS_FP16)
// ==================================================================
// 4. FP16 GEMM 内核 (Fallback)
// ==================================================================

// ---------- [FP16 BSHD 回退版] ----------
__kernel void gemm_fp16(
    __global const half *A, __global const half *B, __global half *C,
    const int M, const int K, const int N, const int H, const int K_b) {
    const int s = get_global_id(1);
    const int n = get_global_id(0);
    const int bh_idx = get_global_id(2);
    const int b = bh_idx / H;
    const int h = bh_idx % H;
    if (s >= M || n >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        long a_idx = (long)b * M * H * K + (long)s * H * K + (long)h * K + k;
        long b_idx = (long)b * K_b * H * N + (long)k * H * N + (long)h * N + n;
        acc += (float)A[a_idx] * (float)B[b_idx];
    }
    C[(long)b * M * H * N + (long)s * H * N + (long)h * N + n] = (half)acc;
}

// ---------- [FP16 BHSD 回退版] (New) ----------
__kernel void gemm_fp16_bhsd(
    __global const half *A, __global const half *B, __global half *C,
    const int M, const int K, const int N, const int H, const int K_b) {
    const int s = get_global_id(1);
    const int n = get_global_id(0);
    const int bh_idx = get_global_id(2);
    const int b = bh_idx / H;
    const int h = bh_idx % H;
    if (s >= M || n >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        long a_idx = (long)b * H * M * K + (long)h * M * K + (long)s * K + k;
        long b_idx = (long)b * H * K_b * N + (long)h * K_b * N + (long)k * N + n;
        acc += (float)A[a_idx] * (float)B[b_idx];
    }
    C[(long)b * H * M * N + (long)h * M * N + (long)s * N + n] = (half)acc;
}

#else
// ==================================================================
// 5. FP16 GEMM 内核 (High-Performance)
// ==================================================================

// ---------- [FP16 BSHD 高性能版] ----------
__kernel void gemm_fp16(
    __global const half *A, __global const half *B, __global half *C,
    const int M, const int K, const int N, const int H, const int K_b) {
    const int s = get_global_id(1);
    const int n = get_global_id(0);
    const int bh_idx = get_global_id(2);
    const int b = bh_idx / H;
    const int h = bh_idx % H;

    const int local_row = get_local_id(1);
    const int local_col = get_local_id(0);

    __local half a_tile[TILE_SIZE][TILE_SIZE];
    __local half b_tile[TILE_SIZE][TILE_SIZE];

    half acc = 0.0h;
    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < num_tiles; ++t) {
        const int k_start = t * TILE_SIZE;
        const int a_k_idx = k_start + local_col;
        const int b_k_idx = k_start + local_row;
        if (s < M && a_k_idx < K) {
            a_tile[local_row][local_col] = A[(long)b * M * H * K + (long)s * H * K + (long)h * K + a_k_idx];
        } else {
            a_tile[local_row][local_col] = 0.0h;
        }
        if (n < N && b_k_idx < K_b) {
            b_tile[local_row][local_col] = B[(long)b * K_b * H * N + (long)b_k_idx * H * N + (long)h * N + n];
        } else {
            b_tile[local_row][local_col] = 0.0h;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k_tile = 0; k_tile < TILE_SIZE; ++k_tile) {
            acc += a_tile[local_row][k_tile] * b_tile[k_tile][local_col];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (s < M && n < N) {
        C[(long)b * M * H * N + (long)s * H * N + (long)h * N + n] = acc;
    }
}

// ---------- [FP16 BHSD 高性能版] ----------

__kernel void gemm_fp16_bhsd(
    __global const half *A, __global const half *B, __global half *C,
    const int M, const int K, const int N, const int H, const int K_b) {
    const int s_out = get_global_id(1);
    const int n_out = get_global_id(0);
    const int bh_idx = get_global_id(2);
    const int b = bh_idx / H;
    const int h = bh_idx % H;
    const int local_row = get_local_id(1);
    const int local_col = get_local_id(0);

    __local half a_tile[TILE_SIZE][TILE_SIZE];
    __local half b_tile[TILE_SIZE][TILE_SIZE];
    half acc = 0.0f;
    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < num_tiles; ++t) {
        const int k_start = t * TILE_SIZE;
        const int a_s_idx = get_group_id(1) * TILE_SIZE + local_row;
        const int a_k_idx = k_start + local_col;
        if (a_s_idx < M && a_k_idx < K) {
            a_tile[local_row][local_col] = A[(long)b * H * M * K + (long)h * M * K + (long)a_s_idx * K + a_k_idx];
        } else {
            a_tile[local_row][local_col] = 0.0h;
        }
        const int b_n_idx = get_group_id(0) * TILE_SIZE + local_col;
        const int b_k_idx = k_start + local_row;
        if (b_n_idx < N && b_k_idx < K_b) {
            b_tile[local_col][local_row] = B[(long)b * H * K_b * N + (long)h * K_b * N + (long)b_k_idx * N + b_n_idx];
        } else {
            b_tile[local_col][local_row] = 0.0h;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

#pragma unroll
        for (int k_tile_vec = 0; k_tile_vec < TILE_SIZE; k_tile_vec += 4) {
            half4 a_vec = vload4(0, &a_tile[local_row][k_tile_vec]);
            half4 b_vec = vload4(0, &b_tile[local_col][k_tile_vec]);
            acc += dot(a_vec, b_vec);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (s_out < M && n_out < N) {
        C[(long)b * H * M * N + (long)h * M * N + (long)s_out * N + n_out] = acc;
    }
}

// #define TILE_M 8
// #define TILE_N 4
// #define K_STEP 4

// __kernel void gemm_fp16_bhsd(
//     __global const half *A,
//     __global const half *B,
//     __global half *C,
//     const int M,
//     const int K,
//     const int N,
//     const int H,
//     const int K_b) {
//     const int gx = get_global_id(0);
//     const int gy = get_global_id(1);
//     const int bh_idx = get_global_id(2);
//     const int m_base = gy * TILE_M;
//     const int n_base = gx * TILE_N;
//     if (m_base >= M || n_base >= N) {
//         return;
//     }
//     const int b = bh_idx / H;
//     const int h = bh_idx % H;
//     __global const half *a_ptr = A + ((long)b * H + h) * M * K;
//     __global const half *b_ptr = B + ((long)b * H + h) * K_b * N;
//     half4 a_reg;
//     half4 b_tile_reg[K_STEP];
//     half c_acc[TILE_M][TILE_N] = {{0.0h}};
//     for (int k_outer = 0; k_outer < K; k_outer += K_STEP) {
// #pragma unroll
//         for (int i = 0; i < K_STEP; ++i) {
//             if (k_outer + i < K) {
//                 b_tile_reg[i] = vload4(0, b_ptr + (k_outer + i) * N + n_base);
//             } else {
//                 b_tile_reg[i] = (half4)(0.0h);
//             }
//         }
// #pragma unroll
//         for (int m_local = 0; m_local < TILE_M; ++m_local) {
//             if (m_base + m_local < M) {
//                 a_reg = vload4(0, a_ptr + (m_base + m_local) * K + k_outer);
//                 // 计算 C[m_local][0]
//                 half4 b_col0 = (half4)(b_tile_reg[0].s0, b_tile_reg[1].s0, b_tile_reg[2].s0, b_tile_reg[3].s0);
//                 c_acc[m_local][0] += dot(a_reg, b_col0);
//                 // 计算 C[m_local][1]
//                 half4 b_col1 = (half4)(b_tile_reg[0].s1, b_tile_reg[1].s1, b_tile_reg[2].s1, b_tile_reg[3].s1);
//                 c_acc[m_local][1] += dot(a_reg, b_col1);
//                 // 计算 C[m_local][2]
//                 half4 b_col2 = (half4)(b_tile_reg[0].s2, b_tile_reg[1].s2, b_tile_reg[2].s2, b_tile_reg[3].s2);
//                 c_acc[m_local][2] += dot(a_reg, b_col2);
//                 // 计算 C[m_local][3]
//                 half4 b_col3 = (half4)(b_tile_reg[0].s3, b_tile_reg[1].s3, b_tile_reg[2].s3, b_tile_reg[3].s3);
//                 c_acc[m_local][3] += dot(a_reg, b_col3);
//             }
//         }
//     }
//     __global half *c_ptr = C + ((long)b * H + h) * M * N;
// #pragma unroll
//     for (int i = 0; i < TILE_M; ++i) {
//         if (m_base + i < M) {
// #pragma unroll
//             for (int j = 0; j < TILE_N; ++j) {
//                 if (n_base + j < N) {
//                     c_ptr[(long)(m_base + i) * N + (n_base + j)] = c_acc[i][j];
//                 }
//             }
//         }
//     }
// }

#endif // SUPPORTS_FP16
