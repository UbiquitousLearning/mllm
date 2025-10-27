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
// 2. FP32 GEMM 内核
// ==================================================================
/**
 * @brief 高性能 FP32 GEMM，计算 C = A * B^T
 * @param A 矩阵 A，布局为 (B, M, H, K)
 * @param B 矩阵 B，布局为 (B, N, H, K)
 * @param C 矩阵 C，布局为 (B, M, H, N)
 */
__kernel void gemm_fp32_transb(
    __global const float *A,
    __global const float *B,
    __global float *C,
    const int M, const int K, const int N,
    const int H) {
    // --- 1. 索引计算 ---
    const int s = get_global_id(1);
    const int n = get_global_id(0);
    const int bh_idx = get_global_id(2);
    const int b = bh_idx / H;
    const int h = bh_idx % H;

    const int local_row = get_local_id(1);
    const int local_col = get_local_id(0);

    // --- 2. 初始化和局部内存 ---
    __local float a_tile[TILE_SIZE][TILE_SIZE];
    __local float b_tile[TILE_SIZE][TILE_SIZE];

    float acc = 0.0f;
    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    // --- 3. 沿 K 维度进行分块计算 ---
    for (int t = 0; t < num_tiles; ++t) {
        const int k_start = t * TILE_SIZE;

        // --- 3a. 协作加载 A 和 B 的 tile ---
        // 加载 A 的 tile: A[s, k]
        const int a_k_idx = k_start + local_col;
        if (s < M && a_k_idx < K) {
            a_tile[local_row][local_col] = A[(long)b * M * H * K + (long)s * H * K + (long)h * K + a_k_idx];
        } else {
            a_tile[local_row][local_col] = 0.0f;
        }

        // 加载 B 的 tile: B[n, k]
        // 整个工作组协作加载 B 的一个 TILE_SIZE * TILE_SIZE 区域
        const int b_n_idx = get_group_id(0) * TILE_SIZE + local_row;
        const int b_k_idx = k_start + local_col;
        if (b_n_idx < N && b_k_idx < K) {
            // B 布局为 (B, N, H, K)
            b_tile[local_row][local_col] = B[(long)b * N * H * K + (long)b_n_idx * H * K + (long)h * K + b_k_idx];
        } else {
            b_tile[local_row][local_col] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // --- 3b. 在局部内存中计算点积 ---
        // C[s, n] = sum_k A[s, k] * B[n, k]
        for (int k_tile = 0; k_tile < TILE_SIZE; ++k_tile) {
            // A[s, k] 从 a_tile[local_row][k_tile] 获取
            // B[n, k] 从 b_tile[local_col][k_tile] 获取
            acc += a_tile[local_row][k_tile] * b_tile[local_col][k_tile];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // --- 4. 将结果写回全局内存 ---
    if (s < M && n < N) {
        C[(long)b * M * H * N + (long)s * H * N + (long)h * N + n] = acc;
    }
}

__kernel void gemm_fp32_q4_0_transb(
    __global const float *A,
    __global const uchar *B_q,
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

    __global const block_q4_0 *B = (__global const block_q4_0 *)B_q;
    __local float a_tile[TILE_SIZE][TILE_SIZE];
    __local float b_dequant_tile[TILE_SIZE][TILE_SIZE];
    float acc = 0.0f;
    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        const int k_start = t * TILE_SIZE;

        const int a_k_idx = k_start + local_col;
        if (s < M && a_k_idx < K) {
            a_tile[local_row][local_col] = A[(long)b * M * H * K + (long)s * H * K + (long)h * K + a_k_idx];
        } else {
            a_tile[local_row][local_col] = 0.0f;
        }

        const int n_for_load = get_group_id(0) * TILE_SIZE + local_row;
        const int k_for_load = k_start + local_col;

        if (n_for_load < N && k_for_load < K) {
            const int k_block_idx = k_for_load / QK4_0;
            const int k_in_block = k_for_load % QK4_0;

            const long b_block_mem_idx = (long)b * N * H * (K / QK4_0) + (long)n_for_load * H * (K / QK4_0) + (long)h * (K / QK4_0) + k_block_idx;
            const __global block_q4_0 *b_block_ptr = &B[b_block_mem_idx];

            const float d_b = vload_half(0, (__global half *)(&(b_block_ptr->d)));

            const int qs_idx = k_in_block % (QK4_0 / 2);
            const uchar q_packed = b_block_ptr->qs[qs_idx];

            char q_nibble;
            if (k_in_block >= (QK4_0 / 2)) {
                q_nibble = (q_packed >> 4);
            } else {
                q_nibble = (q_packed & 0x0F);
            }
            b_dequant_tile[local_row][local_col] = (float)(q_nibble - 8) * d_b;
        } else {
            b_dequant_tile[local_row][local_col] = 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k_tile = 0; k_tile < TILE_SIZE; ++k_tile) {
            acc += a_tile[local_row][k_tile] * b_dequant_tile[local_col][k_tile];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (s < M && n < N) {
        C[(long)b * M * H * N + (long)s * H * N + (long)h * N + n] = acc;
    }
}

/**************************************************************************************************/
/* 新增算子: gemm_fp32_fp16_transb                                  */
/* 功能: 高性能混合精度矩阵乘法 C(fp32) = A(fp32) * B(fp16)^T          */
/* 架构: 采用与gemm_fp32_transb相同的Tiling架构，确保高性能和边界安全。 */
/**************************************************************************************************/
__kernel void gemm_fp32_fp16_transb(
    __global const float *A,
    __global const half *B,
    __global float *C,
    const int M, const int K, const int N,
    const int H) {
    // --- 1. 索引计算 ---
    const int s = get_global_id(1);
    const int n = get_global_id(0);
    const int bh_idx = get_global_id(2);
    const int b = bh_idx / H;
    const int h = bh_idx % H;

    const int local_row = get_local_id(1);
    const int local_col = get_local_id(0);

    // --- 2. 初始化和局部内存 ---
    __local float a_tile[TILE_SIZE][TILE_SIZE];
    __local float b_tile[TILE_SIZE][TILE_SIZE]; // B Tile也使用float以保持精度

    float acc = 0.0f;
    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    // --- 3. 沿 K 维度进行分块计算 ---
    for (int t = 0; t < num_tiles; ++t) {
        const int k_start = t * TILE_SIZE;

        // --- 3a. 协作加载 A (FP32) tile ---
        const int a_k_idx = k_start + local_col;
        if (s < M && a_k_idx < K) {
            a_tile[local_row][local_col] = A[(long)b * M * H * K + (long)s * H * K + (long)h * K + a_k_idx];
        } else {
            a_tile[local_row][local_col] = 0.0f;
        }

        // --- 3b. 协作加载 B (FP16) tile 并立即转换为 FP32 ---
        const int b_n_idx = get_group_id(0) * TILE_SIZE + local_row;
        const int b_k_idx = k_start + local_col;
        if (b_n_idx < N && b_k_idx < K) {
            // 从全局内存读取half，转换为float，存入局部内存
            b_tile[local_row][local_col] = (float)B[(long)b * N * H * K + (long)b_n_idx * H * K + (long)h * K + b_k_idx];
        } else {
            b_tile[local_row][local_col] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // --- 3c. 在局部内存中计算点积 (全部为FP32) ---
        for (int k_tile = 0; k_tile < TILE_SIZE; ++k_tile) {
            acc += a_tile[local_row][k_tile] * b_tile[local_col][k_tile];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // --- 4. 将结果写回全局内存 ---
    if (s < M && n < N) {
        C[(long)b * M * H * N + (long)s * H * N + (long)h * N + n] = acc;
    }
}

// ==================================================================
// 3. FP16 GEMM 内核
// ==================================================================

/**
 * @brief 高性能 FP16 GEMM，计算 C = A * B^T
 */
#if !defined(SUPPORTS_FP16)
// ---------- [FP16_transb 回退版] ----------
__kernel void gemm_fp16_transb(
    __global const half *A,
    __global const half *B,
    __global half *C,
    const int M, const int K, const int N,
    const int H) {
    // 1. 索引计算：每个工作项负责计算输出 C 的一个元素
    const int s = get_global_id(1); // C 的行索引 (M 维度)
    const int n = get_global_id(0); // C 的列索引 (N 维度)
    const int bh_idx = get_global_id(2);
    const int b = bh_idx / H;
    const int h = bh_idx % H;

    // 边界检查
    if (s >= M || n >= N) return;

    // 2. 计算：C[s,n] = sum_k(A[s,k] * B[n,k])
    // 使用 float 累加器以保证精度，与原始回退版保持一致
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        // A[s,k] 的索引，布局 (B, M, H, K)
        long a_idx = (long)b * M * H * K + (long)s * H * K + (long)h * K + k;
        // B[n,k] 的索引，布局 (B, N, H, K)
        long b_idx = (long)b * N * H * K + (long)n * H * K + (long)h * K + k;

        acc += (float)A[a_idx] * (float)B[b_idx];
    }

    // 3. 写回结果
    long c_idx = (long)b * M * H * N + (long)s * H * N + (long)h * N + n;
    C[c_idx] = (half)acc;
}
#else
// ---------- [FP16_transb 高性能版] ----------
__kernel void gemm_fp16_transb(
    __global const half *A,
    __global const half *B,
    __global half *C,
    const int M, const int K, const int N,
    const int H) {
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
        if (s < M && a_k_idx < K) {
            a_tile[local_row][local_col] = A[(long)b * M * H * K + (long)s * H * K + (long)h * K + a_k_idx];
        } else {
            a_tile[local_row][local_col] = 0.0h;
        }

        const int b_n_idx = get_group_id(0) * TILE_SIZE + local_row;
        const int b_k_idx = k_start + local_col;
        if (b_n_idx < N && b_k_idx < K) {
            b_tile[local_row][local_col] = B[(long)b * N * H * K + (long)b_n_idx * H * K + (long)h * K + b_k_idx];
        } else {
            b_tile[local_row][local_col] = 0.0h;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k_tile = 0; k_tile < TILE_SIZE; ++k_tile) {
            acc += a_tile[local_row][k_tile] * b_tile[local_col][k_tile];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (s < M && n < N) {
        C[(long)b * M * H * N + (long)s * H * N + (long)h * N + n] = acc;
    }
}
#endif // SUPPORTS_FP16
#if !defined(SUPPORTS_FP16)
// ---------- [FP16_Q4_0 回退版 - 已修正] ----------
__kernel void gemm_fp16_q4_0_transb(
    __global const half *A,
    __global const block_q4_0 *B,
    __global half *C,
    const int M, const int K, const int N,
    const int H, const int K_b) {
    const int s = get_global_id(1);
    const int n = get_global_id(0);
    const int bh_idx = get_global_id(2);
    const int b = bh_idx / H;
    const int h = bh_idx % H;

    if (s >= M || n >= N) {
        return;
    }

    float acc = 0.0f;

    const long a_row_offset = (long)b * M * H * K + (long)s * H * K + (long)h * K;
    const long b_row_offset = (long)b * N * H * (K / QK4_0) + (long)n * H * (K / QK4_0) + (long)h * (K / QK4_0);

    for (int k_block_idx = 0; k_block_idx < K / QK4_0; ++k_block_idx) {
        const __global block_q4_0 *b_block_ptr = &B[b_row_offset + k_block_idx];
        const float d_b = vload_half(0, (__global half *)(&(b_block_ptr->d)));

        const __global half *a_ptr = A + a_row_offset + k_block_idx * QK4_0;

        for (int j = 0; j < QK4_0 / 2; ++j) {
            const uchar q_packed = b_block_ptr->qs[j];

            const float b_val_0 = (float)((q_packed & 0x0F) - 8) * d_b;
            const float b_val_1 = (float)((q_packed >> 4) - 8) * d_b;

            acc += (float)a_ptr[j] * b_val_0;
            acc += (float)a_ptr[j + QK4_0 / 2] * b_val_1;
        }
    }

    C[(long)b * M * H * N + (long)s * H * N + (long)h * N + n] = (half)acc;
}
#else
// ---------- [FP16_Q4_0 高性能版 - 已修正] ----------
__kernel void gemm_fp16_q4_0_transb(
    __global const half *A,
    __global const block_q4_0 *B,
    __global half *C,
    const int M, const int K, const int N,
    const int H, const int K_b) {
    const int s = get_global_id(1);
    const int n = get_global_id(0);
    const int bh_idx = get_global_id(2);
    const int b = bh_idx / H;
    const int h = bh_idx % H;

    const int local_row = get_local_id(1);
    const int local_col = get_local_id(0);

    __local half a_tile[TILE_SIZE][TILE_SIZE];
    __local float b_dequant_tile[TILE_SIZE][TILE_SIZE];
    float acc = 0.0f;
    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        const int k_start = t * TILE_SIZE;

        const int a_k_idx = k_start + local_col;
        if (s < M && a_k_idx < K) {
            a_tile[local_row][local_col] = A[(long)b * M * H * K + (long)s * H * K + (long)h * K + a_k_idx];
        } else {
            a_tile[local_row][local_col] = 0.0h;
        }

        const int n_for_load = get_group_id(0) * TILE_SIZE + local_row;
        const int k_for_load = k_start + local_col;

        if (n_for_load < N && k_for_load < K) {
            const int k_block_idx = k_for_load / QK4_0;
            const int k_in_block = k_for_load % QK4_0;

            const long b_block_mem_idx = (long)b * N * H * (K / QK4_0) + (long)n_for_load * H * (K / QK4_0) + (long)h * (K / QK4_0) + k_block_idx;
            const __global block_q4_0 *b_block_ptr = &B[b_block_mem_idx];

            const float d_b = vload_half(0, (__global half *)(&(b_block_ptr->d)));

            const int qs_idx = k_in_block % (QK4_0 / 2);
            const uchar q_packed = b_block_ptr->qs[qs_idx];

            char q_nibble;
            if (k_in_block >= (QK4_0 / 2)) {
                q_nibble = (q_packed >> 4);
            } else {
                q_nibble = (q_packed & 0x0F);
            }
            b_dequant_tile[local_row][local_col] = (float)(q_nibble - 8) * d_b;
        } else {
            b_dequant_tile[local_row][local_col] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k_tile = 0; k_tile < TILE_SIZE; ++k_tile) {
            acc += (float)a_tile[local_row][k_tile] * b_dequant_tile[local_col][k_tile];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (s < M && n < N) {
        C[(long)b * M * H * N + (long)s * H * N + (long)h * N + n] = (half)acc;
    }
}

#endif // SUPPORTS_FP16