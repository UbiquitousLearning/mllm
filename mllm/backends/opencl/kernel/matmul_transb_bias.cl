// kernel/matmul_transb_bias.cl

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define TILE_SIZE 16
#define QK4_0 32
#define QK8_0 32

typedef struct {
    half d;
    uchar qs[QK4_0 / 2];
} block_q4_0;

// ==================================================================
// 1. FP32 Fused GEMM + Bias Kernel
// ==================================================================
__kernel void gemm_fp32_transb_bias(
    __global const float *A,
    __global const float *B,
    __global const float *bias,
    __global float *C,
    const int M, const int K, const int N,
    const int H, const int K_b,
    const int has_bias) {
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
        if (s < M && a_k_idx < K) {
            a_tile[local_row][local_col] = A[(long)b * M * H * K + (long)s * H * K + (long)h * K + a_k_idx];
        } else {
            a_tile[local_row][local_col] = 0.0f;
        }

        const int b_n_idx = get_group_id(0) * TILE_SIZE + local_row;
        const int b_k_idx = k_start + local_col;
        if (b_n_idx < N && b_k_idx < K) {
            b_tile[local_row][local_col] = B[(long)b * N * H * K + (long)b_n_idx * H * K + (long)h * K + b_k_idx];
        } else {
            b_tile[local_row][local_col] = 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k_tile = 0; k_tile < TILE_SIZE; ++k_tile) {
            acc += a_tile[local_row][k_tile] * b_tile[local_col][k_tile];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (s < M && n < N) {
        if (has_bias != 0) {
            acc += bias[n];
        }
        C[(long)b * M * H * N + (long)s * H * N + (long)h * N + n] = acc;
    }
}

__kernel void gemm_fp32_fp16_transb_bias(
    __global const float *A,
    __global const half *B,
    __global const float *bias,
    __global float *C,
    const int M, const int K, const int N,
    const int H,
    const int has_bias) {
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

    // --- 4. 添加偏置（可选）并将结果写回全局内存 ---
    if (s < M && n < N) {
        if (has_bias != 0) {
            acc += bias[n];
        }
        C[(long)b * M * H * N + (long)s * H * N + (long)h * N + n] = acc;
    }
}

// ==================================================================
// 2. FP16 Fused GEMM + Bias Kernel
// ==================================================================
#if defined(SUPPORTS_FP16)
// ---------- [FP16_transb_bias 高性能版] ----------
__kernel void gemm_fp16_transb_bias(
    __global const half *A,
    __global const half *B,
    __global const float *bias,
    __global half *C,
    const int M, const int K, const int N,
    const int H, const int K_b,
    const int has_bias) {
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
        if (has_bias != 0) {
            acc += bias[n];
        }
        C[(long)b * M * H * N + (long)s * H * N + (long)h * N + n] = acc;
    }
}
#else
// ---------- [FP16_transb_bias 回退版] ----------
__kernel void gemm_fp16_transb_bias(
    __global const half *A,
    __global const half *B,
    __global const float *bias,
    __global half *C,
    const int M, const int K, const int N,
    const int H, const int K_b,
    const int has_bias) {
    const int s = get_global_id(1);
    const int n = get_global_id(0);
    const int bh_idx = get_global_id(2);
    const int b = bh_idx / H;
    const int h = bh_idx % H;

    if (s >= M || n >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        long a_idx = (long)b * M * H * K + (long)s * H * K + (long)h * K + k;
        long b_idx = (long)b * N * H * K + (long)n * H * K + (long)h * K + k;
        acc += (float)A[a_idx] * (float)B[b_idx];
    }

    if (has_bias != 0) {
        acc += bias[n];
    }
    long c_idx = (long)b * M * H * N + (long)s * H * N + (long)h * N + n;
    C[c_idx] = (half)acc;
}
#endif // SUPPORTS_FP16

// ==================================================================
// 3. FP32 * Q4_0 Fused GEMV + Bias Kernels (for M = 1, Decoding)
// ==================================================================
__kernel void gemv_fp32_q4_0_transb_bias(
    __global const float *A,
    __global const block_q4_0 *B,
    __global const float *bias,
    __global float *C,
    const int K, const int N,
    const int H,
    const int has_bias) {
    // [修正] 从 group ID 而非 global ID 获取索引，确保工作组内索引统一
    const int n = get_group_id(0);
    const int bh_idx = get_group_id(1);
    const int b = bh_idx / H;
    const int h = bh_idx % H;

    // 边界检查
    if (n >= N) return;

    const int local_id = get_local_id(0);
    const int wg_size = get_local_size(0);
    __local float partial_sums[256];

    float private_acc = 0.0f;
    const long a_base_idx = (long)b * H * K + (long)h * K;
    const long b_row_offset_blocks = (long)b * N * H * (K / QK4_0) + (long)n * H * (K / QK4_0) + (long)h * (K / QK4_0);

    // 并行计算，每个线程计算一部分
    for (int k = local_id; k < K; k += wg_size) {
        const int k_block_idx = k / QK4_0;
        const int k_in_block = k % QK4_0;

        const __global block_q4_0 *b_block_ptr = &B[b_row_offset_blocks + k_block_idx];
#if defined(SUPPORTS_FP16)
        const float d_b = vload_half(0, (__global half *)(&(b_block_ptr->d)));
#else
        const float d_b = (float)(b_block_ptr->d); // TODO Change here [gemini]
#endif
        // 反量化逻辑 (这部分与 gemm 内核一致，是正确的)
        const uchar q_packed = b_block_ptr->qs[k_in_block % 16];
        char q_nibble = (k_in_block < 16) ? (q_packed & 0x0F) : (q_packed >> 4);
        const float b_val = (float)(q_nibble - 8) * d_b;

        private_acc += A[a_base_idx + k] * b_val;
    }

    // 将各自的部分和存入局部内存
    partial_sums[local_id] = private_acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    // 在工作组内进行规约求和
    for (int offset = wg_size / 2; offset > 0; offset >>= 1) {
        if (local_id < offset) {
            partial_sums[local_id] += partial_sums[local_id + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // 由工作组的第一个线程写入最终结果
    if (local_id == 0) {
        float final_val = partial_sums[0];
        if (has_bias != 0) {
            final_val += bias[n];
        }
        const long c_idx = (long)b * H * N + (long)h * N + n;
        C[c_idx] = final_val;
    }
}
// ==================================================================
// 4. FP32 * Q4_0 Fused GEMM + Bias Kernels (for M > 1, Training)
// ==================================================================

__kernel void gemm_fp32_q4_0_transb_bias(
    __global const float *A,
    __global const block_q4_0 *B,
    __global const float *bias,
    __global float *C,
    const int M, const int K, const int N,
    const int H, const int K_b,
    const int has_bias) {
    // --- 1. 索引计算 (与原版保持一致) ---
    const int s = get_global_id(1);
    const int n = get_global_id(0);
    const int bh_idx = get_global_id(2);
    const int b = bh_idx / H;
    const int h = bh_idx % H;

    // --- 2. 边界检查 ---
    if (s >= M || n >= N) {
        return;
    }

    // --- 3. 基于寄存器的优化计算 ---

    float acc = 0.0f;

    const long a_row_offset = (long)b * M * H * K + (long)s * H * K + (long)h * K;
    const long b_row_offset_blocks = (long)b * N * H * (K / QK4_0) + (long)n * H * (K / QK4_0) + (long)h * (K / QK4_0);

    // 主循环：沿K维度进行，每次处理一个大小为 QK4_0 (32) 的块
    for (int k_block_idx = 0; k_block_idx < K / QK4_0; ++k_block_idx) {
        const __global block_q4_0 *b_block_ptr = &B[b_row_offset_blocks + k_block_idx];
        // const float d_b = vload_half(0, (__global half *)(&(b_block_ptr->d)));

#if defined(SUPPORTS_FP16)
        const float d_b = vload_half(0, (__global half *)(&(b_block_ptr->d)));
#else
        const float d_b = (float)(b_block_ptr->d); // TODO Change here [gemini]
#endif
        const __global float *a_ptr = A + a_row_offset + k_block_idx * QK4_0;

        // ** 向量化核心 **
        // 循环展开，每次处理块内16个 'uchar' 中的4个，对应8个浮点数
        for (int j = 0; j < QK4_0 / 2; j += 4) { // QK4_0/2 = 16

            // ** 修正点：安全地加载和解包B矩阵的值 **
            // 之前使用*((__global uint*))是不安全的，现改为逐字节加载
            const uchar q_packed0 = b_block_ptr->qs[j + 0];
            const uchar q_packed1 = b_block_ptr->qs[j + 1];
            const uchar q_packed2 = b_block_ptr->qs[j + 2];
            const uchar q_packed3 = b_block_ptr->qs[j + 3];

            // A矩阵的向量化加载是安全的
            const float4 a_vals_lo = vload4(0, a_ptr + j);
            const float4 a_vals_hi = vload4(0, a_ptr + j + (QK4_0 / 2)); // (QK4_0 / 2) = 16

            // 将4个uchar解包成两个float4向量
            float4 b_dequant_lo;
            b_dequant_lo.x = (float)((q_packed0 & 0x0F) - 8) * d_b; // qs[j] 的低4位
            b_dequant_lo.y = (float)((q_packed1 & 0x0F) - 8) * d_b; // qs[j+1] 的低4位
            b_dequant_lo.z = (float)((q_packed2 & 0x0F) - 8) * d_b; // qs[j+2] 的低4位
            b_dequant_lo.w = (float)((q_packed3 & 0x0F) - 8) * d_b; // qs[j+3] 的低4位

            float4 b_dequant_hi;
            b_dequant_hi.x = (float)((q_packed0 >> 4) - 8) * d_b; // qs[j] 的高4位
            b_dequant_hi.y = (float)((q_packed1 >> 4) - 8) * d_b; // qs[j+1] 的高4位
            b_dequant_hi.z = (float)((q_packed2 >> 4) - 8) * d_b; // qs[j+2] 的高4位
            b_dequant_hi.w = (float)((q_packed3 >> 4) - 8) * d_b; // qs[j+3] 的高4位

            // ** 核心计算 **
            acc += dot(a_vals_lo, b_dequant_lo);
            acc += dot(a_vals_hi, b_dequant_hi);
        }
    }

    if (has_bias != 0) {
        acc += bias[n];
    }

    const long c_idx = (long)b * M * H * N + (long)s * H * N + (long)h * N + n;
    C[c_idx] = acc;
}

// ==================================================================
// 5. FP16 * Q4_0 Fused GEMV + Bias Kernel (for M=1, Decoding)
// ==================================================================
#if defined(SUPPORTS_FP16)

// ---------- [高性能版 - 向量化 + 并行规约] ----------

__kernel void gemv_fp16_q4_0_transb_bias(
    __global const half *A,
    __global const block_q4_0 *B,
    __global const float *bias,
    __global half *C,
    const int K,
    const int N,
    const int H,
    const int has_bias) {
    const int n = get_group_id(0);
    const int bh_idx = get_group_id(1);
    const int b = bh_idx / H;
    const int h = bh_idx % H;
    if (n >= N) return;
    const int local_id = get_local_id(0);
    const int wg_size = get_local_size(0);
    __local float partial_sums[256];
    float private_acc = 0.0f;
    const long a_base_idx = (long)b * H * K + (long)h * K;
    const long b_row_offset_blocks = (long)b * N * H * (K / QK4_0) + (long)n * H * (K / QK4_0) + (long)h * (K / QK4_0);
    const int num_k_blocks = K / QK4_0;

    for (int k_block_idx = local_id; k_block_idx < num_k_blocks; k_block_idx += wg_size) {
        const __global block_q4_0 *b_block_ptr = &B[b_row_offset_blocks + k_block_idx];
        const float d_b = vload_half(0, (__global half *)(&(b_block_ptr->d)));
        const __global half *a_ptr = A + a_base_idx + k_block_idx * QK4_0;

#pragma unroll
        for (int j = 0; j < QK4_0 / 2; j += 4) { // j = 0, 4, 8, 12
            const uchar q_packed0 = b_block_ptr->qs[j + 0];
            const uchar q_packed1 = b_block_ptr->qs[j + 1];
            const uchar q_packed2 = b_block_ptr->qs[j + 2];
            const uchar q_packed3 = b_block_ptr->qs[j + 3];
            const int vec_offset_lo = j / 4;
            const int vec_offset_hi = j / 4 + 4;
            const float4 a_vals_lo = convert_float4(vload4(vec_offset_lo, a_ptr));
            const float4 a_vals_hi = convert_float4(vload4(vec_offset_hi, a_ptr));
            float4 b_dequant_lo, b_dequant_hi;
            b_dequant_lo.x = (float)((q_packed0 & 0x0F) - 8) * d_b;
            b_dequant_lo.y = (float)((q_packed1 & 0x0F) - 8) * d_b;
            b_dequant_lo.z = (float)((q_packed2 & 0x0F) - 8) * d_b;
            b_dequant_lo.w = (float)((q_packed3 & 0x0F) - 8) * d_b;
            b_dequant_hi.x = (float)((q_packed0 >> 4) - 8) * d_b;
            b_dequant_hi.y = (float)((q_packed1 >> 4) - 8) * d_b;
            b_dequant_hi.z = (float)((q_packed2 >> 4) - 8) * d_b;
            b_dequant_hi.w = (float)((q_packed3 >> 4) - 8) * d_b;
            private_acc += dot(a_vals_lo, b_dequant_lo);
            private_acc += dot(a_vals_hi, b_dequant_hi);
        }
    }
    partial_sums[local_id] = private_acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int offset = wg_size / 2; offset > 0; offset >>= 1) {
        if (local_id < offset) {
            partial_sums[local_id] += partial_sums[local_id + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0) {
        float final_val = partial_sums[0];
        if (has_bias != 0) {
            final_val += bias[n];
        }
        const long c_idx = (long)b * H * N + (long)h * N + n;
        vstore_half_rte(final_val, 0, &C[c_idx]);
    }
}

// #pragma OPENCL EXTENSION cl_khr_fp16 : enable

// 定义一个 half16 向量的横向求和辅助函数
inline float hsum_half16(half16 v) {
    half8 r1 = v.lo + v.hi;
    half4 r2 = r1.lo + r1.hi;
    half2 r3 = r2.lo + r2.hi;
    return (float)(r3.x + r3.y);
}

__kernel void gemv_fp16_q4_0_transb_bias_half16(
    __global const half *A,
    __global const block_q4_0 *B,
    __global const float *bias,
    __global half *C,
    const int K,
    const int N,
    const int H,
    const int has_bias) {
    const int n = get_group_id(0);
    const int bh_idx = get_group_id(1);
    const int b = bh_idx / H;
    const int h = bh_idx % H;
    if (n >= N) return;

    const int local_id = get_local_id(0);
    const int wg_size = get_local_size(0);

    // 使用 float 累加器保证精度（推荐）
    float private_acc = 0.0f;
    // 若要追求极致速度（有精度风险），可替换为下一行
    // half private_acc = 0.0h;

    // 使用 float 局部内存保证精度（推荐）
    __local float partial_sums[256];
    // 若要追求极致速度，可替换为下一行
    // __local half partial_sums[256];

    const long a_base_idx = (long)b * H * K + (long)h * K;
    const long b_row_offset_blocks = (long)b * N * H * (K / QK4_0) + (long)n * H * (K / QK4_0) + (long)h * (K / QK4_0);

    // K维度被切分为大小为 QK4_0 (32) 的块
    const int num_k_blocks = K / QK4_0;

    for (int k_block_idx = local_id; k_block_idx < num_k_blocks; k_block_idx += wg_size) {
        const __global block_q4_0 *b_block_ptr = &B[b_row_offset_blocks + k_block_idx];
        const half d_b = b_block_ptr->d;

        // QK4_0 = 32, a_ptr 指向一个32个half的块
        const __global half *a_ptr = A + a_base_idx + k_block_idx * QK4_0;

        // ** 优化核心：一次处理16个元素 **
        // 处理块的前半部分 (0-15)
        const half16 a_vals_lo = vload16(0, a_ptr);
        const uchar8 q_packed_lo = vload8(0, b_block_ptr->qs); // 加载8个uchar (包含16个4-bit值)

        // 高效解量化
        char16 b_s_lo;
        b_s_lo.s0 = (q_packed_lo.s0 & 0x0F) - 8;
        b_s_lo.s1 = (q_packed_lo.s0 >> 4) - 8;
        b_s_lo.s2 = (q_packed_lo.s1 & 0x0F) - 8;
        b_s_lo.s3 = (q_packed_lo.s1 >> 4) - 8;
        b_s_lo.s4 = (q_packed_lo.s2 & 0x0F) - 8;
        b_s_lo.s5 = (q_packed_lo.s2 >> 4) - 8;
        b_s_lo.s6 = (q_packed_lo.s3 & 0x0F) - 8;
        b_s_lo.s7 = (q_packed_lo.s3 >> 4) - 8;
        b_s_lo.s8 = (q_packed_lo.s4 & 0x0F) - 8;
        b_s_lo.s9 = (q_packed_lo.s4 >> 4) - 8;
        b_s_lo.sa = (q_packed_lo.s5 & 0x0F) - 8;
        b_s_lo.sb = (q_packed_lo.s5 >> 4) - 8;
        b_s_lo.sc = (q_packed_lo.s6 & 0x0F) - 8;
        b_s_lo.sd = (q_packed_lo.s6 >> 4) - 8;
        b_s_lo.se = (q_packed_lo.s7 & 0x0F) - 8;
        b_s_lo.sf = (q_packed_lo.s7 >> 4) - 8;

        const half16 b_vals_dequant_lo = convert_half16(b_s_lo) * d_b;
        private_acc += hsum_half16(a_vals_lo * b_vals_dequant_lo);

        // 处理块的后半部分 (16-31)
        const half16 a_vals_hi = vload16(0, a_ptr + 16);
        const uchar8 q_packed_hi = vload8(0, b_block_ptr->qs + 8); // 加载后8个uchar

        char16 b_s_hi;
        b_s_hi.s0 = (q_packed_hi.s0 & 0x0F) - 8;
        b_s_hi.s1 = (q_packed_hi.s0 >> 4) - 8;
        b_s_hi.s2 = (q_packed_hi.s1 & 0x0F) - 8;
        b_s_hi.s3 = (q_packed_hi.s1 >> 4) - 8;
        b_s_hi.s4 = (q_packed_hi.s2 & 0x0F) - 8;
        b_s_hi.s5 = (q_packed_hi.s2 >> 4) - 8;
        b_s_hi.s6 = (q_packed_hi.s3 & 0x0F) - 8;
        b_s_hi.s7 = (q_packed_hi.s3 >> 4) - 8;
        b_s_hi.s8 = (q_packed_hi.s4 & 0x0F) - 8;
        b_s_hi.s9 = (q_packed_hi.s4 >> 4) - 8;
        b_s_hi.sa = (q_packed_hi.s5 & 0x0F) - 8;
        b_s_hi.sb = (q_packed_hi.s5 >> 4) - 8;
        b_s_hi.sc = (q_packed_hi.s6 & 0x0F) - 8;
        b_s_hi.sd = (q_packed_hi.s6 >> 4) - 8;
        b_s_hi.se = (q_packed_hi.s7 & 0x0F) - 8;
        b_s_hi.sf = (q_packed_hi.s7 >> 4) - 8;

        const half16 b_vals_dequant_hi = convert_half16(b_s_hi) * d_b;
        private_acc += hsum_half16(a_vals_hi * b_vals_dequant_hi);
    }

    partial_sums[local_id] = private_acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    // 并行规约
    for (int offset = wg_size / 2; offset > 0; offset >>= 1) {
        if (local_id < offset) {
            partial_sums[local_id] += partial_sums[local_id + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // 第一个线程写入最终结果
    if (local_id == 0) {
        float final_val = partial_sums[0];
        if (has_bias != 0) {
            final_val += bias[n];
        }
        const long c_idx = (long)b * H * N + (long)h * N + n;
        vstore_half_rte(final_val, 0, &C[c_idx]);
    }
}

#else
// ---------- [兼容版 - 纯标量 + 并行规约] ----------
__kernel void gemv_fp16_q4_0_transb_bias(
    __global const half *A,
    __global const block_q4_0 *B,
    __global const float *bias,
    __global half *C,
    const int K,
    const int N,
    const int H,
    const int has_bias) {
    // --- 1. 索引计算 (与原版一致) ---
    const int n = get_group_id(0);
    const int bh_idx = get_group_id(1);
    const int b = bh_idx / H;
    const int h = bh_idx % H;
    if (n >= N) return;
    const int local_id = get_local_id(0);
    const int wg_size = get_local_size(0);
    __local float partial_sums[256];
    float private_acc = 0.0f;
    const long a_base_idx = (long)b * H * K + (long)h * K;
    const long b_row_offset_blocks = (long)b * N * H * (K / QK4_0) + (long)n * H * (K / QK4_0) + (long)h * (K / QK4_0);
    const int num_k_blocks = K / QK4_0;
    for (int k_block_idx = local_id; k_block_idx < num_k_blocks; k_block_idx += wg_size) {
        const __global block_q4_0 *b_block_ptr = &B[b_row_offset_blocks + k_block_idx];
        const float d_b = vload_half(0, (__global half *)(&(b_block_ptr->d)));
        const __global half *a_ptr = A + a_base_idx + k_block_idx * QK4_0;
#pragma unroll
        for (int j = 0; j < QK4_0 / 2; j += 4) {
            const uchar q_packed0 = b_block_ptr->qs[j + 0];
            const uchar q_packed1 = b_block_ptr->qs[j + 1];
            const uchar q_packed2 = b_block_ptr->qs[j + 2];
            const uchar q_packed3 = b_block_ptr->qs[j + 3];
            // 将vload4替换为手动的标量加载，以避免内存对齐问题。
            float4 a_vals_lo, a_vals_hi;
            a_vals_lo.x = (float)a_ptr[j + 0];
            a_vals_lo.y = (float)a_ptr[j + 1];
            a_vals_lo.z = (float)a_ptr[j + 2];
            a_vals_lo.w = (float)a_ptr[j + 3];
            a_vals_hi.x = (float)a_ptr[j + 16 + 0];
            a_vals_hi.y = (float)a_ptr[j + 16 + 1];
            a_vals_hi.z = (float)a_ptr[j + 16 + 2];
            a_vals_hi.w = (float)a_ptr[j + 16 + 3];
            float4 b_dequant_lo, b_dequant_hi;
            b_dequant_lo.x = (float)((q_packed0 & 0x0F) - 8) * d_b;
            b_dequant_lo.y = (float)((q_packed1 & 0x0F) - 8) * d_b;
            b_dequant_lo.z = (float)((q_packed2 & 0x0F) - 8) * d_b;
            b_dequant_lo.w = (float)((q_packed3 & 0x0F) - 8) * d_b;
            b_dequant_hi.x = (float)((q_packed0 >> 4) - 8) * d_b;
            b_dequant_hi.y = (float)((q_packed1 >> 4) - 8) * d_b;
            b_dequant_hi.z = (float)((q_packed2 >> 4) - 8) * d_b;
            b_dequant_hi.w = (float)((q_packed3 >> 4) - 8) * d_b;
            private_acc += dot(a_vals_lo, b_dequant_lo);
            private_acc += dot(a_vals_hi, b_dequant_hi);
        }
    }
    partial_sums[local_id] = private_acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int offset = wg_size / 2; offset > 0; offset >>= 1) {
        if (local_id < offset) {
            partial_sums[local_id] += partial_sums[local_id + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0) {
        float final_val = partial_sums[0];
        if (has_bias != 0) {
            final_val += bias[n];
        }
        const long c_idx = (long)b * H * N + (long)h * N + n;
        vstore_half_rte(final_val, 0, &C[c_idx]);
    }
}

#endif // SUPPORTS_FP16

// ==================================================================
// 6. FP16 * Q4_0 Fused GEMM + Bias Kernel (for M>1, Prefill)
// ==================================================================
#if defined(SUPPORTS_FP16)
// ---------- [高性能版 - Tiling + 寄存器] ----------

#define TILE_M 64
#define TILE_N 64
#define TILE_K 16
#define WPT_M 8
#define WPT_N 8
#define THREADS_X (TILE_N / WPT_N) // 8
#define THREADS_Y (TILE_M / WPT_M) // 8

__kernel void gemm_fp16_q4_0_transb_bias(
    __global const half *A,
    __global const block_q4_0 *B,
    __global const float *bias,
    __global half *C,
    const int M, const int K, const int N,
    const int H, const int K_b,
    const int has_bias) {
    const int group_m_idx = get_group_id(1);
    const int group_n_idx = get_group_id(0);
    const int local_m_idx = get_local_id(1);
    const int local_n_idx = get_local_id(0);
    const int bh_idx = get_global_id(2);
    const int b = bh_idx / H;
    const int h = bh_idx % H;
    __local half a_tile[TILE_M][TILE_K + 1];
    __local half b_tile[TILE_K][TILE_N + 1];
    float acc[WPT_M][WPT_N];
#pragma unroll
    for (int i = 0; i < WPT_M; ++i) {
#pragma unroll
        for (int j = 0; j < WPT_N; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    const long base_a_offset = (long)b * M * H * K + (long)h * K;
    const long base_b_offset_blocks = (long)b * N * H * (K / QK4_0) + (long)h * (K / QK4_0);
    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;
    for (int t = 0; t < num_k_tiles; ++t) {
        const int k_start = t * TILE_K;
#pragma unroll
        for (int i = 0; i < WPT_M; ++i) {
            const int m_local = local_m_idx * WPT_M + i;
            const int k_local = local_n_idx;
            const int m_global = group_m_idx * TILE_M + m_local;
            if (m_global < M) {
                for (int k_load_step = 0; k_load_step < TILE_K / THREADS_X; ++k_load_step) {
                    int k_global = k_start + k_local + k_load_step * THREADS_X;
                    if (k_global < K) {
                        a_tile[m_local][k_local + k_load_step * THREADS_X] = A[base_a_offset + m_global * K + k_global];
                    } else {
                        a_tile[m_local][k_local + k_load_step * THREADS_X] = 0.0h;
                    }
                }
            } else {
                for (int k_load_step = 0; k_load_step < TILE_K / THREADS_X; ++k_load_step) {
                    a_tile[m_local][k_local + k_load_step * THREADS_X] = 0.0h;
                }
            }
        }

#pragma unroll
        for (int i = 0; i < WPT_N; ++i) {
            const int n_local = local_n_idx * WPT_N + i;
            const int k_local = local_m_idx;
            const int n_global = group_n_idx * TILE_N + n_local;
            if (n_global < N) {
                for (int k_load_step = 0; k_load_step < TILE_K / THREADS_Y; ++k_load_step) {
                    int k_global = k_start + k_local + k_load_step * THREADS_Y;
                    if (k_global < K) {
                        const int k_block_idx = k_global / QK4_0;
                        const int k_in_block = k_global % QK4_0;
                        const __global block_q4_0 *b_block_ptr = &B[base_b_offset_blocks + n_global * (K / QK4_0) + k_block_idx];

                        const float d_b = vload_half(0, (__global half *)(&(b_block_ptr->d)));

                        // --- **修正核心** ---
                        // 1. 根据 k_in_block (0-31) 计算在 qs 数组 (0-15) 中的索引
                        const uchar qs_sub_idx = k_in_block % 16;
                        // 2. 从 qs 数组中取出打包好的两个 4-bit 值
                        const uchar q_packed = b_block_ptr->qs[qs_sub_idx];
                        // 3. 判断 k_in_block 是在块的前半部分还是后半部分，来决定是取高4位还是低4位
                        const bool is_low_nibble = (k_in_block < 16);
                        char q_nibble = is_low_nibble ? ((q_packed & 0x0F) - 8) : ((q_packed >> 4) - 8);

                        // 4. 解量化并存入 b_tile
                        b_tile[k_local + k_load_step * THREADS_Y][n_local] = (half)((float)q_nibble * d_b);

                    } else {
                        b_tile[k_local + k_load_step * THREADS_Y][n_local] = 0.0h;
                    }
                }
            } else {
                for (int k_load_step = 0; k_load_step < TILE_K / THREADS_Y; ++k_load_step) {
                    b_tile[k_local + k_load_step * THREADS_Y][n_local] = 0.0h;
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
        for (int k_tile = 0; k_tile < TILE_K; ++k_tile) {
#pragma unroll
            for (int m = 0; m < WPT_M; ++m) {
                half a_val = a_tile[local_m_idx * WPT_M + m][k_tile];
#pragma unroll
                for (int n = 0; n < WPT_N; ++n) {
                    half b_val = b_tile[k_tile][local_n_idx * WPT_N + n];
                    acc[m][n] = mad((float)a_val, (float)b_val, acc[m][n]);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    long c_offset = (long)b * M * H * N + (long)h * N;
#pragma unroll
    for (int m = 0; m < WPT_M; ++m) {
        int m_global = group_m_idx * TILE_M + local_m_idx * WPT_M + m;
        if (m_global < M) {
#pragma unroll
            for (int n = 0; n < WPT_N; ++n) {
                int n_global = group_n_idx * TILE_N + local_n_idx * WPT_N + n;
                if (n_global < N) {
                    float result = acc[m][n];
                    if (has_bias) {
                        result += bias[n_global];
                    }
                    C[c_offset + m_global * N + n_global] = (half)result;
                }
            }
        }
    }
}

#else
// ---------- [兼容版] ----------
__kernel void gemm_fp16_q4_0_transb_bias(
    __global const half *A,
    __global const block_q4_0 *B,
    __global const float *bias,
    __global half *C,
    const int M, const int K, const int N,
    const int H, const int K_b,
    const int has_bias) {
    // --- 1. 索引计算 ---
    const int s = get_global_id(1);
    const int n = get_global_id(0);
    const int bh_idx = get_global_id(2);
    const int b = bh_idx / H;
    const int h = bh_idx % H;

    // --- 2. 边界检查 ---
    if (s >= M || n >= N) {
        return;
    }

    // --- 3. 核心计算 ---
    float acc = 0.0f;
    const long a_row_offset = (long)b * M * H * K + (long)s * H * K + (long)h * K;
    const long b_row_offset_blocks = (long)b * N * H * (K / QK4_0) + (long)n * H * (K / QK4_0) + (long)h * (K / QK4_0);

    for (int k_block_idx = 0; k_block_idx < K / QK4_0; ++k_block_idx) {
        const __global block_q4_0 *b_block_ptr = &B[b_row_offset_blocks + k_block_idx];
        // 使用 vload_half 确保从 half* 安全加载到 float
        const float d_b = vload_half(0, (__global half *)(&(b_block_ptr->d)));
        const __global half *a_ptr = A + a_row_offset + k_block_idx * QK4_0;

        for (int j = 0; j < QK4_0 / 2; ++j) {
            const uchar q_packed = b_block_ptr->qs[j];
            const char q_lo = (q_packed & 0x0F) - 8;
            const char q_hi = (q_packed >> 4) - 8;

            acc += vload_half(j, a_ptr) * (float)q_lo * d_b;
            acc += vload_half(j + QK4_0 / 2, a_ptr) * (float)q_hi * d_b;
        }
    }

    // --- 4. 偏置和写回 ---
    if (has_bias != 0) {
        acc += bias[n];
    }
    const long c_idx = (long)b * M * H * N + (long)s * H * N + (long)h * N + n;

    // 使用 vstore_half_rte 进行精确舍入并存储
    // vstore_half_rte 将 float 类型的 acc 舍入为最接近的 half 值，并将结果存入 C 的 c_idx 位置
    vstore_half_rte(acc, 0, &C[c_idx]);
}
#endif // SUPPORTS_FP16

// ==================================================================
// 7. FP32 * Q4_0 Fused GEMM + Bias Kernel (Image Pipe)
//    [最终修正版 - 动态解量化 & 修复索引]
// ==================================================================
__kernel void gemm_fp32_q4_0_transb_bias_image_pipe(
    sampler_t sampler,
    __read_only image2d_t A,
    __global const block_q4_0 *B,
    __global const float *bias,
    __write_only image2d_t C,
    const int M, const int K, const int N,
    const int H, const int K_b,
    const int has_bias) {
    // 1. 索引计算
    const int gx = get_global_id(0);
    const int gy = get_global_id(1);
    const int n_start = gx * 4;
    if (n_start >= N) { return; }

    float4 acc = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    const int K_blocks = K / 32;
    const long base_offset = 0; // B=1, H=1
    const long n_stride = K_blocks;

    // 3. 沿 K 维度循环计算
    for (int k = 0; k < K; ++k) {
        // ** [核心修正] **
        // a. 正确地从 Image A 中读取 FP32 值
        int pixel_x = k / 4;
        int component = k % 4;
        float4 a_pixel = read_imagef(A, sampler, (int2)(pixel_x, gy));
        float a_val;
        if (component == 0)
            a_val = a_pixel.x;
        else if (component == 1)
            a_val = a_pixel.y;
        else if (component == 2)
            a_val = a_pixel.z;
        else
            a_val = a_pixel.w;

        // b. 实时解量化 4 个 B 值 (这部分逻辑是正确的)
        const int k_block_idx = k / 32;
        const int k_in_block = k % 32;
        const long b_block_mem_idx0 = base_offset + (long)(n_start + 0) * n_stride + k_block_idx;
        const long b_block_mem_idx1 = base_offset + (long)(n_start + 1) * n_stride + k_block_idx;
        const long b_block_mem_idx2 = base_offset + (long)(n_start + 2) * n_stride + k_block_idx;
        const long b_block_mem_idx3 = base_offset + (long)(n_start + 3) * n_stride + k_block_idx;

        const float d_b0 = vload_half(0, (__global half *)(&(B[b_block_mem_idx0].d)));
        const float d_b1 = vload_half(0, (__global half *)(&(B[b_block_mem_idx1].d)));
        const float d_b2 = vload_half(0, (__global half *)(&(B[b_block_mem_idx2].d)));
        const float d_b3 = vload_half(0, (__global half *)(&(B[b_block_mem_idx3].d)));

        const uchar q_packed0 = B[b_block_mem_idx0].qs[k_in_block % 16];
        const uchar q_packed1 = B[b_block_mem_idx1].qs[k_in_block % 16];
        const uchar q_packed2 = B[b_block_mem_idx2].qs[k_in_block % 16];
        const uchar q_packed3 = B[b_block_mem_idx3].qs[k_in_block % 16];

        const char q_nibble0 = (k_in_block < 16) ? ((q_packed0 & 0x0F) - 8) : ((q_packed0 >> 4) - 8);
        const char q_nibble1 = (k_in_block < 16) ? ((q_packed1 & 0x0F) - 8) : ((q_packed1 >> 4) - 8);
        const char q_nibble2 = (k_in_block < 16) ? ((q_packed2 & 0x0F) - 8) : ((q_packed2 >> 4) - 8);
        const char q_nibble3 = (k_in_block < 16) ? ((q_packed3 & 0x0F) - 8) : ((q_packed3 >> 4) - 8);

        float4 b_vals = (float4)((float)q_nibble0 * d_b0, (float)q_nibble1 * d_b1, (float)q_nibble2 * d_b2, (float)q_nibble3 * d_b3);

        acc = mad(a_val, b_vals, acc);
    }

    if (has_bias != 0) {
        float4 bias_vals = vload4(0, bias + n_start);
        acc += bias_vals;
    }

    write_imagef(C, (int2)(gx, gy), acc);
}

// ==================================================================
// 8. FP16 * Q4_0 Fused GEMM + Bias Kernel (Image Pipe)
// ==================================================================
#if defined(SUPPORTS_FP16)
/*
__kernel void gemm_fp16_q4_0_transb_bias_image_pipe(
    sampler_t sampler,
    __read_only image2d_t A,
    __global const block_q4_0 *B,
    __global const float *bias,
    __write_only image2d_t C,
    const int K, const int N,
    const int H,
    const int has_bias) {
    // 1. 并行策略: 每个工作项计算一个输出像素 (float4)
    const int gx = get_global_id(0);
    const int n_start = gx * 4;
    if (n_start >= N) { return; }

    // 2. 寄存器累加器
    float4 acc = (float4)(0.0f);
    const int K_blocks = K / 32;
    const long n_stride = K_blocks;
    const int K_vec_size = K / 4;

    // 3. 核心计算循环：沿K维度4倍展开, 深度优化指令级并行
    for (int k_vec = 0; k_vec < K_vec_size; ++k_vec) {
        const int k = k_vec * 4;

        // a. 向量化读取A
        half4 a_pixel_h = read_imageh(A, sampler, (int2)(k_vec, 0));
        float4 a_vals = convert_float4(a_pixel_h);

        // b. 为 a_vals 的每个分量并行准备权重向量，打破依赖
        // --- 解量化与 a_vals.x (k+0) 相乘的 b_vals_k0 ---
        const int k0_block_idx = (k + 0) / 32;
        const int k0_in_block = (k + 0) % 32;
        const long b_idx0_k0 = (long)(n_start + 0) * n_stride + k0_block_idx;
        const long b_idx1_k0 = (long)(n_start + 1) * n_stride + k0_block_idx;
        const long b_idx2_k0 = (long)(n_start + 2) * n_stride + k0_block_idx;
        const long b_idx3_k0 = (long)(n_start + 3) * n_stride + k0_block_idx;
        const float d0_k0 = vload_half(0, (__global half *)(&(B[b_idx0_k0].d)));
        const float d1_k0 = vload_half(0, (__global half *)(&(B[b_idx1_k0].d)));
        const float d2_k0 = vload_half(0, (__global half *)(&(B[b_idx2_k0].d)));
        const float d3_k0 = vload_half(0, (__global half *)(&(B[b_idx3_k0].d)));
        const uchar qp0_k0 = B[b_idx0_k0].qs[k0_in_block % 16];
        const uchar qp1_k0 = B[b_idx1_k0].qs[k0_in_block % 16];
        const uchar qp2_k0 = B[b_idx2_k0].qs[k0_in_block % 16];
        const uchar qp3_k0 = B[b_idx3_k0].qs[k0_in_block % 16];
        const char qn0_k0 = (k0_in_block < 16) ? ((qp0_k0 & 0x0F) - 8) : ((qp0_k0 >> 4) - 8);
        const char qn1_k0 = (k0_in_block < 16) ? ((qp1_k0 & 0x0F) - 8) : ((qp1_k0 >> 4) - 8);
        const char qn2_k0 = (k0_in_block < 16) ? ((qp2_k0 & 0x0F) - 8) : ((qp2_k0 >> 4) - 8);
        const char qn3_k0 = (k0_in_block < 16) ? ((qp3_k0 & 0x0F) - 8) : ((qp3_k0 >> 4) - 8);
        float4 b_vals_k0 = (float4)((float)qn0_k0 * d0_k0, (float)qn1_k0 * d1_k0, (float)qn2_k0 * d2_k0, (float)qn3_k0 * d3_k0);

        // --- 解量化与 a_vals.y (k+1) 相乘的 b_vals_k1 ---
        const int k1_block_idx = (k + 1) / 32;
        const int k1_in_block = (k + 1) % 32;
        const long b_idx0_k1 = (long)(n_start + 0) * n_stride + k1_block_idx;
        const long b_idx1_k1 = (long)(n_start + 1) * n_stride + k1_block_idx;
        const long b_idx2_k1 = (long)(n_start + 2) * n_stride + k1_block_idx;
        const long b_idx3_k1 = (long)(n_start + 3) * n_stride + k1_block_idx;
        const float d0_k1 = vload_half(0, (__global half *)(&(B[b_idx0_k1].d)));
        const float d1_k1 = vload_half(0, (__global half *)(&(B[b_idx1_k1].d)));
        const float d2_k1 = vload_half(0, (__global half *)(&(B[b_idx2_k1].d)));
        const float d3_k1 = vload_half(0, (__global half *)(&(B[b_idx3_k1].d)));
        const uchar qp0_k1 = B[b_idx0_k1].qs[k1_in_block % 16];
        const uchar qp1_k1 = B[b_idx1_k1].qs[k1_in_block % 16];
        const uchar qp2_k1 = B[b_idx2_k1].qs[k1_in_block % 16];
        const uchar qp3_k1 = B[b_idx3_k1].qs[k1_in_block % 16];
        const char qn0_k1 = (k1_in_block < 16) ? ((qp0_k1 & 0x0F) - 8) : ((qp0_k1 >> 4) - 8);
        const char qn1_k1 = (k1_in_block < 16) ? ((qp1_k1 & 0x0F) - 8) : ((qp1_k1 >> 4) - 8);
        const char qn2_k1 = (k1_in_block < 16) ? ((qp2_k1 & 0x0F) - 8) : ((qp2_k1 >> 4) - 8);
        const char qn3_k1 = (k1_in_block < 16) ? ((qp3_k1 & 0x0F) - 8) : ((qp3_k1 >> 4) - 8);
        float4 b_vals_k1 = (float4)((float)qn0_k1 * d0_k1, (float)qn1_k1 * d1_k1, (float)qn2_k1 * d2_k1, (float)qn3_k1 * d3_k1);

        // --- 解量化与 a_vals.z (k+2) 相乘的 b_vals_k2 ---
        const int k2_block_idx = (k + 2) / 32;
        const int k2_in_block = (k + 2) % 32;
        const long b_idx0_k2 = (long)(n_start + 0) * n_stride + k2_block_idx;
        const long b_idx1_k2 = (long)(n_start + 1) * n_stride + k2_block_idx;
        const long b_idx2_k2 = (long)(n_start + 2) * n_stride + k2_block_idx;
        const long b_idx3_k2 = (long)(n_start + 3) * n_stride + k2_block_idx;
        const float d0_k2 = vload_half(0, (__global half *)(&(B[b_idx0_k2].d)));
        const float d1_k2 = vload_half(0, (__global half *)(&(B[b_idx1_k2].d)));
        const float d2_k2 = vload_half(0, (__global half *)(&(B[b_idx2_k2].d)));
        const float d3_k2 = vload_half(0, (__global half *)(&(B[b_idx3_k2].d)));
        const uchar qp0_k2 = B[b_idx0_k2].qs[k2_in_block % 16];
        const uchar qp1_k2 = B[b_idx1_k2].qs[k2_in_block % 16];
        const uchar qp2_k2 = B[b_idx2_k2].qs[k2_in_block % 16];
        const uchar qp3_k2 = B[b_idx3_k2].qs[k2_in_block % 16];
        const char qn0_k2 = (k2_in_block < 16) ? ((qp0_k2 & 0x0F) - 8) : ((qp0_k2 >> 4) - 8);
        const char qn1_k2 = (k2_in_block < 16) ? ((qp1_k2 & 0x0F) - 8) : ((qp1_k2 >> 4) - 8);
        const char qn2_k2 = (k2_in_block < 16) ? ((qp2_k2 & 0x0F) - 8) : ((qp2_k2 >> 4) - 8);
        const char qn3_k2 = (k2_in_block < 16) ? ((qp3_k2 & 0x0F) - 8) : ((qp3_k2 >> 4) - 8);
        float4 b_vals_k2 = (float4)((float)qn0_k2 * d0_k2, (float)qn1_k2 * d1_k2, (float)qn2_k2 * d2_k2, (float)qn3_k2 * d3_k2);

        // --- 解量化与 a_vals.w (k+3) 相乘的 b_vals_k3 ---
        const int k3_block_idx = (k + 3) / 32;
        const int k3_in_block = (k + 3) % 32;
        const long b_idx0_k3 = (long)(n_start + 0) * n_stride + k3_block_idx;
        const long b_idx1_k3 = (long)(n_start + 1) * n_stride + k3_block_idx;
        const long b_idx2_k3 = (long)(n_start + 2) * n_stride + k3_block_idx;
        const long b_idx3_k3 = (long)(n_start + 3) * n_stride + k3_block_idx;
        const float d0_k3 = vload_half(0, (__global half *)(&(B[b_idx0_k3].d)));
        const float d1_k3 = vload_half(0, (__global half *)(&(B[b_idx1_k3].d)));
        const float d2_k3 = vload_half(0, (__global half *)(&(B[b_idx2_k3].d)));
        const float d3_k3 = vload_half(0, (__global half *)(&(B[b_idx3_k3].d)));
        const uchar qp0_k3 = B[b_idx0_k3].qs[k3_in_block % 16];
        const uchar qp1_k3 = B[b_idx1_k3].qs[k3_in_block % 16];
        const uchar qp2_k3 = B[b_idx2_k3].qs[k3_in_block % 16];
        const uchar qp3_k3 = B[b_idx3_k3].qs[k3_in_block % 16];
        const char qn0_k3 = (k3_in_block < 16) ? ((qp0_k3 & 0x0F) - 8) : ((qp0_k3 >> 4) - 8);
        const char qn1_k3 = (k3_in_block < 16) ? ((qp1_k3 & 0x0F) - 8) : ((qp1_k3 >> 4) - 8);
        const char qn2_k3 = (k3_in_block < 16) ? ((qp2_k3 & 0x0F) - 8) : ((qp2_k3 >> 4) - 8);
        const char qn3_k3 = (k3_in_block < 16) ? ((qp3_k3 & 0x0F) - 8) : ((qp3_k3 >> 4) - 8);
        float4 b_vals_k3 = (float4)((float)qn0_k3 * d0_k3, (float)qn1_k3 * d1_k3, (float)qn2_k3 * d2_k3, (float)qn3_k3 * d3_k3);

        // c. 并行执行 4次独立的MAD指令
        acc = mad(a_vals.x, b_vals_k0, acc);
        acc = mad(a_vals.y, b_vals_k1, acc);
        acc = mad(a_vals.z, b_vals_k2, acc);
        acc = mad(a_vals.w, b_vals_k3, acc);
    }

    // 4. 扫尾处理: K不是4的倍数时
    for (int k = K_vec_size * 4; k < K; ++k) {
        int pixel_x = k / 4;
        int component = k % 4;
        half4 a_pixel = read_imageh(A, sampler, (int2)(pixel_x, 0));
        float a_val;
        if (component == 0)
            a_val = (float)a_pixel.x;
        else if (component == 1)
            a_val = (float)a_pixel.y;
        else if (component == 2)
            a_val = (float)a_pixel.z;
        else
            a_val = (float)a_pixel.w;

        const int k_block_idx = k / 32;
        const int k_in_block = k % 32;
        const long b_idx0 = (long)(n_start + 0) * n_stride + k_block_idx;
        const long b_idx1 = (long)(n_start + 1) * n_stride + k_block_idx;
        const long b_idx2 = (long)(n_start + 2) * n_stride + k_block_idx;
        const long b_idx3 = (long)(n_start + 3) * n_stride + k_block_idx;
        const float d0 = vload_half(0, (__global half *)(&(B[b_idx0].d)));
        const float d1 = vload_half(0, (__global half *)(&(B[b_idx1].d)));
        const float d2 = vload_half(0, (__global half *)(&(B[b_idx2].d)));
        const float d3 = vload_half(0, (__global half *)(&(B[b_idx3].d)));
        const uchar qp0 = B[b_idx0].qs[k_in_block % 16];
        const uchar qp1 = B[b_idx1].qs[k_in_block % 16];
        const uchar qp2 = B[b_idx2].qs[k_in_block % 16];
        const uchar qp3 = B[b_idx3].qs[k_in_block % 16];
        const char qn0 = (k_in_block < 16) ? ((qp0 & 0x0F) - 8) : ((qp0 >> 4) - 8);
        const char qn1 = (k_in_block < 16) ? ((qp1 & 0x0F) - 8) : ((qp1 >> 4) - 8);
        const char qn2 = (k_in_block < 16) ? ((qp2 & 0x0F) - 8) : ((qp2 >> 4) - 8);
        const char qn3 = (k_in_block < 16) ? ((qp3 & 0x0F) - 8) : ((qp3 >> 4) - 8);
        float4 b_vals = (float4)((float)qn0 * d0, (float)qn1 * d1, (float)qn2 * d2, (float)qn3 * d3);
        acc = mad(a_val, b_vals, acc);
    }

    // 5. 添加偏置
    if (has_bias != 0) {
        acc += vload4(0, bias + n_start);
    }

    // 6. 将结果像素写入输出 Image C
    write_imageh(C, (int2)(gx, 0), convert_half4_rte(acc));
}
*/
__kernel void gemm_fp16_q4_0_transb_bias_image_pipe(
    sampler_t sampler,
    __read_only image2d_t A,
    __global const block_q4_0 *B,
    __global const float *bias,
    __write_only image2d_t C,
    const int M, const int K, const int N,
    const int H, const int K_b,
    const int has_bias) {
    const int gx = get_global_id(0);
    const int gy = get_global_id(1); // gy 对应 M 维度
    const int n_start = gx * 4;

    if (n_start >= N || gy >= M) { return; }

    half4 acc = (half4)(0.0h);

    const int K_blocks = K / 32;
    const long n_stride = K_blocks;
    const int K_vec_size_x8 = K / 8;

    for (int k_vec = 0; k_vec < K_vec_size_x8; ++k_vec) {
        const int k = k_vec * 8;

        // **核心区别**: 读取A时，y坐标使用 gy
        half4 a_vals_lo = read_imageh(A, sampler, (int2)(k / 4, gy));
        half4 a_vals_hi = read_imageh(A, sampler, (int2)(k / 4 + 1, gy));

        // --- 8倍展开的解量化与计算（代码同GEMV版，此处为简洁省略）---
        // --- k+0 ---
        const int k0_block_idx = (k + 0) / 32;
        const int k0_in_block = (k + 0) % 32;
        const long b_idx0_k0 = (long)(n_start + 0) * n_stride + k0_block_idx;
        const long b_idx1_k0 = (long)(n_start + 1) * n_stride + k0_block_idx;
        const long b_idx2_k0 = (long)(n_start + 2) * n_stride + k0_block_idx;
        const long b_idx3_k0 = (long)(n_start + 3) * n_stride + k0_block_idx;
        const half d0_k0 = vload_half(0, (__global half *)(&(B[b_idx0_k0].d)));
        const half d1_k0 = vload_half(0, (__global half *)(&(B[b_idx1_k0].d)));
        const half d2_k0 = vload_half(0, (__global half *)(&(B[b_idx2_k0].d)));
        const half d3_k0 = vload_half(0, (__global half *)(&(B[b_idx3_k0].d)));
        const uchar qp0_k0 = B[b_idx0_k0].qs[k0_in_block % 16];
        const uchar qp1_k0 = B[b_idx1_k0].qs[k0_in_block % 16];
        const uchar qp2_k0 = B[b_idx2_k0].qs[k0_in_block % 16];
        const uchar qp3_k0 = B[b_idx3_k0].qs[k0_in_block % 16];
        const half qn0_k0 = (half)((k0_in_block < 16) ? ((qp0_k0 & 0x0F) - 8) : ((qp0_k0 >> 4) - 8));
        const half qn1_k0 = (half)((k0_in_block < 16) ? ((qp1_k0 & 0x0F) - 8) : ((qp1_k0 >> 4) - 8));
        const half qn2_k0 = (half)((k0_in_block < 16) ? ((qp2_k0 & 0x0F) - 8) : ((qp2_k0 >> 4) - 8));
        const half qn3_k0 = (half)((k0_in_block < 16) ? ((qp3_k0 & 0x0F) - 8) : ((qp3_k0 >> 4) - 8));
        half4 b_vals_k0 = (half4)(qn0_k0 * d0_k0, qn1_k0 * d1_k0, qn2_k0 * d2_k0, qn3_k0 * d3_k0);
        // ... k+1 到 k+7 的代码 ...
        const int k1_block_idx = (k + 1) / 32;
        const int k1_in_block = (k + 1) % 32;
        const long b_idx0_k1 = (long)(n_start + 0) * n_stride + k1_block_idx;
        const long b_idx1_k1 = (long)(n_start + 1) * n_stride + k1_block_idx;
        const long b_idx2_k1 = (long)(n_start + 2) * n_stride + k1_block_idx;
        const long b_idx3_k1 = (long)(n_start + 3) * n_stride + k1_block_idx;
        const half d0_k1 = vload_half(0, (__global half *)(&(B[b_idx0_k1].d)));
        const half d1_k1 = vload_half(0, (__global half *)(&(B[b_idx1_k1].d)));
        const half d2_k1 = vload_half(0, (__global half *)(&(B[b_idx2_k1].d)));
        const half d3_k1 = vload_half(0, (__global half *)(&(B[b_idx3_k1].d)));
        const uchar qp0_k1 = B[b_idx0_k1].qs[k1_in_block % 16];
        const uchar qp1_k1 = B[b_idx1_k1].qs[k1_in_block % 16];
        const uchar qp2_k1 = B[b_idx2_k1].qs[k1_in_block % 16];
        const uchar qp3_k1 = B[b_idx3_k1].qs[k1_in_block % 16];
        const half qn0_k1 = (half)((k1_in_block < 16) ? ((qp0_k1 & 0x0F) - 8) : ((qp0_k1 >> 4) - 8));
        const half qn1_k1 = (half)((k1_in_block < 16) ? ((qp1_k1 & 0x0F) - 8) : ((qp1_k1 >> 4) - 8));
        const half qn2_k1 = (half)((k1_in_block < 16) ? ((qp2_k1 & 0x0F) - 8) : ((qp2_k1 >> 4) - 8));
        const half qn3_k1 = (half)((k1_in_block < 16) ? ((qp3_k1 & 0x0F) - 8) : ((qp3_k1 >> 4) - 8));
        half4 b_vals_k1 = (half4)(qn0_k1 * d0_k1, qn1_k1 * d1_k1, qn2_k1 * d2_k1, qn3_k1 * d3_k1);
        const int k2_block_idx = (k + 2) / 32;
        const int k2_in_block = (k + 2) % 32;
        const long b_idx0_k2 = (long)(n_start + 0) * n_stride + k2_block_idx;
        const long b_idx1_k2 = (long)(n_start + 1) * n_stride + k2_block_idx;
        const long b_idx2_k2 = (long)(n_start + 2) * n_stride + k2_block_idx;
        const long b_idx3_k2 = (long)(n_start + 3) * n_stride + k2_block_idx;
        const half d0_k2 = vload_half(0, (__global half *)(&(B[b_idx0_k2].d)));
        const half d1_k2 = vload_half(0, (__global half *)(&(B[b_idx1_k2].d)));
        const half d2_k2 = vload_half(0, (__global half *)(&(B[b_idx2_k2].d)));
        const half d3_k2 = vload_half(0, (__global half *)(&(B[b_idx3_k2].d)));
        const uchar qp0_k2 = B[b_idx0_k2].qs[k2_in_block % 16];
        const uchar qp1_k2 = B[b_idx1_k2].qs[k2_in_block % 16];
        const uchar qp2_k2 = B[b_idx2_k2].qs[k2_in_block % 16];
        const uchar qp3_k2 = B[b_idx3_k2].qs[k2_in_block % 16];
        const half qn0_k2 = (half)((k2_in_block < 16) ? ((qp0_k2 & 0x0F) - 8) : ((qp0_k2 >> 4) - 8));
        const half qn1_k2 = (half)((k2_in_block < 16) ? ((qp1_k2 & 0x0F) - 8) : ((qp1_k2 >> 4) - 8));
        const half qn2_k2 = (half)((k2_in_block < 16) ? ((qp2_k2 & 0x0F) - 8) : ((qp2_k2 >> 4) - 8));
        const half qn3_k2 = (half)((k2_in_block < 16) ? ((qp3_k2 & 0x0F) - 8) : ((qp3_k2 >> 4) - 8));
        half4 b_vals_k2 = (half4)(qn0_k2 * d0_k2, qn1_k2 * d1_k2, qn2_k2 * d2_k2, qn3_k2 * d3_k2);
        const int k3_block_idx = (k + 3) / 32;
        const int k3_in_block = (k + 3) % 32;
        const long b_idx0_k3 = (long)(n_start + 0) * n_stride + k3_block_idx;
        const long b_idx1_k3 = (long)(n_start + 1) * n_stride + k3_block_idx;
        const long b_idx2_k3 = (long)(n_start + 2) * n_stride + k3_block_idx;
        const long b_idx3_k3 = (long)(n_start + 3) * n_stride + k3_block_idx;
        const half d0_k3 = vload_half(0, (__global half *)(&(B[b_idx0_k3].d)));
        const half d1_k3 = vload_half(0, (__global half *)(&(B[b_idx1_k3].d)));
        const half d2_k3 = vload_half(0, (__global half *)(&(B[b_idx2_k3].d)));
        const half d3_k3 = vload_half(0, (__global half *)(&(B[b_idx3_k3].d)));
        const uchar qp0_k3 = B[b_idx0_k3].qs[k3_in_block % 16];
        const uchar qp1_k3 = B[b_idx1_k3].qs[k3_in_block % 16];
        const uchar qp2_k3 = B[b_idx2_k3].qs[k3_in_block % 16];
        const uchar qp3_k3 = B[b_idx3_k3].qs[k3_in_block % 16];
        const half qn0_k3 = (half)((k3_in_block < 16) ? ((qp0_k3 & 0x0F) - 8) : ((qp0_k3 >> 4) - 8));
        const half qn1_k3 = (half)((k3_in_block < 16) ? ((qp1_k3 & 0x0F) - 8) : ((qp1_k3 >> 4) - 8));
        const half qn2_k3 = (half)((k3_in_block < 16) ? ((qp2_k3 & 0x0F) - 8) : ((qp2_k3 >> 4) - 8));
        const half qn3_k3 = (half)((k3_in_block < 16) ? ((qp3_k3 & 0x0F) - 8) : ((qp3_k3 >> 4) - 8));
        half4 b_vals_k3 = (half4)(qn0_k3 * d0_k3, qn1_k3 * d1_k3, qn2_k3 * d2_k3, qn3_k3 * d3_k3);
        const int k4_block_idx = (k + 4) / 32;
        const int k4_in_block = (k + 4) % 32;
        const long b_idx0_k4 = (long)(n_start + 0) * n_stride + k4_block_idx;
        const long b_idx1_k4 = (long)(n_start + 1) * n_stride + k4_block_idx;
        const long b_idx2_k4 = (long)(n_start + 2) * n_stride + k4_block_idx;
        const long b_idx3_k4 = (long)(n_start + 3) * n_stride + k4_block_idx;
        const half d0_k4 = vload_half(0, (__global half *)(&(B[b_idx0_k4].d)));
        const half d1_k4 = vload_half(0, (__global half *)(&(B[b_idx1_k4].d)));
        const half d2_k4 = vload_half(0, (__global half *)(&(B[b_idx2_k4].d)));
        const half d3_k4 = vload_half(0, (__global half *)(&(B[b_idx3_k4].d)));
        const uchar qp0_k4 = B[b_idx0_k4].qs[k4_in_block % 16];
        const uchar qp1_k4 = B[b_idx1_k4].qs[k4_in_block % 16];
        const uchar qp2_k4 = B[b_idx2_k4].qs[k4_in_block % 16];
        const uchar qp3_k4 = B[b_idx3_k4].qs[k4_in_block % 16];
        const half qn0_k4 = (half)((k4_in_block < 16) ? ((qp0_k4 & 0x0F) - 8) : ((qp0_k4 >> 4) - 8));
        const half qn1_k4 = (half)((k4_in_block < 16) ? ((qp1_k4 & 0x0F) - 8) : ((qp1_k4 >> 4) - 8));
        const half qn2_k4 = (half)((k4_in_block < 16) ? ((qp2_k4 & 0x0F) - 8) : ((qp2_k4 >> 4) - 8));
        const half qn3_k4 = (half)((k4_in_block < 16) ? ((qp3_k4 & 0x0F) - 8) : ((qp3_k4 >> 4) - 8));
        half4 b_vals_k4 = (half4)(qn0_k4 * d0_k4, qn1_k4 * d1_k4, qn2_k4 * d2_k4, qn3_k4 * d3_k4);
        const int k5_block_idx = (k + 5) / 32;
        const int k5_in_block = (k + 5) % 32;
        const long b_idx0_k5 = (long)(n_start + 0) * n_stride + k5_block_idx;
        const long b_idx1_k5 = (long)(n_start + 1) * n_stride + k5_block_idx;
        const long b_idx2_k5 = (long)(n_start + 2) * n_stride + k5_block_idx;
        const long b_idx3_k5 = (long)(n_start + 3) * n_stride + k5_block_idx;
        const half d0_k5 = vload_half(0, (__global half *)(&(B[b_idx0_k5].d)));
        const half d1_k5 = vload_half(0, (__global half *)(&(B[b_idx1_k5].d)));
        const half d2_k5 = vload_half(0, (__global half *)(&(B[b_idx2_k5].d)));
        const half d3_k5 = vload_half(0, (__global half *)(&(B[b_idx3_k5].d)));
        const uchar qp0_k5 = B[b_idx0_k5].qs[k5_in_block % 16];
        const uchar qp1_k5 = B[b_idx1_k5].qs[k5_in_block % 16];
        const uchar qp2_k5 = B[b_idx2_k5].qs[k5_in_block % 16];
        const uchar qp3_k5 = B[b_idx3_k5].qs[k5_in_block % 16];
        const half qn0_k5 = (half)((k5_in_block < 16) ? ((qp0_k5 & 0x0F) - 8) : ((qp0_k5 >> 4) - 8));
        const half qn1_k5 = (half)((k5_in_block < 16) ? ((qp1_k5 & 0x0F) - 8) : ((qp1_k5 >> 4) - 8));
        const half qn2_k5 = (half)((k5_in_block < 16) ? ((qp2_k5 & 0x0F) - 8) : ((qp2_k5 >> 4) - 8));
        const half qn3_k5 = (half)((k5_in_block < 16) ? ((qp3_k5 & 0x0F) - 8) : ((qp3_k5 >> 4) - 8));
        half4 b_vals_k5 = (half4)(qn0_k5 * d0_k5, qn1_k5 * d1_k5, qn2_k5 * d2_k5, qn3_k5 * d3_k5);
        const int k6_block_idx = (k + 6) / 32;
        const int k6_in_block = (k + 6) % 32;
        const long b_idx0_k6 = (long)(n_start + 0) * n_stride + k6_block_idx;
        const long b_idx1_k6 = (long)(n_start + 1) * n_stride + k6_block_idx;
        const long b_idx2_k6 = (long)(n_start + 2) * n_stride + k6_block_idx;
        const long b_idx3_k6 = (long)(n_start + 3) * n_stride + k6_block_idx;
        const half d0_k6 = vload_half(0, (__global half *)(&(B[b_idx0_k6].d)));
        const half d1_k6 = vload_half(0, (__global half *)(&(B[b_idx1_k6].d)));
        const half d2_k6 = vload_half(0, (__global half *)(&(B[b_idx2_k6].d)));
        const half d3_k6 = vload_half(0, (__global half *)(&(B[b_idx3_k6].d)));
        const uchar qp0_k6 = B[b_idx0_k6].qs[k6_in_block % 16];
        const uchar qp1_k6 = B[b_idx1_k6].qs[k6_in_block % 16];
        const uchar qp2_k6 = B[b_idx2_k6].qs[k6_in_block % 16];
        const uchar qp3_k6 = B[b_idx3_k6].qs[k6_in_block % 16];
        const half qn0_k6 = (half)((k6_in_block < 16) ? ((qp0_k6 & 0x0F) - 8) : ((qp0_k6 >> 4) - 8));
        const half qn1_k6 = (half)((k6_in_block < 16) ? ((qp1_k6 & 0x0F) - 8) : ((qp1_k6 >> 4) - 8));
        const half qn2_k6 = (half)((k6_in_block < 16) ? ((qp2_k6 & 0x0F) - 8) : ((qp2_k6 >> 4) - 8));
        const half qn3_k6 = (half)((k6_in_block < 16) ? ((qp3_k6 & 0x0F) - 8) : ((qp3_k6 >> 4) - 8));
        half4 b_vals_k6 = (half4)(qn0_k6 * d0_k6, qn1_k6 * d1_k6, qn2_k6 * d2_k6, qn3_k6 * d3_k6);
        const int k7_block_idx = (k + 7) / 32;
        const int k7_in_block = (k + 7) % 32;
        const long b_idx0_k7 = (long)(n_start + 0) * n_stride + k7_block_idx;
        const long b_idx1_k7 = (long)(n_start + 1) * n_stride + k7_block_idx;
        const long b_idx2_k7 = (long)(n_start + 2) * n_stride + k7_block_idx;
        const long b_idx3_k7 = (long)(n_start + 3) * n_stride + k7_block_idx;
        const half d0_k7 = vload_half(0, (__global half *)(&(B[b_idx0_k7].d)));
        const half d1_k7 = vload_half(0, (__global half *)(&(B[b_idx1_k7].d)));
        const half d2_k7 = vload_half(0, (__global half *)(&(B[b_idx2_k7].d)));
        const half d3_k7 = vload_half(0, (__global half *)(&(B[b_idx3_k7].d)));
        const uchar qp0_k7 = B[b_idx0_k7].qs[k7_in_block % 16];
        const uchar qp1_k7 = B[b_idx1_k7].qs[k7_in_block % 16];
        const uchar qp2_k7 = B[b_idx2_k7].qs[k7_in_block % 16];
        const uchar qp3_k7 = B[b_idx3_k7].qs[k7_in_block % 16];
        const half qn0_k7 = (half)((k7_in_block < 16) ? ((qp0_k7 & 0x0F) - 8) : ((qp0_k7 >> 4) - 8));
        const half qn1_k7 = (half)((k7_in_block < 16) ? ((qp1_k7 & 0x0F) - 8) : ((qp1_k7 >> 4) - 8));
        const half qn2_k7 = (half)((k7_in_block < 16) ? ((qp2_k7 & 0x0F) - 8) : ((qp2_k7 >> 4) - 8));
        const half qn3_k7 = (half)((k7_in_block < 16) ? ((qp3_k7 & 0x0F) - 8) : ((qp3_k7 >> 4) - 8));
        half4 b_vals_k7 = (half4)(qn0_k7 * d0_k7, qn1_k7 * d1_k7, qn2_k7 * d2_k7, qn3_k7 * d3_k7);

        acc = mad(a_vals_lo.x, b_vals_k0, acc);
        acc = mad(a_vals_lo.y, b_vals_k1, acc);
        acc = mad(a_vals_lo.z, b_vals_k2, acc);
        acc = mad(a_vals_lo.w, b_vals_k3, acc);
        acc = mad(a_vals_hi.x, b_vals_k4, acc);
        acc = mad(a_vals_hi.y, b_vals_k5, acc);
        acc = mad(a_vals_hi.z, b_vals_k6, acc);
        acc = mad(a_vals_hi.w, b_vals_k7, acc);
    }

    // 扫尾和偏置
    for (int k = K_vec_size_x8 * 8; k < K; ++k) {
        int pixel_x = k / 4;
        int component = k % 4;
        half4 a_pixel = read_imageh(A, sampler, (int2)(pixel_x, gy));
        half a_val;
        if (component == 0)
            a_val = a_pixel.x;
        else if (component == 1)
            a_val = a_pixel.y;
        else if (component == 2)
            a_val = a_pixel.z;
        else
            a_val = a_pixel.w;
        const int k_block_idx = k / 32;
        const int k_in_block = k % 32;
        const long b_idx0 = (long)(n_start + 0) * n_stride + k_block_idx;
        const long b_idx1 = (long)(n_start + 1) * n_stride + k_block_idx;
        const long b_idx2 = (long)(n_start + 2) * n_stride + k_block_idx;
        const long b_idx3 = (long)(n_start + 3) * n_stride + k_block_idx;
        const half d0 = vload_half(0, (__global half *)(&(B[b_idx0].d)));
        const half d1 = vload_half(0, (__global half *)(&(B[b_idx1].d)));
        const half d2 = vload_half(0, (__global half *)(&(B[b_idx2].d)));
        const half d3 = vload_half(0, (__global half *)(&(B[b_idx3].d)));
        const uchar qp0 = B[b_idx0].qs[k_in_block % 16];
        const uchar qp1 = B[b_idx1].qs[k_in_block % 16];
        const uchar qp2 = B[b_idx2].qs[k_in_block % 16];
        const uchar qp3 = B[b_idx3].qs[k_in_block % 16];
        const half qn0 = (half)((k_in_block < 16) ? ((qp0 & 0x0F) - 8) : ((qp0 >> 4) - 8));
        const half qn1 = (half)((k_in_block < 16) ? ((qp1 & 0x0F) - 8) : ((qp1 >> 4) - 8));
        const half qn2 = (half)((k_in_block < 16) ? ((qp2 & 0x0F) - 8) : ((qp2 >> 4) - 8));
        const half qn3 = (half)((k_in_block < 16) ? ((qp3 & 0x0F) - 8) : ((qp3 >> 4) - 8));
        half4 b_vals = (half4)(qn0 * d0, qn1 * d1, qn2 * d2, qn3 * d3);
        acc = mad(a_val, b_vals, acc);
    }

    if (has_bias != 0) {
        half4 bias_h = convert_half4_rte(vload4(0, bias + n_start));
        if (n_start < N) acc.x += bias_h.x;
        if (n_start + 1 < N) acc.y += bias_h.y;
        if (n_start + 2 < N) acc.z += bias_h.z;
        if (n_start + 3 < N) acc.w += bias_h.w;
    }

    write_imageh(C, (int2)(gx, gy), acc);
}
#else
// ---------- [兼容回退版 - Fallback Version] ----------
__kernel void gemm_fp16_q4_0_transb_bias_image_pipe(
    sampler_t sampler,
    __read_only image2d_t A,
    __global const block_q4_0 *B,
    __global const float *bias,
    __write_only image2d_t C,
    const int M, const int K, const int N,
    const int H, const int K_b,
    const int has_bias) {
    // 逻辑与FP32版本完全相同，因为Host端会准备好FP32的Image
    const int gx = get_global_id(0);
    const int gy = get_global_id(1);
    const int n_start = gx * 4;
    if (n_start >= N) { return; }

    float4 acc = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    const int K_blocks = K / 32;
    const long base_offset = 0;
    const long n_stride = K_blocks;

    for (int k = 0; k < K; ++k) {
        int pixel_x = k / 4;
        int component = k % 4;
        // 使用 read_imagef 读取，因为Image是CL_FLOAT格式
        float4 a_pixel = read_imagef(A, sampler, (int2)(pixel_x, gy));
        float a_val;
        if (component == 0)
            a_val = a_pixel.x;
        else if (component == 1)
            a_val = a_pixel.y;
        else if (component == 2)
            a_val = a_pixel.z;
        else
            a_val = a_pixel.w;

        const int k_block_idx = k / 32;
        const int k_in_block = k % 32;
        const long b_block_mem_idx0 = base_offset + (long)(n_start + 0) * n_stride + k_block_idx;
        const long b_block_mem_idx1 = base_offset + (long)(n_start + 1) * n_stride + k_block_idx;
        const long b_block_mem_idx2 = base_offset + (long)(n_start + 2) * n_stride + k_block_idx;
        const long b_block_mem_idx3 = base_offset + (long)(n_start + 3) * n_stride + k_block_idx;

        const float d_b0 = vload_half(0, (__global half *)(&(B[b_block_mem_idx0].d)));
        const float d_b1 = vload_half(0, (__global half *)(&(B[b_block_mem_idx1].d)));
        const float d_b2 = vload_half(0, (__global half *)(&(B[b_block_mem_idx2].d)));
        const float d_b3 = vload_half(0, (__global half *)(&(B[b_block_mem_idx3].d)));

        const uchar q_packed0 = B[b_block_mem_idx0].qs[k_in_block % 16];
        const uchar q_packed1 = B[b_block_mem_idx1].qs[k_in_block % 16];
        const uchar q_packed2 = B[b_block_mem_idx2].qs[k_in_block % 16];
        const uchar q_packed3 = B[b_block_mem_idx3].qs[k_in_block % 16];

        const char q_nibble0 = (k_in_block < 16) ? ((q_packed0 & 0x0F) - 8) : ((q_packed0 >> 4) - 8);
        const char q_nibble1 = (k_in_block < 16) ? ((q_packed1 & 0x0F) - 8) : ((q_packed1 >> 4) - 8);
        const char q_nibble2 = (k_in_block < 16) ? ((q_packed2 & 0x0F) - 8) : ((q_packed2 >> 4) - 8);
        const char q_nibble3 = (k_in_block < 16) ? ((q_packed3 & 0x0F) - 8) : ((q_packed3 >> 4) - 8);

        float4 b_vals = (float4)((float)q_nibble0 * d_b0, (float)q_nibble1 * d_b1, (float)q_nibble2 * d_b2, (float)q_nibble3 * d_b3);

        acc = mad(a_val, b_vals, acc);
    }

    if (has_bias != 0) {
        float4 bias_vals = vload4(0, bias + n_start);
        acc += bias_vals;
    }

    // 使用 write_imagef 写入，因为输出Image也是CL_FLOAT格式
    write_imagef(C, (int2)(gx, gy), acc);
}
#endif // SUPPORTS_FP16

// ==================================================================
// 9. FP32 * Q4_0 Fused GEMV + Bias Kernel (All Image Pipe) [最终修正版]
// C = A(Image) * B(Q4_0 Buffer)^T + Bias
// ==================================================================
__kernel void gemv_fp32_q4_0_transb_bias_image_pipe(
    sampler_t sampler,
    __read_only image2d_t A,
    __global const block_q4_0 *B,
    __global const float *bias,
    __write_only image2d_t C, // 【修改】C 也是 Image
    const int K, const int N,
    const int H,
    const int has_bias) {
    // 1. 并行策略: 每个工作项计算一个输出像素 (float4)
    const int gx = get_global_id(0); // 对应 C 的 x 坐标, 范围 [0, N/4 - 1]
    const int n_start = gx * 4;
    if (n_start >= N) { return; }

    // 2. 初始化累加器
    float4 acc = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    const int K_blocks = K / 32;
    const long n_stride = K_blocks;

    // 3. 沿 K 维度循环计算
    for (int k = 0; k < K; ++k) {
        // a. 正确地从 Image A 中读取 FP32 值 (y坐标永远是0)
        int pixel_x = k / 4;
        int component = k % 4;
        float4 a_pixel = read_imagef(A, sampler, (int2)(pixel_x, 0));
        float a_val;
        if (component == 0)
            a_val = a_pixel.x;
        else if (component == 1)
            a_val = a_pixel.y;
        else if (component == 2)
            a_val = a_pixel.z;
        else
            a_val = a_pixel.w;

        // b. 解量化 B 的4个值
        const int k_block_idx = k / 32;
        const int k_in_block = k % 32;
        const long b_block_mem_idx0 = (long)(n_start + 0) * n_stride + k_block_idx;
        const long b_block_mem_idx1 = (long)(n_start + 1) * n_stride + k_block_idx;
        const long b_block_mem_idx2 = (long)(n_start + 2) * n_stride + k_block_idx;
        const long b_block_mem_idx3 = (long)(n_start + 3) * n_stride + k_block_idx;

        const float d_b0 = vload_half(0, (__global half *)(&(B[b_block_mem_idx0].d)));
        const float d_b1 = vload_half(0, (__global half *)(&(B[b_block_mem_idx1].d)));
        const float d_b2 = vload_half(0, (__global half *)(&(B[b_block_mem_idx2].d)));
        const float d_b3 = vload_half(0, (__global half *)(&(B[b_block_mem_idx3].d)));

        const uchar q_packed0 = B[b_block_mem_idx0].qs[k_in_block % 16];
        const char q_nibble0 = (k_in_block < 16) ? ((q_packed0 & 0x0F) - 8) : ((q_packed0 >> 4) - 8);
        const uchar q_packed1 = B[b_block_mem_idx1].qs[k_in_block % 16];
        const char q_nibble1 = (k_in_block < 16) ? ((q_packed1 & 0x0F) - 8) : ((q_packed1 >> 4) - 8);
        const uchar q_packed2 = B[b_block_mem_idx2].qs[k_in_block % 16];
        const char q_nibble2 = (k_in_block < 16) ? ((q_packed2 & 0x0F) - 8) : ((q_packed2 >> 4) - 8);
        const uchar q_packed3 = B[b_block_mem_idx3].qs[k_in_block % 16];
        const char q_nibble3 = (k_in_block < 16) ? ((q_packed3 & 0x0F) - 8) : ((q_packed3 >> 4) - 8);

        float4 b_vals = (float4)((float)q_nibble0 * d_b0, (float)q_nibble1 * d_b1, (float)q_nibble2 * d_b2, (float)q_nibble3 * d_b3);

        acc = mad(a_val, b_vals, acc);
    }

    if (has_bias != 0) {
        float4 bias_vals = vload4(0, bias + n_start);
        acc += bias_vals;
    }

    // 4. 将结果像素写入输出 Image C (y坐标永远是0)
    write_imagef(C, (int2)(gx, 0), acc);
}

// ==================================================================
// 10. FP16 * Q4_0 Fused GEMV + Bias Kernel (All Image Pipe) [最终修正版]
// ==================================================================
#if defined(SUPPORTS_FP16)
// ---------- [高性能版] ----------
/*
__kernel void gemv_fp16_q4_0_transb_bias_image_pipe(
    sampler_t sampler,
    __read_only image2d_t A,      // 输入向量 A, 形状 [1, K], 映射为 [K/4, 1] 的 half4 Image
    __global const block_q4_0 *B, // 权重矩阵 B, 形状 [N, K]
    __global const float *bias,   // 偏置向量, 形状 [N]
    __write_only image2d_t C,     // 输出向量 C, 形状 [1, N], 映射为 [N/4, 1] 的 half4 Image
    const int K, const int N,
    const int H, // H=1, 为了兼容性保留
    const int has_bias) {
    // 1. 并行策略: 每个工作项计算一个输出像素 (float4)
    const int gx = get_global_id(0); // C 的 x 坐标, 范围 [0, N/4 - 1]
    const int n_start = gx * 4;
    if (n_start >= N) { return; }

    // 2. 寄存器累加器：使用float4在寄存器中进行高精度累加
    float4 acc = (float4)(0.0f);
    const int K_blocks = K / 32;
    const long n_stride = K_blocks;
    const int K_vec_size = K / 4;

    // 3. 核心计算循环：沿K维度4倍展开, 深度优化指令级并行
    // 每次循环处理K维度的4个元素
    for (int k_vec = 0; k_vec < K_vec_size; ++k_vec) {
        const int k = k_vec * 4;

        // a. 向量化读取A: 一次读取4个half, 组成一个half4
        half4 a_pixel_h = read_imageh(A, sampler, (int2)(k_vec, 0));
        float4 a_vals = convert_float4(a_pixel_h);

        // b. 为 a_vals 的每个分量(x,y,z,w) 分别解量化B的值
        //    这部分代码虽然冗长，但将所有计算和访存操作完全展开，
        //    能让编译器和硬件最大程度地并行调度，隐藏延迟。

        // --- 解量化与 a_vals.x (k+0) 相乘的 b_vals_k0 ---
        const int k0_block_idx = (k + 0) / 32;
        const int k0_in_block = (k + 0) % 32;
        const long b_idx0_k0 = (long)(n_start + 0) * n_stride + k0_block_idx;
        const long b_idx1_k0 = (long)(n_start + 1) * n_stride + k0_block_idx;
        const long b_idx2_k0 = (long)(n_start + 2) * n_stride + k0_block_idx;
        const long b_idx3_k0 = (long)(n_start + 3) * n_stride + k0_block_idx;
        const float d0_k0 = vload_half(0, (__global half *)(&(B[b_idx0_k0].d)));
        const float d1_k0 = vload_half(0, (__global half *)(&(B[b_idx1_k0].d)));
        const float d2_k0 = vload_half(0, (__global half *)(&(B[b_idx2_k0].d)));
        const float d3_k0 = vload_half(0, (__global half *)(&(B[b_idx3_k0].d)));
        const uchar qp0_k0 = B[b_idx0_k0].qs[k0_in_block % 16];
        const uchar qp1_k0 = B[b_idx1_k0].qs[k0_in_block % 16];
        const uchar qp2_k0 = B[b_idx2_k0].qs[k0_in_block % 16];
        const uchar qp3_k0 = B[b_idx3_k0].qs[k0_in_block % 16];
        const char qn0_k0 = (k0_in_block < 16) ? ((qp0_k0 & 0x0F) - 8) : ((qp0_k0 >> 4) - 8);
        const char qn1_k0 = (k0_in_block < 16) ? ((qp1_k0 & 0x0F) - 8) : ((qp1_k0 >> 4) - 8);
        const char qn2_k0 = (k0_in_block < 16) ? ((qp2_k0 & 0x0F) - 8) : ((qp2_k0 >> 4) - 8);
        const char qn3_k0 = (k0_in_block < 16) ? ((qp3_k0 & 0x0F) - 8) : ((qp3_k0 >> 4) - 8);
        float4 b_vals_k0 = (float4)((float)qn0_k0 * d0_k0, (float)qn1_k0 * d1_k0, (float)qn2_k0 * d2_k0, (float)qn3_k0 * d3_k0);

        // --- 解量化与 a_vals.y (k+1) 相乘的 b_vals_k1 ---
        const int k1_block_idx = (k + 1) / 32;
        const int k1_in_block = (k + 1) % 32;
        const long b_idx0_k1 = (long)(n_start + 0) * n_stride + k1_block_idx;
        const long b_idx1_k1 = (long)(n_start + 1) * n_stride + k1_block_idx;
        const long b_idx2_k1 = (long)(n_start + 2) * n_stride + k1_block_idx;
        const long b_idx3_k1 = (long)(n_start + 3) * n_stride + k1_block_idx;
        const float d0_k1 = vload_half(0, (__global half *)(&(B[b_idx0_k1].d)));
        const float d1_k1 = vload_half(0, (__global half *)(&(B[b_idx1_k1].d)));
        const float d2_k1 = vload_half(0, (__global half *)(&(B[b_idx2_k1].d)));
        const float d3_k1 = vload_half(0, (__global half *)(&(B[b_idx3_k1].d)));
        const uchar qp0_k1 = B[b_idx0_k1].qs[k1_in_block % 16];
        const uchar qp1_k1 = B[b_idx1_k1].qs[k1_in_block % 16];
        const uchar qp2_k1 = B[b_idx2_k1].qs[k1_in_block % 16];
        const uchar qp3_k1 = B[b_idx3_k1].qs[k1_in_block % 16];
        const char qn0_k1 = (k1_in_block < 16) ? ((qp0_k1 & 0x0F) - 8) : ((qp0_k1 >> 4) - 8);
        const char qn1_k1 = (k1_in_block < 16) ? ((qp1_k1 & 0x0F) - 8) : ((qp1_k1 >> 4) - 8);
        const char qn2_k1 = (k1_in_block < 16) ? ((qp2_k1 & 0x0F) - 8) : ((qp2_k1 >> 4) - 8);
        const char qn3_k1 = (k1_in_block < 16) ? ((qp3_k1 & 0x0F) - 8) : ((qp3_k1 >> 4) - 8);
        float4 b_vals_k1 = (float4)((float)qn0_k1 * d0_k1, (float)qn1_k1 * d1_k1, (float)qn2_k1 * d2_k1, (float)qn3_k1 * d3_k1);

        // --- 解量化与 a_vals.z (k+2) 相乘的 b_vals_k2 ---
        const int k2_block_idx = (k + 2) / 32;
        const int k2_in_block = (k + 2) % 32;
        const long b_idx0_k2 = (long)(n_start + 0) * n_stride + k2_block_idx;
        const long b_idx1_k2 = (long)(n_start + 1) * n_stride + k2_block_idx;
        const long b_idx2_k2 = (long)(n_start + 2) * n_stride + k2_block_idx;
        const long b_idx3_k2 = (long)(n_start + 3) * n_stride + k2_block_idx;
        const float d0_k2 = vload_half(0, (__global half *)(&(B[b_idx0_k2].d)));
        const float d1_k2 = vload_half(0, (__global half *)(&(B[b_idx1_k2].d)));
        const float d2_k2 = vload_half(0, (__global half *)(&(B[b_idx2_k2].d)));
        const float d3_k2 = vload_half(0, (__global half *)(&(B[b_idx3_k2].d)));
        const uchar qp0_k2 = B[b_idx0_k2].qs[k2_in_block % 16];
        const uchar qp1_k2 = B[b_idx1_k2].qs[k2_in_block % 16];
        const uchar qp2_k2 = B[b_idx2_k2].qs[k2_in_block % 16];
        const uchar qp3_k2 = B[b_idx3_k2].qs[k2_in_block % 16];
        const char qn0_k2 = (k2_in_block < 16) ? ((qp0_k2 & 0x0F) - 8) : ((qp0_k2 >> 4) - 8);
        const char qn1_k2 = (k2_in_block < 16) ? ((qp1_k2 & 0x0F) - 8) : ((qp1_k2 >> 4) - 8);
        const char qn2_k2 = (k2_in_block < 16) ? ((qp2_k2 & 0x0F) - 8) : ((qp2_k2 >> 4) - 8);
        const char qn3_k2 = (k2_in_block < 16) ? ((qp3_k2 & 0x0F) - 8) : ((qp3_k2 >> 4) - 8);
        float4 b_vals_k2 = (float4)((float)qn0_k2 * d0_k2, (float)qn1_k2 * d1_k2, (float)qn2_k2 * d2_k2, (float)qn3_k2 * d3_k2);

        // --- 解量化与 a_vals.w (k+3) 相乘的 b_vals_k3 ---
        const int k3_block_idx = (k + 3) / 32;
        const int k3_in_block = (k + 3) % 32;
        const long b_idx0_k3 = (long)(n_start + 0) * n_stride + k3_block_idx;
        const long b_idx1_k3 = (long)(n_start + 1) * n_stride + k3_block_idx;
        const long b_idx2_k3 = (long)(n_start + 2) * n_stride + k3_block_idx;
        const long b_idx3_k3 = (long)(n_start + 3) * n_stride + k3_block_idx;
        const float d0_k3 = vload_half(0, (__global half *)(&(B[b_idx0_k3].d)));
        const float d1_k3 = vload_half(0, (__global half *)(&(B[b_idx1_k3].d)));
        const float d2_k3 = vload_half(0, (__global half *)(&(B[b_idx2_k3].d)));
        const float d3_k3 = vload_half(0, (__global half *)(&(B[b_idx3_k3].d)));
        const uchar qp0_k3 = B[b_idx0_k3].qs[k3_in_block % 16];
        const uchar qp1_k3 = B[b_idx1_k3].qs[k3_in_block % 16];
        const uchar qp2_k3 = B[b_idx2_k3].qs[k3_in_block % 16];
        const uchar qp3_k3 = B[b_idx3_k3].qs[k3_in_block % 16];
        const char qn0_k3 = (k3_in_block < 16) ? ((qp0_k3 & 0x0F) - 8) : ((qp0_k3 >> 4) - 8);
        const char qn1_k3 = (k3_in_block < 16) ? ((qp1_k3 & 0x0F) - 8) : ((qp1_k3 >> 4) - 8);
        const char qn2_k3 = (k3_in_block < 16) ? ((qp2_k3 & 0x0F) - 8) : ((qp2_k3 >> 4) - 8);
        const char qn3_k3 = (k3_in_block < 16) ? ((qp3_k3 & 0x0F) - 8) : ((qp3_k3 >> 4) - 8);
        float4 b_vals_k3 = (float4)((float)qn0_k3 * d0_k3, (float)qn1_k3 * d1_k3, (float)qn2_k3 * d2_k3, (float)qn3_k3 * d3_k3);

        // c. 累加: 4次独立的MAD指令，可以被硬件并行执行
        acc = mad(a_vals.x, b_vals_k0, acc);
        acc = mad(a_vals.y, b_vals_k1, acc);
        acc = mad(a_vals.z, b_vals_k2, acc);
        acc = mad(a_vals.w, b_vals_k3, acc);
    }

    // 4. 扫尾处理: K不是4的倍数时，处理余下的1-3个元素 (保持不变)
    for (int k = K_vec_size * 4; k < K; ++k) {
        int pixel_x = k / 4;
        int component = k % 4;
        half4 a_pixel = read_imageh(A, sampler, (int2)(pixel_x, 0));
        float a_val;
        if (component == 0)
            a_val = (float)a_pixel.x;
        else if (component == 1)
            a_val = (float)a_pixel.y;
        else if (component == 2)
            a_val = (float)a_pixel.z;
        else
            a_val = (float)a_pixel.w;

        const int k_block_idx = k / 32;
        const int k_in_block = k % 32;
        const long b_idx0 = (long)(n_start + 0) * n_stride + k_block_idx;
        const long b_idx1 = (long)(n_start + 1) * n_stride + k_block_idx;
        const long b_idx2 = (long)(n_start + 2) * n_stride + k_block_idx;
        const long b_idx3 = (long)(n_start + 3) * n_stride + k_block_idx;
        const float d0 = vload_half(0, (__global half *)(&(B[b_idx0].d)));
        const float d1 = vload_half(0, (__global half *)(&(B[b_idx1].d)));
        const float d2 = vload_half(0, (__global half *)(&(B[b_idx2].d)));
        const float d3 = vload_half(0, (__global half *)(&(B[b_idx3].d)));
        const uchar qp0 = B[b_idx0].qs[k_in_block % 16];
        const uchar qp1 = B[b_idx1].qs[k_in_block % 16];
        const uchar qp2 = B[b_idx2].qs[k_in_block % 16];
        const uchar qp3 = B[b_idx3].qs[k_in_block % 16];
        const char qn0 = (k_in_block < 16) ? ((qp0 & 0x0F) - 8) : ((qp0 >> 4) - 8);
        const char qn1 = (k_in_block < 16) ? ((qp1 & 0x0F) - 8) : ((qp1 >> 4) - 8);
        const char qn2 = (k_in_block < 16) ? ((qp2 & 0x0F) - 8) : ((qp2 >> 4) - 8);
        const char qn3 = (k_in_block < 16) ? ((qp3 & 0x0F) - 8) : ((qp3 >> 4) - 8);
        float4 b_vals = (float4)((float)qn0 * d0, (float)qn1 * d1, (float)qn2 * d2, (float)qn3 * d3);

        acc = mad(a_val, b_vals, acc);
    }

    // 5. 添加偏置 (优化为单次vload4，因为N通常是4的倍数)
    if (has_bias != 0) {
        float4 bias_vals = vload4(0, bias + n_start);
        acc += bias_vals;
    }

    // 6. 将结果像素写入输出 Image C
    write_imageh(C, (int2)(gx, 0), convert_half4_rte(acc));
}
*/
__kernel void gemv_fp16_q4_0_transb_bias_image_pipe(
    sampler_t sampler,
    __read_only image2d_t A,
    __global const block_q4_0 *B,
    __global const float *bias,
    __write_only image2d_t C,
    const int K, const int N,
    const int H,
    const int has_bias) {
    const int gx = get_global_id(0);
    const int n_start = gx * 4;
    if (n_start >= N) { return; }

    half4 acc = (half4)(0.0h);

    const int K_blocks = K / 32;
    const long n_stride = K_blocks;
    const int K_vec_size_x8 = K / 8;

    for (int k_vec = 0; k_vec < K_vec_size_x8; ++k_vec) {
        const int k = k_vec * 8;

        // 读取A时，y坐标固定为0，因为是GEMV
        half4 a_vals_lo = read_imageh(A, sampler, (int2)(k / 4, 0));
        half4 a_vals_hi = read_imageh(A, sampler, (int2)(k / 4 + 1, 0));

        // --- 8倍展开的解量化与计算（代码同上一版，此处为简洁省略）---
        // --- k+0 ---
        const int k0_block_idx = (k + 0) / 32;
        const int k0_in_block = (k + 0) % 32;
        const long b_idx0_k0 = (long)(n_start + 0) * n_stride + k0_block_idx;
        const long b_idx1_k0 = (long)(n_start + 1) * n_stride + k0_block_idx;
        const long b_idx2_k0 = (long)(n_start + 2) * n_stride + k0_block_idx;
        const long b_idx3_k0 = (long)(n_start + 3) * n_stride + k0_block_idx;
        const half d0_k0 = vload_half(0, (__global half *)(&(B[b_idx0_k0].d)));
        const half d1_k0 = vload_half(0, (__global half *)(&(B[b_idx1_k0].d)));
        const half d2_k0 = vload_half(0, (__global half *)(&(B[b_idx2_k0].d)));
        const half d3_k0 = vload_half(0, (__global half *)(&(B[b_idx3_k0].d)));
        const uchar qp0_k0 = B[b_idx0_k0].qs[k0_in_block % 16];
        const uchar qp1_k0 = B[b_idx1_k0].qs[k0_in_block % 16];
        const uchar qp2_k0 = B[b_idx2_k0].qs[k0_in_block % 16];
        const uchar qp3_k0 = B[b_idx3_k0].qs[k0_in_block % 16];
        const half qn0_k0 = (half)((k0_in_block < 16) ? ((qp0_k0 & 0x0F) - 8) : ((qp0_k0 >> 4) - 8));
        const half qn1_k0 = (half)((k0_in_block < 16) ? ((qp1_k0 & 0x0F) - 8) : ((qp1_k0 >> 4) - 8));
        const half qn2_k0 = (half)((k0_in_block < 16) ? ((qp2_k0 & 0x0F) - 8) : ((qp2_k0 >> 4) - 8));
        const half qn3_k0 = (half)((k0_in_block < 16) ? ((qp3_k0 & 0x0F) - 8) : ((qp3_k0 >> 4) - 8));
        half4 b_vals_k0 = (half4)(qn0_k0 * d0_k0, qn1_k0 * d1_k0, qn2_k0 * d2_k0, qn3_k0 * d3_k0);
        // ... k+1 到 k+7 的代码 ...
        const int k1_block_idx = (k + 1) / 32;
        const int k1_in_block = (k + 1) % 32;
        const long b_idx0_k1 = (long)(n_start + 0) * n_stride + k1_block_idx;
        const long b_idx1_k1 = (long)(n_start + 1) * n_stride + k1_block_idx;
        const long b_idx2_k1 = (long)(n_start + 2) * n_stride + k1_block_idx;
        const long b_idx3_k1 = (long)(n_start + 3) * n_stride + k1_block_idx;
        const half d0_k1 = vload_half(0, (__global half *)(&(B[b_idx0_k1].d)));
        const half d1_k1 = vload_half(0, (__global half *)(&(B[b_idx1_k1].d)));
        const half d2_k1 = vload_half(0, (__global half *)(&(B[b_idx2_k1].d)));
        const half d3_k1 = vload_half(0, (__global half *)(&(B[b_idx3_k1].d)));
        const uchar qp0_k1 = B[b_idx0_k1].qs[k1_in_block % 16];
        const uchar qp1_k1 = B[b_idx1_k1].qs[k1_in_block % 16];
        const uchar qp2_k1 = B[b_idx2_k1].qs[k1_in_block % 16];
        const uchar qp3_k1 = B[b_idx3_k1].qs[k1_in_block % 16];
        const half qn0_k1 = (half)((k1_in_block < 16) ? ((qp0_k1 & 0x0F) - 8) : ((qp0_k1 >> 4) - 8));
        const half qn1_k1 = (half)((k1_in_block < 16) ? ((qp1_k1 & 0x0F) - 8) : ((qp1_k1 >> 4) - 8));
        const half qn2_k1 = (half)((k1_in_block < 16) ? ((qp2_k1 & 0x0F) - 8) : ((qp2_k1 >> 4) - 8));
        const half qn3_k1 = (half)((k1_in_block < 16) ? ((qp3_k1 & 0x0F) - 8) : ((qp3_k1 >> 4) - 8));
        half4 b_vals_k1 = (half4)(qn0_k1 * d0_k1, qn1_k1 * d1_k1, qn2_k1 * d2_k1, qn3_k1 * d3_k1);
        const int k2_block_idx = (k + 2) / 32;
        const int k2_in_block = (k + 2) % 32;
        const long b_idx0_k2 = (long)(n_start + 0) * n_stride + k2_block_idx;
        const long b_idx1_k2 = (long)(n_start + 1) * n_stride + k2_block_idx;
        const long b_idx2_k2 = (long)(n_start + 2) * n_stride + k2_block_idx;
        const long b_idx3_k2 = (long)(n_start + 3) * n_stride + k2_block_idx;
        const half d0_k2 = vload_half(0, (__global half *)(&(B[b_idx0_k2].d)));
        const half d1_k2 = vload_half(0, (__global half *)(&(B[b_idx1_k2].d)));
        const half d2_k2 = vload_half(0, (__global half *)(&(B[b_idx2_k2].d)));
        const half d3_k2 = vload_half(0, (__global half *)(&(B[b_idx3_k2].d)));
        const uchar qp0_k2 = B[b_idx0_k2].qs[k2_in_block % 16];
        const uchar qp1_k2 = B[b_idx1_k2].qs[k2_in_block % 16];
        const uchar qp2_k2 = B[b_idx2_k2].qs[k2_in_block % 16];
        const uchar qp3_k2 = B[b_idx3_k2].qs[k2_in_block % 16];
        const half qn0_k2 = (half)((k2_in_block < 16) ? ((qp0_k2 & 0x0F) - 8) : ((qp0_k2 >> 4) - 8));
        const half qn1_k2 = (half)((k2_in_block < 16) ? ((qp1_k2 & 0x0F) - 8) : ((qp1_k2 >> 4) - 8));
        const half qn2_k2 = (half)((k2_in_block < 16) ? ((qp2_k2 & 0x0F) - 8) : ((qp2_k2 >> 4) - 8));
        const half qn3_k2 = (half)((k2_in_block < 16) ? ((qp3_k2 & 0x0F) - 8) : ((qp3_k2 >> 4) - 8));
        half4 b_vals_k2 = (half4)(qn0_k2 * d0_k2, qn1_k2 * d1_k2, qn2_k2 * d2_k2, qn3_k2 * d3_k2);
        const int k3_block_idx = (k + 3) / 32;
        const int k3_in_block = (k + 3) % 32;
        const long b_idx0_k3 = (long)(n_start + 0) * n_stride + k3_block_idx;
        const long b_idx1_k3 = (long)(n_start + 1) * n_stride + k3_block_idx;
        const long b_idx2_k3 = (long)(n_start + 2) * n_stride + k3_block_idx;
        const long b_idx3_k3 = (long)(n_start + 3) * n_stride + k3_block_idx;
        const half d0_k3 = vload_half(0, (__global half *)(&(B[b_idx0_k3].d)));
        const half d1_k3 = vload_half(0, (__global half *)(&(B[b_idx1_k3].d)));
        const half d2_k3 = vload_half(0, (__global half *)(&(B[b_idx2_k3].d)));
        const half d3_k3 = vload_half(0, (__global half *)(&(B[b_idx3_k3].d)));
        const uchar qp0_k3 = B[b_idx0_k3].qs[k3_in_block % 16];
        const uchar qp1_k3 = B[b_idx1_k3].qs[k3_in_block % 16];
        const uchar qp2_k3 = B[b_idx2_k3].qs[k3_in_block % 16];
        const uchar qp3_k3 = B[b_idx3_k3].qs[k3_in_block % 16];
        const half qn0_k3 = (half)((k3_in_block < 16) ? ((qp0_k3 & 0x0F) - 8) : ((qp0_k3 >> 4) - 8));
        const half qn1_k3 = (half)((k3_in_block < 16) ? ((qp1_k3 & 0x0F) - 8) : ((qp1_k3 >> 4) - 8));
        const half qn2_k3 = (half)((k3_in_block < 16) ? ((qp2_k3 & 0x0F) - 8) : ((qp2_k3 >> 4) - 8));
        const half qn3_k3 = (half)((k3_in_block < 16) ? ((qp3_k3 & 0x0F) - 8) : ((qp3_k3 >> 4) - 8));
        half4 b_vals_k3 = (half4)(qn0_k3 * d0_k3, qn1_k3 * d1_k3, qn2_k3 * d2_k3, qn3_k3 * d3_k3);
        const int k4_block_idx = (k + 4) / 32;
        const int k4_in_block = (k + 4) % 32;
        const long b_idx0_k4 = (long)(n_start + 0) * n_stride + k4_block_idx;
        const long b_idx1_k4 = (long)(n_start + 1) * n_stride + k4_block_idx;
        const long b_idx2_k4 = (long)(n_start + 2) * n_stride + k4_block_idx;
        const long b_idx3_k4 = (long)(n_start + 3) * n_stride + k4_block_idx;
        const half d0_k4 = vload_half(0, (__global half *)(&(B[b_idx0_k4].d)));
        const half d1_k4 = vload_half(0, (__global half *)(&(B[b_idx1_k4].d)));
        const half d2_k4 = vload_half(0, (__global half *)(&(B[b_idx2_k4].d)));
        const half d3_k4 = vload_half(0, (__global half *)(&(B[b_idx3_k4].d)));
        const uchar qp0_k4 = B[b_idx0_k4].qs[k4_in_block % 16];
        const uchar qp1_k4 = B[b_idx1_k4].qs[k4_in_block % 16];
        const uchar qp2_k4 = B[b_idx2_k4].qs[k4_in_block % 16];
        const uchar qp3_k4 = B[b_idx3_k4].qs[k4_in_block % 16];
        const half qn0_k4 = (half)((k4_in_block < 16) ? ((qp0_k4 & 0x0F) - 8) : ((qp0_k4 >> 4) - 8));
        const half qn1_k4 = (half)((k4_in_block < 16) ? ((qp1_k4 & 0x0F) - 8) : ((qp1_k4 >> 4) - 8));
        const half qn2_k4 = (half)((k4_in_block < 16) ? ((qp2_k4 & 0x0F) - 8) : ((qp2_k4 >> 4) - 8));
        const half qn3_k4 = (half)((k4_in_block < 16) ? ((qp3_k4 & 0x0F) - 8) : ((qp3_k4 >> 4) - 8));
        half4 b_vals_k4 = (half4)(qn0_k4 * d0_k4, qn1_k4 * d1_k4, qn2_k4 * d2_k4, qn3_k4 * d3_k4);
        const int k5_block_idx = (k + 5) / 32;
        const int k5_in_block = (k + 5) % 32;
        const long b_idx0_k5 = (long)(n_start + 0) * n_stride + k5_block_idx;
        const long b_idx1_k5 = (long)(n_start + 1) * n_stride + k5_block_idx;
        const long b_idx2_k5 = (long)(n_start + 2) * n_stride + k5_block_idx;
        const long b_idx3_k5 = (long)(n_start + 3) * n_stride + k5_block_idx;
        const half d0_k5 = vload_half(0, (__global half *)(&(B[b_idx0_k5].d)));
        const half d1_k5 = vload_half(0, (__global half *)(&(B[b_idx1_k5].d)));
        const half d2_k5 = vload_half(0, (__global half *)(&(B[b_idx2_k5].d)));
        const half d3_k5 = vload_half(0, (__global half *)(&(B[b_idx3_k5].d)));
        const uchar qp0_k5 = B[b_idx0_k5].qs[k5_in_block % 16];
        const uchar qp1_k5 = B[b_idx1_k5].qs[k5_in_block % 16];
        const uchar qp2_k5 = B[b_idx2_k5].qs[k5_in_block % 16];
        const uchar qp3_k5 = B[b_idx3_k5].qs[k5_in_block % 16];
        const half qn0_k5 = (half)((k5_in_block < 16) ? ((qp0_k5 & 0x0F) - 8) : ((qp0_k5 >> 4) - 8));
        const half qn1_k5 = (half)((k5_in_block < 16) ? ((qp1_k5 & 0x0F) - 8) : ((qp1_k5 >> 4) - 8));
        const half qn2_k5 = (half)((k5_in_block < 16) ? ((qp2_k5 & 0x0F) - 8) : ((qp2_k5 >> 4) - 8));
        const half qn3_k5 = (half)((k5_in_block < 16) ? ((qp3_k5 & 0x0F) - 8) : ((qp3_k5 >> 4) - 8));
        half4 b_vals_k5 = (half4)(qn0_k5 * d0_k5, qn1_k5 * d1_k5, qn2_k5 * d2_k5, qn3_k5 * d3_k5);
        const int k6_block_idx = (k + 6) / 32;
        const int k6_in_block = (k + 6) % 32;
        const long b_idx0_k6 = (long)(n_start + 0) * n_stride + k6_block_idx;
        const long b_idx1_k6 = (long)(n_start + 1) * n_stride + k6_block_idx;
        const long b_idx2_k6 = (long)(n_start + 2) * n_stride + k6_block_idx;
        const long b_idx3_k6 = (long)(n_start + 3) * n_stride + k6_block_idx;
        const half d0_k6 = vload_half(0, (__global half *)(&(B[b_idx0_k6].d)));
        const half d1_k6 = vload_half(0, (__global half *)(&(B[b_idx1_k6].d)));
        const half d2_k6 = vload_half(0, (__global half *)(&(B[b_idx2_k6].d)));
        const half d3_k6 = vload_half(0, (__global half *)(&(B[b_idx3_k6].d)));
        const uchar qp0_k6 = B[b_idx0_k6].qs[k6_in_block % 16];
        const uchar qp1_k6 = B[b_idx1_k6].qs[k6_in_block % 16];
        const uchar qp2_k6 = B[b_idx2_k6].qs[k6_in_block % 16];
        const uchar qp3_k6 = B[b_idx3_k6].qs[k6_in_block % 16];
        const half qn0_k6 = (half)((k6_in_block < 16) ? ((qp0_k6 & 0x0F) - 8) : ((qp0_k6 >> 4) - 8));
        const half qn1_k6 = (half)((k6_in_block < 16) ? ((qp1_k6 & 0x0F) - 8) : ((qp1_k6 >> 4) - 8));
        const half qn2_k6 = (half)((k6_in_block < 16) ? ((qp2_k6 & 0x0F) - 8) : ((qp2_k6 >> 4) - 8));
        const half qn3_k6 = (half)((k6_in_block < 16) ? ((qp3_k6 & 0x0F) - 8) : ((qp3_k6 >> 4) - 8));
        half4 b_vals_k6 = (half4)(qn0_k6 * d0_k6, qn1_k6 * d1_k6, qn2_k6 * d2_k6, qn3_k6 * d3_k6);
        const int k7_block_idx = (k + 7) / 32;
        const int k7_in_block = (k + 7) % 32;
        const long b_idx0_k7 = (long)(n_start + 0) * n_stride + k7_block_idx;
        const long b_idx1_k7 = (long)(n_start + 1) * n_stride + k7_block_idx;
        const long b_idx2_k7 = (long)(n_start + 2) * n_stride + k7_block_idx;
        const long b_idx3_k7 = (long)(n_start + 3) * n_stride + k7_block_idx;
        const half d0_k7 = vload_half(0, (__global half *)(&(B[b_idx0_k7].d)));
        const half d1_k7 = vload_half(0, (__global half *)(&(B[b_idx1_k7].d)));
        const half d2_k7 = vload_half(0, (__global half *)(&(B[b_idx2_k7].d)));
        const half d3_k7 = vload_half(0, (__global half *)(&(B[b_idx3_k7].d)));
        const uchar qp0_k7 = B[b_idx0_k7].qs[k7_in_block % 16];
        const uchar qp1_k7 = B[b_idx1_k7].qs[k7_in_block % 16];
        const uchar qp2_k7 = B[b_idx2_k7].qs[k7_in_block % 16];
        const uchar qp3_k7 = B[b_idx3_k7].qs[k7_in_block % 16];
        const half qn0_k7 = (half)((k7_in_block < 16) ? ((qp0_k7 & 0x0F) - 8) : ((qp0_k7 >> 4) - 8));
        const half qn1_k7 = (half)((k7_in_block < 16) ? ((qp1_k7 & 0x0F) - 8) : ((qp1_k7 >> 4) - 8));
        const half qn2_k7 = (half)((k7_in_block < 16) ? ((qp2_k7 & 0x0F) - 8) : ((qp2_k7 >> 4) - 8));
        const half qn3_k7 = (half)((k7_in_block < 16) ? ((qp3_k7 & 0x0F) - 8) : ((qp3_k7 >> 4) - 8));
        half4 b_vals_k7 = (half4)(qn0_k7 * d0_k7, qn1_k7 * d1_k7, qn2_k7 * d2_k7, qn3_k7 * d3_k7);

        acc = mad(a_vals_lo.x, b_vals_k0, acc);
        acc = mad(a_vals_lo.y, b_vals_k1, acc);
        acc = mad(a_vals_lo.z, b_vals_k2, acc);
        acc = mad(a_vals_lo.w, b_vals_k3, acc);
        acc = mad(a_vals_hi.x, b_vals_k4, acc);
        acc = mad(a_vals_hi.y, b_vals_k5, acc);
        acc = mad(a_vals_hi.z, b_vals_k6, acc);
        acc = mad(a_vals_hi.w, b_vals_k7, acc);
    }

    for (int k = K_vec_size_x8 * 8; k < K; ++k) {
        int pixel_x = k / 4;
        int component = k % 4;
        half4 a_pixel = read_imageh(A, sampler, (int2)(pixel_x, 0));
        half a_val;
        if (component == 0)
            a_val = a_pixel.x;
        else if (component == 1)
            a_val = a_pixel.y;
        else if (component == 2)
            a_val = a_pixel.z;
        else
            a_val = a_pixel.w;
        const int k_block_idx = k / 32;
        const int k_in_block = k % 32;
        const long b_idx0 = (long)(n_start + 0) * n_stride + k_block_idx;
        const long b_idx1 = (long)(n_start + 1) * n_stride + k_block_idx;
        const long b_idx2 = (long)(n_start + 2) * n_stride + k_block_idx;
        const long b_idx3 = (long)(n_start + 3) * n_stride + k_block_idx;
        const half d0 = vload_half(0, (__global half *)(&(B[b_idx0].d)));
        const half d1 = vload_half(0, (__global half *)(&(B[b_idx1].d)));
        const half d2 = vload_half(0, (__global half *)(&(B[b_idx2].d)));
        const half d3 = vload_half(0, (__global half *)(&(B[b_idx3].d)));
        const uchar qp0 = B[b_idx0].qs[k_in_block % 16];
        const uchar qp1 = B[b_idx1].qs[k_in_block % 16];
        const uchar qp2 = B[b_idx2].qs[k_in_block % 16];
        const uchar qp3 = B[b_idx3].qs[k_in_block % 16];
        const half qn0 = (half)((k_in_block < 16) ? ((qp0 & 0x0F) - 8) : ((qp0 >> 4) - 8));
        const half qn1 = (half)((k_in_block < 16) ? ((qp1 & 0x0F) - 8) : ((qp1 >> 4) - 8));
        const half qn2 = (half)((k_in_block < 16) ? ((qp2 & 0x0F) - 8) : ((qp2 >> 4) - 8));
        const half qn3 = (half)((k_in_block < 16) ? ((qp3 & 0x0F) - 8) : ((qp3 >> 4) - 8));
        half4 b_vals = (half4)(qn0 * d0, qn1 * d1, qn2 * d2, qn3 * d3);
        acc = mad(a_val, b_vals, acc);
    }

    if (has_bias != 0) {
        acc += convert_half4_rte(vload4(0, bias + n_start));
    }
    write_imageh(C, (int2)(gx, 0), acc);
}
#else
// ---------- [兼容回退版] ----------
__kernel void gemv_fp16_q4_0_transb_bias_image_pipe(
    sampler_t sampler,
    __read_only image2d_t A,
    __global const block_q4_0 *B,
    __global const float *bias,
    __write_only image2d_t C,
    const int K, const int N,
    const int H,
    const int has_bias) {
    // 兼容版逻辑与FP32版本完全相同
    const int gx = get_global_id(0);
    const int n_start = gx * 4;
    if (n_start >= N) { return; }

    float4 acc = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    const int K_blocks = K / 32;
    const long n_stride = K_blocks;

    for (int k = 0; k < K; ++k) {
        int pixel_x = k / 4;
        int component = k % 4;
        float4 a_pixel = read_imagef(A, sampler, (int2)(pixel_x, 0));
        float a_val;
        if (component == 0)
            a_val = a_pixel.x;
        else if (component == 1)
            a_val = a_pixel.y;
        else if (component == 2)
            a_val = a_pixel.z;
        else
            a_val = a_pixel.w;

        const int k_block_idx = k / 32;
        const int k_in_block = k % 32;
        const long b_block_mem_idx0 = (long)(n_start + 0) * n_stride + k_block_idx;
        const long b_block_mem_idx1 = (long)(n_start + 1) * n_stride + k_block_idx;
        const long b_block_mem_idx2 = (long)(n_start + 2) * n_stride + k_block_idx;
        const long b_block_mem_idx3 = (long)(n_start + 3) * n_stride + k_block_idx;

        const float d_b0 = vload_half(0, (__global half *)(&(B[b_block_mem_idx0].d)));
        const float d_b1 = vload_half(0, (__global half *)(&(B[b_block_mem_idx1].d)));
        const float d_b2 = vload_half(0, (__global half *)(&(B[b_block_mem_idx2].d)));
        const float d_b3 = vload_half(0, (__global half *)(&(B[b_block_mem_idx3].d)));

        const uchar q_packed0 = B[b_block_mem_idx0].qs[k_in_block % 16];
        const char q_nibble0 = (k_in_block < 16) ? ((q_packed0 & 0x0F) - 8) : ((q_packed0 >> 4) - 8);
        const uchar q_packed1 = B[b_block_mem_idx1].qs[k_in_block % 16];
        const char q_nibble1 = (k_in_block < 16) ? ((q_packed1 & 0x0F) - 8) : ((q_packed1 >> 4) - 8);
        const uchar q_packed2 = B[b_block_mem_idx2].qs[k_in_block % 16];
        const char q_nibble2 = (k_in_block < 16) ? ((q_packed2 & 0x0F) - 8) : ((q_packed2 >> 4) - 8);
        const uchar q_packed3 = B[b_block_mem_idx3].qs[k_in_block % 16];
        const char q_nibble3 = (k_in_block < 16) ? ((q_packed3 & 0x0F) - 8) : ((q_packed3 >> 4) - 8);

        float4 b_vals = (float4)((float)q_nibble0 * d_b0, (float)q_nibble1 * d_b1, (float)q_nibble2 * d_b2, (float)q_nibble3 * d_b3);

        acc = mad(a_val, b_vals, acc);
    }

    if (has_bias != 0) {
        float4 bias_vals = vload4(0, bias + n_start);
        acc += bias_vals;
    }

    // 兼容版输出的Image也是FP32格式
    write_imagef(C, (int2)(gx, 0), acc);
}
#endif // SUPPORTS_FP16