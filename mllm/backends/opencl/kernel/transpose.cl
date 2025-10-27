#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// 定义在本地内存中转置的块大小
#define BLOCK_DIM 16

// // ============================================================================
// // ========================== FP32 (float) Kernel =============================
// // ============================================================================
// __kernel void transpose_float_2d(
//     const __global float *src, // 源矩阵指针 (指向一个 S x D 块)
//     __global float *dst,       // 目标矩阵指针 (指向一个 D x S 块)
//     const int S,               // 源矩阵的行数
//     const int D                // 源矩阵的列数
// ) {
//     // 1. 定义一个本地内存块
//     __local float tile[BLOCK_DIM][BLOCK_DIM + 1]; // +1 用于避免 bank conflict

//     // 2. 计算全局ID (在 S x D 矩阵内的坐标)
//     int d = get_global_id(0);
//     int s = get_global_id(1);

//     // 3. 从全局内存加载数据到本地内存
//     if (d < D && s < S) {
//         tile[get_local_id(1)][get_local_id(0)] = src[s * D + d];
//     }

//     // 4. 同步工作组，确保所有数据都已加载
//     barrier(CLK_LOCAL_MEM_FENCE);

//     // 5. 计算转置后的新坐标
//     int new_d = get_group_id(1) * BLOCK_DIM + get_local_id(0);
//     int new_s = get_group_id(0) * BLOCK_DIM + get_local_id(1);

//     // 6. 从本地内存读取转置后的元素并写回全局内存
//     if (new_d < S && new_s < D) {
//         // 读取时索引交换，实现转置
//         dst[new_s * S + new_d] = tile[get_local_id(0)][get_local_id(1)];
//     }
// }

// // ============================================================================
// // ========================== FP16 (half) Kernel ==============================
// // ============================================================================
// __kernel void transpose_fp16_2d(
//     const __global half *src,
//     __global half *dst,
//     const int S,
//     const int D) {
//     __local half tile[BLOCK_DIM][BLOCK_DIM + 1];

//     int d = get_global_id(0);
//     int s = get_global_id(1);

//     if (d < D && s < S) {
//         tile[get_local_id(1)][get_local_id(0)] = src[s * D + d];
//     }

//     barrier(CLK_LOCAL_MEM_FENCE);

//     int new_d = get_group_id(1) * BLOCK_DIM + get_local_id(0);
//     int new_s = get_group_id(0) * BLOCK_DIM + get_local_id(1);

//     if (new_d < S && new_s < D) {
//         dst[new_s * S + new_d] = tile[get_local_id(0)][get_local_id(1)];
//     }
// }

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// 定义在本地内存中转置的块大小
#define BLOCK_DIM 16

// ============================================================================
// ========================== FP32 (float) Kernel =============================
// ============================================================================
__kernel void transpose_float_2d(
    const __global float *src,     // 源矩阵的基地址指针
    __global float *dst,           // 目标矩阵的基地址指针
    const int S,                   // 子矩阵的行数
    const int D,                   // 子矩阵的列数
    const int src_offset_elements, // 【新增】源数据在基地址上的偏移量 (以元素为单位)
    const int dst_offset_elements  // 【新增】目标数据在基地址上的偏移量 (以元素为单位)
) {
    // 1. 定义一个本地内存块
    __local float tile[BLOCK_DIM][BLOCK_DIM + 1]; // +1 用于避免 bank conflict

    // 【修改】根据偏移量获取子矩阵的实际指针
    const __global float *src_ptr = src + src_offset_elements;
    __global float *dst_ptr = dst + dst_offset_elements;

    // 2. 计算全局ID (在 S x D 子矩阵内的坐标)
    int d = get_global_id(0);
    int s = get_global_id(1);

    // 3. 从全局内存加载数据到本地内存
    if (d < D && s < S) {
        tile[get_local_id(1)][get_local_id(0)] = src_ptr[s * D + d];
    }

    // 4. 同步工作组，确保所有数据都已加载
    barrier(CLK_LOCAL_MEM_FENCE);

    // 5. 计算转置后的新坐标
    int new_d = get_group_id(1) * BLOCK_DIM + get_local_id(0);
    int new_s = get_group_id(0) * BLOCK_DIM + get_local_id(1);

    // 6. 从本地内存读取转置后的元素并写回全局内存
    if (new_d < S && new_s < D) {
        // 读取时索引交换，实现转置
        dst_ptr[new_s * S + new_d] = tile[get_local_id(0)][get_local_id(1)];
    }
}

// ============================================================================
// ========================== FP16 (half) Kernel ==============================
// ============================================================================
__kernel void transpose_fp16_2d(
    const __global half *src,
    __global half *dst,
    const int S,
    const int D,
    const int src_offset_elements, // 【新增】源数据在基地址上的偏移量
    const int dst_offset_elements  // 【新增】目标数据在基地址上的偏移量
) {
    __local half tile[BLOCK_DIM][BLOCK_DIM + 1];

    // 【修改】根据偏移量获取子矩阵的实际指针
    const __global half *src_ptr = src + src_offset_elements;
    __global half *dst_ptr = dst + dst_offset_elements;

    int d = get_global_id(0);
    int s = get_global_id(1);

    if (d < D && s < S) {
        tile[get_local_id(1)][get_local_id(0)] = src_ptr[s * D + d];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int new_d = get_group_id(1) * BLOCK_DIM + get_local_id(0);
    int new_s = get_group_id(0) * BLOCK_DIM + get_local_id(1);

    if (new_d < S && new_s < D) {
        dst_ptr[new_s * S + new_d] = tile[get_local_id(0)][get_local_id(1)];
    }
}

// ============================================================================
// ============ BSHD -> BHSD FP32 (float) Kernel ==============================
// ============================================================================
__kernel void transpose_bshd2bhsd_fp32(
    const __global float *src, // 源张量指针 (BSHD 布局)
    __global float *dst,       // 目标张量指针 (BHSD 布局)
    const int B,               // Batch size
    const int H,               // Head count
    const int S,               // Sequence length
    const int D                // Dimension
) {
    // 使用3D工作项ID来唯一标识目标张量中的每一个元素
    const int d = get_global_id(0);
    const int s = get_global_id(1);
    const int hb = get_global_id(2); // h 和 b 的组合索引

    // 检查边界
    if (d >= D || s >= S || hb >= H * B) {
        return;
    }

    // 从组合索引中分解出 h 和 b
    const int h = hb % H;
    const int b = hb / H;

    // 根据 BSHD 布局计算源索引
    // src[b][s][h][d]
    size_t src_idx = (size_t)b * S * H * D + (size_t)s * H * D + (size_t)h * D + d;

    // 根据 BHSD 布局计算目标索引
    // dst[b][h][s][d]
    size_t dst_idx = (size_t)b * H * S * D + (size_t)h * S * D + (size_t)s * D + d;

    // 执行数据拷贝
    dst[dst_idx] = src[src_idx];
}

// ============================================================================
// ============ BSHD -> BHSD FP16 (half) Kernel ===============================
// ============================================================================
__kernel void transpose_bshd2bhsd_fp16(
    const __global half *src,
    __global half *dst,
    const int B,
    const int H,
    const int S,
    const int D) {
    const int d = get_global_id(0);
    const int s = get_global_id(1);
    const int hb = get_global_id(2);

    if (d >= D || s >= S || hb >= H * B) {
        return;
    }

    const int h = hb % H;
    const int b = hb / H;

    size_t src_idx = (size_t)b * S * H * D + (size_t)s * H * D + (size_t)h * D + d;
    size_t dst_idx = (size_t)b * H * S * D + (size_t)h * S * D + (size_t)s * D + d;

    dst[dst_idx] = src[src_idx];
}

// ============================================================================
// ============ BHSD -> BSHD FP32 (float) Kernel (新增) ========================
// ============================================================================
__kernel void transpose_bhsd2bshd_fp32(
    const __global float *src, // 源张量指针 (BHSD 布局)
    __global float *dst,       // 目标张量指针 (BSHD 布局)
    const int B,
    const int H,
    const int S,
    const int D) {
    const int d = get_global_id(0);
    const int s = get_global_id(1);
    const int hb = get_global_id(2);

    if (d >= D || s >= S || hb >= H * B) {
        return;
    }

    const int h = hb % H;
    const int b = hb / H;

    // 根据 BHSD 布局计算源索引
    // src[b][h][s][d]
    size_t src_idx = (size_t)b * H * S * D + (size_t)h * S * D + (size_t)s * D + d;

    // 根据 BSHD 布局计算目标索引
    // dst[b][s][h][d]
    size_t dst_idx = (size_t)b * S * H * D + (size_t)s * H * D + (size_t)h * D + d;

    dst[dst_idx] = src[src_idx];
}

// ============================================================================
// ============ BHSD -> BSHD FP16 (half) Kernel (新增) ========================
// ============================================================================
__kernel void transpose_bhsd2bshd_fp16(
    const __global half *src,
    __global half *dst,
    const int B,
    const int H,
    const int S,
    const int D) {
    const int d = get_global_id(0);
    const int s = get_global_id(1);
    const int hb = get_global_id(2);

    if (d >= D || s >= S || hb >= H * B) {
        return;
    }

    const int h = hb % H;
    const int b = hb / H;

    // 根据 BHSD 布局计算源索引
    size_t src_idx = (size_t)b * H * S * D + (size_t)h * S * D + (size_t)s * D + d;
    // 根据 BSHD 布局计算目标索引
    size_t dst_idx = (size_t)b * S * H * D + (size_t)s * H * D + (size_t)h * D + d;

    dst[dst_idx] = src[src_idx];
}