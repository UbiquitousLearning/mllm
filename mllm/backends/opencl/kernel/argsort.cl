#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// ============================================================================
//  通用内核 (FP32 和 索引初始化)
// ============================================================================

/**
 * @brief 初始化索引数组，值为 0, 1, 2, ..., N-1。
 */
__kernel void init_indices(
    __global int *indices,
    const int N) {
    const int gid = get_global_id(0);
    const int local_idx = gid % N;
    indices[gid] = local_idx;
}

/**
 * @brief Bitonic Sort 的一个步骤 (FP32 版本)。
 */
__kernel void bitonic_argsort_step_fp32(
    __global float *values,
    __global int *indices,
    const int N,
    const int stage,
    const int pass,
    const int descending) {
    const int i = get_global_id(0);
    const int row = get_global_id(1);
    int k = 1 << pass;
    int j = i & (k - 1);
    int ix = (i << 1) - j;
    int iy = ix + k;
    int sort_increasing = ((ix >> (stage + 1)) & 1) == 0;
    if (descending) {
        sort_increasing = !sort_increasing;
    }

    if (ix < N) { // 仅当比较对的第一个元素在界内时才处理
        const int row_offset = row * N;
        const int global_ix = row_offset + ix;
        const int global_iy = row_offset + iy;

        float val1 = values[global_ix];
        // 如果 iy 越界, 根据排序方向，将其值视作 INFINITY (无穷大) 或 -INFINITY (负无穷大)
        float val2 = (iy < N) ? values[global_iy] : (descending ? -INFINITY : INFINITY);

        if ((val1 > val2) == sort_increasing) {
            // 当需要交换时，只对真实存在的元素进行写操作
            int index_from_ix = indices[global_ix];
            int index_from_iy = (iy < N) ? indices[global_iy] : -1; // 用-1作为无效索引哨兵

            values[global_ix] = val2;
            indices[global_ix] = index_from_iy;

            // 只有当 iy 也在界内时，才把 val1 的值和原始索引写入 iy 位置
            if (iy < N) {
                values[global_iy] = val1;
                indices[global_iy] = index_from_ix;
            }
        }
    }
}

/**
 * @brief 将排序后的 int 类型索引转换为 float 类型并写入输出。
 */
__kernel void cast_indices_to_fp32(
    __global const int *sorted_indices,
    __global float *output) {
    const int gid = get_global_id(0);
    output[gid] = (float)sorted_indices[gid];
}

// ============================================================================
//  FP16 内核 (根据硬件支持情况进行条件编译)
// ============================================================================

#if defined(SUPPORTS_FP16)

// =================== 方案A: 设备支持原生 FP16 =====================

/**
 * @brief Bitonic Sort 的一个步骤 (原生FP16版本)。
 */
__kernel void bitonic_argsort_step_fp16(
    __global half *values,
    __global int *indices,
    const int N,
    const int stage,
    const int pass,
    const int descending) {
    const int i = get_global_id(0);
    const int row = get_global_id(1);
    int k = 1 << pass;
    int j = i & (k - 1);
    int ix = (i << 1) - j;
    int iy = ix + k;
    int sort_increasing = ((ix >> (stage + 1)) & 1) == 0;
    if (descending) {
        sort_increasing = !sort_increasing;
    }

    if (ix < N) {
        const int row_offset = row * N;
        const int global_ix = row_offset + ix;
        const int global_iy = row_offset + iy;

        half val1 = values[global_ix];

        // ====================== ✨✨✨ 核心修正区域 as_half ✨✨✨ ======================
        // 修正: 显式地将整型字面量转换为 ushort (16-bit) 来解决 as_half 的歧义
        half infinity_h = as_half((ushort)0x7C00);
        half neg_infinity_h = as_half((ushort)0xFC00);
        // ==========================================================================

        half val2 = (iy < N) ? values[global_iy] : (descending ? neg_infinity_h : infinity_h);

        if ((val1 > val2) == sort_increasing) {
            int index_from_ix = indices[global_ix];
            int index_from_iy = (iy < N) ? indices[global_iy] : -1;

            values[global_ix] = val2;
            indices[global_ix] = index_from_iy;

            if (iy < N) {
                values[global_iy] = val1;
                indices[global_iy] = index_from_ix;
            }
        }
    }
}

/**
 * @brief 将排序后的 int 类型索引转换为 half 类型并写入输出。
 */
__kernel void cast_indices_to_fp16(
    __global const int *sorted_indices,
    __global half *output) {
    const int gid = get_global_id(0);
    output[gid] = (half)sorted_indices[gid];
}

#else

// =================== 方案B: 软件模拟 FP16 (兼容版) =====================
// ✨✨✨ 核心修正区域：严格仿照您的榜样代码 ✨✨✨

// 辅助函数: 将 ushort (存储着 half 的二进制位) 转换为 float
// (该函数仿照您项目中的 kernel/convert_fp.cl 和 scatter_add.cl)
static float ushort_to_float(ushort u) {
    uint sign = (u >> 15) & 1;
    uint exponent = (u >> 10) & 0x1F;
    uint mantissa = u & 0x3FF;
    uint result_uint;
    if (exponent == 0) {
        if (mantissa == 0) {
            result_uint = sign << 31;
        } else {
            exponent = 1;
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                exponent++;
            }
            mantissa &= 0x3FF;
            exponent = 127 - 15 - exponent + 1;
            result_uint = (sign << 31) | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1F) {
        result_uint = (sign << 31) | (0xFF << 23) | (mantissa << 13);
    } else {
        exponent = exponent - 15 + 127;
        result_uint = (sign << 31) | (exponent << 23) | (mantissa << 13);
    }
    return as_float(result_uint);
}

// 辅助函数: 将 float 转换为 ushort
static ushort float_to_ushort(float f) {
    uint u = as_uint(f);
    uint sign = (u >> 16) & 0x8000;
    int exponent = ((u >> 23) & 0xFF) - 127;
    uint mantissa = u & 0x7FFFFF;
    if (exponent > 15) {
        return sign | 0x7C00;
    } // Infinity
    if (exponent < -14) {
        mantissa = (mantissa | 0x800000) >> (1 - exponent);
        return sign | (mantissa >> 13);
    }
    return sign | ((exponent + 15) << 10) | (mantissa >> 13);
}

/**
 * @brief Bitonic Sort 的一个步骤 (兼容版FP16)。
 * 使用与原生版相同的名称，但输入输出为 ushort*。
 */
__kernel void bitonic_argsort_step_fp16(
    __global ushort *values, // 数据以 ushort 形式存储
    __global int *indices,
    const int N,
    const int stage,
    const int pass,
    const int descending) {
    const int i = get_global_id(0);
    const int row = get_global_id(1);
    int k = 1 << pass;
    int j = i & (k - 1);
    int ix = (i << 1) - j;
    int iy = ix + k;
    int sort_increasing = ((ix >> (stage + 1)) & 1) == 0;
    if (descending) {
        sort_increasing = !sort_increasing;
    }

    if (ix < N) {
        const int row_offset = row * N;
        const int global_ix = row_offset + ix;
        const int global_iy = row_offset + iy;

        // 核心: 读出 ushort, 转换为 float 进行比较
        float val1 = ushort_to_float(values[global_ix]);
        float val2 = (iy < N) ? ushort_to_float(values[global_iy]) : (descending ? -INFINITY : INFINITY);

        if ((val1 > val2) == sort_increasing) {
            // 交换时，直接交换原始的 ushort 值
            ushort val_from_ix = values[global_ix];
            // 如果 iy 越界，需要一个代表无穷大的 ushort 值
            ushort val_from_iy = (iy < N) ? values[global_iy] : float_to_ushort(val2);
            values[global_ix] = val_from_iy;

            // 交换索引
            int temp_idx_from_ix = indices[global_ix];
            int temp_idx_from_iy = (iy < N) ? indices[global_iy] : -1;
            indices[global_ix] = temp_idx_from_iy;

            if (iy < N) {
                values[global_iy] = val_from_ix;
                indices[global_iy] = temp_idx_from_ix;
            }
        }
    }
}

/**
 * @brief 将排序后的 int 类型索引转换为 ushort 类型并写入输出 (兼容版)。
 */
__kernel void cast_indices_to_fp16(
    __global const int *sorted_indices,
    __global ushort *output) { // 输出是 ushort*
    const int gid = get_global_id(0);
    // 先将 int 转为 float, 再将 float 的二进制位表示转为 ushort
    output[gid] = float_to_ushort((float)sorted_indices[gid]);
}

#endif // SUPPORTS_FP16