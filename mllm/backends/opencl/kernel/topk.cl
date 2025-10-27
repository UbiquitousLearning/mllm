#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define WG_SIZE 256 // 工作组大小，必须是2的幂

// 交换两个 pair
void swap_pairs(volatile __local float *values, volatile __local int *indices, int i, int j) {
    float temp_val = values[i];
    values[i] = values[j];
    values[j] = temp_val;

    int temp_idx = indices[i];
    indices[i] = indices[j];
    indices[j] = temp_idx;
}

// ========================== FP32 (float) Kernel =============================
__kernel void topk_fp32(
    __global const float *input,
    __global float *topk_values,
    __global float *topk_indices,
    const int D, // dimension to sort along
    const int k) {
    // ** FIX: All __local memory declarations moved to the top-level scope **
    __local float local_values[WG_SIZE];
    __local int local_indices[WG_SIZE];
    __local float wg_max_vals[WG_SIZE];
    __local int wg_max_indices[WG_SIZE];

    const int row_id = get_group_id(0);
    const int local_id = get_local_id(0);

    // 1. 并行加载一行数据到本地内存
    for (int i = local_id; i < D; i += WG_SIZE) {
        local_values[i] = input[row_id * D + i];
        local_indices[i] = i;
    }
    // 处理 D < WG_SIZE 的情况，将多余的本地内存元素初始化为最小值
    if (local_id >= D && local_id < WG_SIZE) {
        local_values[local_id] = -FLT_MAX;
        local_indices[local_id] = -1;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // 2. 在本地内存中进行部分排序 (选择排序的变体)
    for (int i = 0; i < k; ++i) {
        // 在工作组内并行查找 [i, D) 区间内的最大值
        float thread_max_val = -FLT_MAX;
        int thread_max_idx = -1;
        for (int j = i + local_id; j < D; j += WG_SIZE) {
            if (local_values[j] > thread_max_val) {
                thread_max_val = local_values[j];
                thread_max_idx = j;
            }
        }

        // 使用本地内存进行归约，找到整个工作组的最大值
        wg_max_vals[local_id] = thread_max_val;
        wg_max_indices[local_id] = thread_max_idx;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int s = WG_SIZE / 2; s > 0; s >>= 1) {
            if (local_id < s) {
                if (wg_max_vals[local_id + s] > wg_max_vals[local_id]) {
                    wg_max_vals[local_id] = wg_max_vals[local_id + s];
                    wg_max_indices[local_id] = wg_max_indices[local_id + s];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        int max_idx = -1;
        if (local_id == 0) {
            max_idx = wg_max_indices[0];
        }
        barrier(CLK_LOCAL_MEM_FENCE); // 确保所有线程都看到了max_idx

        // 将找到的最大值与第 i 个元素交换
        if (max_idx != -1 && max_idx != i) {
            if (local_id == 0) {
                swap_pairs(local_values, local_indices, i, max_idx);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // 3. 由前 k 个线程将结果写回全局内存
    if (local_id < k) {
        topk_values[row_id * k + local_id] = local_values[local_id];
        // 将索引转换为 float 类型写入
        topk_indices[row_id * k + local_id] = (float)local_indices[local_id];
    }
}

// ========================== FP16 (half) Kernel ==============================
__kernel void topk_fp16(
    __global const half *input,
    __global half *topk_values,
    __global half *topk_indices,
    const int D,
    const int k) {
    // ** FIX: All __local memory declarations moved to the top-level scope **
    __local float local_values[WG_SIZE];
    __local int local_indices[WG_SIZE];
    __local float wg_max_vals[WG_SIZE];
    __local int wg_max_indices[WG_SIZE];

    const int row_id = get_group_id(0);
    const int local_id = get_local_id(0);

    for (int i = local_id; i < D; i += WG_SIZE) {
        local_values[i] = (float)input[row_id * D + i];
        local_indices[i] = i;
    }
    if (local_id >= D && local_id < WG_SIZE) {
        local_values[local_id] = -FLT_MAX;
        local_indices[local_id] = -1;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 0; i < k; ++i) {
        float thread_max_val = -FLT_MAX;
        int thread_max_idx = -1;
        for (int j = i + local_id; j < D; j += WG_SIZE) {
            if (local_values[j] > thread_max_val) {
                thread_max_val = local_values[j];
                thread_max_idx = j;
            }
        }

        wg_max_vals[local_id] = thread_max_val;
        wg_max_indices[local_id] = thread_max_idx;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int s = WG_SIZE / 2; s > 0; s >>= 1) {
            if (local_id < s) {
                if (wg_max_vals[local_id + s] > wg_max_vals[local_id]) {
                    wg_max_vals[local_id] = wg_max_vals[local_id + s];
                    wg_max_indices[local_id] = wg_max_indices[local_id + s];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        int max_idx = -1;
        if (local_id == 0) {
            max_idx = wg_max_indices[0];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (max_idx != -1 && max_idx != i) {
            if (local_id == 0) {
                swap_pairs(local_values, local_indices, i, max_idx);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id < k) {
        topk_values[row_id * k + local_id] = (half)local_values[local_id];
        topk_indices[row_id * k + local_id] = (half)local_indices[local_id];
    }
}