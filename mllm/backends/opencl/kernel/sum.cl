#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define SUM_WG_SIZE 256 // 工作组大小, 必须是2的幂

// ========================== FP32 (float) Kernel =============================
__kernel void sum_fp32(
    __global const float *input,
    __global float *output,
    const int outer_size,
    const int inner_size,
    const int reduce_size) {
    // 1. 获取全局和局部ID
    const int inner_idx = get_group_id(0);
    const int outer_idx = get_group_id(1);
    const int local_id = get_local_id(0);

    // 2. 在本地内存中声明共享数组，用于存储部分和
    __local float partial_sums[SUM_WG_SIZE];

    // 3. 并行计算部分和
    float thread_sum = 0.0f;
    // 每个线程负责累加 `reduce_size` 维度上的一部分数据
    for (int i = local_id; i < reduce_size; i += SUM_WG_SIZE) {
        size_t offset = (size_t)outer_idx * reduce_size * inner_size + (size_t)i * inner_size + inner_idx;
        thread_sum += input[offset];
    }
    partial_sums[local_id] = thread_sum;

    // 4. 工作组内同步，确保所有线程都已算完自己的部分和
    barrier(CLK_LOCAL_MEM_FENCE);

    // 5. 在本地内存中进行并行规约
    for (int s = SUM_WG_SIZE / 2; s > 0; s >>= 1) {
        if (local_id < s) {
            partial_sums[local_id] += partial_sums[local_id + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // 6. 由工作组的第一个线程将最终结果写回全局内存
    if (local_id == 0) {
        size_t out_offset = (size_t)outer_idx * inner_size + inner_idx;
        output[out_offset] = partial_sums[0];
    }
}

// ========================== FP16 (half) Kernel ==============================
__kernel void sum_fp16(
    __global const half *input,
    __global half *output,
    const int outer_size,
    const int inner_size,
    const int reduce_size) {
    const int inner_idx = get_group_id(0);
    const int outer_idx = get_group_id(1);
    const int local_id = get_local_id(0);

    // 使用 float 累加器以保证精度
    __local float partial_sums[SUM_WG_SIZE];
    float thread_sum = 0.0f;

    for (int i = local_id; i < reduce_size; i += SUM_WG_SIZE) {
        size_t offset = (size_t)outer_idx * reduce_size * inner_size + (size_t)i * inner_size + inner_idx;
        thread_sum += (float)input[offset];
    }
    partial_sums[local_id] = thread_sum;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = SUM_WG_SIZE / 2; s > 0; s >>= 1) {
        if (local_id < s) {
            partial_sums[local_id] += partial_sums[local_id + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        size_t out_offset = (size_t)outer_idx * inner_size + inner_idx;
        output[out_offset] = (half)partial_sums[0];
    }
}