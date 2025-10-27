// 文件名: kernel/silu.cl

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// ============================================================================
// ========================== FP32 (float) Kernel =============================
// ============================================================================
/**
 * @brief 对 float 张量执行 SiLU (Swish) 激活函数。
 * out = x * (1 / (1 + exp(-x)))
 */
__kernel void silu_fp32(
    __global const float *input,
    __global float *output) {
    const int i = get_global_id(0);
    const float x = input[i];

    // 计算 SiLU
    output[i] = x / (1.0f + exp(-x));
}

// ============================================================================
// ========================== FP16 (half) Kernel ==============================
// ============================================================================
/**
 * @brief 对 half 张量使用向量指令执行 SiLU (Swish) 激活函数。
 * 一次处理4个 half 元素。
 */
// __kernel void silu_fp16_vector(
//     __global const half *input,
//     __global half *output) {
//     const int i = get_global_id(0);

//     // 高效地加载 4 个 half 数据
//     half4 x = vload4(i, input);

//     // 计算 SiLU
//     // 注意: OpenCL C 中，exp() 等数学函数可以直接作用于向量类型
//     half4 result = x / ((half4)(1.0h) + exp(-x));

//     // 高效地写回 4 个 half 数据
//     vstore4(result, i, output);
// }

__kernel void silu_fp16(
    __global const half *input,
    __global half *output,
    const int count) {
    // --- 向量化部分 ---
    const int vec_idx = get_global_id(0);
    const int vec_limit = count / 4;

    if (vec_idx < vec_limit) {
        const int i = vec_idx * 4;
        half4 x = vload4(0, input + i);
        half4 result = x / ((half4)(1.0h) + exp(-x));
        vstore4(result, 0, output + i);
    }

    // --- 标量处理部分 (处理余数) ---
    // ✨ **核心修正点** ✨: 'local_id' 替换为 'get_local_id(0)'
    // 让第一个工作组中ID合适的线程来处理余下的部分
    const int remainder_start = vec_limit * 4;
    if (get_local_id(0) < (count - remainder_start) && get_group_id(0) == 0) {
        const int i = remainder_start + get_local_id(0);
        if (i < count) { // 再次检查边界，确保安全
            const half x = input[i];
            output[i] = x / (1.0h + exp(-x));
        }
    }
}