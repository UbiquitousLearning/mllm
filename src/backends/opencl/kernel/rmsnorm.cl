// 文件名: rmsnorm.cl

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// 定义工作组内用于并行归约的线程数（必须是2的幂）
#define RMSNORM_WG_SIZE 256

// 与 C++ DataType.hpp 中定义匹配的量化块结构
typedef struct {
    half d;
    uchar qs[16]; // QK4_0 / 2 = 16
} block_q4_0;

// 内核辅助函数：实时从 Q4_0 块中解量化一个元素
inline float dequantize_q4_0(const __global block_q4_0 *blocks, int index) {
    const int block_idx = index / 32; // QK4_0 is 32
    const int quant_idx_in_block = index % 32;

    // 定位到对应的 Q4_0 块
    const __global block_q4_0 *b = &blocks[block_idx];

    // 从 uchar 中提取 4-bit 的量化值
    const uchar quant_pair = b->qs[quant_idx_in_block / 2];
    const int nibble = (quant_idx_in_block % 2 == 0) ? (quant_pair & 0x0F) : (quant_pair >> 4);

    // 应用反量化公式
    return (float)b->d * (float)(nibble - 8);
}

__kernel void rmsnorm_f32_q4(
    __global const float *src,    // 输入张量 (fp32)
    __global float *dst,          // 输出张量 (fp32)
    __global const void *weights, // 权重张量 (可以是 fp32 或 q4_0)
    const int weight_is_q4,       // 标志位：0 表示权重是 fp32, 1 表示是 q4_0
    const int D,                  // Dimension, 即每行的长度
    const float epsilon,          // epsilon 值，防止除以零
    const int add_unit_offset     // 标志位：是否对权重执行 +1 操作
) {
    // 1. 获取ID
    const int row_id = get_group_id(0);   // 每个工作组处理一行，行ID由工作组ID决定
    const int local_id = get_local_id(0); // 工作组内的线程ID

    // 2. 在本地内存中声明共享数组
    __local float local_sum_sq[RMSNORM_WG_SIZE];

    // 3. 并行计算平方和
    float thread_sum_sq = 0.0f; // 每个线程计算一部分元素的平方和
    for (int i = local_id; i < D; i += RMSNORM_WG_SIZE) {
        float val = src[row_id * D + i];
        thread_sum_sq += val * val;
    }
    local_sum_sq[local_id] = thread_sum_sq;

    // 4. 工作组内归约（Reduction），计算整个行的总平方和
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = RMSNORM_WG_SIZE / 2; s > 0; s >>= 1) {
        if (local_id < s) {
            local_sum_sq[local_id] += local_sum_sq[local_id + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // 此时，local_sum_sq[0] 中存放了整个行的平方和

    // 5. 计算 RMS 缩放因子并安全地广播
    float rms_val;
    // 只有工作组的第一个线程进行这个标量计算
    if (local_id == 0) {
        float variance = local_sum_sq[0] / D;
        rms_val = rsqrt(variance + epsilon);
        local_sum_sq[0] = rms_val; // 线程0计算结果并存入共享内存
    }

    // 同步点：确保所有线程都等待线程0将rms_val写入共享内存
    barrier(CLK_LOCAL_MEM_FENCE);

    // 所有线程（包括线程0）从共享内存中读取广播的值
    rms_val = local_sum_sq[0];

    // 6. 并行执行归一化和应用权重
    for (int i = local_id; i < D; i += RMSNORM_WG_SIZE) {
        // a. 获取权重值
        float weight_val;
        if (weight_is_q4) {
            weight_val = dequantize_q4_0((const __global block_q4_0 *)weights, i);
        } else {
            weight_val = ((const __global float *)weights)[i];
        }

        // b. 根据标志位决定是否加1
        if (add_unit_offset) {
            weight_val += 1.0f;
        }

        // c. 计算最终结果并写回全局内存
        size_t index = row_id * D + i;
        dst[index] = src[index] * rms_val * weight_val;
    }
}

// ==================================================================
// 2.  FP16 Input Kernel (rmsnorm_f16_q4)
// ==================================================================
__kernel void rmsnorm_f16_q4(
    __global const half *src,     // 输入张量 (fp16)
    __global half *dst,           // 输出张量 (fp16)
    __global const void *weights, // 权重张量 (可以是 fp32 或 q4_0)
    const int weight_is_q4,       // 标志位
    const int D,                  // Dimension
    const float epsilon,          // Epsilon (仍然是 float)
    const int add_unit_offset     // 标志位
) {
    const int row_id = get_group_id(0);
    const int local_id = get_local_id(0);

    __local float local_sum_sq[RMSNORM_WG_SIZE];

    // 使用 float 累加器以保证精度
    float thread_sum_sq = 0.0f;
    for (int i = local_id; i < D; i += RMSNORM_WG_SIZE) {
        // 从 half 转换为 float 进行计算
        float val = (float)src[row_id * D + i];
        thread_sum_sq += val * val;
    }
    local_sum_sq[local_id] = thread_sum_sq;

    // 工作组内归约 (与fp32版本完全相同)
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = RMSNORM_WG_SIZE / 2; s > 0; s >>= 1) {
        if (local_id < s) {
            local_sum_sq[local_id] += local_sum_sq[local_id + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // 计算 RMS 缩放因子 (与fp32版本完全相同)
    float rms_val;
    if (local_id == 0) {
        float variance = local_sum_sq[0] / D;
        rms_val = rsqrt(variance + epsilon);
        local_sum_sq[0] = rms_val;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    rms_val = local_sum_sq[0];

    // 归一化和应用权重
    for (int i = local_id; i < D; i += RMSNORM_WG_SIZE) {
        float weight_val;
        if (weight_is_q4) {
            weight_val = dequantize_q4_0((const __global block_q4_0 *)weights, i);
        } else {
            weight_val = ((const __global float *)weights)[i];
        }
        if (add_unit_offset) {
            weight_val += 1.0f;
        }

        size_t index = row_id * D + i;
        // 计算结果为 float，最后转换回 half 存入 dst
        float src_val = (float)src[index];
        dst[index] = (half)(src_val * rms_val * weight_val);
    }
}