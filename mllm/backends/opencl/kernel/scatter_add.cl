// 请用此完整代码替换您的 kernel/scatter_add.cl 文件

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

// ========================================================================
// 通用部分：fp32 内核 和 fp32 原子加法
// 这部分代码已经是正确的，保持不变。
// ========================================================================

void atomic_add_float(__global float *addr, float val) {
    union {
        unsigned int u32;
        float f32;
    } next, expected, current;
    current.f32 = *addr;
    do {
        expected.f32 = current.f32;
        next.f32 = expected.f32 + val;
        current.u32 = atomic_cmpxchg((volatile __global unsigned int *)addr, expected.u32, next.u32);
    } while (current.u32 != expected.u32);
}

__kernel void scatter_add_fp32(
    __global float *self_data,
    __global const float *value_data,
    __global const float *index_data,
    const int B, const int H, const int D,
    const int S_self, const int S_value) {
    const int d = get_global_id(0);
    const int h = get_global_id(1);
    const int bs_val = get_global_id(2);
    const int b = bs_val / S_value;
    const int s_val = bs_val % S_value;

    if (d >= D || h >= H || b >= B) {
        return;
    }
    size_t value_offset = (size_t)b * S_value * H * D + (size_t)s_val * H * D + (size_t)h * D + d;
    float value_to_add = value_data[value_offset];
    int target_seq_index = (int)index_data[s_val];
    size_t self_offset = (size_t)b * S_self * H * D + (size_t)target_seq_index * H * D + (size_t)h * D + d;
    atomic_add_float(&self_data[self_offset], value_to_add);
}

// ========================================================================
// FP16 实现部分：根据 SUPPORTS_FP16 宏进行条件编译
// ========================================================================

#ifdef SUPPORTS_FP16

// A. 如果设备支持 FP16，我们编译这部分代码

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/*
 * [最终修正] atomic_add_half:
 * 1. 使用 uintptr_t 和位运算代替取模，提高可移植性，解决编译失败问题。
 * 2. 使用 float 进行中间加法计算，保证精度和兼容性。
 */
void atomic_add_half(__global half *addr, half val) {
    // uintptr_t 是专门为存储指针而设计的整数类型，转换更安全。
    uintptr_t ptr_val = (uintptr_t)addr;
    // 使用位运算判断地址是在一个32位字的前半部分(0)还是后半部分(1)。
    // (ptr_val >> 1) & 1  等价于 ((ptr_val % 4) / 2)
    int half_idx_in_uint = (ptr_val >> 1) & 1;

    // 找到包含当前 half 的那个4字节对齐的地址
    volatile __global uint *addr_as_uint = (volatile __global uint *)(ptr_val - (half_idx_in_uint * 2));

    union {
        uint u32;
        half2 f16x2;
    } next, expected, current;

    do {
        current.u32 = *addr_as_uint;
        expected.u32 = current.u32;

        if (half_idx_in_uint == 0) { // 更新前半部分 (s0)
            float sum = (float)expected.f16x2.s0 + (float)val;
            next.f16x2.s0 = (half)sum;
            next.f16x2.s1 = expected.f16x2.s1;
        } else { // 更新后半部分 (s1)
            float sum = (float)expected.f16x2.s1 + (float)val;
            next.f16x2.s0 = expected.f16x2.s0;
            next.f16x2.s1 = (half)sum;
        }
    } while (atomic_cmpxchg(addr_as_uint, expected.u32, next.u32) != expected.u32);
}

__kernel void scatter_add_fp16(
    __global half *self_data,
    __global const half *value_data,
    __global const half *index_data,
    const int B, const int H, const int D,
    const int S_self, const int S_value) {
    const int d = get_global_id(0);
    const int h = get_global_id(1);
    const int bs_val = get_global_id(2);
    const int b = bs_val / S_value;
    const int s_val = bs_val % S_value;
    if (d >= D || h >= H || b >= B) {
        return;
    }
    size_t value_offset = (size_t)b * S_value * H * D + (size_t)s_val * H * D + (size_t)h * D + d;
    half value_to_add = value_data[value_offset];
    int target_seq_index = (int)convert_float(index_data[s_val]);
    size_t self_offset = (size_t)b * S_self * H * D + (size_t)target_seq_index * H * D + (size_t)h * D + d;
    atomic_add_half(&self_data[self_offset], value_to_add);
}
#else
// 辅助函数: 将 ushort (存储着 half 的二进制位) 转换为 float
// 注意: 这个函数内部不创建 half 变量，只进行位运算和类型双关转换
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
    if (exponent > 15) { return sign | 0x7C00; } // Infinity
    if (exponent < -14) {
        mantissa = (mantissa | 0x800000) >> (1 - exponent);
        return sign | (mantissa >> 13);
    }
    return sign | ((exponent + 15) << 10) | (mantissa >> 13);
}

// 辅助函数: 在一个 ushort 地址上，原子地加上一个 float 值
void atomic_add_float_to_ushort_location(__global ushort *addr, float val_to_add) {
    uintptr_t ptr_val = (uintptr_t)addr;
    int ushort_idx_in_uint = (ptr_val >> 1) & 1;
    volatile __global uint *addr_as_uint = (volatile __global uint *)(ptr_val - (ushort_idx_in_uint * 2));

    union {
        uint u32;
        ushort us16[2];
    } next, expected, current;

    do {
        current.u32 = *addr_as_uint;
        expected.u32 = current.u32;

        // 核心逻辑：解包 -> 转float -> 计算 -> 转ushort -> 打包
        ushort old_val_ushort = expected.us16[ushort_idx_in_uint];
        float old_val_float = ushort_to_float(old_val_ushort);
        float new_val_float = old_val_float + val_to_add;
        ushort new_val_ushort = float_to_ushort(new_val_float);

        next.u32 = expected.u32;                        // 先复制
        next.us16[ushort_idx_in_uint] = new_val_ushort; // 再修改目标部分

    } while (atomic_cmpxchg(addr_as_uint, expected.u32, next.u32) != expected.u32);
}

__kernel void scatter_add_fp16(
    __global ushort *self_data,
    __global const ushort *value_data,
    __global const ushort *index_data,
    const int B, const int H, const int D,
    const int S_self, const int S_value) {
    const int d = get_global_id(0);
    const int h = get_global_id(1);
    const int bs_val = get_global_id(2);
    const int b = bs_val / S_value;
    const int s_val = bs_val % S_value;
    if (d >= D || h >= H || b >= B) { return; }

    // 1. 读取源数据和索引，并立即转换为 float
    size_t value_offset = (size_t)b * S_value * H * D + (size_t)s_val * H * D + (size_t)h * D + d;
    // 使用 vload_half 读取并转换，遵从编译器建议
    float value_to_add = vload_half(value_offset, (__global half *)value_data);

    // 读取索引也同样处理
    float index_as_float = vload_half(s_val, (__global half *)index_data);
    int target_seq_index = (int)index_as_float;

    // 2. 计算目标地址
    size_t self_offset = (size_t)b * S_self * H * D + (size_t)target_seq_index * H * D + (size_t)h * D + d;

    // 3. 执行软件模拟的原子加法
    // 因为 vload/vstore 不支持原子操作，所以必须使用手动实现的原子函数
    atomic_add_float_to_ushort_location(&self_data[self_offset], value_to_add);
}
#endif // SUPPORTS_FP16