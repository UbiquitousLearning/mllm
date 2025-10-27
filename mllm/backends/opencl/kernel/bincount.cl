// opencl/kernel/bincount.cl

#if defined(SUPPORTS_FP16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

// ===================== FP16 (ushort) <-> FP32 (float) 转换辅助函数 =====================
// 只有在不支持原生 FP16 时才需要
#if !defined(SUPPORTS_FP16)
static float ushort_to_float(ushort u) {
    uint sign = (u >> 15) & 1;
    uint exponent = (u >> 10) & 0x1F;
    uint mantissa = u & 0x3FF;
    if (exponent == 0x1F) { return as_float((sign << 31) | (0xFF << 23) | (mantissa << 13)); }
    if (exponent == 0) {
        if (mantissa == 0) { return as_float(sign << 31); }
        exponent = 1;
        while ((mantissa & 0x400) == 0) {
            mantissa <<= 1;
            exponent++;
        }
        mantissa &= 0x3FF;
        exponent = 127 - 15 - exponent + 1;
        return as_float((sign << 31) | (exponent << 23) | (mantissa << 13));
    }
    return as_float((sign << 31) | ((exponent - 15 + 127) << 23) | (mantissa << 13));
}

static ushort float_to_ushort(float f) {
    uint u = as_uint(f);
    uint sign = (u >> 16) & 0x8000;
    int exponent = ((u >> 23) & 0xFF) - 127;
    uint mantissa = u & 0x7FFFFF;
    if (exponent > 15) { return sign | 0x7C00; }
    if (exponent < -14) {
        mantissa = (mantissa | 0x800000) >> (1 - exponent);
        return sign | (mantissa >> 13);
    }
    return sign | ((exponent + 15) << 10) | (mantissa >> 13);
}
#endif

// ========================== 内核 1: 执行 Bincount 到整数缓冲区 ==========================
__kernel void bincount_count(
#if defined(SUPPORTS_FP16)
    __global const half *input,
#else
    __global const ushort *input,
#endif
    __global int *out_counts,
    const int size,
    const int max_val) {

    for (int i = get_global_id(0); i < size; i += get_global_size(0)) {
        float val;
#if defined(SUPPORTS_FP16)
        val = input[i];
#else
        val = ushort_to_float(input[i]);
#endif
        int index = (int)val;
        if (index >= 0 && index <= max_val) {
            atomic_add(&out_counts[index], 1);
        }
    }
}

// Float32 版本的 bincount 内核
__kernel void bincount_count_fp32(
    __global const float *input,
    __global int *out_counts,
    const int size,
    const int max_val) {
    for (int i = get_global_id(0); i < size; i += get_global_size(0)) {
        int index = (int)input[i];
        if (index >= 0 && index <= max_val) {
            atomic_add(&out_counts[index], 1);
        }
    }
}

// ========================== 内核 2: 将整数计数转换为 FP32/FP16 ==========================
__kernel void cast_int_to_float(
    __global const int *int_buffer,
    __global float *float_buffer,
    const int count) {
    for (int i = get_global_id(0); i < count; i += get_global_size(0)) {
        float_buffer[i] = (float)int_buffer[i];
    }
}

__kernel void cast_int_to_half(
    __global const int *int_buffer,
#if defined(SUPPORTS_FP16)
    __global half *half_buffer,
#else
    __global ushort *half_buffer,
#endif
    const int count) {
    for (int i = get_global_id(0); i < count; i += get_global_size(0)) {
#if defined(SUPPORTS_FP16)
        half_buffer[i] = (half)int_buffer[i];
#else
        half_buffer[i] = float_to_ushort((float)int_buffer[i]);
#endif
    }
}