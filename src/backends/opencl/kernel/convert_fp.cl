// ===================================================================
// **部分一：兼容模式内核（适用于无原生FP16支持的硬件）**
// ===================================================================

// 将一个32位float稳健地转换为16位ushort (模拟FP16)
ushort float_to_half_bits(const float f) {
    // 将 float 的位模式存储在 uint 中
    const uint u = as_uint(f);
    // 1. 正确提取各部分
    // 提取符号位 (bit 31)
    const uint sign_bit = u & 0x80000000;
    // 提取指数位 (bits 30-23)
    int exponent = (int)((u >> 23) & 0xff) - 127; // FP32 bias is 127
    // 提取尾数位 (bits 22-0)
    uint mantissa = u & 0x007fffff;
    // 将32位符号位移动到16位的位置
    const ushort half_sign = (ushort)(sign_bit >> 16);
    // 2. 处理特殊值：NaN 和 Infinity
    if (exponent == 128) { // FP32 exponent is all 1s
        if (mantissa == 0) {
            // FP32 is Infinity, convert to FP16 Infinity
            return half_sign | 0x7c00;
        } else {
            // FP32 is NaN, convert to FP16 NaN
            // 保留尾数最高位以标识为NaN
            return half_sign | 0x7c00 | (ushort)(mantissa >> 13);
        }
    }
    // 3. 处理上溢到 Infinity
    if (exponent > 15) {           // Exponent too large for FP16 normal
        return half_sign | 0x7c00; // Overflow -> Infinity
    }
    // 4. 处理正规数 (Normal Numbers)
    if (exponent >= -14) {
        // 将FP32指数转换为FP16指数 (FP16 bias is 15)
        ushort half_exponent = (ushort)(exponent + 15);
        // 添加被截断部分的最高位，以实现简单的“四舍五入”
        mantissa = mantissa + (1 << 12);
        // 如果舍入导致尾数上溢，需要调整指数
        if (mantissa & 0x00800000) {
            mantissa = 0;
            half_exponent++;
            if (half_exponent == 31) { // 溢出到 Infinity
                return half_sign | 0x7c00;
            }
        }
        ushort half_mantissa = (ushort)(mantissa >> 13);
        return half_sign | (half_exponent << 10) | half_mantissa;
    }
    // 5. 处理非正规数 (Denormalized Numbers)
    if (exponent >= -24) {
        // 加上隐藏位，然后根据指数进行移位
        mantissa = (mantissa | 0x00800000) >> (14 - exponent);
        // 同样可以增加舍入逻辑
        mantissa += 1;
        return half_sign | (ushort)(mantissa >> 1);
    }
    // 6. 处理下溢到 0
    return half_sign; // Underflow -> Zero
}

// [已优化] 将一个16位ushort (模拟FP16) 转换回32位float
float half_bits_to_float(const ushort h) {
    const uint sign = (uint)(h & 0x8000) << 16;
    uint exponent = (h & 0x7c00) >> 10;
    uint mantissa = h & 0x03ff;

    if (exponent == 0x1f) { // Infinity or NaN
        // 直接构造对应的FP32 Infinity/NaN
        return as_float(sign | 0x7f800000 | (mantissa << 13));
    }

    if (exponent == 0) {     // Zero or Denormal
        if (mantissa == 0) { // Zero
            return as_float(sign);
        }
        // Denormalized: 找到隐藏的 '1'
        while ((mantissa & 0x0400) == 0) {
            mantissa <<= 1;
            exponent--;
        }
        exponent++;         // 补偿循环多减的一次
        mantissa &= 0x03ff; // 移除隐藏的 '1'
    }

    // 转换为FP32的指数和尾数
    exponent = exponent + (127 - 15);
    mantissa = mantissa << 13;

    return as_float(sign | (exponent << 23) | mantissa);
}

__kernel void convert_fp32_to_fp16_buffer_compat( // 兼容版内核
    __global const float *input,
    __global ushort *output,
    const int count) {
    int i = get_global_id(0);
    if (i < count) output[i] = float_to_half_bits(input[i]);
}

__kernel void convert_fp16_to_fp32_buffer_compat( // 兼容版内核
    __global const ushort *input,
    __global float *output,
    const int count) {
    int i = get_global_id(0);
    if (i < count) output[i] = half_bits_to_float(input[i]);
}

// ===================================================================
// **部分二：高性能内核（需要硬件原生支持FP16）**
// ===================================================================
#ifdef SUPPORTS_FP16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// --- Buffer Kernels (高性能版) ---
__kernel void convert_fp32_to_fp16_buffer_ext(
    __global const float *input,
    __global half *output,
    const int count) {
    int i = get_global_id(0);
    if (i < count) output[i] = convert_half(input[i]);
}

__kernel void convert_fp16_to_fp32_buffer_ext(
    __global const half *input,
    __global float *output,
    const int count) {
    int i = get_global_id(0);
    if (i < count) output[i] = vload_half(i, input);
}

// --- Image Kernels (高性能版) ---
__kernel void convert_fp32_to_fp16_image2d(
    sampler_t sampler,
    __read_only image2d_t input_fp32,
    __write_only image2d_t output_fp16,
    const int width,
    const int height) {
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));
    if (pos.x >= width || pos.y >= height) return;
    float4 data_fp32 = read_imagef(input_fp32, sampler, pos);
    half4 data_fp16 = convert_half4(data_fp32);
    write_imageh(output_fp16, pos, data_fp16);
}

__kernel void convert_fp16_to_fp32_image2d(
    sampler_t sampler,
    __read_only image2d_t input_fp16,
    __write_only image2d_t output_fp32,
    const int width,
    const int height) {
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));
    if (pos.x >= width || pos.y >= height) return;
    half4 data_fp16 = read_imageh(input_fp16, sampler, pos);
    float4 data_fp32 = convert_float4(data_fp16);
    write_imagef(output_fp32, pos, data_fp32);
}

#endif // SUPPORTS_FP16