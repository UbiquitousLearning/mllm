
// 将一个32位float转换为16位ushort (模拟FP16)
ushort float_to_half_bits(const float f) {
    const uint u = as_uint(f);
    const uint sign = (u >> 16) & 0x8000;
    uint exponent = ((u >> 23) & 0xff) - 127;
    uint mantissa = u & 0x7fffff;

    if (exponent > 15) return sign | 0x7c00;
    if (exponent >= -14) {
        exponent = (exponent + 15) << 10;
        mantissa >>= 13;
        return sign | exponent | mantissa;
    }
    if (exponent >= -24) {
        mantissa = (mantissa | 0x800000) >> (14 - (-exponent));
        return sign | mantissa;
    }
    return sign;
}

// 将一个16位ushort (模拟FP16) 转换回32位float
float half_bits_to_float(const ushort h) {
    const uint sign = (h >> 15) & 1;
    uint exponent = (h >> 10) & 0x1f;
    uint mantissa = h & 0x3ff;

    if (exponent == 0x1f) return as_float((sign << 31) | 0x7f800000 | (mantissa << 13));
    if (exponent == 0) {
        if (mantissa == 0) return as_float(sign << 31);
        mantissa <<= 1;
        while ((mantissa & 0x400) == 0) { mantissa <<= 1; exponent--; }
        mantissa &= 0x3ff; exponent += 1;
    }
    exponent = exponent + (127 - 15);
    mantissa <<= 13;
    return as_float((sign << 31) | (exponent << 23) | mantissa);
}

__kernel void convert_fp32_to_fp16_buffer_compat( // 兼容版内核
    __global const float* input,
    __global ushort* output,
    const int count)
{
    int i = get_global_id(0);
    if (i < count) output[i] = float_to_half_bits(input[i]);
}

__kernel void convert_fp16_to_fp32_buffer_compat( // 兼容版内核
    __global const ushort* input,
    __global float* output,
    const int count)
{
    int i = get_global_id(0);
    if (i < count) output[i] = half_bits_to_float(input[i]);
}

// ===================================================================
// **部分二：高性能内核（需要硬件原生支持FP16）**
// 只有在C++端定义了 "SUPPORTS_FP16" 宏时，这部分代码才会被编译。
// ===================================================================
#ifdef SUPPORTS_FP16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// --- Buffer Kernels (高性能版) ---
__kernel void convert_fp32_to_fp16_buffer_ext( // 高性能版内核
    __global const float* input,
    __global half* output,
    const int count)
{
    int i = get_global_id(0);
    if (i < count) output[i] = convert_half(input[i]);
}

__kernel void convert_fp16_to_fp32_buffer_ext( // 高性能版内核
    __global const half* input,
    __global float* output,
    const int count)
{
    int i = get_global_id(0);
    if (i < count) output[i] = vload_half(i, input);
}

// --- Image Kernels (高性能版) ---
// Image的FP16转换本身就依赖硬件支持，所以它只在高性能路径下才有意义。
__kernel void convert_fp32_to_fp16_image2d(
    sampler_t sampler,
    __read_only image2d_t input_fp32,
    __write_only image2d_t output_fp16,
    const int width,
    const int height)
{
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
    const int height)
{
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));
    if (pos.x >= width || pos.y >= height) return;
    half4 data_fp16 = read_imageh(input_fp16, sampler, pos);
    float4 data_fp32 = convert_float4(data_fp16);
    write_imagef(output_fp32, pos, data_fp32);
}

#endif // SUPPORTS_FP16