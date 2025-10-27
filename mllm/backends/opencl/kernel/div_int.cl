// 文件名: kernel/div_int.cl

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// ==================================================================
// 1. Tensor / Scalar Division with Integer Truncation (FP32)
// ==================================================================

__kernel void div_int_scalar_float(
    __global const float *A,
    const float B,
    __global float *C) {
    size_t index = get_global_id(0);
    int val_a = convert_int_rtz(A[index]);
    int val_b = convert_int_rtz(B);
    int result = (val_b == 0) ? 0 : (val_a / val_b);
    C[index] = (float)result;
}

__kernel void div_int_scalar_float_image2d(
    sampler_t sampler,
    __read_only image2d_t inputA,
    const float B,
    __write_only image2d_t output,
    const int width,
    const int height) {
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));
    if (pos.x >= width || pos.y >= height) { return; }
    int4 val_a = convert_int4_rtz(read_imagef(inputA, sampler, pos));
    int val_b = convert_int_rtz(B);
    int4 result = (val_b == 0) ? (int4)(0) : (val_a / val_b);
    write_imagef(output, pos, convert_float4(result));
}

// ==================================================================
// 2. FP16 Kernels (Hardware vs. Software Fallback)
// ==================================================================

#ifdef SUPPORTS_FP16

// A. FP16实现 (硬件原生支持)
__kernel void div_int_scalar_fp16_vector(
    __global const half *A,
    const float B,
    __global half *C) {
    const int i = get_global_id(0);
    float4 val_a_f = convert_float4(vload4(i, A));
    int4 val_a = convert_int4_rtz(val_a_f);
    int val_b = convert_int_rtz(B);
    int4 result = (val_b == 0) ? (int4)(0) : (val_a / val_b);
    vstore4(convert_half4_rte(result), i, C);
}

__kernel void div_int_scalar_fp16_image2d(
    sampler_t sampler,
    __read_only image2d_t inputA,
    const float B,
    __write_only image2d_t output,
    const int width,
    const int height) {
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));
    if (pos.x >= width || pos.y >= height) { return; }
    float4 val_a_f = convert_float4(read_imageh(inputA, sampler, pos));
    int4 val_a = convert_int4_rtz(val_a_f);
    int val_b = convert_int_rtz(B);
    int4 result = (val_b == 0) ? (int4)(0) : (val_a / val_b);
    write_imageh(output, pos, convert_half4_rte(result));
}

// ADDED: Scalar Kernel (for any data count)
__kernel void div_int_scalar_fp16(
    __global const half *A,
    const float B,
    __global half *C) {
    size_t index = get_global_id(0);
    float val_a_f = convert_float(A[index]);
    int val_a = convert_int_rtz(val_a_f);
    int val_b = convert_int_rtz(B);
    int result = (val_b == 0) ? 0 : (val_a / val_b);
    C[index] = convert_half_rte((float)result);
}

#else // !SUPPORTS_FP16

// B. FP16实现 (软件回退)
inline float half_to_float(ushort h) {
    const uint s = (h >> 15) & 0x0001;
    const uint e = (h >> 10) & 0x001f;
    const uint f = h & 0x03ff;
    uint float_val;
    if (e == 0) {
        if (f == 0) {
            float_val = s << 31;
        } else {
            uint f_shifted = f;
            uint e_shifted = e;
            while ((f_shifted & 0x0400) == 0) {
                f_shifted <<= 1;
                e_shifted--;
            }
            e_shifted++;
            f_shifted &= ~0x0400;
            float_val = (s << 31) | ((e_shifted + 112) << 23) | (f_shifted << 13);
        }
    } else if (e == 31) {
        if (f == 0) {
            float_val = (s << 31) | 0x7f800000;
        } else {
            float_val = (s << 31) | 0x7f800000 | (f << 13);
        }
    } else {
        float_val = (s << 31) | ((e + 112) << 23) | (f << 13);
    }
    return as_float(float_val);
}

inline ushort float_to_half(float f) {
    uint u = as_uint(f);
    uint s = (u >> 16) & 0x8000;
    int e = ((u >> 23) & 0xFF) - 127;
    uint f_mant = u & 0x7FFFFF;

    if (e > 15) return (ushort)(s | 0x7C00);
    if (e < -14) {
        f_mant |= 0x800000;
        return (ushort)(s | (f_mant >> (-e - 14)));
    }
    return (ushort)(s | ((e + 15) << 10) | (f_mant >> 13));
}

__kernel void div_int_scalar_fp16_vector(
    __global const ushort *A,
    const float B,
    __global ushort *C) {
    const int i = get_global_id(0) * 4;
    int val_b = convert_int_rtz(B);
    if (val_b == 0) {
        C[i + 0] = float_to_half(0.0f);
        C[i + 1] = float_to_half(0.0f);
        C[i + 2] = float_to_half(0.0f);
        C[i + 3] = float_to_half(0.0f);
        return;
    }
    int val_a0 = convert_int_rtz(half_to_float(A[i + 0]));
    int val_a1 = convert_int_rtz(half_to_float(A[i + 1]));
    int val_a2 = convert_int_rtz(half_to_float(A[i + 2]));
    int val_a3 = convert_int_rtz(half_to_float(A[i + 3]));
    int res0 = val_a0 / val_b;
    int res1 = val_a1 / val_b;
    int res2 = val_a2 / val_b;
    int res3 = val_a3 / val_b;
    C[i + 0] = float_to_half((float)res0);
    C[i + 1] = float_to_half((float)res1);
    C[i + 2] = float_to_half((float)res2);
    C[i + 3] = float_to_half((float)res3);
}

__kernel void div_int_scalar_fp16_image2d(
    sampler_t sampler,
    __read_only image2d_t inputA,
    const float B,
    __write_only image2d_t output,
    const int width,
    const int height) {
    // Stub implementation as this path is not used for non-aligned data
    return;
}

// ADDED: Scalar Kernel Fallback (for any data count)
__kernel void div_int_scalar_fp16(
    __global const ushort *A,
    const float B,
    __global ushort *C) {
    size_t index = get_global_id(0);
    int val_a = convert_int_rtz(half_to_float(A[index]));
    int val_b = convert_int_rtz(B);
    int result = (val_b == 0) ? 0 : (val_a / val_b);
    C[index] = float_to_half((float)result);
}

#endif // SUPPORTS_FP16