// 文件名: kernel/div.cl

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
// ==================================================================
// 1. Tensor / Tensor Division Kernels (FP32)
// ==================================================================

__kernel void div_float(
    __global const float *A,
    __global const float *B,
    __global float *C,
    const int b_dim,
    const int a_dim) {
    size_t index = get_global_id(0);
    float b_val;

    // If b_dim is 1 and a_dim is greater than 1, apply broadcasting
    if (b_dim == 1 && a_dim > 1) {
        // Correct Broadcasting Logic:
        // 1. Get the current d coordinate from the global index of A
        int d_coord = index % a_dim;
        // 2. Get the BSH part of the index for A
        size_t a_bsh_index = index / a_dim;
        // 3. Since B's dimension is 1, its BSH index is the same as its global index
        size_t b_index = a_bsh_index;
        // This is the correct index for B
        b_val = B[b_index];
    } else {
        // Original element-wise division
        b_val = B[index];
    }

    // 添加保护防止除以零
    C[index] = b_val == 0.0f ? 0.0f : A[index] / b_val;
}

__kernel void div_float_image2d(
    sampler_t sampler,
    __read_only image2d_t inputA,
    __read_only image2d_t inputB,
    __write_only image2d_t output,
    const int width,
    const int height) {
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));
    if (pos.x >= width || pos.y >= height) { return; }
    float4 inA = read_imagef(inputA, sampler, pos);
    float4 inB = read_imagef(inputB, sampler, pos);
    // 添加保护防止除以零
    float4 result = (inB.x == 0.0f && inB.y == 0.0f && inB.z == 0.0f && inB.w == 0.0f) ?
                        (float4)(0.0f) :
                        inA / inB;
    write_imagef(output, pos, result);
}

// ==================================================================
// 2. Tensor / Scalar Division Kernels (FP32)
// ==================================================================

__kernel void div_scalar_float(
    __global const float *A,
    const float B,
    __global float *C) {
    size_t index = get_global_id(0);
    C[index] = B == 0.0f ? 0.0f : A[index] / B;
}

__kernel void div_scalar_float_image2d(
    sampler_t sampler,
    __read_only image2d_t inputA,
    const float B,
    __write_only image2d_t output,
    const int width,
    const int height) {
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));
    if (pos.x >= width || pos.y >= height) { return; }
    float4 inA = read_imagef(inputA, sampler, pos);
    float4 inB = (float4)(B);
    float4 result = (B == 0.0f) ? (float4)(0.0f) : inA / inB;
    write_imagef(output, pos, result);
}

// ==================================================================
// FP16 Kernels Implementations with Preprocessor Guards
// ==================================================================

#ifdef SUPPORTS_FP16

__kernel void div_fp16_vector(
    __global const half *A,
    __global const half *B,
    __global half *C,
    const int b_dim,
    const int a_dim) {
    const int i = get_global_id(0);
    // If b_dim is 1 and a_dim is greater than 1, apply broadcasting
    if (b_dim == 1 && a_dim > 1) {
        // Broadcasting case with correct indexing
        const int start_idx_A = i * 4;
        for (int j = 0; j < 4; ++j) {
            int current_idx_A = start_idx_A + j;
            // Correct Broadcasting Logic:
            size_t a_bsh_index = current_idx_A / a_dim;
            size_t b_index = a_bsh_index; // This is the correct index for B

            half b_val = B[b_index];
            half a_val = A[current_idx_A];
            C[current_idx_A] = (b_val == (half)0.0h) ? (half)0.0h : a_val / b_val;
        }
    } else {
        // Original element-wise vectorized division
        half4 a_vec = vload4(i, A);
        half4 b_vec = vload4(i, B);
        // 添加保护防止除以零 (转换为 float 进行比较)
        half4 c_vec = (all(convert_float4(b_vec) == 0.0f)) ?
                          (half4)(0.0h) :
                          a_vec / b_vec;
        vstore4(c_vec, i, C);
    }
}

// 新增的标量内核，用于处理任意尺寸的张量
__kernel void div_fp16_scalar(
    __global const half *A,
    __global const half *B,
    __global half *C,
    const int b_dim,
    const int a_dim) {
    size_t index = get_global_id(0);
    half b_val;
    if (b_dim == 1 && a_dim > 1) {
        size_t a_bsh_index = index / a_dim;
        size_t b_index = a_bsh_index;
        b_val = B[b_index];
    } else {
        b_val = B[index];
    }
    C[index] = (b_val == (half)0.0h) ? (half)0.0h : A[index] / b_val;
}

__kernel void div_fp16_image2d(
    sampler_t sampler,
    __read_only image2d_t inputA,
    __read_only image2d_t inputB,
    __write_only image2d_t output,
    const int width,
    const int height) {
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));
    if (pos.x >= width || pos.y >= height) { return; }
    half4 inA = read_imageh(inputA, sampler, pos);
    half4 inB = read_imageh(inputB, sampler, pos);
    half4 result = (all(convert_float4(inB) == 0.0f)) ? (half4)(0.0h) : inA / inB;
    write_imageh(output, pos, result);
}

__kernel void div_scalar_fp16_vector(
    __global const half *A,
    const float B,
    __global half *C) {
    const int i = get_global_id(0);
    float4 a_vec_f = convert_float4(vload4(i, A));

    // B 已经是 float，无需转换
    float4 c_vec_f = a_vec_f / B;
    vstore4(convert_half4_rte(c_vec_f), i, C);
}

__kernel void div_scalar_fp16_image2d(
    sampler_t sampler,
    __read_only image2d_t inputA,
    const float B,
    __write_only image2d_t output,
    const int width,
    const int height) {
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));
    if (pos.x >= width || pos.y >= height) {
        return;
    }

    // 1. 读取 half4 数据并立即提升到 float4
    float4 inA_f = convert_float4(read_imageh(inputA, sampler, pos));
    // 2. 在 float 精度下进行计算
    float4 result_f = inA_f / B;
    // 3. 将结果转换回 half4 并写入
    write_imageh(output, pos, convert_half4_rte(result_f));
}

#else // !SUPPORTS_FP16

// ===================== B. FP16实现 (软件回退) =====================
// 当硬件不支持时, 不使用'half'类型.
// 我们用'ushort'来存储16位数据, 并手动转换到'float'进行计算.
inline float half_to_float(ushort h) {
    const uint s = (h >> 15) & 0x0001;
    const uint e = (h >> 10) & 0x001f;
    const uint f = h & 0x03ff;
    uint float_val;
    if (e == 0) {
        if (f == 0) { // +0 or -0
            float_val = s << 31;
        } else { // Denormalized number to normalized float
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
    } else if (e == 31) { // Inf or NaN
        if (f == 0) {     // +/- Infinity
            float_val = (s << 31) | 0x7f800000;
        } else { // NaN
            float_val = (s << 31) | 0x7f800000 | (f << 13);
        }
    } else { // Normalized number
        float_val = (s << 31) | ((e + 112) << 23) | (f << 13);
    }

    return as_float(float_val);
}

// 帮助函数: 将 float 转换为 ushort (存储为half)
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

__kernel void div_fp16_vector(
    __global const ushort *A,
    __global const ushort *B,
    __global ushort *C,
    const int b_dim,
    const int a_dim) {
    const int i = get_global_id(0);
    // If b_dim is 1 and a_dim is greater than 1, apply broadcasting
    if (b_dim == 1 && a_dim > 1) {
        const int start_idx_A = i * 4;
        for (int j = 0; j < 4; ++j) {
            int current_idx_A = start_idx_A + j;
            // Correct Broadcasting Logic:
            size_t a_bsh_index = current_idx_A / a_dim;
            size_t b_index = a_bsh_index;

            float a_val = half_to_float(A[current_idx_A]);
            float b_val = half_to_float(B[b_index]);

            float result = b_val == 0.0f ?
                               0.0f :
                               a_val / b_val;
            C[current_idx_A] = float_to_half(result);
        }
    } else {
        // Original element-wise division for software fallback
        const int start_idx = i * 4;
        float4 a_vec = (float4)(half_to_float(A[start_idx]), half_to_float(A[start_idx + 1]), half_to_float(A[start_idx + 2]), half_to_float(A[start_idx + 3]));
        float4 b_vec = (float4)(half_to_float(B[start_idx]), half_to_float(B[start_idx + 1]), half_to_float(B[start_idx + 2]), half_to_float(B[start_idx + 3]));

        float4 c_vec;
        c_vec.x = b_vec.x == 0.0f ? 0.0f : a_vec.x / b_vec.x;
        c_vec.y = b_vec.y == 0.0f ?
                      0.0f :
                      a_vec.y / b_vec.y;
        c_vec.z = b_vec.z == 0.0f ? 0.0f : a_vec.z / b_vec.z;
        c_vec.w = b_vec.w == 0.0f ? 0.0f : a_vec.w / b_vec.w;

        C[start_idx] = float_to_half(c_vec.x);
        C[start_idx + 1] = float_to_half(c_vec.y);
        C[start_idx + 2] = float_to_half(c_vec.z);
        C[start_idx + 3] = float_to_half(c_vec.w);
    }
}

// 新增的标量内核的软件回退实现
__kernel void div_fp16_scalar(
    __global const ushort *A,
    __global const ushort *B,
    __global ushort *C,
    const int b_dim,
    const int a_dim) {
    size_t index = get_global_id(0);
    float b_val;

    if (b_dim == 1 && a_dim > 1) {
        size_t a_bsh_index = index / a_dim;
        size_t b_index = a_bsh_index;
        b_val = half_to_float(B[b_index]);
    } else {
        b_val = half_to_float(B[index]);
    }

    float a_val = half_to_float(A[index]);
    float result = b_val == 0.0f ? 0.0f : a_val / b_val;
    C[index] = float_to_half(result);
}

__kernel void div_fp16_image2d(
    sampler_t sampler,
    __read_only image2d_t inputA,
    __read_only image2d_t inputB,
    __write_only image2d_t output,
    const int width,
    const int height) {
    // 这是一个存根(stub)实现, 因为不支持cl_khr_fp16的平台
    // 通常也不支持CL_HALF_FLOAT图像格式.
    // 主机代码中该路径已通过&& false禁用.
    // 仅用于保证内核能被创建.
    return;
}

__kernel void div_scalar_fp16_vector(
    __global const ushort *A,
    const float B,
    __global ushort *C) {
    // 每个工作项依然负责4个元素，但我们将逐个处理它们
    const int i = get_global_id(0) * 4;
    // 临时存储4个float类型的结果
    float results[4];

    // 核心安全检查
    if (B == 0.0f) {
        results[0] = 0.0f;
        results[1] = 0.0f;
        results[2] = 0.0f;
        results[3] = 0.0f;
    } else {
        // 【关键改动】像 flash_attention.cl 一样，逐个加载、转换、计算
        results[0] = half_to_float(A[i + 0]) / B;
        results[1] = half_to_float(A[i + 1]) / B;
        results[2] = half_to_float(A[i + 2]) / B;
        results[3] = half_to_float(A[i + 3]) / B;
    }

    // 逐个转换回 half 并存储
    C[i + 0] = float_to_half(results[0]);
    C[i + 1] = float_to_half(results[1]);
    C[i + 2] = float_to_half(results[2]);
    C[i + 3] = float_to_half(results[3]);
}

__kernel void div_scalar_fp16_image2d(
    sampler_t sampler,
    __read_only image2d_t inputA,
    const ushort B,
    __write_only image2d_t output,
    const int width,
    const int height) {
    // 存根(stub)实现.
    return;
}

#endif // SUPPORTS_FP16