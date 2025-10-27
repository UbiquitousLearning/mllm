// 文件名: kernel/sub.cl

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// ==================================================================
// 1. Tensor - Tensor Subtraction Kernels
// ==================================================================

/**
 * @brief [FP32 Buffer版] 两个 float 张量的元素级减法 C = A - B
 */
__kernel void sub_float(
    __global const float *A,
    __global const float *B,
    __global float *C) {
    size_t index = get_global_id(0);
    C[index] = A[index] - B[index];
}

/**
 * @brief [FP32 Image版] 两个 float 图像的元素级减法
 */
__kernel void sub_float_image2d(
    sampler_t sampler,
    __read_only image2d_t inputA,
    __read_only image2d_t inputB,
    __write_only image2d_t output,
    const int width,
    const int height) {
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));

    if (pos.x >= width || pos.y >= height) {
        return;
    }

    float4 inA = read_imagef(inputA, sampler, pos);
    float4 inB = read_imagef(inputB, sampler, pos);
    float4 result = inA - inB;
    write_imagef(output, pos, result);
}

/**
 * @brief [FP16 Buffer版] 两个 half 张量的向量化元素级减法
 */
__kernel void sub_fp16_vector(
    __global const half *A,
    __global const half *B,
    __global half *C) {
    const int i = get_global_id(0);
    half4 a_vec = vload4(i, A);
    half4 b_vec = vload4(i, B);
    half4 c_vec = a_vec - b_vec;
    vstore4(c_vec, i, C);
}

/**
 * @brief [FP16 Image版] 两个 half 图像的元素级减法
 */
__kernel void sub_fp16_image2d(
    sampler_t sampler,
    __read_only image2d_t inputA,
    __read_only image2d_t inputB,
    __write_only image2d_t output,
    const int width,
    const int height) {
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));

    if (pos.x >= width || pos.y >= height) {
        return;
    }

    half4 inA = read_imageh(inputA, sampler, pos);
    half4 inB = read_imageh(inputB, sampler, pos);
    half4 result = inA - inB;
    write_imageh(output, pos, result);
}