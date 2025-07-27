__kernel void add_float(
    __global const float *A,
    __global const float *B,
    __global float *C) {
    size_t index = get_global_id(0);
    C[index] = A[index] + B[index];
}

/*
 * 定义一个全局的采样器（sampler）。采样器是用于配置如何从图像对象中读取数据的。
 * CLK_NORMALIZED_COORDS_FALSE: 使用非归一化的整数坐标（像素坐标），而不是[0.0, 1.0]的浮点坐标。
 * CLK_ADDRESS_CLAMP_TO_EDGE: 当读取坐标超出图像边界时，自动返回最接近的边界上的像素值，可有效防止越界读取。
 * CLK_FILTER_NEAREST: 读取最接近坐标的那个像素，不做任何插值，这对于数据计算是必须的。
 */
// const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
/**
 * @brief 使用 image2d_t 对两个 float 类型的张量进行高效的元素级相加。
 * @param sampler     用于读取图像的采样器对象。
 * @param inputA      输入张量 A，作为只读的 2D 图像对象。
 * @param inputB      输入张量 B，作为只读的 2D 图像对象。
 * @param output      输出张量 C，作为只写的 2D 图像对象。
 * @param width       图像的逻辑宽度（单位：像素）。
 * @param height      图像的逻辑高度（单位：像素）。
 */
__kernel void add_float_image2d(
    sampler_t sampler, // 采样器现在是第一个参数
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
    float4 result = inA + inB;
    write_imagef(output, pos, result);
}

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/**
 * @brief [FP16 Buffer版] 使用向量指令对两个 half 类型的张量进行高效的元素级相加。
 * 内核利用 vload4/vstore4 一次处理4个 half 元素。
 * @param A           输入张量 A (__global const half*)
 * @param B           输入张量 B (__global const half*)
 * @param C           输出张量 C (__global half*)
 */
__kernel void add_fp16_vector(
    __global const half *A,
    __global const half *B,
    __global half *C) {
    const int i = get_global_id(0);

    // 高效地加载 4 个 half (共64位) 数据
    half4 a_vec = vload4(i, A);
    half4 b_vec = vload4(i, B);

    // 向量加法
    half4 c_vec = a_vec + b_vec;

    // 高效地写回 4 个 half 数据
    vstore4(c_vec, i, C);
}

/**
 * @brief [FP16 Image版] 使用 image2d_t 对两个 half 类型的张量进行高效的元素级相加。
 * 利用硬件纹理缓存和 read_imageh/write_imageh 函数。
 * @param sampler     用于读取图像的采样器对象。
 * @param inputA      输入张量 A，作为只读的 2D 图像对象 (数据类型为 half)。
 * @param inputB      输入张量 B，作为只读的 2D 图像对象 (数据类型为 half)。
 * @param output      输出张量 C，作为只写的 2D 图像对象 (数据类型为 half)。
 * @param width       图像的逻辑宽度（单位：像素）。
 * @param height      图像的逻辑高度（单位：像素）。
 */
__kernel void add_fp16_image2d(
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

    // 使用 read_imageh 读取 half4 向量
    half4 inA = read_imageh(inputA, sampler, pos);
    half4 inB = read_imageh(inputB, sampler, pos);

    half4 result = inA + inB;

    // 使用 write_imageh 写回 half4 向量
    write_imageh(output, pos, result);
}

// ==================================================================
// 4. Tensor + Scalar 内核
// ==================================================================

/**
 * @brief [FP32 Buffer版] 将一个标量 `B` 加到张量 `A` 的每个元素上。
 */
__kernel void add_scalar_float(
    __global const float *A,
    const float B,
    __global float *C) {
    size_t index = get_global_id(0);
    C[index] = A[index] + B;
}

/**
 * @brief [FP16 Buffer版] 使用向量指令将一个标量 `B` 加到张量 `A` 的每个元素上。
 */
__kernel void add_scalar_fp16_vector(
    __global const half *A,
    const half B,
    __global half *C) {
    const int i = get_global_id(0);
    half4 a_vec = vload4(i, A);
    // 将标量 B 广播成一个 half4 向量
    half4 b_vec = (half4)(B);
    half4 c_vec = a_vec + b_vec;
    vstore4(c_vec, i, C);
}

/**
 * @brief [FP32 Image版] 将一个标量 `B` 加到张量 `A` 的每个像素上。
 */
__kernel void add_scalar_float_image2d(
    sampler_t sampler,
    __read_only image2d_t inputA,
    const float B,
    __write_only image2d_t output,
    const int width,
    const int height) {
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));
    if (pos.x >= width || pos.y >= height) { return; }

    float4 inA = read_imagef(inputA, sampler, pos);
    // 将标量 B 广播成一个 float4 向量
    float4 inB = (float4)(B);
    float4 result = inA + inB;
    write_imagef(output, pos, result);
}

/**
 * @brief [FP16 Image版] 将一个标量 `B` 加到张量 `A` 的每个像素上。
 */
__kernel void add_scalar_fp16_image2d(
    sampler_t sampler,
    __read_only image2d_t inputA,
    const half B,
    __write_only image2d_t output,
    const int width,
    const int height) {
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));
    if (pos.x >= width || pos.y >= height) { return; }

    half4 inA = read_imageh(inputA, sampler, pos);
    // 将标量 B 广播成一个 half4 向量
    half4 inB = (half4)(B);
    half4 result = inA + inB;
    write_imageh(output, pos, result);
}