__kernel void add_float(__global const float *A, __global const float *B,
                        __global float *C) {
  size_t index = get_global_id(0);
  C[index] = A[index] + B[index];
}

/*
 * Define a global sampler. A sampler is used to configure how data is read from
 * image objects. CLK_NORMALIZED_COORDS_FALSE: Use non-normalized integer
 * coordinates (pixel coordinates) instead of floating point coordinates in
 * [0.0, 1.0]. CLK_ADDRESS_CLAMP_TO_EDGE: When reading coordinates beyond the
 * image boundary, automatically return the pixel value of the nearest edge to
 * prevent out-of-bounds reading. CLK_FILTER_NEAREST: Read the pixel closest to
 * the coordinate without interpolation, which is essential for data
 * calculation.
 */
// const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
// CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
/**
 * @brief Efficiently perform element-wise addition of two float tensors using
 * image2d_t.
 * @param sampler     Sampler object used to read images.
 * @param inputA      Input tensor A, as a read-only 2D image object.
 * @param inputB      Input tensor B, as a read-only 2D image object.
 * @param output      Output tensor C, as a write-only 2D image object.
 * @param width       Logical width of the image (in pixels).
 * @param height      Logical height of the image (in pixels).
 */
__kernel void
add_float_image2d(sampler_t sampler, // Sampler is now the first parameter
                  __read_only image2d_t inputA, __read_only image2d_t inputB,
                  __write_only image2d_t output, const int width,
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
 * @brief [FP16 Buffer Version] Efficiently perform element-wise addition of two
 * half tensors using vector instructions. The kernel uses vload4/vstore4 to
 * process 4 half elements at once.
 * @param A           Input tensor A (__global const half*)
 * @param B           Input tensor B (__global const half*)
 * @param C           Output tensor C (__global half*)
 */
__kernel void add_fp16_vector(__global const half *A, __global const half *B,
                              __global half *C) {
  const int i = get_global_id(0);

  // Efficiently load 4 half (total 64 bits) data
  half4 a_vec = vload4(i, A);
  half4 b_vec = vload4(i, B);

  // Vector addition
  half4 c_vec = a_vec + b_vec;

  // Efficiently write back 4 half data
  vstore4(c_vec, i, C);
}

/**
 * @brief [FP16 Image Version] Efficiently perform element-wise addition of two
 * half tensors using image2d_t. Utilize hardware texture cache and
 * read_imageh/write_imageh functions.
 * @param sampler     Sampler object used to read images.
 * @param inputA      Input tensor A, as a read-only 2D image object (data type
 * is half).
 * @param inputB      Input tensor B, as a read-only 2D image object (data type
 * is half).
 * @param output      Output tensor C, as a write-only 2D image object (data
 * type is half).
 * @param width       Logical width of the image (in pixels).
 * @param height      Logical height of the image (in pixels).
 */
__kernel void add_fp16_image2d(sampler_t sampler, __read_only image2d_t inputA,
                               __read_only image2d_t inputB,
                               __write_only image2d_t output, const int width,
                               const int height) {
  const int2 pos = (int2)(get_global_id(0), get_global_id(1));

  if (pos.x >= width || pos.y >= height) {
    return;
  }

  // Use read_imageh to read half4 vector
  half4 inA = read_imageh(inputA, sampler, pos);
  half4 inB = read_imageh(inputB, sampler, pos);

  half4 result = inA + inB;

  // Use write_imageh to write back half4 vector
  write_imageh(output, pos, result);
}

// ==================================================================
// 4. Tensor + Scalar Kernels
// ==================================================================

/**
 * @brief [FP32 Buffer Version] Add a scalar `B` to each element of tensor `A`.
 */
__kernel void add_scalar_float(__global const float *A, const float B,
                               __global float *C) {
  size_t index = get_global_id(0);
  C[index] = A[index] + B;
}

/**
 * @brief [FP16 Buffer Version] Use vector instructions to add a scalar `B` to
 * each element of tensor `A`.
 */
__kernel void add_scalar_fp16_vector(__global const half *A, const half B,
                                     __global half *C) {
  const int i = get_global_id(0);
  half4 a_vec = vload4(i, A);
  // Broadcast scalar B to a half4 vector
  half4 b_vec = (half4)(B);
  half4 c_vec = a_vec + b_vec;
  vstore4(c_vec, i, C);
}

/**
 * @brief [FP32 Image Version] Add a scalar `B` to each pixel of tensor `A`.
 */
__kernel void add_scalar_float_image2d(sampler_t sampler,
                                       __read_only image2d_t inputA,
                                       const float B,
                                       __write_only image2d_t output,
                                       const int width, const int height) {
  const int2 pos = (int2)(get_global_id(0), get_global_id(1));
  if (pos.x >= width || pos.y >= height) {
    return;
  }

  float4 inA = read_imagef(inputA, sampler, pos);
  // Broadcast scalar B to a float4 vector
  float4 inB = (float4)(B);
  float4 result = inA + inB;
  write_imagef(output, pos, result);
}

/**
 * @brief [FP16 Image Version] Add a scalar `B` to each pixel of tensor `A`.
 */
__kernel void add_scalar_fp16_image2d(sampler_t sampler,
                                      __read_only image2d_t inputA,
                                      const half B,
                                      __write_only image2d_t output,
                                      const int width, const int height) {
  const int2 pos = (int2)(get_global_id(0), get_global_id(1));
  if (pos.x >= width || pos.y >= height) {
    return;
  }

  half4 inA = read_imageh(inputA, sampler, pos);
  // Broadcast scalar B to a half4 vector
  half4 inB = (half4)(B);
  half4 result = inA + inB;
  write_imageh(output, pos, result);
}