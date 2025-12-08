__kernel void div_float(__global const float *A, __global const float *B,
                        __global float *C) {
  size_t index = get_global_id(0);
  C[index] = A[index] / B[index];
}
