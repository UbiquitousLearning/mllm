__kernel void mul_float(__global const float *A, __global const float *B,
                        __global float *C) {
  size_t index = get_global_id(0);
  C[index] = A[index] * B[index];
}

__kernel void mul_broadcast_float(__global const float *A, __global const float *B, __global float *C, 
                            int batch_size, int loop_size, int vector_size, 
                            int batch_stride_a, int loop_stride_a, 
                            int batch_stride_b, int loop_stride_b) {
    int global_id = get_global_id(0);
    
    int v = global_id % vector_size;
    int tmp = global_id / vector_size;
    int l = tmp % loop_size;
    int b = tmp / loop_size;
    
    if (b >= batch_size) return;

    int offset_a = b * batch_stride_a + l * loop_stride_a + v;
    int offset_b = b * batch_stride_b + l * loop_stride_b + v;
    
    C[global_id] = A[offset_a] * B[offset_b];
}
