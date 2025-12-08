__kernel void matmul_buffer_nt_nt(const int M, const int N, const int K,
                                  __global float *A, const int A_batch_stride,
                                  __global float *B, const int B_batch_stride,
                                  __global float *C, const int C_batch_stride) {
  const int m = get_global_id(0);
  const int n = get_global_id(1);
  const int b = get_global_id(2);

  if (m >= M || n >= N)
    return;

  const int a_offset = b * A_batch_stride;
  const int b_offset = b * B_batch_stride;
  const int c_offset = b * C_batch_stride;

  float sum = 0.0f;
  for (int k = 0; k < K; ++k) {
    sum += A[a_offset + m * K + k] * B[b_offset + k * N + n];
  }

  C[c_offset + m * N + n] = sum;
}

__kernel void matmul_buffer_nt_t(const int M, const int N, const int K,
                                 __global float *A, const int A_batch_stride,
                                 __global float *B, const int B_batch_stride,
                                 __global float *C, const int C_batch_stride) {
  const int m = get_global_id(0);
  const int n = get_global_id(1);
  const int b = get_global_id(2);

  if (m >= M || n >= N)
    return;

  const int a_offset = b * A_batch_stride;
  const int b_offset = b * B_batch_stride;
  const int c_offset = b * C_batch_stride;

  float sum = 0.0f;
  for (int k = 0; k < K; ++k) {
    sum += A[a_offset + m * K + k] * B[b_offset + n * K + k];
  }

  C[c_offset + m * N + n] = sum;
}
