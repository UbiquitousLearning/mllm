#define TILE_SIZE 16

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

__kernel void
matmul_buffer_nt_nt_opt(const int M, const int N, const int K,
                        __global float *A, const int A_batch_stride,
                        __global float *B, const int B_batch_stride,
                        __global float *C, const int C_batch_stride) {
  const int loc_m = get_local_id(0);
  const int loc_n = get_local_id(1);

  const int global_m = get_global_id(0);
  const int global_n = get_global_id(1);
  const int b = get_global_id(2);

  __local float Asub[TILE_SIZE][TILE_SIZE];
  __local float Bsub[TILE_SIZE][TILE_SIZE];

  const int a_offset = b * A_batch_stride;
  const int b_offset = b * B_batch_stride;
  const int c_offset = b * C_batch_stride;

  float sum = 0.0f;

  const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

  for (int t = 0; t < num_tiles; t++) {
    const int tiled_k = t * TILE_SIZE;

    const int a_row = global_m;
    const int a_col = tiled_k + loc_n;
    if (a_row < M && a_col < K) {
      Asub[loc_m][loc_n] = A[a_offset + a_row * K + a_col];
    } else {
      Asub[loc_m][loc_n] = 0.0f;
    }

    const int b_row = tiled_k + loc_m;
    const int b_col = global_n;
    if (b_row < K && b_col < N) {
      Bsub[loc_m][loc_n] = B[b_offset + b_row * N + b_col];
    } else {
      Bsub[loc_m][loc_n] = 0.0f;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = 0; k < TILE_SIZE; ++k) {
      sum += Asub[loc_m][k] * Bsub[k][loc_n];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (global_m < M && global_n < N) {
    C[c_offset + global_m * N + global_n] = sum;
  }
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

__kernel void
matmul_buffer_nt_t_opt(const int M, const int N, const int K, __global float *A,
                       const int A_batch_stride, __global float *B,
                       const int B_batch_stride, __global float *C,
                       const int C_batch_stride) {
  const int loc_m = get_local_id(0);
  const int loc_n = get_local_id(1);

  const int global_m = get_global_id(0);
  const int global_n = get_global_id(1);
  const int b = get_global_id(2);

  __local float Asub[TILE_SIZE][TILE_SIZE];
  __local float Bsub[TILE_SIZE][TILE_SIZE];

  const int a_offset = b * A_batch_stride;
  const int b_offset = b * B_batch_stride;
  const int c_offset = b * C_batch_stride;

  float sum = 0.0f;
  const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

  for (int t = 0; t < num_tiles; t++) {
    const int tiled_k = t * TILE_SIZE;

    if (global_m < M && (tiled_k + loc_n) < K) {
      Asub[loc_m][loc_n] = A[a_offset + global_m * K + (tiled_k + loc_n)];
    } else {
      Asub[loc_m][loc_n] = 0.0f;
    }

    if (global_m < M && (tiled_k + loc_n) < K) {
      Asub[loc_m][loc_n] = A[a_offset + global_m * K + (tiled_k + loc_n)];
    } else {
      Asub[loc_m][loc_n] = 0.0f;
    }

    int t_b_row = get_group_id(1) * TILE_SIZE + loc_n;
    int t_b_col = tiled_k + loc_m;

    if (t_b_row < N && t_b_col < K) {
      Bsub[loc_n][loc_m] = B[b_offset + t_b_row * K + t_b_col];
    } else {
      Bsub[loc_n][loc_m] = 0.0f;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = 0; k < TILE_SIZE; ++k) {
      sum += Asub[loc_m][k] * Bsub[loc_n][k];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (global_m < M && global_n < N) {
    C[c_offset + global_m * N + global_n] = sum;
  }
}