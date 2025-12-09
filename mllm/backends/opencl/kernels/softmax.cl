#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define SOFTMAX_WG_SIZE 256

__kernel void softmax_fp32(__global const float *src, __global float *dst,
                           const int D) {
  const int row_id = get_group_id(0);
  const int local_id = get_local_id(0);
  const int local_size = get_local_size(0);

  // Move pointers to the current row
  const __global float *src_row = src + row_id * D;
  __global float *dst_row = dst + row_id * D;

  // 1. Find Max
  float thread_max = -INFINITY;
  for (int i = local_id; i < D; i += local_size) {
    float val = src_row[i];
    thread_max = fmax(thread_max, val);
  }

  // Reduction for Max
  __local float local_max_storage[SOFTMAX_WG_SIZE];
  local_max_storage[local_id] = thread_max;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int offset = local_size / 2; offset > 0; offset >>= 1) {
    if (local_id < offset) {
      local_max_storage[local_id] = fmax(local_max_storage[local_id],
                                         local_max_storage[local_id + offset]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float row_max = local_max_storage[0];

  // 2. Compute Sum of Exp
  float thread_sum = 0.0f;
  for (int i = local_id; i < D; i += local_size) {
    float val = src_row[i];
    thread_sum += exp(val - row_max);
  }

  // Reduction for Sum
  __local float local_sum_storage[SOFTMAX_WG_SIZE];
  local_sum_storage[local_id] = thread_sum;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int offset = local_size / 2; offset > 0; offset >>= 1) {
    if (local_id < offset) {
      local_sum_storage[local_id] += local_sum_storage[local_id + offset];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float row_sum = local_sum_storage[0];

  // 3. Write Output
  for (int i = local_id; i < D; i += local_size) {
    float val = src_row[i];
    dst_row[i] = exp(val - row_max) / row_sum;
  }
}

__kernel void softmax_fp16(__global const half *src, __global half *dst,
                           const int D) {
  const int row_id = get_group_id(0);
  const int local_id = get_local_id(0);
  const int local_size = get_local_size(0);

  const __global half *src_row = src + row_id * D;
  __global half *dst_row = dst + row_id * D;

  // 1. Find Max
  float thread_max = -INFINITY; // Use float for accumulation/reduction to avoid
                                // overflow/precision issues
  for (int i = local_id; i < D; i += local_size) {
    float val = (float)src_row[i];
    thread_max = fmax(thread_max, val);
  }

  __local float local_max_storage[SOFTMAX_WG_SIZE];
  local_max_storage[local_id] = thread_max;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int offset = local_size / 2; offset > 0; offset >>= 1) {
    if (local_id < offset) {
      local_max_storage[local_id] = fmax(local_max_storage[local_id],
                                         local_max_storage[local_id + offset]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float row_max = local_max_storage[0];

  // 2. Compute Sum of Exp
  float thread_sum = 0.0f;
  for (int i = local_id; i < D; i += local_size) {
    float val = (float)src_row[i];
    thread_sum += exp(val - row_max);
  }

  __local float local_sum_storage[SOFTMAX_WG_SIZE];
  local_sum_storage[local_id] = thread_sum;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int offset = local_size / 2; offset > 0; offset >>= 1) {
    if (local_id < offset) {
      local_sum_storage[local_id] += local_sum_storage[local_id + offset];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float row_sum = local_sum_storage[0];

  // 3. Write Output
  for (int i = local_id; i < D; i += local_size) {
    float val = (float)src_row[i];
    dst_row[i] = (half)(exp(val - row_max) / row_sum);
  }
}
