// Filename: rmsnorm.cl

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Define the number of threads used for parallel reduction within a workgroup
// (must be a power of 2)
#define RMSNORM_WG_SIZE 256

// Quantized block structure matching definition in C++ DataType.hpp
typedef struct {
  half d;
  uchar qs[16]; // QK4_0 / 2 = 16
} block_q4_0;

// Kernel helper function: Dequantize an element from Q4_0 block on-the-fly
inline float dequantize_q4_0(const __global block_q4_0 *blocks, int index) {
  const int block_idx = index / 32; // QK4_0 is 32
  const int quant_idx_in_block = index % 32;

  // Locate the corresponding Q4_0 block
  const __global block_q4_0 *b = &blocks[block_idx];

  // Extract 4-bit quantized value from uchar
  const uchar quant_pair = b->qs[quant_idx_in_block / 2];
  const int nibble =
      (quant_idx_in_block % 2 == 0) ? (quant_pair & 0x0F) : (quant_pair >> 4);

  // Apply dequantization formula
  return (float)b->d * (float)(nibble - 8);
}

__kernel void rmsnorm_f32_q4(
    __global const float *src,    // Input tensor (fp32)
    __global float *dst,          // Output tensor (fp32)
    __global const void *weights, // Weight tensor (can be fp32 or q4_0)
    const int weight_is_q4, // Flag: 0 means weights are fp32, 1 means q4_0
    const int D,            // Dimension, i.e., length of each row
    const float epsilon,    // Epsilon value to prevent division by zero
    const int
        add_unit_offset // Flag: whether to perform +1 operation on weights
) {
  // 1. Get IDs
  const int row_id = get_group_id(
      0); // Each workgroup processes one row, row ID determined by workgroup ID
  const int local_id = get_local_id(0); // Thread ID within workgroup

  // 2. Declare shared array in local memory
  __local float local_sum_sq[RMSNORM_WG_SIZE];

  // 3. Parallel computation of sum of squares
  float thread_sum_sq =
      0.0f; // Each thread calculates sum of squares for part of elements
  for (int i = local_id; i < D; i += RMSNORM_WG_SIZE) {
    float val = src[row_id * D + i];
    thread_sum_sq += val * val;
  }
  local_sum_sq[local_id] = thread_sum_sq;

  // 4. Workgroup reduction to compute total sum of squares for the entire row
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = RMSNORM_WG_SIZE / 2; s > 0; s >>= 1) {
    if (local_id < s) {
      local_sum_sq[local_id] += local_sum_sq[local_id + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  // At this point, local_sum_sq[0] contains the sum of squares for the entire
  // row

  // 5. Calculate RMS scaling factor and broadcast safely
  float rms_val;
  // Only the first thread in the workgroup performs this scalar calculation
  if (local_id == 0) {
    float variance = local_sum_sq[0] / D;
    rms_val = rsqrt(variance + epsilon);
    local_sum_sq[0] =
        rms_val; // Thread 0 calculates result and stores in shared memory
  }

  // Synchronization point: Ensure all threads wait for thread 0 to write
  // rms_val to shared memory
  barrier(CLK_LOCAL_MEM_FENCE);

  // All threads (including thread 0) read the broadcast value from shared
  // memory
  rms_val = local_sum_sq[0];

  // 6. Parallel normalization and apply weights
  for (int i = local_id; i < D; i += RMSNORM_WG_SIZE) {
    // a. Get weight value
    float weight_val;
    if (weight_is_q4) {
      weight_val = dequantize_q4_0((const __global block_q4_0 *)weights, i);
    } else {
      weight_val = ((const __global float *)weights)[i];
    }

    // b. Decide whether to add 1 based on flag
    if (add_unit_offset) {
      weight_val += 1.0f;
    }

    // c. Calculate final result and write back to global memory
    size_t index = row_id * D + i;
    dst[index] = src[index] * rms_val * weight_val;
  }
}

// ==================================================================
// 2.  FP16 Input Kernel (rmsnorm_f16_q4)
// ==================================================================
__kernel void rmsnorm_f16_q4(
    __global const half *src,     // Input tensor (fp16)
    __global half *dst,           // Output tensor (fp16)
    __global const void *weights, // Weight tensor (can be fp32 or q4_0)
    const int weight_is_q4,       // Flag
    const int D,                  // Dimension
    const float epsilon,          // Epsilon (still float)
    const int add_unit_offset     // Flag
) {
  const int row_id = get_group_id(0);
  const int local_id = get_local_id(0);

  __local float local_sum_sq[RMSNORM_WG_SIZE];

  // Use float accumulator to ensure precision
  float thread_sum_sq = 0.0f;
  for (int i = local_id; i < D; i += RMSNORM_WG_SIZE) {
    // Convert from half to float for calculation
    float val = (float)src[row_id * D + i];
    thread_sum_sq += val * val;
  }
  local_sum_sq[local_id] = thread_sum_sq;

  // Workgroup reduction (identical to fp32 version)
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = RMSNORM_WG_SIZE / 2; s > 0; s >>= 1) {
    if (local_id < s) {
      local_sum_sq[local_id] += local_sum_sq[local_id + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Calculate RMS scaling factor (identical to fp32 version)
  float rms_val;
  if (local_id == 0) {
    float variance = local_sum_sq[0] / D;
    rms_val = rsqrt(variance + epsilon);
    local_sum_sq[0] = rms_val;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  rms_val = local_sum_sq[0];

  // Normalization and apply weights
  for (int i = local_id; i < D; i += RMSNORM_WG_SIZE) {
    float weight_val;
    if (weight_is_q4) {
      weight_val = dequantize_q4_0((const __global block_q4_0 *)weights, i);
    } else {
      weight_val = ((const __global float *)weights)[i];
    }
    if (add_unit_offset) {
      weight_val += 1.0f;
    }

    size_t index = row_id * D + i;
    // Calculation result is float, finally convert back to half and store in
    // dst
    float src_val = (float)src[index];
    dst[index] = (half)(src_val * rms_val * weight_val);
  }
}
