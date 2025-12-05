// File name: kernel/embedding.cl
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

typedef struct {
  half d;
  unsigned char qs[16];
} block_q4_0;

// ============================================================================
// ========================== FP32 Embedding Kernel ===========================
// ============================================================================
__kernel void embedding_fp32(__global const long *input_ids_long,
                             __global const float *weights,
                             __global float *output, const int vocab_size,
                             const int hidden_size, const int sequence_len) {
  const int d_idx = get_global_id(0);
  const int token_idx = get_global_id(1);

  if (d_idx >= hidden_size || token_idx >= sequence_len) {
    return;
  }

  const int token_id = (int)input_ids_long[token_idx];
  if (token_id < 0 || token_id >= vocab_size) {
    output[token_idx * hidden_size + d_idx] = 0.0f;
    return;
  }

  const size_t src_idx = (size_t)token_id * hidden_size + d_idx;
  const size_t dst_idx = (size_t)token_idx * hidden_size + d_idx;

  output[dst_idx] = weights[src_idx];
}

// ============================================================================
// ========================== Q4_0 Embedding Kernel ===========================
// ============================================================================
__kernel void embedding_q4_0(__global const long *input_ids_long,
                             __global const block_q4_0 *weights,
                             __global float *output, const int vocab_size,
                             const int hidden_size, const int sequence_len) {
  const int d_idx = get_global_id(0);
  const int token_idx = get_global_id(1);

  if (d_idx >= hidden_size || token_idx >= sequence_len) {
    return;
  }

  const int token_id = (int)input_ids_long[token_idx];
  if (token_id < 0 || token_id >= vocab_size) {
    output[token_idx * hidden_size + d_idx] = 0.0f;
    return;
  }

  const size_t weight_idx = (size_t)token_id * hidden_size + d_idx;
  const int block_idx = weight_idx / 32;
  const int quant_idx_in_block = weight_idx % 32;
  const __global block_q4_0 *b = &weights[block_idx];

  const int qs_index = quant_idx_in_block % 16;
  const unsigned char quant_pair = b->qs[qs_index];
  int nibble;

  if (quant_idx_in_block < 16) {
    nibble = (quant_pair & 0x0F);
  } else {
    nibble = (quant_pair >> 4);
  }

  const float dequantized_value = (float)b->d * (float)(nibble - 8);
  const size_t dst_idx = (size_t)token_idx * hidden_size + d_idx;
  output[dst_idx] = dequantized_value;
}
// ============================================================================
// =================  Q4_0 Embedding Kernel (Output: FP16) ============
// ============================================================================
__kernel void embedding_q4_0_fp16(__global const long *input_ids_long,
                                  __global const block_q4_0 *weights,
                                  __global half *output, const int vocab_size,
                                  const int hidden_size,
                                  const int sequence_len) {
  const int d_idx = get_global_id(0);
  const int token_idx = get_global_id(1);

  if (d_idx >= hidden_size || token_idx >= sequence_len) {
    return;
  }

  const int token_id = (int)input_ids_long[token_idx];
  const size_t dst_idx = (size_t)token_idx * hidden_size + d_idx;

  if (token_id < 0 || token_id >= vocab_size) {
    output[dst_idx] = (half)0.0f;
    return;
  }

  const size_t weight_idx = (size_t)token_id * hidden_size + d_idx;
  const int block_idx = weight_idx / 32;
  const int quant_idx_in_block = weight_idx % 32;
  const __global block_q4_0 *b = &weights[block_idx];
  const int qs_index = quant_idx_in_block % 16;
  const unsigned char quant_pair = b->qs[qs_index];
  int nibble =
      (quant_idx_in_block < 16) ? (quant_pair & 0x0F) : (quant_pair >> 4);

  const float dequantized_value = (float)b->d * (float)(nibble - 8);

  output[dst_idx] = (half)dequantized_value;
}