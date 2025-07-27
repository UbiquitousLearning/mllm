// 文件名: kernel/embedding.cl
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// 与 C++ DataType.hpp 中定义匹配的 Q4_0 量化块结构
typedef struct {
    half d;
    unsigned char qs[16];
} block_q4_0;

// ============================================================================
// ========================== FP32 Embedding Kernel ===========================
// ============================================================================
__kernel void embedding_fp32(
    __global const float *input_ids_float,
    __global const float *weights,
    __global float *output,
    const int vocab_size,
    const int hidden_size,
    const int sequence_len) {
    const int d_idx = get_global_id(0);
    const int token_idx = get_global_id(1);

    if (d_idx >= hidden_size || token_idx >= sequence_len) {
        return;
    }

    const int token_id = (int)input_ids_float[token_idx];
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
__kernel void embedding_q4_0(
    __global const float *input_ids_float,
    __global const block_q4_0 *weights,
    __global float *output,
    const int vocab_size,
    const int hidden_size,
    const int sequence_len) {
    const int d_idx = get_global_id(0);
    const int token_idx = get_global_id(1);

    if (d_idx >= hidden_size || token_idx >= sequence_len) {
        return;
    }

    const int token_id = (int)input_ids_float[token_idx];
    if (token_id < 0 || token_id >= vocab_size) {
        output[token_idx * hidden_size + d_idx] = 0.0f;
        return;
    }

    const size_t weight_idx = (size_t)token_id * hidden_size + d_idx;
    const int block_idx = weight_idx / 32;
    const int quant_idx_in_block = weight_idx % 32;
    const __global block_q4_0 *b = &weights[block_idx];

    // 正确的解量化逻辑
    const int qs_index = quant_idx_in_block % 16;
    // [修正] 将 uchar 替换为 unsigned char 提高兼容性
    const unsigned char quant_pair = b->qs[qs_index];
    int nibble;

    if (quant_idx_in_block < 16) {
        // 元素在块的前半部分 (0-15), 取低4位
        nibble = (quant_pair & 0x0F);
    } else {
        // 元素在块的后半部分 (16-31), 取高4位
        nibble = (quant_pair >> 4);
    }

    const float dequantized_value = (float)b->d * (float)(nibble - 8);
    const size_t dst_idx = (size_t)token_idx * hidden_size + d_idx;
    output[dst_idx] = dequantized_value;
}
// ============================================================================
// =================  Q4_0 Embedding Kernel (Output: FP16) ============
// ============================================================================
__kernel void embedding_q4_0_fp16(
    __global const half *input_ids_half, // ✨ **核心修正点**: 输入类型改为 half
    __global const block_q4_0 *weights,
    __global half *output,
    const int vocab_size,
    const int hidden_size,
    const int sequence_len) {
    const int d_idx = get_global_id(0);
    const int token_idx = get_global_id(1);

    if (d_idx >= hidden_size || token_idx >= sequence_len) {
        return;
    }

    // ✨ **核心修正点**: 从 half 类型的输入中读取 token ID
    const int token_id = (int)input_ids_half[token_idx];
    const size_t dst_idx = (size_t)token_idx * hidden_size + d_idx;

    if (token_id < 0 || token_id >= vocab_size) {
        output[dst_idx] = (half)0.0f;
        return;
    }

    // 解量化逻辑与 embedding_q4_0 完全相同
    const size_t weight_idx = (size_t)token_id * hidden_size + d_idx;
    const int block_idx = weight_idx / 32;
    const int quant_idx_in_block = weight_idx % 32;
    const __global block_q4_0 *b = &weights[block_idx];
    const int qs_index = quant_idx_in_block % 16;
    const unsigned char quant_pair = b->qs[qs_index];
    int nibble = (quant_idx_in_block < 16) ? (quant_pair & 0x0F) : (quant_pair >> 4);

    // 计算结果为 float
    const float dequantized_value = (float)b->d * (float)(nibble - 8);

    // 将 float 结果转换为 half 并存储
    output[dst_idx] = (half)dequantized_value;
}