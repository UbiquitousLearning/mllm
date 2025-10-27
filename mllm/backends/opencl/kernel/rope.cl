// 文件名: kernel/rope.cl
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// LLaMA-style RoPE 内核 (FP32)
__kernel void rope_llama_fp32(
    __global float *data,
    __global const float *sin_table,
    __global const float *cos_table,
    const int partial_dim,
    const int head_dim,
    const int seq_len,
    const int pos_offset) {
    const int d = get_global_id(0); // Dimension index within a pair (0 to partial_dim/2 - 1)
    const int s = get_global_id(1); // Sequence index
    const int h = get_global_id(2); // Head index

    if (d >= (partial_dim / 2) || s >= seq_len || h >= head_dim) {
        return;
    }

    const int pos = s + pos_offset;
    const int idx_d = d * 2;

    // BSHD 布局: index = s * (H * D) + h * D + d
    size_t base_idx = (size_t)s * head_dim * partial_dim + (size_t)h * partial_dim + idx_d;

    float in0 = data[base_idx];
    float in1 = data[base_idx + 1];

    float sin_val = sin_table[pos * (partial_dim / 2) + d];
    float cos_val = cos_table[pos * (partial_dim / 2) + d];

    data[base_idx] = in0 * cos_val - in1 * sin_val;
    data[base_idx + 1] = in0 * sin_val + in1 * cos_val;
}

// HuggingFace-style RoPE 内核 (FP32)
__kernel void rope_hf_fp32(
    __global float *data,
    __global const float *sin_table,
    __global const float *cos_table,
    const int partial_dim,
    const int head_dim,
    const int seq_len,
    const int pos_offset) {
    const int d = get_global_id(0); // Half dimension index
    const int s = get_global_id(1); // Sequence index
    const int h = get_global_id(2); // Head index

    const int half_dim = partial_dim / 2;
    if (d >= half_dim || s >= seq_len || h >= head_dim) {
        return;
    }

    const int pos = s + pos_offset;

    // BSHD 布局
    size_t base_idx0 = (size_t)s * head_dim * partial_dim + (size_t)h * partial_dim + d;
    size_t base_idx1 = base_idx0 + half_dim;

    float in0 = data[base_idx0];
    float in1 = data[base_idx1];

    float sin_val = sin_table[pos * half_dim + d];
    float cos_val = cos_table[pos * half_dim + d];

    data[base_idx0] = in0 * cos_val - in1 * sin_val;
    data[base_idx1] = in0 * sin_val + in1 * cos_val;
}

// ==================================================================
// =================== 新增: FP16 Kernels =====================
// ==================================================================

// LLaMA-style RoPE 内核 (FP16)
__kernel void rope_llama_fp16(
    __global half *data,
    __global const half *sin_table,
    __global const half *cos_table,
    const int partial_dim,
    const int head_dim,
    const int seq_len,
    const int pos_offset) {
    const int d = get_global_id(0);
    const int s = get_global_id(1);
    const int h = get_global_id(2);

    if (d >= (partial_dim / 2) || s >= seq_len || h >= head_dim) {
        return;
    }

    const int pos = s + pos_offset;
    const int idx_d = d * 2;
    size_t base_idx = (size_t)s * head_dim * partial_dim + (size_t)h * partial_dim + idx_d;

    half in0 = data[base_idx];
    half in1 = data[base_idx + 1];

    half sin_val = sin_table[pos * partial_dim + idx_d];
    half cos_val = cos_table[pos * partial_dim + idx_d];

    data[base_idx] = in0 * cos_val - in1 * sin_val;
    data[base_idx + 1] = in0 * sin_val + in1 * cos_val;
}

// HuggingFace-style RoPE 内核 (FP16)
__kernel void rope_hf_fp16(
    __global half *data,
    __global const half *sin_table,
    __global const half *cos_table,
    const int partial_dim,
    const int head_dim,
    const int seq_len,
    const int pos_offset) {
    const int d = get_global_id(0);
    const int s = get_global_id(1);
    const int h = get_global_id(2);

    const int half_dim = partial_dim / 2;
    if (d >= half_dim || s >= seq_len || h >= head_dim) {
        return;
    }

    const int pos = s + pos_offset;
    size_t base_idx0 = (size_t)s * head_dim * partial_dim + (size_t)h * partial_dim + d;
    size_t base_idx1 = base_idx0 + half_dim;

    half in0 = data[base_idx0];
    half in1 = data[base_idx1];

    half sin_val = sin_table[pos * half_dim + d];
    half cos_val = cos_table[pos * half_dim + d];

    data[base_idx0] = in0 * cos_val - in1 * sin_val;
    data[base_idx1] = in0 * sin_val + in1 * cos_val;
}