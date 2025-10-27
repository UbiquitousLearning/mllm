#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Kernel for clipping along the SEQUENCE dimension
__kernel void clip_sequence_fp32(
    __global const float *src_data,
    __global const float *indices,
    __global float *dst_data,
    const int B, const int H, const int S_in, const int D, const int S_out) {
    // Each work-item handles one element in the H*D plane of the output
    const int d = get_global_id(0);
    const int h = get_global_id(1);
    const int bs = get_global_id(2); // Combined batch and output sequence index

    const int b = bs / S_out;
    const int s_out = bs % S_out;

    if (d >= D || h >= H || b >= B) {
        return;
    }

    // Get the source sequence index from the indices tensor
    const int s_in = (int)indices[s_out];

    // Assuming BSHD layout for both input and output
    // src_offset for [b, s_in, h, d]
    size_t src_offset = ((size_t)b * S_in + s_in) * H * D + (size_t)h * D + d;
    // dst_offset for [b, s_out, h, d]
    size_t dst_offset = ((size_t)b * S_out + s_out) * H * D + (size_t)h * D + d;

    dst_data[dst_offset] = src_data[src_offset];
}

// Kernel for clipping along the DIMENSION dimension
__kernel void clip_dimension_fp32(
    __global const float *src_data,
    __global const float *indices,
    __global float *dst_data,
    const int B, const int H, const int S, const int D_in, const int D_out) {
    // Each work-item handles one element in the output tensor
    const int d_out = get_global_id(0);
    const int s = get_global_id(1);
    const int bh = get_global_id(2);
    const int b = bh / H;
    const int h = bh % H;

    if (d_out >= D_out || s >= S || b >= B) {
        return;
    }

    // Get the source dimension index from the indices tensor
    const int d_in = (int)indices[d_out];

    // Assuming BSHD layout for both input and output
    // src_offset for [b, s, h, d_in]
    size_t src_offset = ((size_t)b * S + s) * H * D_in + (size_t)h * D_in + d_in;
    // dst_offset for [b, s, h, d_out]
    size_t dst_offset = ((size_t)b * S + s) * H * D_out + (size_t)h * D_out + d_out;

    dst_data[dst_offset] = src_data[src_offset];
}

// ========================== FP16 Versions ==============================

__kernel void clip_sequence_fp16(
    __global const half *src_data,
    __global const half *indices,
    __global half *dst_data,
    const int B, const int H, const int S_in, const int D, const int S_out) {
    const int d = get_global_id(0);
    const int h = get_global_id(1);
    const int bs = get_global_id(2);
    const int b = bs / S_out;
    const int s_out = bs % S_out;

    if (d >= D || h >= H || b >= B) {
        return;
    }
    const int s_in = (int)indices[s_out];
    size_t src_offset = ((size_t)b * S_in + s_in) * H * D + (size_t)h * D + d;
    size_t dst_offset = ((size_t)b * S_out + s_out) * H * D + (size_t)h * D + d;
    dst_data[dst_offset] = src_data[src_offset];
}

__kernel void clip_dimension_fp16(
    __global const half *src_data,
    __global const half *indices,
    __global half *dst_data,
    const int B, const int H, const int S, const int D_in, const int D_out) {
    const int d_out = get_global_id(0);
    const int s = get_global_id(1);
    const int bh = get_global_id(2);
    const int b = bh / H;
    const int h = bh % H;

    if (d_out >= D_out || s >= S || b >= B) {
        return;
    }
    const int d_in = (int)indices[d_out];
    size_t src_offset = ((size_t)b * S + s) * H * D_in + (size_t)h * D_in + d_in;
    size_t dst_offset = ((size_t)b * S + s) * H * D_out + (size_t)h * D_out + d_out;
    dst_data[dst_offset] = src_data[src_offset];
}