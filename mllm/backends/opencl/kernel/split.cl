#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void split_fp32(
    __global const float *input,
    __global float *output,
    const int outer_size,
    const int split_dim_size,
    const int inner_size,
    const int offset,
    const int total_split_dim_size) {
    const int inner_idx = get_global_id(0);
    const int split_idx = get_global_id(1);
    const int outer_idx = get_global_id(2);

    if (inner_idx >= inner_size || split_idx >= split_dim_size || outer_idx >= outer_size) {
        return;
    }

    size_t src_offset = (size_t)outer_idx * total_split_dim_size * inner_size + (size_t)(split_idx + offset) * inner_size + inner_idx;

    size_t dst_offset = (size_t)outer_idx * split_dim_size * inner_size + (size_t)split_idx * inner_size + inner_idx;

    output[dst_offset] = input[src_offset];
}

__kernel void split_fp16(
    __global const half *input,
    __global half *output,
    const int outer_size,
    const int split_dim_size,
    const int inner_size,
    const int offset,
    const int total_split_dim_size) {
    const int inner_idx = get_global_id(0);
    const int split_idx = get_global_id(1);
    const int outer_idx = get_global_id(2);

    if (inner_idx >= inner_size || split_idx >= split_dim_size || outer_idx >= outer_size) {
        return;
    }

    size_t src_offset = (size_t)outer_idx * total_split_dim_size * inner_size + (size_t)(split_idx + offset) * inner_size + inner_idx;

    size_t dst_offset = (size_t)outer_idx * split_dim_size * inner_size + (size_t)split_idx * inner_size + inner_idx;

    output[dst_offset] = input[src_offset];
}