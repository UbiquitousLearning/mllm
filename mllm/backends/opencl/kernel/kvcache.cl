// 文件名: kernel/kvcache.cl

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// ==================================================================
// Kernels for BSHD (Batch, Sequence, Head, Dim) Layout
// ==================================================================
__kernel void update_kv_cache_fp32_bshd(
    __global const float *src,
    __global float *cache,
    const int H_in,
    const int S_in,
    const int D,
    const int H_cache,
    const int S_cache,
    const int n_rep,
    const int cache_offset) {
    const int d_vec_idx = get_global_id(0);
    const int s_idx = get_global_id(1);
    const int h_idx_dst = get_global_id(2);

    const int D_vec = D / 4;
    if (d_vec_idx >= D_vec || s_idx >= S_in || h_idx_dst >= H_cache) {
        return;
    }

    const int h_idx_src = h_idx_dst / n_rep;
    const int s_idx_dst = cache_offset + s_idx;

    // BSHD [Batch, Sequence, Head, Dim] layout indexing
    size_t src_offset = (size_t)s_idx * H_in * D + (size_t)h_idx_src * D + (size_t)d_vec_idx * 4;
    size_t dst_offset = (size_t)s_idx_dst * H_cache * D + (size_t)h_idx_dst * D + (size_t)d_vec_idx * 4;

    float4 data_vec = vload4(0, src + src_offset);
    vstore4(data_vec, 0, cache + dst_offset);
}

__kernel void update_kv_cache_fp16_bshd(
    __global const half *src,
    __global half *cache,
    const int H_in,
    const int S_in,
    const int D,
    const int H_cache,
    const int S_cache,
    const int n_rep,
    const int cache_offset) {
    const int d_vec_idx = get_global_id(0);
    const int s_idx = get_global_id(1);
    const int h_idx_dst = get_global_id(2);

    const int D_vec = D / 4;
    if (d_vec_idx >= D_vec || s_idx >= S_in || h_idx_dst >= H_cache) {
        return;
    }

    const int h_idx_src = h_idx_dst / n_rep;
    const int s_idx_dst = cache_offset + s_idx;

    // BSHD [Batch, Sequence, Head, Dim] layout indexing
    size_t src_offset = (size_t)s_idx * H_in * D + (size_t)h_idx_src * D + (size_t)d_vec_idx * 4;
    size_t dst_offset = (size_t)s_idx_dst * H_cache * D + (size_t)h_idx_dst * D + (size_t)d_vec_idx * 4;

    half4 data_vec = vload4(0, src + src_offset);
    vstore4(data_vec, 0, cache + dst_offset);
}

// ==================================================================
// Kernels for BHSD (Batch, Head, Sequence, Dim) Layout (New)
// ==================================================================
__kernel void update_kv_cache_fp32_bhsd(
    __global const float *src,
    __global float *cache,
    const int H_in,
    const int S_in,
    const int D,
    const int H_cache,
    const int S_cache,
    const int n_rep,
    const int cache_offset) {
    const int d_vec_idx = get_global_id(0);
    const int s_idx = get_global_id(1);
    const int h_idx_dst = get_global_id(2);

    const int D_vec = D / 4;
    if (d_vec_idx >= D_vec || s_idx >= S_in || h_idx_dst >= H_cache) {
        return;
    }

    const int h_idx_src = h_idx_dst / n_rep;
    const int s_idx_dst = cache_offset + s_idx;

    // BHSD [Batch, Head, Sequence, Dim] layout indexing
    size_t src_offset = (size_t)h_idx_src * S_in * D + (size_t)s_idx * D + (size_t)d_vec_idx * 4;
    size_t dst_offset = (size_t)h_idx_dst * S_cache * D + (size_t)s_idx_dst * D + (size_t)d_vec_idx * 4;

    float4 data_vec = vload4(0, src + src_offset);
    vstore4(data_vec, 0, cache + dst_offset);
}

__kernel void update_kv_cache_fp16_bhsd(
    __global const half *src,
    __global half *cache,
    const int H_in,
    const int S_in,
    const int D,
    const int H_cache,
    const int S_cache,
    const int n_rep,
    const int cache_offset) {
    const int d_vec_idx = get_global_id(0);
    const int s_idx = get_global_id(1);
    const int h_idx_dst = get_global_id(2);

    const int D_vec = D / 4;
    if (d_vec_idx >= D_vec || s_idx >= S_in || h_idx_dst >= H_cache) {
        return;
    }

    const int h_idx_src = h_idx_dst / n_rep;
    const int s_idx_dst = cache_offset + s_idx;

    // BHSD [Batch, Head, Sequence, Dim] layout indexing
    size_t src_offset = (size_t)h_idx_src * S_in * D + (size_t)s_idx * D + (size_t)d_vec_idx * 4;
    size_t dst_offset = (size_t)h_idx_dst * S_cache * D + (size_t)s_idx_dst * D + (size_t)d_vec_idx * 4;

    half4 data_vec = vload4(0, src + src_offset);
    vstore4(data_vec, 0, cache + dst_offset);
}