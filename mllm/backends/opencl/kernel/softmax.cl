#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define SOFTMAX_BLOCK_SIZE 256

#define HALF_MAX 65504.0h
#ifdef SUPPORTS_FP16
// ============================================================================
// ========================== FP16 (half) 高性能版 =============================
// ============================================================================
__kernel void softmax_fp16_along_d(
    const __global half *src,
    __global half *dst,
    const int B, const int H, const int S, const int D,
    const int do_causal_mask) {
    const int row_id = get_group_id(0);
    if (row_id >= B * H * S) return;
    const int local_id = get_local_id(0);
    int effective_D = D;
    if (do_causal_mask) {
        const int current_s = row_id % S;
        effective_D = current_s + 1;
    }
    __local half local_max[SOFTMAX_BLOCK_SIZE];
    __local float local_sum[SOFTMAX_BLOCK_SIZE];
    half thread_max = -HALF_MAX;
    for (int i = local_id; i < effective_D; i += SOFTMAX_BLOCK_SIZE) {
        thread_max = max(thread_max, src[row_id * D + i]);
    }
    local_max[local_id] = thread_max;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = SOFTMAX_BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (local_id < s) { local_max[local_id] = max(local_max[local_id], local_max[local_id + s]); }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const half row_max = local_max[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    float thread_sum = 0.0f;
    for (int i = local_id; i < effective_D; i += SOFTMAX_BLOCK_SIZE) {
        half val = exp(src[row_id * D + i] - row_max);
        thread_sum += (float)val;
        dst[row_id * D + i] = val;
    }
    local_sum[local_id] = thread_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = SOFTMAX_BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (local_id < s) { local_sum[local_id] += local_sum[local_id + s]; }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float row_sum = local_sum[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = local_id; i < effective_D; i += SOFTMAX_BLOCK_SIZE) {
        dst[row_id * D + i] /= (half)row_sum;
    }
    if (do_causal_mask) {
        for (int i = local_id + effective_D; i < D; i += SOFTMAX_BLOCK_SIZE) {
            dst[row_id * D + i] = (half)0.0f;
        }
    }
}
#else
// ============================================================================
// ========================== FP16 (half) 兼容版 ==============================
// ============================================================================
__kernel void softmax_fp16_along_d(
    const __global half *src,
    __global half *dst,
    const int B, const int H, const int S, const int D,
    const int do_causal_mask) {
    const int row_id = get_global_id(0);
    if (row_id >= B * H * S) return;

    int effective_D = D;
    if (do_causal_mask) {
        effective_D = (row_id % S) + 1;
    }
    const __global half *p_src = src + row_id * D;
    __global half *p_dst = dst + row_id * D;

    float max_val = -INFINITY;
    for (int i = 0; i < effective_D; ++i) {
        max_val = max(max_val, (float)p_src[i]);
    }
    float sum = 0.0f;
    for (int i = 0; i < effective_D; ++i) {
        float val_f32 = exp((float)p_src[i] - max_val);
        p_dst[i] = (half)val_f32;
        sum += val_f32;
    }
    for (int i = 0; i < effective_D; ++i) {
        p_dst[i] = (half)((float)p_dst[i] / sum);
    }
    if (do_causal_mask) {
        for (int i = effective_D; i < D; ++i) {
            p_dst[i] = (half)0.0f;
        }
    }
}
#endif // SUPPORTS_FP16

// ============================================================================
// ========================== FP32 (float) Kernel =============================
// ============================================================================
__kernel void softmax_f32_along_d(
    const __global float *src,
    __global float *dst,
    const int B, const int H, const int S, const int D,
    const int do_causal_mask) {
    const int row_id = get_group_id(0);
    if (row_id >= B * H * S) return;

    const int local_id = get_local_id(0);
    int effective_D = D;
    if (do_causal_mask) {
        effective_D = (row_id % S) + 1;
    }

    __local float local_max[SOFTMAX_BLOCK_SIZE];
    __local float local_sum[SOFTMAX_BLOCK_SIZE];

    float thread_max = -INFINITY;
    for (int i = local_id; i < effective_D; i += SOFTMAX_BLOCK_SIZE) {
        thread_max = max(thread_max, src[row_id * D + i]);
    }
    local_max[local_id] = thread_max;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = SOFTMAX_BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (local_id < s) { local_max[local_id] = max(local_max[local_id], local_max[local_id + s]); }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float row_max = local_max[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    float thread_sum = 0.0f;
    for (int i = local_id; i < effective_D; i += SOFTMAX_BLOCK_SIZE) {
        float val = exp(src[row_id * D + i] - row_max);
        thread_sum += val;
        dst[row_id * D + i] = val;
    }
    local_sum[local_id] = thread_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = SOFTMAX_BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (local_id < s) { local_sum[local_id] += local_sum[local_id + s]; }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float row_sum = local_sum[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = local_id; i < effective_D; i += SOFTMAX_BLOCK_SIZE) {
        dst[row_id * D + i] /= row_sum;
    }
    if (do_causal_mask) {
        for (int i = local_id + effective_D; i < D; i += SOFTMAX_BLOCK_SIZE) {
            dst[row_id * D + i] = 0.0f;
        }
    }
}