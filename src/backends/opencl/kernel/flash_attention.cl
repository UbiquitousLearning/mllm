// 文件: opencl/kernel/flash_attention.cl
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define F32_MAX 3.402823466e+38f
// 根据您的硬件能力和模型维度进行调整
#define DIM_MAX 128
// K/V瓦片的序列长度，32或64是常见选择
#define TILE_S 32
// 工作组大小，128或256是常见选择
#define WGS 128

__kernel void flash_attention_2_prefill_fp32(
    __global const float *Q,
    __global const float *K,
    __global const float *V,
    __global float *O,
    const int q_head_size,
    const int kv_head_size,
    const int seq_size_q,
    const int seq_size_k,
    const int dim_size,
    const int causal_mask_flag) {
    __local float k_tile[TILE_S][DIM_MAX];
    __local float v_tile[TILE_S][DIM_MAX];

    // 1. 并行模型与索引：一个工作组计算一个输出行 O[b, s_q, h, :]
    // 全局ID的每一行代表一个输出向量
    const int row_idx = get_group_id(0);
    const int local_id = get_local_id(0); // 线程在工作组内的ID (0 to WGS-1)

    const int b = row_idx / (seq_size_q * q_head_size);
    const int s_q_h_idx = row_idx % (seq_size_q * q_head_size);
    const int s_q = s_q_h_idx / q_head_size;
    const int h = s_q_h_idx % q_head_size;

    // 处理GQA/MQA: 多个Q头可能对应同一个KV头
    const int kv_h_idx = h / (q_head_size / kv_head_size);

    // 2. 初始化线程私有累加器 (在寄存器中，速度最快)
    float my_max_score = -F32_MAX;
    float my_sum_exp = 0.0f;
    float my_acc_o[DIM_MAX];
    for (int d = 0; d < dim_size; ++d) {
        my_acc_o[d] = 0.0f;
    }

    // 将当前查询向量Q加载到私有内存(寄存器)中，以便在循环中反复使用
    float my_q[DIM_MAX];
    const long q_base_addr = (long)b * seq_size_q * q_head_size * dim_size + (long)s_q * q_head_size * dim_size + (long)h * dim_size;
    for (int d = 0; d < dim_size; ++d) {
        my_q[d] = Q[q_base_addr + d];
    }
    // 注意：此处需要一个屏障，因为每个线程只加载了Q的一部分，但下面计算需要完整的Q
    // 但由于并行模型是冗余计算，每个线程最终都会有完整的my_q，所以也可以不加。
    // 为了逻辑严谨和未来可能的优化，我们这里假设每个线程都拿到了完整的Q。
    // 在实践中，更优化的方式是让一个warp加载Q然后广播，但为简单起见，这里每个线程都读。

    const float scale = rsqrt((float)dim_size);
    const int max_s_k = (causal_mask_flag && seq_size_q > 1) ? (s_q + 1) : seq_size_k;

    // 3. 沿K/V序列长度的分块循环
    for (int s_k_start = 0; s_k_start < max_s_k; s_k_start += TILE_S) {
        // a. 工作组协作加载K, V块到高速的__local内存
        for (int i = local_id; i < TILE_S * dim_size; i += WGS) {
            int s_local = i / dim_size;
            int d_local = i % dim_size;
            int s_k_global = s_k_start + s_local;

            if (s_k_global < seq_size_k) {
                long kv_offset = (long)b * seq_size_k * kv_head_size * dim_size + (long)s_k_global * kv_head_size * dim_size + (long)kv_h_idx * dim_size + d_local;
                k_tile[s_local][d_local] = K[kv_offset];
                v_tile[s_local][d_local] = V[kv_offset];
            } else {
                // 超出范围的数据用0填充
                k_tile[s_local][d_local] = 0.0f;
                v_tile[s_local][d_local] = 0.0f;
            }
        }
        // 同步点：确保所有线程都完成了加载，K/V tile现在对所有线程可见
        barrier(CLK_LOCAL_MEM_FENCE);

        // b. 每个线程独立处理已加载的K/V瓦片，执行在线Softmax
        for (int s_local = 0; s_local < TILE_S; ++s_local) {
            int s_k_global = s_k_start + s_local;
            if (s_k_global < max_s_k) {
                // 计算分数 (Q * K^T)
                float score = 0.0f;
                for (int d = 0; d < dim_size; ++d) {
                    score += my_q[d] * k_tile[s_local][d];
                }
                score *= scale;

                float old_max = my_max_score;
                my_max_score = fmax(old_max, score);

                float scale_factor = exp(old_max - my_max_score);
                my_sum_exp *= scale_factor;

                for (int d = 0; d < dim_size; ++d) {
                    my_acc_o[d] *= scale_factor;
                }

                float p = exp(score - my_max_score);
                my_sum_exp += p;

                for (int d = 0; d < dim_size; ++d) {
                    my_acc_o[d] += p * v_tile[s_local][d];
                }
            }
        }
        // 同步点：确保所有线程都处理完当前块，才能进入下一轮循环加载新块
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // 4. 最终写回
    // 只需要一个线程来执行最后的归一化和写回操作，避免写入冲突
    if (local_id == 0) {
        float inv_sum_exp = 1.0f / (my_sum_exp + 1e-6f);
        long o_offset = (long)b * seq_size_q * q_head_size * dim_size + (long)s_q * q_head_size * dim_size + (long)h * dim_size;
        for (int d = 0; d < dim_size; ++d) {
            O[o_offset + d] = my_acc_o[d] * inv_sum_exp;
        }
    }
}

__kernel void flash_attention_2_decode_fp32(
    __global const float *Q,
    __global const float *K,
    __global const float *V,
    __global float *O,
    const int q_head_size,
    const int kv_head_size,
    const int seq_size_k,
    const int dim_size) {
    __local float q_vec[DIM_MAX];
    __local float l_max_score;
    __local float l_sum_exp;
    __local float partial_sums[WGS];
    __local float p, scale_factor;

    // 1. 并行模型与索引
    const int row_idx = get_group_id(0);
    const int local_id = get_local_id(0);

    const int b = row_idx / q_head_size;
    const int h = row_idx % q_head_size;
    const int kv_h_idx = h / (q_head_size / kv_head_size);

    // 2. 协作加载Q向量
    const long q_addr = (long)b * q_head_size * dim_size + (long)h * dim_size;
    for (int d = local_id; d < dim_size; d += WGS) {
        q_vec[d] = Q[q_addr + d];
    }

    // 3. 初始化
    if (local_id == 0) {
        l_max_score = -F32_MAX;
        l_sum_exp = 0.0f;
    }

    const int dims_per_thread = (dim_size + WGS - 1) / WGS;
    float my_acc_o[DIM_MAX / WGS + 1];
    for (int i = 0; i < dims_per_thread; ++i) my_acc_o[i] = 0.0f;

    const float scale = rsqrt((float)dim_size);
    barrier(CLK_LOCAL_MEM_FENCE);

    // 4. 遍历所有Key向量
    for (int s_k = 0; s_k < seq_size_k; ++s_k) {
        // ... 并行计算点积与规约 (这部分逻辑不变) ...
        float partial_sum = 0.0f;
        long k_addr = (long)b * seq_size_k * kv_head_size * dim_size + (long)s_k * kv_head_size * dim_size + (long)kv_h_idx * dim_size;
        for (int i = 0; i < dims_per_thread; ++i) {
            int d = local_id + i * WGS;
            if (d < dim_size) {
                partial_sum += q_vec[d] * K[k_addr + d];
            }
        }
        partial_sums[local_id] = partial_sum;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int offset = WGS / 2; offset > 0; offset /= 2) {
            if (local_id < offset) {
                partial_sums[local_id] += partial_sums[local_id + offset];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // c. 由线程0计算并更新共享的统计量
        // __local float p, scale_factor; // <--- 从这里移除
        if (local_id == 0) {
            float score = partial_sums[0] * scale;
            float old_max = l_max_score;
            l_max_score = fmax(old_max, score);
            scale_factor = exp(old_max - l_max_score);
            p = exp(score - l_max_score);
            l_sum_exp = l_sum_exp * scale_factor + p;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // d. 所有线程并行累加V向量
        long v_addr = (long)b * seq_size_k * kv_head_size * dim_size + (long)s_k * kv_head_size * dim_size + (long)kv_h_idx * dim_size;
        for (int i = 0; i < dims_per_thread; ++i) {
            int d = local_id + i * WGS;
            if (d < dim_size) {
                my_acc_o[i] = my_acc_o[i] * scale_factor + p * V[v_addr + d];
            }
        }
    }

    // 5. 最终写回
    barrier(CLK_LOCAL_MEM_FENCE);
    float inv_sum_exp = 1.0f / (l_sum_exp + 1e-6f);
    long o_addr = (long)b * q_head_size * dim_size + (long)h * dim_size;

    for (int i = 0; i < dims_per_thread; ++i) {
        int d = local_id + i * WGS;
        if (d < dim_size) {
            O[o_addr + d] = my_acc_o[i] * inv_sum_exp;
        }
    }
}

// =================================================================================================
//
// FP16 KERNELS START HERE
//
// =================================================================================================
#if !defined(SUPPORTS_FP16)
// -------------------------------------------------------------------------------------------------
// [回退版] FP16 Prefill (Tiled) Kernel
// -------------------------------------------------------------------------------------------------
__kernel void flash_attention_2_prefill_fp16(
    __global const half *Q,
    __global const half *K,
    __global const half *V,
    __global half *O,
    const int q_head_size,
    const int kv_head_size,
    const int seq_size_q,
    const int seq_size_k,
    const int dim_size,
    const int causal_mask_flag) {
    __local half k_tile[TILE_S][DIM_MAX];
    __local half v_tile[TILE_S][DIM_MAX];

    const int row_idx = get_group_id(0);
    const int local_id = get_local_id(0);
    const int b = row_idx / (seq_size_q * q_head_size);
    const int s_q_h_idx = row_idx % (seq_size_q * q_head_size);
    const int s_q = s_q_h_idx / q_head_size;
    const int h = s_q_h_idx % q_head_size;
    const int kv_h_idx = h / (q_head_size / kv_head_size);

    float my_max_score = -F32_MAX;
    float my_sum_exp = 0.0f;
    float my_acc_o[DIM_MAX];
    for (int d = 0; d < dim_size; ++d) {
        my_acc_o[d] = 0.0f;
    }

    half my_q[DIM_MAX];
    const long q_base_addr = (long)b * seq_size_q * q_head_size * dim_size + (long)s_q * q_head_size * dim_size + (long)h * dim_size;
    for (int d = 0; d < dim_size; ++d) {
        my_q[d] = Q[q_base_addr + d];
    }

    const float scale = rsqrt((float)dim_size);
    const int max_s_k = (causal_mask_flag && seq_size_q > 1) ? (s_q + 1) : seq_size_k;

    for (int s_k_start = 0; s_k_start < max_s_k; s_k_start += TILE_S) {
        for (int i = local_id; i < TILE_S * dim_size; i += WGS) {
            int s_local = i / dim_size;
            int d_local = i % dim_size;
            int s_k_global = s_k_start + s_local;
            if (s_k_global < seq_size_k) {
                long kv_offset = (long)b * seq_size_k * kv_head_size * dim_size + (long)s_k_global * kv_head_size * dim_size + (long)kv_h_idx * dim_size + d_local;
                k_tile[s_local][d_local] = K[kv_offset];
                v_tile[s_local][d_local] = V[kv_offset];
            } else {
                k_tile[s_local][d_local] = 0.0h;
                v_tile[s_local][d_local] = 0.0h;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int s_local = 0; s_local < TILE_S; ++s_local) {
            int s_k_global = s_k_start + s_local;
            if (s_k_global < max_s_k) {
                float score = 0.0f;
                for (int d = 0; d < dim_size; ++d) {
                    score += (float)my_q[d] * (float)k_tile[s_local][d];
                }
                score *= scale;

                float old_max = my_max_score;
                my_max_score = fmax(old_max, score);

                float scale_factor = exp(old_max - my_max_score);
                my_sum_exp *= scale_factor;
                for (int d = 0; d < dim_size; ++d) {
                    my_acc_o[d] *= scale_factor;
                }

                float p = exp(score - my_max_score);
                my_sum_exp += p;

                for (int d = 0; d < dim_size; ++d) {
                    my_acc_o[d] += p * (float)v_tile[s_local][d];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        float inv_sum_exp = 1.0f / (my_sum_exp + 1e-6f);
        long o_offset = (long)b * seq_size_q * q_head_size * dim_size + (long)s_q * q_head_size * dim_size + (long)h * dim_size;
        for (int d = 0; d < dim_size; ++d) {
            O[o_offset + d] = (half)(my_acc_o[d] * inv_sum_exp);
        }
    }
}

#else
// -------------------------------------------------------------------------------------------------
// [高性能版] FP16 Prefill (Tiled) Kernel
// -------------------------------------------------------------------------------------------------

#define VEC_SIZE 8
__kernel void flash_attention_2_prefill_fp16(
    __global const half *Q,
    __global const half *K,
    __global const half *V,
    __global half *O,
    const int q_head_size,
    const int kv_head_size,
    const int seq_size_q,
    const int seq_size_k,
    const int dim_size,
    const int causal_mask_flag) {
    __local half k_tile[TILE_S * DIM_MAX];
    __local half v_tile[TILE_S * DIM_MAX];
    __local half q_local[DIM_MAX];

    const int row_idx = get_group_id(0);
    const int local_id = get_local_id(0);

    const int b = row_idx / (seq_size_q * q_head_size);
    const int s_q_h_idx = row_idx % (seq_size_q * q_head_size);
    const int s_q = s_q_h_idx / q_head_size;
    const int h = s_q_h_idx % q_head_size;
    const int kv_h_idx = h / (q_head_size / kv_head_size);

    float my_max_score = -F32_MAX;
    float my_sum_exp = 0.0f;
    float8 my_acc_o_vec[DIM_MAX / VEC_SIZE];
    for (int i = 0; i < dim_size / VEC_SIZE; ++i) {
        my_acc_o_vec[i] = (float8)(0.0f);
    }

    const long q_base_addr = (long)b * seq_size_q * q_head_size * dim_size + (long)s_q * q_head_size * dim_size + (long)h * dim_size;
    for (int d = local_id; d < dim_size; d += WGS) {
        q_local[d] = Q[q_base_addr + d];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const float scale = rsqrt((float)dim_size);
    const int max_s_k = (causal_mask_flag && seq_size_q > 1) ? (s_q + 1) : seq_size_k;
    const int dim_vec = dim_size / VEC_SIZE;

    for (int s_k_start = 0; s_k_start < max_s_k; s_k_start += TILE_S) {
        for (int i = local_id; i < TILE_S * dim_vec; i += WGS) {
            int s_local = i / dim_vec;
            int d_vec_idx = i % dim_vec;
            int s_k_global = s_k_start + s_local;

            if (s_k_global < seq_size_k) {
                long kv_offset = (long)b * seq_size_k * kv_head_size * dim_size + (long)s_k_global * kv_head_size * dim_size + (long)kv_h_idx * dim_size + (long)d_vec_idx * VEC_SIZE;
                *((__local half8 *)k_tile + i) = *((__global const half8 *)(K + kv_offset));
                *((__local half8 *)v_tile + i) = *((__global const half8 *)(V + kv_offset));
            } else {
                *((__local half8 *)k_tile + i) = (half8)(0.0h);
                *((__local half8 *)v_tile + i) = (half8)(0.0h);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int s_local = 0; s_local < TILE_S; ++s_local) {
            int s_k_global = s_k_start + s_local;
            if (s_k_global < max_s_k) {
                float score = 0.0f;
                for (int d_vec = 0; d_vec < dim_vec; ++d_vec) {
                    float8 q_f_vec = vload_half8(d_vec, q_local);
                    float8 k_f_vec = vload_half8(s_local * dim_vec + d_vec, k_tile);
                    float8 mul_res = q_f_vec * k_f_vec;
                    score += mul_res.s0 + mul_res.s1 + mul_res.s2 + mul_res.s3 + mul_res.s4 + mul_res.s5 + mul_res.s6 + mul_res.s7;
                }
                score *= scale;
                float old_max = my_max_score;
                my_max_score = fmax(old_max, score);
                float scale_factor = exp(old_max - my_max_score);
                my_sum_exp *= scale_factor;
                float p = exp(score - my_max_score);
                my_sum_exp += p;
                for (int d_vec = 0; d_vec < dim_vec; ++d_vec) {
                    float8 v_f_vec = vload_half8(s_local * dim_vec + d_vec, v_tile);
                    my_acc_o_vec[d_vec] = mad((float8)(p), v_f_vec, my_acc_o_vec[d_vec] * scale_factor);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0) {
        float inv_sum_exp = 1.0f / (my_sum_exp + 1e-6f);
        long o_offset = (long)b * seq_size_q * q_head_size * dim_size + (long)s_q * q_head_size * dim_size + (long)h * dim_size;
        for (int d_vec = 0; d_vec < dim_vec; ++d_vec) {
            float8 out_f_vec = my_acc_o_vec[d_vec] * inv_sum_exp;
            vstore_half8(out_f_vec, 0, (__global half *)(O + o_offset + d_vec * VEC_SIZE));
        }
    }
}

#endif // SUPPORTS_FP16
// -------------------------------------------------------------------------------------------------
//  FP16 Decode Kernel
// -------------------------------------------------------------------------------------------------
__kernel void flash_attention_2_decode_fp16(
    __global const half *Q,
    __global const half *K,
    __global const half *V,
    __global half *O,
    const int q_head_size,
    const int kv_head_size,
    const int seq_size_k,
    const int dim_size) {
    __local half q_vec[DIM_MAX];
    __local float l_max_score;
    __local float l_sum_exp;
    __local float partial_sums[WGS];
    __local float p, scale_factor;

    const int row_idx = get_group_id(0);
    const int local_id = get_local_id(0);

    const int b = row_idx / q_head_size;
    const int h = row_idx % q_head_size;
    const int kv_h_idx = h / (q_head_size / kv_head_size);

    const long q_addr = (long)b * q_head_size * dim_size + (long)h * dim_size;
    for (int d = local_id; d < dim_size; d += WGS) {
        q_vec[d] = Q[q_addr + d];
    }

    if (local_id == 0) {
        l_max_score = -F32_MAX;
        l_sum_exp = 0.0f;
    }

    const int dims_per_thread = (dim_size + WGS - 1) / WGS;
    float my_acc_o[DIM_MAX / WGS + 1];
    for (int i = 0; i < dims_per_thread; ++i) my_acc_o[i] = 0.0f;

    const float scale = rsqrt((float)dim_size);
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s_k = 0; s_k < seq_size_k; ++s_k) {
        float partial_sum = 0.0f;
        long k_addr = (long)b * seq_size_k * kv_head_size * dim_size + (long)s_k * kv_head_size * dim_size + (long)kv_h_idx * dim_size;
        for (int i = 0; i < dims_per_thread; ++i) {
            int d = local_id + i * WGS;
            if (d < dim_size) {
                partial_sum += (float)q_vec[d] * (float)K[k_addr + d];
            }
        }
        partial_sums[local_id] = partial_sum;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int offset = WGS / 2; offset > 0; offset /= 2) {
            if (local_id < offset) {
                partial_sums[local_id] += partial_sums[local_id + offset];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (local_id == 0) {
            float score = partial_sums[0] * scale;
            float old_max = l_max_score;
            l_max_score = fmax(old_max, score);
            scale_factor = exp(old_max - l_max_score);
            p = exp(score - l_max_score);
            l_sum_exp = l_sum_exp * scale_factor + p;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        long v_addr = (long)b * seq_size_k * kv_head_size * dim_size + (long)s_k * kv_head_size * dim_size + (long)kv_h_idx * dim_size;
        for (int i = 0; i < dims_per_thread; ++i) {
            int d = local_id + i * WGS;
            if (d < dim_size) {
                my_acc_o[i] = my_acc_o[i] * scale_factor + p * (float)V[v_addr + d];
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    float inv_sum_exp = 1.0f / (l_sum_exp + 1e-6f);
    long o_addr = (long)b * q_head_size * dim_size + (long)h * dim_size;
    for (int i = 0; i < dims_per_thread; ++i) {
        int d = local_id + i * WGS;
        if (d < dim_size) {
            O[o_addr + d] = (half)(my_acc_o[i] * inv_sum_exp);
        }
    }
}

// ---------- [Image 版] FP32 Prefill Kernel ----------
__kernel void flash_attention_2_prefill_fp32_image(
    sampler_t sampler,
    __read_only image2d_t Q, __read_only image2d_t K, __read_only image2d_t V,
    __write_only image2d_t O,
    const int q_head_size, const int kv_head_size,
    const int seq_size_q, const int seq_size_k,
    const int dim_size, const int causal_mask_flag) {
    const int row_idx = get_group_id(0);
    const int local_id = get_local_id(0);
    const int b = row_idx / (seq_size_q * q_head_size);
    const int s_h_idx = row_idx % (seq_size_q * q_head_size);
    const int s_q = s_h_idx / q_head_size;
    const int h = s_h_idx % q_head_size;
    const int kv_h_idx = h / (q_head_size / kv_head_size);
    float my_max_score = -F32_MAX;
    float my_sum_exp = 0.0f;
    float my_acc_o[128];
    for (int d = 0; d < dim_size; ++d) my_acc_o[d] = 0.0f;
    float my_q[128];
    const int q_y_coord = (b * seq_size_q * q_head_size) + (s_q * q_head_size) + h;
    for (int d_pixel = 0; d_pixel < dim_size / 4; ++d_pixel) {
        float4 q_pix = read_imagef(Q, sampler, (int2)(d_pixel, q_y_coord));
        my_q[d_pixel * 4 + 0] = q_pix.x;
        my_q[d_pixel * 4 + 1] = q_pix.y;
        my_q[d_pixel * 4 + 2] = q_pix.z;
        my_q[d_pixel * 4 + 3] = q_pix.w;
    }
    const float scale = rsqrt((float)dim_size);
    const int max_s_k = (causal_mask_flag && seq_size_q > 1) ? (s_q + 1) : seq_size_k;
    for (int s_k_start = 0; s_k_start < max_s_k; s_k_start += TILE_S) {
        for (int s_local = 0; s_local < TILE_S; ++s_local) {
            int s_k_global = s_k_start + s_local;
            if (s_k_global < max_s_k) {
                float score = 0.0f;
                const int k_y_coord = (b * seq_size_k * kv_head_size) + (s_k_global * kv_head_size) + kv_h_idx;
                for (int d_pixel = 0; d_pixel < dim_size / 4; ++d_pixel) {
                    float4 k_pix = read_imagef(K, sampler, (int2)(d_pixel, k_y_coord));
                    score += my_q[d_pixel * 4 + 0] * k_pix.x;
                    score += my_q[d_pixel * 4 + 1] * k_pix.y;
                    score += my_q[d_pixel * 4 + 2] * k_pix.z;
                    score += my_q[d_pixel * 4 + 3] * k_pix.w;
                }
                score *= scale;
                float old_max = my_max_score;
                my_max_score = fmax(old_max, score);
                float scale_factor = exp(old_max - my_max_score);
                my_sum_exp *= scale_factor;
                float p = exp(score - my_max_score);
                const int v_y_coord = k_y_coord;
                for (int d_pixel = 0; d_pixel < dim_size / 4; ++d_pixel) {
                    float4 v_pix = read_imagef(V, sampler, (int2)(d_pixel, v_y_coord));
                    my_acc_o[d_pixel * 4 + 0] = my_acc_o[d_pixel * 4 + 0] * scale_factor + p * v_pix.x;
                    my_acc_o[d_pixel * 4 + 1] = my_acc_o[d_pixel * 4 + 1] * scale_factor + p * v_pix.y;
                    my_acc_o[d_pixel * 4 + 2] = my_acc_o[d_pixel * 4 + 2] * scale_factor + p * v_pix.z;
                    my_acc_o[d_pixel * 4 + 3] = my_acc_o[d_pixel * 4 + 3] * scale_factor + p * v_pix.w;
                }
                my_sum_exp += p;
            }
        }
    }
    if (local_id == 0) {
        float inv_sum_exp = 1.0f / (my_sum_exp + 1e-6f);
        const int o_y_coord = (b * seq_size_q * q_head_size) + (s_q * q_head_size) + h;
        for (int d_pixel = 0; d_pixel < dim_size / 4; ++d_pixel) {
            float4 out_pixel;
            out_pixel.x = my_acc_o[d_pixel * 4 + 0] * inv_sum_exp;
            out_pixel.y = my_acc_o[d_pixel * 4 + 1] * inv_sum_exp;
            out_pixel.z = my_acc_o[d_pixel * 4 + 2] * inv_sum_exp;
            out_pixel.w = my_acc_o[d_pixel * 4 + 3] * inv_sum_exp;
            write_imagef(O, (int2)(d_pixel, o_y_coord), out_pixel);
        }
    }
}

// ---------- [Image 版] FP32 Decode Kernel [最终修正] ----------
__kernel void flash_attention_2_decode_fp32_image(
    sampler_t sampler,
    __read_only image2d_t Q, __read_only image2d_t K, __read_only image2d_t V,
    __write_only image2d_t O,
    const int q_head_size, const int kv_head_size,
    const int seq_size_k, const int dim_size) {
    const int gx = get_global_id(0);
    const int gy = get_global_id(1);
    if (gx >= (dim_size / 4)) return;
    const int b = gy / q_head_size;
    const int h = gy % q_head_size;
    const int kv_h_idx = h / (q_head_size / kv_head_size);
    float max_score = -F32_MAX;
    float sum_exp = 0.0f;
    float4 acc_o = (float4)(0.0f);
    const float scale = rsqrt((float)dim_size);
    const int q_y_coord = gy;
    for (int s_k = 0; s_k < seq_size_k; ++s_k) {
        float score = 0.0f;
        const int k_y_coord = (b * seq_size_k * kv_head_size) + (s_k * kv_head_size) + kv_h_idx;
        for (int d_pixel = 0; d_pixel < dim_size / 4; ++d_pixel) {
            float4 q_pix = read_imagef(Q, sampler, (int2)(d_pixel, q_y_coord));
            float4 k_pix = read_imagef(K, sampler, (int2)(d_pixel, k_y_coord));
            score += dot(q_pix, k_pix);
        }
        score *= scale;
        float old_max = max_score;
        max_score = fmax(old_max, score);
        float scale_factor = exp(old_max - max_score);
        sum_exp *= scale_factor;
        acc_o *= scale_factor;
        float p = exp(score - max_score);
        sum_exp += p;
        const int v_y_coord = k_y_coord;
        float4 v_pix = read_imagef(V, sampler, (int2)(gx, v_y_coord));
        acc_o = mad(p, v_pix, acc_o);
    }
    float inv_sum_exp = 1.0f / (sum_exp + 1e-6f);
    acc_o *= inv_sum_exp;
    write_imagef(O, (int2)(gx, gy), acc_o);
}

// =================================================================================================
// 3. FP16 全 Image 版本的内核
// =================================================================================================
#if defined(SUPPORTS_FP16)
// =================================================================================================
// FP16 Prefill Kernel (Image)
// =================================================================================================
__kernel void flash_attention_2_prefill_fp16_image(
    sampler_t sampler,
    __read_only image2d_t Q, __read_only image2d_t K, __read_only image2d_t V,
    __write_only image2d_t O,
    const int q_head_size, const int kv_head_size,
    const int seq_size_q, const int seq_size_k,
    const int dim_size, const int causal_mask_flag) {
    const int row_idx = get_group_id(0);
    const int local_id = get_local_id(0);
    const int b = row_idx / (seq_size_q * q_head_size);
    const int s_h_idx = row_idx % (seq_size_q * q_head_size);
    const int s_q = s_h_idx / q_head_size;
    const int h = s_h_idx % q_head_size;
    const int kv_h_idx = h / (q_head_size / kv_head_size);
    float my_max_score = -F32_MAX;
    float my_sum_exp = 0.0f;
    float4 my_acc_o[32];
    float4 my_q[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        my_acc_o[i] = (float4)(0.0f);
    }
    const int q_y_coord = (b * seq_size_q * q_head_size) + (s_q * q_head_size) + h;
    const int dim_vec_size = dim_size / 4;
#pragma unroll
    for (int d_vec = 0; d_vec < dim_vec_size; ++d_vec) {
        my_q[d_vec] = convert_float4(read_imageh(Q, sampler, (int2)(d_vec, q_y_coord)));
    }
    const float scale = rsqrt((float)dim_size);
    const int max_s_k = (causal_mask_flag && seq_size_q > 1) ? (s_q + 1) : seq_size_k;
    for (int s_k_global = 0; s_k_global < max_s_k; ++s_k_global) {
        float score = 0.0f;
        const int k_y_coord = (b * seq_size_k * kv_head_size) + (s_k_global * kv_head_size) + kv_h_idx;
#pragma unroll
        for (int d_vec = 0; d_vec < dim_vec_size; ++d_vec) {
            float4 k_pix = convert_float4(read_imageh(K, sampler, (int2)(d_vec, k_y_coord)));
            score += dot(my_q[d_vec], k_pix);
        }
        score *= scale;
        float old_max = my_max_score;
        my_max_score = fmax(old_max, score);
        float scale_factor = exp(old_max - my_max_score);
        my_sum_exp *= scale_factor;
        float p = exp(score - my_max_score);
        const int v_y_coord = k_y_coord;
#pragma unroll
        for (int d_vec = 0; d_vec < dim_vec_size; ++d_vec) {
            float4 v_pix = convert_float4(read_imageh(V, sampler, (int2)(d_vec, v_y_coord)));
            my_acc_o[d_vec] = my_acc_o[d_vec] * scale_factor + p * v_pix;
        }
        my_sum_exp += p;
    }
    if (local_id == 0) {
        float inv_sum_exp = 1.0f / (my_sum_exp + 1e-6f);
        const int o_y_coord = (b * seq_size_q * q_head_size) + (s_q * q_head_size) + h;
#pragma unroll
        for (int d_vec = 0; d_vec < dim_vec_size; ++d_vec) {
            float4 out_pixel_f = my_acc_o[d_vec] * inv_sum_exp;
            write_imageh(O, (int2)(d_vec, o_y_coord), convert_half4_rte(out_pixel_f));
        }
    }
}

// =================================================================================================
// FP16 Decode Kernel (Image)
// =================================================================================================
__kernel void flash_attention_2_decode_fp16_image(
    sampler_t sampler,
    __read_only image2d_t Q, __read_only image2d_t K, __read_only image2d_t V,
    __write_only image2d_t O,
    const int q_head_size, const int kv_head_size,
    const int seq_size_k, const int dim_size) {
    const int gx = get_global_id(0); // Corresponds to dimension vector index
    const int gy = get_global_id(1); // Corresponds to batch/head/sequence index
    if (gx >= (dim_size / 4)) return;
    const int b = gy / q_head_size;
    const int h = gy % q_head_size;
    const int kv_h_idx = h / (q_head_size / kv_head_size);
    float max_score = -F32_MAX;
    float sum_exp = 0.0f;
    float4 acc_o = (float4)(0.0f);
    const float scale = rsqrt((float)dim_size);
    const int q_y_coord = gy;
    const int dim_vec_size = dim_size / 4;
    for (int s_k = 0; s_k < seq_size_k; ++s_k) {
        float score = 0.0f;
        const int k_y_coord = (b * seq_size_k * kv_head_size) + (s_k * kv_head_size) + kv_h_idx;
#pragma unroll
        for (int d_vec = 0; d_vec < dim_vec_size; ++d_vec) {
            float4 q_pix = convert_float4(read_imageh(Q, sampler, (int2)(d_vec, q_y_coord)));
            float4 k_pix = convert_float4(read_imageh(K, sampler, (int2)(d_vec, k_y_coord)));
            score += dot(q_pix, k_pix);
        }
        score *= scale;
        float old_max = max_score;
        max_score = fmax(old_max, score);
        float scale_factor = exp(old_max - max_score);
        sum_exp *= scale_factor;
        acc_o *= scale_factor;
        float p = exp(score - max_score);
        sum_exp += p;
        const int v_y_coord = k_y_coord;
        float4 v_pix = convert_float4(read_imageh(V, sampler, (int2)(gx, v_y_coord)));
        acc_o = mad(p, v_pix, acc_o);
    }
    float inv_sum_exp = 1.0f / (sum_exp + 1e-6f);
    acc_o *= inv_sum_exp;
    write_imageh(O, (int2)(gx, gy), convert_half4_rte(acc_o));
}

#else

// ---------- [Image 版] FP16 Prefill Kernel [兼容回退版] ----------
__kernel void flash_attention_2_prefill_fp16_image(
    sampler_t sampler,
    __read_only image2d_t Q, __read_only image2d_t K, __read_only image2d_t V,
    __write_only image2d_t O,
    const int q_head_size, const int kv_head_size,
    const int seq_size_q, const int seq_size_k,
    const int dim_size, const int causal_mask_flag) {
    const int row_idx = get_group_id(0);
    const int local_id = get_local_id(0);
    const int b = row_idx / (seq_size_q * q_head_size);
    const int s_h_idx = row_idx % (seq_size_q * q_head_size);
    const int s_q = s_h_idx / q_head_size;
    const int h = s_h_idx % q_head_size;
    const int kv_h_idx = h / (q_head_size / kv_head_size);

    float my_max_score = -F32_MAX;
    float my_sum_exp = 0.0f;
    float my_acc_o[128]; // 假设 DIM_MAX <= 128
    for (int d = 0; d < dim_size; ++d) my_acc_o[d] = 0.0f;

    float my_q[128];
    const int q_y_coord = (b * seq_size_q * q_head_size) + (s_q * q_head_size) + h;
    for (int d_pixel = 0; d_pixel < dim_size / 4; ++d_pixel) {
        // 直接读取CL_FLOAT图像
        float4 q_pix = read_imagef(Q, sampler, (int2)(d_pixel, q_y_coord));
        my_q[d_pixel * 4 + 0] = q_pix.x;
        my_q[d_pixel * 4 + 1] = q_pix.y;
        my_q[d_pixel * 4 + 2] = q_pix.z;
        my_q[d_pixel * 4 + 3] = q_pix.w;
    }

    const float scale = rsqrt((float)dim_size);
    const int max_s_k = (causal_mask_flag && seq_size_q > 1) ? (s_q + 1) : seq_size_k;

    for (int s_k_global = 0; s_k_global < max_s_k; ++s_k_global) {
        float score = 0.0f;
        const int k_y_coord = (b * seq_size_k * kv_head_size) + (s_k_global * kv_head_size) + kv_h_idx;
        for (int d_pixel = 0; d_pixel < dim_size / 4; ++d_pixel) {
            float4 k_pix = read_imagef(K, sampler, (int2)(d_pixel, k_y_coord));
            score += my_q[d_pixel * 4 + 0] * k_pix.x;
            score += my_q[d_pixel * 4 + 1] * k_pix.y;
            score += my_q[d_pixel * 4 + 2] * k_pix.z;
            score += my_q[d_pixel * 4 + 3] * k_pix.w;
        }
        score *= scale;

        float old_max = my_max_score;
        my_max_score = fmax(old_max, score);
        float scale_factor = exp(old_max - my_max_score);
        my_sum_exp *= scale_factor;
        float p = exp(score - my_max_score);

        const int v_y_coord = k_y_coord;
        for (int d_pixel = 0; d_pixel < dim_size / 4; ++d_pixel) {
            float4 v_pix = read_imagef(V, sampler, (int2)(d_pixel, v_y_coord));
            my_acc_o[d_pixel * 4 + 0] = my_acc_o[d_pixel * 4 + 0] * scale_factor + p * v_pix.x;
            my_acc_o[d_pixel * 4 + 1] = my_acc_o[d_pixel * 4 + 1] * scale_factor + p * v_pix.y;
            my_acc_o[d_pixel * 4 + 2] = my_acc_o[d_pixel * 4 + 2] * scale_factor + p * v_pix.z;
            my_acc_o[d_pixel * 4 + 3] = my_acc_o[d_pixel * 4 + 3] * scale_factor + p * v_pix.w;
        }
        my_sum_exp += p;
    }

    if (local_id == 0) {
        float inv_sum_exp = 1.0f / (my_sum_exp + 1e-6f);
        const int o_y_coord = (b * seq_size_q * q_head_size) + (s_q * q_head_size) + h;
        for (int d_pixel = 0; d_pixel < dim_size / 4; ++d_pixel) {
            float4 out_pixel;
            out_pixel.x = my_acc_o[d_pixel * 4 + 0] * inv_sum_exp;
            out_pixel.y = my_acc_o[d_pixel * 4 + 1] * inv_sum_exp;
            out_pixel.z = my_acc_o[d_pixel * 4 + 2] * inv_sum_exp;
            out_pixel.w = my_acc_o[d_pixel * 4 + 3] * inv_sum_exp;
            // 直接写入CL_FLOAT图像
            write_imagef(O, (int2)(d_pixel, o_y_coord), out_pixel);
        }
    }
}

// ---------- [Image 版] FP16 Decode Kernel [兼容回退版] ----------
__kernel void flash_attention_2_decode_fp16_image(
    sampler_t sampler,
    __read_only image2d_t Q, __read_only image2d_t K, __read_only image2d_t V,
    __write_only image2d_t O,
    const int q_head_size, const int kv_head_size,
    const int seq_size_k, const int dim_size) {
    const int gx = get_global_id(0);
    const int gy = get_global_id(1);

    if (gx >= (dim_size / 4)) return;

    const int b = gy / q_head_size;
    const int h = gy % q_head_size;
    const int kv_h_idx = h / (q_head_size / kv_head_size);

    float max_score = -F32_MAX;
    float sum_exp = 0.0f;
    float4 acc_o = (float4)(0.0f);

    const float scale = rsqrt((float)dim_size);
    const int q_y_coord = gy;

    for (int s_k = 0; s_k < seq_size_k; ++s_k) {
        float score = 0.0f;
        const int k_y_coord = (b * seq_size_k * kv_head_size) + (s_k * kv_head_size) + kv_h_idx;
        for (int d_pixel = 0; d_pixel < dim_size / 4; ++d_pixel) {
            float4 q_pix = read_imagef(Q, sampler, (int2)(d_pixel, q_y_coord));
            float4 k_pix = read_imagef(K, sampler, (int2)(d_pixel, k_y_coord));
            score += dot(q_pix, k_pix);
        }
        score *= scale;

        float old_max = max_score;
        max_score = fmax(old_max, score);
        float scale_factor = exp(old_max - max_score);
        sum_exp *= scale_factor;
        acc_o *= scale_factor;
        float p = exp(score - max_score);
        sum_exp += p;

        const int v_y_coord = k_y_coord;
        float4 v_pix = read_imagef(V, sampler, (int2)(gx, v_y_coord));
        acc_o = mad(p, v_pix, acc_o);
    }

    float inv_sum_exp = 1.0f / (sum_exp + 1e-6f);
    acc_o *= inv_sum_exp;

    // 直接写入CL_FLOAT图像
    write_imagef(O, (int2)(gx, gy), acc_o);
}
#endif // SUPPORTS_FP16
