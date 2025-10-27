#ifndef SAGE_QUANT_H
#define SAGE_QUANT_H

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <algorithm>
#include "Types.hpp"

// --- SIMD Intrinsics (依赖项) ---
#ifdef __AVX2__
#include <immintrin.h>
#endif

// 外部依赖，确保QK8_0F在包含此文件前已定义
// 或者直接在这里定义
// #ifndef QK8_0F
// #define QK8_0F 128
// #endif

// 为了让工具函数独立，把 hmax_ps 这种辅助函数也移过来
#ifdef __AVX2__
namespace { // 放在匿名空间中，使其为内部链接
inline float _mm256_hmax_ps(__m256 x) {
    __m128 lo = _mm256_castps256_ps128(x);
    __m128 hi = _mm256_extractf128_ps(x, 1);
    __m128 max_val = _mm_max_ps(lo, hi);
    max_val = _mm_max_ps(max_val, _mm_shuffle_ps(max_val, max_val, _MM_SHUFFLE(0, 0, 2, 2)));
    max_val = _mm_max_ps(max_val, _mm_shuffle_ps(max_val, max_val, _MM_SHUFFLE(0, 0, 0, 1)));
    return _mm_cvtss_f32(max_val);
}
} // namespace
#endif

// struct block_q8_0f {
//     float scale;
//     int8_t qs[QK8_0F];
// };
#pragma region "SAGE KV Cache Utils"
namespace sage_kv_cache {
inline void compute_sage_mean_vector(const float *head_start_ptr, float *out_mean,
                                     int seq_size_k, int dim_size, size_t stride_s) {
    if (seq_size_k == 0)
        return;
    std::vector<double> sum(dim_size, 0.0);
#pragma omp parallel
    {
        std::vector<double> local_sum(dim_size, 0.0);
#pragma omp for
        for (int s = 0; s < seq_size_k; ++s) {
            const float *row_ptr = head_start_ptr + s * stride_s;
            for (int d = 0; d < dim_size; ++d)
                local_sum[d] += row_ptr[d];
        }
#pragma omp critical
        {
            for (int d = 0; d < dim_size; ++d)
                sum[d] += local_sum[d];
        }
    }
    const float inv_seq_len = 1.0f / static_cast<float>(seq_size_k);
#pragma omp parallel for
    for (int d = 0; d < dim_size; ++d)
        out_mean[d] = sum[d] * inv_seq_len;
}

inline void update_sage_mean_vector_incremental(float *mean_vector_io,
                                                const float *new_token_vector,
                                                int old_seq_len, int dim_size) {
    if (old_seq_len <= 0) {
        memcpy(mean_vector_io, new_token_vector, dim_size * sizeof(float));
        return;
    }
    const float N = static_cast<float>(old_seq_len);
    const float factor1 = N / (N + 1.0f);
    const float factor2 = 1.0f / (N + 1.0f);
#pragma omp parallel for
    for (int d = 0; d < dim_size; ++d)
        mean_vector_io[d] =
            mean_vector_io[d] * factor1 + new_token_vector[d] * factor2;
}

template <typename T>
inline void quantize_new_token_to_sage_blocks(const float *new_token_vector,
                                              const float *current_mean_data,
                                              T *out_blocks_for_token, int dim_size) {
    static_assert(std::is_same_v<T, block_q8_0f>,
                  "This function can only quantize to block_q8_0f.");
    block_q8_0f *out_ptr = reinterpret_cast<block_q8_0f *>(out_blocks_for_token);
    const int num_k_blocks = dim_size / QK8_0F;
    std::vector<float> smoothed_row(dim_size);
    for (int d = 0; d < dim_size; ++d)
        smoothed_row[d] = new_token_vector[d] - current_mean_data[d];
    for (int g = 0; g < num_k_blocks; ++g) {
        const int offset = g * QK8_0F;
        const float *smoothed_block_ptr = smoothed_row.data() + offset;
        float max_abs_val = 0.0f;
        for (int d = 0; d < QK8_0F; ++d)
            max_abs_val = std::max(max_abs_val, fabsf(smoothed_block_ptr[d]));
        const float scale = (max_abs_val > 1e-9f) ? max_abs_val / 127.0f : 0.0f;
        out_ptr[g].scale = scale;
        const float inv_scale = (scale > 1e-9f) ? 1.0f / scale : 0.0f;
        for (int d = 0; d < QK8_0F; ++d)
            out_ptr[g].qs[d] =
                static_cast<int8_t>(roundf(smoothed_block_ptr[d] * inv_scale));
    }
}

// [修正] 改为只为单个头计算均值
inline void compute_sage_mean_for_one_head_bshd(
    const float *head_start_ptr, // 指向单个头的起始地址
    float *out_mean_for_head,    // 指向该头对应的均值输出位置
    int seq_size, int dim_size,
    size_t s_stride // BSHD布局下，序列方向的步长
) {
    if (seq_size == 0) return;

    std::vector<double> sum(dim_size, 0.0);
    for (int s = 0; s < seq_size; ++s) {
        // 直接通过步长访问一个头内的所有序列
        const float *row_ptr = head_start_ptr + s * s_stride;
        for (int d = 0; d < dim_size; ++d) {
            sum[d] += row_ptr[d];
        }
    }

    const float inv_seq_len = 1.0f / static_cast<float>(seq_size);
    for (int d = 0; d < dim_size; ++d) {
        out_mean_for_head[d] = sum[d] * inv_seq_len;
    }
}

} // namespace sage_kv_cache
#pragma endregion
#endif // SAGE_QUANT_H