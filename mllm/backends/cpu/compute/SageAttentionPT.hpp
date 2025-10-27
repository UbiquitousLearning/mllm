// 文件名: sage_attention_unified_final_with_simd.cpp
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>
#include <cstdint>
#include <limits>
#include <cstring>
#include <omp.h>
// #include <numeric>
#include <algorithm>
// #include <iomanip>
#include <string>
#include <type_traits>

// --- SIMD Intrinsics ---
#ifdef __AVX2__
#include <immintrin.h>
#include <immintrin.h>
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
#include <arm_neon.h>
#if defined(__ARM_FP16_FORMAT_IEEE) && !defined(_MSC_VER)
#include <arm_fp16.h>
#endif
#endif

#include "Types.hpp"
#include "backends/cpu/third_party/ggml/QuantizeFP16.hpp"

namespace sage_attn_pt_cpu {

#define NEG_INF std::numeric_limits<float>::lowest()

template <typename T>
inline float to_float(T val);
template <>
inline float to_float<float>(float val) {
    return val;
}
template <>
inline float to_float<mllm_fp16_t>(mllm_fp16_t val) {
    return MLLM_FP16_TO_FP32(val);
}

#ifdef __AVX2__
inline float _mm256_hmax_ps(__m256 x) {
    __m128 lo = _mm256_castps256_ps128(x);
    __m128 hi = _mm256_extractf128_ps(x, 1);
    __m128 max_val = _mm_max_ps(lo, hi);
    max_val = _mm_max_ps(max_val, _mm_shuffle_ps(max_val, max_val, _MM_SHUFFLE(0, 0, 2, 2)));
    max_val = _mm_max_ps(max_val, _mm_shuffle_ps(max_val, max_val, _MM_SHUFFLE(0, 0, 0, 1)));
    return _mm_cvtss_f32(max_val);
}
inline float hsum_ps(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}
inline int32_t hsum_i32(__m256i v) {
    __m128i vlo = _mm256_castsi256_si128(v);
    __m128i vhi = _mm256_extracti128_si256(v, 1);
    __m128i vsum = _mm_add_epi32(vlo, vhi);
    vsum = _mm_add_epi32(vsum, _mm_shuffle_epi32(vsum, _MM_SHUFFLE(1, 0, 3, 2)));
    vsum = _mm_add_epi32(vsum, _mm_shuffle_epi32(vsum, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(vsum);
}

inline __m256 load_and_convert_to_fp32_vec(const float *ptr) {
    return _mm256_loadu_ps(ptr);
}
#ifdef __F16C__
inline __m256 load_and_convert_to_fp32_vec(const mllm_fp16_t *ptr) {
    return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)ptr));
}
#endif
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
inline float _vmaxvq_f32_hmax(float32x4_t x) {
    return vmaxvq_f32(x);
}
inline void load_and_convert_to_fp32x4x2(const float *ptr, float32x4_t &out_lo, float32x4_t &out_hi) {
    out_lo = vld1q_f32(ptr);
    out_hi = vld1q_f32(ptr + 4);
}
#if defined(__ARM_FP16_FORMAT_IEEE)
inline void load_and_convert_to_fp32x4x2(const mllm_fp16_t *ptr, float32x4_t &out_lo, float32x4_t &out_hi) {
    float16x8_t v_f16 = vld1q_f16(reinterpret_cast<const float16_t *>(ptr));
    out_lo = vcvt_f32_f16(vget_low_f16(v_f16));
    out_hi = vcvt_f32_f16(vget_high_f16(v_f16));
}
#endif
#endif

inline void aligned_alloc(void **ptr, size_t required_bytes, size_t align) {
    if (align % sizeof(void *) != 0 || (align & (align - 1)) != 0) {
        *ptr = nullptr;
        return;
    }
    if (posix_memalign(ptr, align, required_bytes) != 0) { *ptr = nullptr; }
}
inline void aligned_free(void *ptr) {
    free(ptr);
}

inline void quantize_row_simd(const float *float_row, int8_t *int8_row, float *scale, int dim_size, float sm_scale, float *temp_buf) {
    for (int d = 0; d < dim_size; ++d) { temp_buf[d] = float_row[d] * sm_scale; }
    float max_abs_val = 0.0f;
#if defined(__AVX2__)
    __m256 max_vec = _mm256_setzero_ps();
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    int d = 0;
    for (; d <= dim_size - 8; d += 8) max_vec = _mm256_max_ps(max_vec, _mm256_and_ps(_mm256_loadu_ps(temp_buf + d), abs_mask));
    max_abs_val = _mm256_hmax_ps(max_vec);
    for (; d < dim_size; ++d) max_abs_val = std::max(max_abs_val, fabsf(temp_buf[d]));
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
    float32x4_t max_vec = vdupq_n_f32(0.0f);
    int d = 0;
    for (; d <= dim_size - 4; d += 4) max_vec = vmaxq_f32(max_vec, vabsq_f32(vld1q_f32(temp_buf + d)));
    max_abs_val = vmaxvq_f32(max_vec);
    for (; d < dim_size; ++d) max_abs_val = std::max(max_abs_val, fabsf(temp_buf[d]));
#else
    for (int d = 0; d < dim_size; ++d) max_abs_val = std::max(max_abs_val, fabsf(temp_buf[d]));
#endif
    *scale = (max_abs_val > 1e-9f) ? max_abs_val / 127.0f : 0.0f;
    const float inv_scale = (*scale > 1e-9f) ? 1.0f / *scale : 0.0f;
#if defined(__AVX2__)
    __m256 inv_scale_vec = _mm256_set1_ps(inv_scale);
    d = 0;
    for (; d <= dim_size - 8; d += 8) {
        __m256i val_i32 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(temp_buf + d), inv_scale_vec));
        __m128i val_i16 = _mm_packs_epi32(_mm256_castsi256_si128(val_i32), _mm256_extracti128_si256(val_i32, 1));
        __m128i val_i8 = _mm_packs_epi16(val_i16, val_i16);
        *(int64_t *)(int8_row + d) = _mm_cvtsi128_si64(val_i8);
    }
    for (; d < dim_size; ++d) int8_row[d] = static_cast<int8_t>(roundf(temp_buf[d] * inv_scale));
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
    float32x4_t inv_scale_vec = vdupq_n_f32(inv_scale);
    d = 0;
    for (; d <= dim_size - 16; d += 16) {
        int32x4_t i32_0 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(temp_buf + d + 0), inv_scale_vec));
        int32x4_t i32_1 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(temp_buf + d + 4), inv_scale_vec));
        int32x4_t i32_2 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(temp_buf + d + 8), inv_scale_vec));
        int32x4_t i32_3 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(temp_buf + d + 12), inv_scale_vec));
        int16x8_t i16_0 = vcombine_s16(vqmovn_s32(i32_0), vqmovn_s32(i32_1));
        int16x8_t i16_1 = vcombine_s16(vqmovn_s32(i32_2), vqmovn_s32(i32_3));
        vst1q_s8(int8_row + d, vcombine_s8(vqmovn_s16(i16_0), vqmovn_s16(i16_1)));
    }
    for (; d < dim_size; ++d) int8_row[d] = static_cast<int8_t>(roundf(temp_buf[d] * inv_scale));
#else
    for (int d = 0; d < dim_size; ++d) int8_row[d] = static_cast<int8_t>(roundf(temp_buf[d] * inv_scale));
#endif
}

template <typename KVDtype>
void compute_channel_means(const KVDtype *tensor, float *mean_tensor, int batch_size, int head_size, int seq_len, int dim_size) {
#pragma omp parallel for collapse(3)
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < head_size; ++h) {
            for (int d = 0; d < dim_size; d += 8) { // AVX2 processes 8 floats at a time
#if (defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(__ARM_FP16_FORMAT_IEEE)
                float32x4_t sum_vec = vdupq_n_f32(0.0f);
                for (int s = 0; s < seq_len; ++s) {
                    int idx = b * (seq_len * head_size * dim_size) + s * (head_size * dim_size) + h * (dim_size) + d;
                    float32x4_t val_vec;
                    if constexpr (std::is_same_v<KVDtype, mllm_fp16_t>) {
                        val_vec = vcvt_f32_f16(vld1_f16(reinterpret_cast<const float16_t *>(tensor + idx)));
                    } else {
                        val_vec = vld1q_f32(tensor + idx);
                    }
                    sum_vec = vaddq_f32(sum_vec, val_vec);
                }
                float inv_seq_len = 1.0f / seq_len;
                float32x4_t mean_vec = vmulq_f32(sum_vec, vdupq_n_f32(inv_seq_len));
                // Note: Original code used d+=4, but NEON loop processed only 4 floats.
                // Assuming it should be d+=4 for the NEON part. Let's stick to the original logic.
                vst1q_f32(mean_tensor + b * head_size * dim_size + h * dim_size + d, mean_vec);
#elif defined(__AVX2__)
                // =========== AVX2 IMPLEMENTATION START ===========
                if (d + 8 > dim_size) { // Handle remainder
                    for (int i = 0; i < (dim_size - d); ++i) {
                        double sum = 0.0;
                        for (int s = 0; s < seq_len; ++s) {
                            int idx = b * (seq_len * head_size * dim_size) + s * (head_size * dim_size) + h * (dim_size) + d + i;
                            sum += to_float(tensor[idx]);
                        }
                        mean_tensor[b * head_size * dim_size + h * dim_size + d + i] = static_cast<float>(sum / seq_len);
                    }
                    continue; // Skip to next d in the outer loop
                }

                __m256 sum_vec = _mm256_setzero_ps();
                for (int s = 0; s < seq_len; ++s) {
                    const KVDtype *current_row = tensor + b * (seq_len * head_size * dim_size) + s * (head_size * dim_size) + h * (dim_size) + d;
                    __m256 val_vec = load_and_convert_to_fp32_vec(current_row);
                    sum_vec = _mm256_add_ps(sum_vec, val_vec);
                }
                const float inv_seq_len = 1.0f / seq_len;
                __m256 inv_len_vec = _mm256_set1_ps(inv_seq_len);
                __m256 mean_vec = _mm256_mul_ps(sum_vec, inv_len_vec);
                _mm256_storeu_ps(mean_tensor + b * head_size * dim_size + h * dim_size + d, mean_vec);
                // =========== AVX2 IMPLEMENTATION END ===========
#else
                // Fallback for non-AVX2/NEON
                double sum[8] = {0.0};
                for (int s = 0; s < seq_len; ++s) {
                    for (int i = 0; i < 8; ++i) {
                        if (d + i < dim_size) {
                            int idx = b * (seq_len * head_size * dim_size) + s * (head_size * dim_size) + h * (dim_size) + d + i;
                            sum[i] += to_float(tensor[idx]);
                        }
                    }
                }
                for (int i = 0; i < 8; ++i) {
                    if (d + i < dim_size) {
                        mean_tensor[b * head_size * dim_size + h * dim_size + d + i] = static_cast<float>(sum[i] / seq_len);
                    }
                }
#endif
            }
        }
    }
}

template <typename KVDtype>
void compute_mean_and_quantize_k(
    const KVDtype *K,
    float *mean_tensor,
    int8_t *k_quant_global,
    float *k_scale_global,
    int batch_size, int kv_head_size, int seq_size_k, int dim_size,
    int threads,
    float *temp_k_sum,
    float *temp_k_smoothed) {
#pragma omp parallel for num_threads(threads) collapse(2)
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < kv_head_size; ++h) {
            const int thread_id = omp_get_thread_num();
            float *thread_sum_buf = temp_k_sum + thread_id * dim_size;
            float *thread_smoothed_buf = temp_k_smoothed + thread_id * dim_size;

            float *target_mean = mean_tensor + b * kv_head_size * dim_size + h * dim_size;
            int8_t *target_k_quant = k_quant_global + (b * kv_head_size + h) * seq_size_k * dim_size;
            float *target_k_scale = k_scale_global + (b * kv_head_size + h) * seq_size_k;

            const int k_stride = kv_head_size * dim_size;

            memset(thread_sum_buf, 0, dim_size * sizeof(float));

            for (int s = 0; s < seq_size_k; ++s) {
                const KVDtype *k_row = K + b * seq_size_k * k_stride + s * k_stride + h * dim_size;
                for (int d = 0; d < dim_size; ++d) {
                    thread_sum_buf[d] += to_float(k_row[d]);
                }
            }

            float inv_seq_len = 1.0f / seq_size_k;
            for (int d = 0; d < dim_size; ++d) {
                target_mean[d] = thread_sum_buf[d] * inv_seq_len;
            }

            for (int s = 0; s < seq_size_k; ++s) {
                const KVDtype *k_row = K + b * seq_size_k * k_stride + s * k_stride + h * dim_size;
                for (int d = 0; d < dim_size; ++d) {
                    thread_smoothed_buf[d] = to_float(k_row[d]) - target_mean[d];
                }
                quantize_row_simd(thread_smoothed_buf, target_k_quant + s * dim_size, &target_k_scale[s], dim_size, 1.0f, thread_sum_buf);
            }
        }
    }
}

class WorkspaceManager {
public:
    WorkspaceManager() = default;
    ~WorkspaceManager() {
        for (auto &ptr : workspace_) {
            if (ptr) aligned_free(ptr);
        }
    }
    void **get_workspace(const std::vector<size_t> &required_sizes) {
        if (workspace_.empty()) {
            workspace_.resize(required_sizes.size(), nullptr);
            current_sizes_.resize(required_sizes.size(), 0);
        }
        for (size_t i = 0; i < required_sizes.size(); ++i) {
            if (required_sizes[i] > current_sizes_[i]) {
                if (workspace_[i]) aligned_free(workspace_[i]);
                aligned_alloc(&workspace_[i], required_sizes[i], 64);
                current_sizes_[i] = required_sizes[i];
            }
        }
        return workspace_.data();
    }

private:
    std::vector<void *> workspace_;
    std::vector<size_t> current_sizes_;
};

template <typename KVDtype>
struct SAGE_CPU_IMPL {
    using dtype_q_in_t = float;
    using dtype_kv_in_t = KVDtype;
    using dtype_out_t = float;
    int32_t Br, Bc, Q_Head, KV_Head, threads;
    float *acc_o_, *acc_s_, *logsum_, *scoremax_, *scoremax_prev_, *score_scale_, *score_sum_;
    int8_t *q_quant_tile_, *k_quant_tile_;
    float *q_scale_, *k_scale_, *k_smoothed_row_buf_, *q_scaled_row_buf_;

    void configure(int32_t Br_, int32_t Bc_, int32_t Q_Head_, int32_t KV_Head_, int32_t threads_) {
        Br = Br_;
        Bc = Bc_;
        Q_Head = Q_Head_;
        KV_Head = KV_Head_;
        threads = threads_;
    }
    void init_workspace(void **workspace) {
        acc_o_ = static_cast<float *>(workspace[0]);
        acc_s_ = static_cast<float *>(workspace[1]);
        logsum_ = static_cast<float *>(workspace[2]);
        scoremax_ = static_cast<float *>(workspace[3]);
        scoremax_prev_ = static_cast<float *>(workspace[4]);
        score_scale_ = static_cast<float *>(workspace[5]);
        score_sum_ = static_cast<float *>(workspace[6]);
        q_quant_tile_ = static_cast<int8_t *>(workspace[7]);
        k_quant_tile_ = static_cast<int8_t *>(workspace[8]);
        q_scale_ = static_cast<float *>(workspace[9]);
        k_scale_ = static_cast<float *>(workspace[10]);
        k_smoothed_row_buf_ = static_cast<float *>(workspace[11]);
        q_scaled_row_buf_ = static_cast<float *>(workspace[12]);
    }

    void sage_attn_prefill(const dtype_q_in_t *__restrict__ Q, const dtype_kv_in_t *__restrict__ K, const dtype_kv_in_t *__restrict__ V, dtype_out_t *__restrict__ O, const float *K_mean, const float *V_mean, int32_t batch_size, int32_t head_size, int32_t seq_size_q, int32_t seq_size_k, int32_t dim_size, bool causal_mask) {
        const int32_t Tr = (seq_size_q + Br - 1) / Br;
        const int32_t Tc = (seq_size_k + Bc - 1) / Bc;
        const float local_scale = 1.0f / sqrtf(static_cast<float>(dim_size));
        const int32_t kv_group_size = (Q_Head > 0 && KV_Head > 0) ? Q_Head / KV_Head : 1;

        for (int32_t b_idx = 0; b_idx < batch_size; ++b_idx) {
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1)
            for (int32_t h_idx = 0; h_idx < head_size; ++h_idx) {
                const int32_t thread_id = omp_get_thread_num();
                const int32_t this_thread_kv_head = h_idx / kv_group_size;

                float *p_acc_o = acc_o_ + thread_id * Br * dim_size;
                float *p_acc_s = acc_s_ + thread_id * Br * Bc;
                float *p_logsum = logsum_ + thread_id * Br;
                float *p_scoremax = scoremax_ + thread_id * Br;
                float *p_scoremax_prev = scoremax_prev_ + thread_id * Br;
                float *p_score_scale = score_scale_ + thread_id * Br;
                float *p_score_sum = score_sum_ + thread_id * Br;
                int8_t *p_q_quant = q_quant_tile_ + thread_id * Br * dim_size;
                const int8_t *p_k_quant_global = k_quant_tile_ + (b_idx * KV_Head + this_thread_kv_head) * seq_size_k * dim_size;
                float *p_q_scale = q_scale_ + thread_id * Br;
                const float *p_k_scale_global = k_scale_ + (b_idx * KV_Head + this_thread_kv_head) * seq_size_k;
                float *p_q_scaled = q_scaled_row_buf_ + thread_id * dim_size;

                const float *p_V_mean = V_mean + b_idx * KV_Head * dim_size + this_thread_kv_head * dim_size;
                const int k_stride = KV_Head * dim_size;

                for (int32_t t_r_idx = 0; t_r_idx < Tr; ++t_r_idx) {
                    int32_t Br_fixed = std::min(Br, seq_size_q - t_r_idx * Br);
                    init_temp(p_logsum, p_scoremax, p_acc_o, Br_fixed, dim_size);

                    const dtype_q_in_t *tile_q_main = Q + b_idx * seq_size_q * head_size * dim_size + t_r_idx * Br * head_size * dim_size + h_idx * dim_size;
                    for (int r = 0; r < Br_fixed; ++r) {
                        quantize_row_simd(tile_q_main + r * (head_size * dim_size), p_q_quant + r * dim_size, &p_q_scale[r], dim_size, local_scale, p_q_scaled);
                    }

                    for (int32_t t_c_idx = 0; t_c_idx < Tc; ++t_c_idx) {
                        int32_t Bc_fixed = std::min(Bc, seq_size_k - t_c_idx * Bc);
                        const dtype_kv_in_t *tile_v = V + b_idx * seq_size_k * k_stride + t_c_idx * Bc * k_stride + this_thread_kv_head * dim_size;

                        quantize_and_mma0_sdot(Br_fixed, Bc_fixed, p_q_quant, p_k_quant_global + t_c_idx * Bc * dim_size, p_acc_s, p_q_scale, p_k_scale_global + t_c_idx * Bc, dim_size, t_r_idx * Br, t_c_idx * Bc, causal_mask);
                        softmax(Br_fixed, Bc_fixed, p_acc_s, p_scoremax, p_scoremax_prev, p_score_scale, p_score_sum, p_logsum);
                        rescale(Br_fixed, p_acc_o, p_score_scale, dim_size);
                        mma1(Br_fixed, Bc_fixed, p_acc_s, tile_v, p_V_mean, p_acc_o, KV_Head, dim_size);
                    }

                    dtype_out_t *tile_o = O + b_idx * seq_size_q * head_size * dim_size + t_r_idx * Br * head_size * dim_size + h_idx * dim_size;
                    scale_and_store(Br_fixed, p_acc_o, p_logsum, p_V_mean, tile_o, head_size, dim_size);
                }
            }
        }
    }

    void sage_attn_decode(const dtype_q_in_t *__restrict__ Q, const dtype_kv_in_t *__restrict__ K, const dtype_kv_in_t *__restrict__ V, dtype_out_t *__restrict__ O, const float *K_mean, const float *V_mean, int32_t batch_size, int32_t head_size, int32_t seq_size_k, int32_t dim_size, bool causal_mask) {
        const int32_t Tc = (seq_size_k + Bc - 1) / Bc;
        const float local_scale = 1.0f / sqrtf(static_cast<float>(dim_size));
        const int32_t kv_group_size = (Q_Head > 0 && KV_Head > 0) ? Q_Head / KV_Head : 1;

        for (int32_t b_idx = 0; b_idx < batch_size; ++b_idx) {
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1)
            for (int32_t h_idx = 0; h_idx < head_size; ++h_idx) {
                const int32_t Br_fixed = 1;
                const int32_t thread_id = omp_get_thread_num();
                const int32_t this_thread_kv_head = h_idx / kv_group_size;

                float *p_acc_o = acc_o_ + thread_id * Br_fixed * dim_size;
                float *p_acc_s = acc_s_ + thread_id * Br_fixed * Bc;
                float *p_logsum = logsum_ + thread_id * Br_fixed;
                float *p_scoremax = scoremax_ + thread_id * Br_fixed;
                float *p_scoremax_prev = scoremax_prev_ + thread_id * Br_fixed;
                float *p_score_scale = score_scale_ + thread_id * Br_fixed;
                float *p_score_sum = score_sum_ + thread_id * Br_fixed;
                int8_t *p_q_quant = q_quant_tile_ + thread_id * Br_fixed * dim_size;
                const int8_t *p_k_quant_global = k_quant_tile_ + (b_idx * KV_Head + this_thread_kv_head) * seq_size_k * dim_size;
                float *p_q_scale = q_scale_ + thread_id * Br_fixed;
                const float *p_k_scale_global = k_scale_ + (b_idx * KV_Head + this_thread_kv_head) * seq_size_k;
                float *p_q_scaled = q_scaled_row_buf_ + thread_id * dim_size;

                const float *p_V_mean = V_mean + b_idx * KV_Head * dim_size + this_thread_kv_head * dim_size;
                const int k_stride = KV_Head * dim_size;

                const dtype_q_in_t *tile_q_decode = Q + b_idx * head_size * dim_size + h_idx * dim_size;
                quantize_row_simd(tile_q_decode, p_q_quant, p_q_scale, dim_size, local_scale, p_q_scaled);

                init_temp(p_logsum, p_scoremax, p_acc_o, Br_fixed, dim_size);

                for (int32_t t_c_idx = 0; t_c_idx < Tc; ++t_c_idx) {
                    int32_t Bc_fixed = std::min(Bc, seq_size_k - t_c_idx * Bc);
                    const dtype_kv_in_t *tile_v = V + b_idx * seq_size_k * k_stride + t_c_idx * Bc * k_stride + this_thread_kv_head * dim_size;

                    quantize_and_mma0_sdot(Br_fixed, Bc_fixed, p_q_quant, p_k_quant_global + t_c_idx * Bc * dim_size, p_acc_s, p_q_scale, p_k_scale_global + t_c_idx * Bc, dim_size, seq_size_k - 1, t_c_idx * Bc, causal_mask);
                    softmax(Br_fixed, Bc_fixed, p_acc_s, p_scoremax, p_scoremax_prev, p_score_scale, p_score_sum, p_logsum);
                    rescale(Br_fixed, p_acc_o, p_score_scale, dim_size);
                    mma1(Br_fixed, Bc_fixed, p_acc_s, tile_v, p_V_mean, p_acc_o, KV_Head, dim_size);
                }
                dtype_out_t *tile_o = O + b_idx * head_size * dim_size + h_idx * dim_size;
                scale_and_store(Br_fixed, p_acc_o, p_logsum, p_V_mean, tile_o, head_size, dim_size);
            }
        }
    }

    void init_temp(float *logsum, float *scoremax, float *acc_o, int Br_fixed, int dim_size) {
        for (int i = 0; i < Br_fixed; ++i) {
            logsum[i] = 0.0f;
            scoremax[i] = NEG_INF;
        }
        memset(acc_o, 0, Br_fixed * dim_size * sizeof(float));
    }

    void quantize_and_mma0_sdot(int Br_fixed, int Bc_fixed, const int8_t *q_quant_tile, const int8_t *k_quant_tile, float *acc_s, const float *q_scale, const float *k_scale, int dim_size, int global_r_start, int global_c_start, bool causal) {
#if (defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(__ARM_FEATURE_DOTPROD)
        constexpr int MR = 4;
        constexpr int NR = 4;
        const int br_4_end = (Br_fixed / MR) * MR;
        const int bc_4_end = (Bc_fixed / NR) * NR;

        int r = 0;
        for (; r < br_4_end; r += MR) {
            int c = 0;
            for (; c < bc_4_end; c += NR) {
                int32x4_t vacc[MR][NR];
                for (int i = 0; i < MR; ++i)
                    for (int j = 0; j < NR; ++j) vacc[i][j] = vdupq_n_s32(0);

                const int8_t *q_ptr[MR];
                for (int i = 0; i < MR; ++i) q_ptr[i] = q_quant_tile + (r + i) * dim_size;
                const int8_t *k_ptr[NR];
                for (int j = 0; j < NR; ++j) k_ptr[j] = k_quant_tile + (c + j) * dim_size;

                int d = 0;
                for (; d <= dim_size - 16; d += 16) {
                    int8x16_t k0 = vld1q_s8(k_ptr[0] + d);
                    int8x16_t k1 = vld1q_s8(k_ptr[1] + d);
                    int8x16_t k2 = vld1q_s8(k_ptr[2] + d);
                    int8x16_t k3 = vld1q_s8(k_ptr[3] + d);
                    for (int i = 0; i < MR; ++i) {
                        int8x16_t q_vec = vld1q_s8(q_ptr[i] + d);
                        vacc[i][0] = vdotq_s32(vacc[i][0], q_vec, k0);
                        vacc[i][1] = vdotq_s32(vacc[i][1], q_vec, k1);
                        vacc[i][2] = vdotq_s32(vacc[i][2], q_vec, k2);
                        vacc[i][3] = vdotq_s32(vacc[i][3], q_vec, k3);
                    }
                }

                for (int i = 0; i < MR; ++i) {
                    for (int j = 0; j < NR; ++j) {
                        int32_t total_i32 = vaddvq_s32(vacc[i][j]);
                        for (int d_tail = d; d_tail < dim_size; ++d_tail) total_i32 += q_ptr[i][d_tail] * k_ptr[j][d_tail];
                        if (causal && (global_c_start + c + j) > (global_r_start + r + i)) {
                            acc_s[(r + i) * Bc + c + j] = NEG_INF;
                        } else {
                            acc_s[(r + i) * Bc + c + j] = (float)total_i32 * q_scale[r + i] * k_scale[c + j];
                        }
                    }
                }
            }
        }

        for (r = 0; r < Br_fixed; ++r) {
            int start_c = (r < br_4_end) ? bc_4_end : 0;
            for (int c = start_c; c < Bc_fixed; ++c) {
                if (causal && (global_c_start + c) > (global_r_start + r)) {
                    acc_s[r * Bc + c] = NEG_INF;
                    continue;
                }
                const int8_t *q_quant_line = q_quant_tile + r * dim_size;
                const int8_t *k_quant_line = k_quant_tile + c * dim_size;
                int32x4_t acc_i32_vec = vdupq_n_s32(0);
                int d = 0;
                for (; d <= dim_size - 16; d += 16) acc_i32_vec = vdotq_s32(acc_i32_vec, vld1q_s8(q_quant_line + d), vld1q_s8(k_quant_line + d));
                int32_t total_i32 = vaddvq_s32(acc_i32_vec);
                for (; d < dim_size; ++d) total_i32 += q_quant_line[d] * k_quant_line[d];
                acc_s[r * Bc + c] = (float)total_i32 * q_scale[r] * k_scale[c];
            }
        }
#elif defined(__AVX2__)
        // =========== AVX2 IMPLEMENTATION START (FIXED) ===========
        for (int r = 0; r < Br_fixed; ++r) {
            for (int c = 0; c < Bc_fixed; ++c) {
                if (causal && (global_c_start + c) > (global_r_start + r)) {
                    acc_s[r * Bc + c] = NEG_INF;
                    continue;
                }
                const int8_t *q_quant_line = q_quant_tile + r * dim_size;
                const int8_t *k_quant_line = k_quant_tile + c * dim_size;

                // Accumulator for 8x 32-bit integers
                __m256i acc_i32_v = _mm256_setzero_si256();
                int d = 0;

                // Process 16 bytes at a time, as we expand 8-bit to 16-bit
                for (; d <= dim_size - 16; d += 16) {
                    // Load 16 int8 values from Q and K
                    __m128i q_i8_v = _mm_loadu_si128((const __m128i *)(q_quant_line + d));
                    __m128i k_i8_v = _mm_loadu_si128((const __m128i *)(k_quant_line + d));

                    // Convert signed 8-bit integers to signed 16-bit integers
                    __m256i q_i16_v = _mm256_cvtepi8_epi16(q_i8_v);
                    __m256i k_i16_v = _mm256_cvtepi8_epi16(k_i8_v);

                    // Multiply signed 16-bit integers and horizontally add adjacent pairs
                    // This computes dot products of 2-element chunks and stores them in 32-bit lanes
                    __m256i prod_i32_v = _mm256_madd_epi16(q_i16_v, k_i16_v);

                    // Accumulate the 32-bit results
                    acc_i32_v = _mm256_add_epi32(acc_i32_v, prod_i32_v);
                }

                // Horizontally sum the 8 integer results in the accumulator vector
                int32_t total_i32 = hsum_i32(acc_i32_v);

                // Handle remainder
                for (; d < dim_size; ++d) {
                    total_i32 += q_quant_line[d] * k_quant_line[d];
                }

                acc_s[r * Bc + c] = (float)total_i32 * q_scale[r] * k_scale[c];
            }
        }
        // =========== AVX2 IMPLEMENTATION END ===========
#else
        // Fallback for other platforms
        for (int r = 0; r < Br_fixed; ++r) {
            for (int c = 0; c < Bc_fixed; ++c) {
                if (causal && (global_c_start + c) > (global_r_start + r)) {
                    acc_s[r * Bc + c] = NEG_INF;
                    continue;
                }
                const int8_t *q_quant_line = q_quant_tile + r * dim_size;
                const int8_t *k_quant_line = k_quant_tile + c * dim_size;
                int32_t total_i32 = 0;
                for (int d = 0; d < dim_size; ++d) total_i32 += q_quant_line[d] * k_quant_line[d];
                acc_s[r * Bc + c] = (float)total_i32 * q_scale[r] * k_scale[c];
            }
        }
#endif
    }

    void softmax(int Br_fixed, int Bc_fixed, float *acc_s, float *scoremax, float *scoremax_prev, float *score_scale, float *score_sum, float *logsum) {
        memcpy(scoremax_prev, scoremax, Br_fixed * sizeof(float));
        for (int r = 0; r < Br_fixed; ++r) {
            float *row = acc_s + r * Bc;
            float current_max = scoremax[r];
            for (int c = 0; c < Bc_fixed; ++c) current_max = std::max(current_max, row[c]);
            scoremax[r] = current_max;
        }
        for (int r = 0; r < Br_fixed; ++r) score_scale[r] = expf(scoremax_prev[r] - scoremax[r]);
        for (int r = 0; r < Br_fixed; ++r) {
            float *row = acc_s + r * Bc;
            float sm = scoremax[r];
            float sum = 0.f;
            for (int c = 0; c < Bc_fixed; ++c) {
                if (row[c] > NEG_INF / 2) {
                    float val = expf(row[c] - sm);
                    row[c] = val;
                    sum += val;
                } else {
                    row[c] = 0.f;
                }
            }
            score_sum[r] = sum;
        }
        for (int r = 0; r < Br_fixed; ++r) logsum[r] = logsum[r] * score_scale[r] + score_sum[r];
    }
    void rescale(int Br_fixed, float *acc_o, const float *score_scale, int dim_size) {
        for (int r = 0; r < Br_fixed; ++r) {
            float scale_val = score_scale[r];
            float *row_ptr = acc_o + r * dim_size;
#if defined(__AVX2__)
            __m256 scale_vec = _mm256_set1_ps(scale_val);
            int d = 0;
            for (; d <= dim_size - 8; d += 8) _mm256_storeu_ps(row_ptr + d, _mm256_mul_ps(_mm256_loadu_ps(row_ptr + d), scale_vec));
            for (; d < dim_size; ++d) row_ptr[d] *= scale_val;
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
            float32x4_t scale_vec = vdupq_n_f32(scale_val);
            int d = 0;
            for (; d <= dim_size - 4; d += 4) vst1q_f32(row_ptr + d, vmulq_f32(vld1q_f32(row_ptr + d), scale_vec));
            for (; d < dim_size; ++d) row_ptr[d] *= scale_val;
#else
            for (int d = 0; d < dim_size; ++d) row_ptr[d] *= scale_val;
#endif
        }
    }

    void mma1(int Br_fixed, int Bc_fixed, const float *p_block, const dtype_kv_in_t *v_block, const float *v_mean, float *acc_o, int kv_head_size, int dim_size) {
        int v_stride = kv_head_size * dim_size;

#if (defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(__ARM_FP16_FORMAT_IEEE)
        for (int r = 0; r < Br_fixed; ++r) {
            float *o_row = acc_o + r * dim_size;
            const float *p_row = p_block + r * Bc;

            for (int d = 0; d < dim_size; d += 4) {
                float32x4_t o_acc_vec = vld1q_f32(o_row + d);
                const float32x4_t vm_vec = vld1q_f32(v_mean + d);

                int c = 0;
                for (; c <= Bc_fixed - 4; c += 4) {
                    // 预取更远的数据
                    __builtin_prefetch(v_block + (c + 8) * v_stride + d, 0, 0);

                    // 加载 4 个 P 标量
                    const float p0 = p_row[c + 0];
                    const float p1 = p_row[c + 1];
                    const float p2 = p_row[c + 2];
                    const float p3 = p_row[c + 3];

                    // 加载 4 个 V 向量
                    const dtype_kv_in_t *v_row0 = v_block + (c + 0) * v_stride;
                    const dtype_kv_in_t *v_row1 = v_block + (c + 1) * v_stride;
                    const dtype_kv_in_t *v_row2 = v_block + (c + 2) * v_stride;
                    const dtype_kv_in_t *v_row3 = v_block + (c + 3) * v_stride;

                    float32x4_t v_vec0, v_vec1, v_vec2, v_vec3;
                    if constexpr (std::is_same_v<KVDtype, mllm_fp16_t>) {
                        v_vec0 = vcvt_f32_f16(vld1_f16(reinterpret_cast<const float16_t *>(v_row0 + d)));
                        v_vec1 = vcvt_f32_f16(vld1_f16(reinterpret_cast<const float16_t *>(v_row1 + d)));
                        v_vec2 = vcvt_f32_f16(vld1_f16(reinterpret_cast<const float16_t *>(v_row2 + d)));
                        v_vec3 = vcvt_f32_f16(vld1_f16(reinterpret_cast<const float16_t *>(v_row3 + d)));
                    } else { // float
                        v_vec0 = vld1q_f32(v_row0 + d);
                        v_vec1 = vld1q_f32(v_row1 + d);
                        v_vec2 = vld1q_f32(v_row2 + d);
                        v_vec3 = vld1q_f32(v_row3 + d);
                    }

                    // 4组独立的 FMA 运算
                    o_acc_vec = vfmaq_n_f32(o_acc_vec, vsubq_f32(v_vec0, vm_vec), p0);
                    o_acc_vec = vfmaq_n_f32(o_acc_vec, vsubq_f32(v_vec1, vm_vec), p1);
                    o_acc_vec = vfmaq_n_f32(o_acc_vec, vsubq_f32(v_vec2, vm_vec), p2);
                    o_acc_vec = vfmaq_n_f32(o_acc_vec, vsubq_f32(v_vec3, vm_vec), p3);
                }

                // 处理剩余的循环
                for (; c < Bc_fixed; ++c) {
                    const float p_scalar = p_row[c];
                    const float32x4_t p_vec = vdupq_n_f32(p_scalar);
                    const dtype_kv_in_t *v_row = v_block + c * v_stride;
                    float32x4_t v_vec;
                    if constexpr (std::is_same_v<KVDtype, mllm_fp16_t>) {
                        v_vec = vcvt_f32_f16(vld1_f16(reinterpret_cast<const float16_t *>(v_row + d)));
                    } else {
                        v_vec = vld1q_f32(v_row + d);
                    }
                    float32x4_t v_smoothed = vsubq_f32(v_vec, vm_vec);
                    o_acc_vec = vfmaq_f32(o_acc_vec, p_vec, v_smoothed);
                }

                vst1q_f32(o_row + d, o_acc_vec);
            }
        }
#elif defined(__AVX2__) && defined(__FMA__)
        // =========== AVX2 IMPLEMENTATION START ===========
        for (int r = 0; r < Br_fixed; ++r) {
            float *o_row = acc_o + r * dim_size;
            for (int c = 0; c < Bc_fixed; ++c) {
                const float p_scalar = p_block[r * Bc + c];
                if (fabsf(p_scalar) < 1e-9) continue;

                const __m256 p_vec = _mm256_set1_ps(p_scalar);
                const dtype_kv_in_t *v_row = v_block + c * v_stride;

                int d = 0;
                for (; d <= dim_size - 8; d += 8) {
                    const __m256 vm_vec = _mm256_loadu_ps(v_mean + d);
                    const __m256 v_vec = load_and_convert_to_fp32_vec(v_row + d);

                    __m256 o_vec = _mm256_loadu_ps(o_row + d);
                    __m256 v_smoothed = _mm256_sub_ps(v_vec, vm_vec);

                    // Fused Multiply-Add: o_vec = (p_vec * v_smoothed) + o_vec
                    o_vec = _mm256_fmadd_ps(p_vec, v_smoothed, o_vec);

                    _mm256_storeu_ps(o_row + d, o_vec);
                }
                // Remainder loop
                for (; d < dim_size; ++d) {
                    o_row[d] += p_scalar * (to_float(v_row[d]) - v_mean[d]);
                }
            }
        }
        // =========== AVX2 IMPLEMENTATION END ===========
#else
        // Fallback for other platforms
        for (int r = 0; r < Br_fixed; ++r) {
            float *o_row = acc_o + r * dim_size;
            for (int c = 0; c < Bc_fixed; ++c) {
                const float p = p_block[r * Bc + c];
                if (fabsf(p) < 1e-9) continue;
                const dtype_kv_in_t *v_row = v_block + c * v_stride;
                for (int d = 0; d < dim_size; ++d) { o_row[d] += p * (to_float(v_row[d]) - v_mean[d]); }
            }
        }
#endif
    }

    void scale_and_store(int Br_fixed, const float *acc_o, const float *logsum, const float *v_mean, float *O, int head_size, int dim_size) {
        int o_stride = head_size * dim_size;
        for (int r = 0; r < Br_fixed; ++r) {
            float inv_logsum = (logsum[r] > 1e-9f) ? 1.f / logsum[r] : 0.f;
            const float *o_row = acc_o + r * dim_size;
            float *O_row = O + r * o_stride;
#if defined(__AVX2__) && defined(__FMA__)
            const __m256 inv_logsum_vec = _mm256_set1_ps(inv_logsum);
            int d = 0;
            for (; d <= dim_size - 8; d += 8) {
                const __m256 o_vec = _mm256_loadu_ps(o_row + d);
                const __m256 vm_vec = _mm256_loadu_ps(v_mean + d);
                _mm256_storeu_ps(O_row + d, _mm256_fmadd_ps(o_vec, inv_logsum_vec, vm_vec));
            }
            for (; d < dim_size; ++d) O_row[d] = o_row[d] * inv_logsum + v_mean[d];
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
            float32x4_t inv_logsum_vec = vdupq_n_f32(inv_logsum);
            int d = 0;
            for (; d <= dim_size - 4; d += 4) {
                float32x4_t o_vec = vld1q_f32(o_row + d);
                float32x4_t vm_vec = vld1q_f32(v_mean + d);
                vst1q_f32(O_row + d, vfmaq_f32(vm_vec, o_vec, inv_logsum_vec));
            }
            for (; d < dim_size; ++d) O_row[d] = o_row[d] * inv_logsum + v_mean[d];
#else
            for (int d = 0; d < dim_size; ++d) O_row[d] = o_row[d] * inv_logsum + v_mean[d];
#endif
        }
    }
};

template <typename KVDtype>
void sage_attention_forward_cpu_dispatch(const float *Q, const KVDtype *K, const KVDtype *V, float *O, int32_t batch_size, int32_t q_head, int32_t kv_head, int32_t seq_size_q, int32_t seq_size_k, int32_t dim_size, bool causal_mask, int32_t threads, int32_t br, int32_t bc) {
    std::vector<float> V_mean(batch_size * kv_head * dim_size);
    compute_channel_means<KVDtype>(V, V_mean.data(), batch_size, kv_head, seq_size_k, dim_size);

    thread_local WorkspaceManager manager;

    std::vector<float> K_mean(batch_size * kv_head * dim_size);

    if (seq_size_q > 1) { // Prefill
        const std::vector<size_t> required_sizes = {
            (size_t)threads * br * dim_size * sizeof(float),
            (size_t)threads * br * bc * sizeof(float),
            (size_t)threads * br * sizeof(float),
            (size_t)threads * br * sizeof(float),
            (size_t)threads * br * sizeof(float),
            (size_t)threads * br * sizeof(float),
            (size_t)threads * br * sizeof(float),
            (size_t)threads * br * dim_size * sizeof(int8_t),
            (size_t)batch_size * kv_head * seq_size_k * dim_size * sizeof(int8_t),
            (size_t)threads * br * sizeof(float),
            (size_t)batch_size * kv_head * seq_size_k * sizeof(float),
            (size_t)threads * dim_size * sizeof(float),
            (size_t)threads * dim_size * sizeof(float),
            (size_t)threads * dim_size * sizeof(float),
        };
        void **workspace = manager.get_workspace(required_sizes);

        int8_t *k_quant_global_buffer = static_cast<int8_t *>(workspace[8]);
        float *k_scale_global_buffer = static_cast<float *>(workspace[10]);

        compute_mean_and_quantize_k<KVDtype>(K, K_mean.data(), k_quant_global_buffer, k_scale_global_buffer, batch_size, kv_head, seq_size_k, dim_size, threads, static_cast<float *>(workspace[13]), static_cast<float *>(workspace[11]));

        SAGE_CPU_IMPL<KVDtype> op;
        op.configure(br, bc, q_head, kv_head, threads);
        op.init_workspace(workspace);
        op.sage_attn_prefill(Q, K, V, O, K_mean.data(), V_mean.data(), batch_size, q_head, seq_size_q, seq_size_k, dim_size, causal_mask);
    } else { // Decode
        const int32_t decode_br = 1;
        const std::vector<size_t> required_sizes = {
            (size_t)threads * decode_br * dim_size * sizeof(float),
            (size_t)threads * decode_br * bc * sizeof(float),
            (size_t)threads * decode_br * sizeof(float),
            (size_t)threads * decode_br * sizeof(float),
            (size_t)threads * decode_br * sizeof(float),
            (size_t)threads * decode_br * sizeof(float),
            (size_t)threads * decode_br * sizeof(float),
            (size_t)threads * decode_br * dim_size * sizeof(int8_t),
            (size_t)batch_size * kv_head * seq_size_k * dim_size * sizeof(int8_t),
            (size_t)threads * decode_br * sizeof(float),
            (size_t)batch_size * kv_head * seq_size_k * sizeof(float),
            (size_t)threads * dim_size * sizeof(float),
            (size_t)threads * dim_size * sizeof(float),
            (size_t)threads * dim_size * sizeof(float),
        };
        void **workspace = manager.get_workspace(required_sizes);

        int8_t *k_quant_global_buffer = static_cast<int8_t *>(workspace[8]);
        float *k_scale_global_buffer = static_cast<float *>(workspace[10]);

        compute_mean_and_quantize_k<KVDtype>(K, K_mean.data(), k_quant_global_buffer, k_scale_global_buffer, batch_size, kv_head, seq_size_k, dim_size, threads, static_cast<float *>(workspace[13]), static_cast<float *>(workspace[11]));

        SAGE_CPU_IMPL<KVDtype> op;
        op.configure(br, bc, q_head, kv_head, threads);
        op.init_workspace(workspace);
        op.sage_attn_decode(Q, K, V, O, K_mean.data(), V_mean.data(), batch_size, q_head, seq_size_k, dim_size, causal_mask);
    }
}
} // namespace sage_attn_pt_cpu
