
#if defined(__aarch64__) || defined(__arm__) || defined(__arm64__)

#include "GemmKleidiai.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cfloat>

// 引入 OpenMP 头文件
#include <omp.h>
//#if defined(USE_QSI4_C32)
// ###################################################################### //
// ##                   Implementation 1: QSI4 (INT4)                  ##
// ###################################################################### //
#include "kai_matmul_clamp_f32_qai8dxp_qsi4c32p_interface.h"
#include "kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0.h"

static const kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel qsi4_ukernel = {
    .get_m_step = kai_get_m_step_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
    .get_n_step = kai_get_n_step_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
    .get_mr = kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
    .get_nr = kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
    .get_kr = kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
    .get_sr = kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
    .get_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
    .get_rhs_packed_offset = kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
    .get_dst_offset = kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
    .run_matmul = kai_run_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
};

size_t mllm_kleidai_get_packed_b_qsi4_size(int N, int K) {
    const int block_len = 32;
    return kai_get_rhs_packed_size_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0(
        N, K, qsi4_ukernel.get_nr(), qsi4_ukernel.get_kr(), qsi4_ukernel.get_sr(),
        block_len, kai_dt_bf16);
}

size_t get_workspace_qsi4_size(int M, int K) {
    return kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(
        M, K, qsi4_ukernel.get_mr(), qsi4_ukernel.get_kr(), qsi4_ukernel.get_sr());
}

void mllm_kleidai_pack_b_and_bias_qsi4(
    uint8_t* packed_b_ptr,
    const float* b_ptr,
    const float* bias_ptr,
    int N,
    int K) {

    // 【新增】创建临时的bias（如果需要）
    const float* bias_to_use = bias_ptr;
    std::vector<float> fake_bias;
    if (bias_to_use == nullptr) {
        fake_bias.assign(N, 0.0f);
        bias_to_use = fake_bias.data();
    }


    const int block_len = 32;
    const size_t num_blocks_k = (K + block_len - 1) / block_len;
    std::vector<uint8_t> temp_quantized_b(K * N / 2);
    std::vector<uint16_t> temp_scales(N * num_blocks_k);

    for (int n = 0; n < N; ++n) {
        for (size_t kb = 0; kb < num_blocks_k; ++kb) {
            float amax = 0.0f;
            float max_with_sign = 0.0f;
            int start_k = kb * block_len;
            int end_k = std::min(start_k + block_len, K);
            for (int k = start_k; k < end_k; ++k) {
                const float val = b_ptr[k * N + n];
                const float abs_val = std::abs(val);
                if (abs_val > amax) {
                    amax = abs_val;
                    max_with_sign = val;
                }
            }
            const float scale = max_with_sign / -8.0f;
            const float inv_scale = scale != 0.0f ? 1.0f / scale : 0.0f;
            temp_scales[n * num_blocks_k + kb] = kai_cast_bf16_f32(scale);
            for (int k = start_k; k < end_k; ++k) {
                const float val = b_ptr[k * N + n];
                int32_t q_val = static_cast<int32_t>(round(val * inv_scale));
                q_val = std::max(-8, std::min(7, q_val));
                uint8_t stored_val = q_val + 8;
                size_t byte_idx = (k * N + n) / 2;
                if ((k * N + n) % 2 == 0) {
                    temp_quantized_b[byte_idx] = stored_val;
                } else {
                    temp_quantized_b[byte_idx] |= (stored_val << 4);
                }
            }
        }
    }
    kai_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0_params params = {};
    params.lhs_zero_point = 1;
    params.rhs_zero_point = 8;
    params.scale_dt = kai_dt_bf16;
    // kai_run_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0(
    //     1, N, K, qsi4_ukernel.get_nr(), qsi4_ukernel.get_kr(), qsi4_ukernel.get_sr(), block_len,
    //     temp_quantized_b.data(), K/2, bias_to_use, (const uint8_t*)temp_scales.data(), num_blocks_k * sizeof(uint16_t),
    //     packed_b_ptr, 0, &params);
    kai_run_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0(
        1, N, K, qsi4_ukernel.get_nr(), qsi4_ukernel.get_kr(), qsi4_ukernel.get_sr(), block_len,
        temp_quantized_b.data(), N/2, bias_to_use, (const uint8_t*)temp_scales.data(), num_blocks_k * sizeof(uint16_t),
        packed_b_ptr, 0, &params);
}

void mllm_kleidai_gemm_qsi4(
    float* c_ptr, const float* a_ptr, const uint8_t* packed_b_ptr,
    int M, int N, int K) {
    
    size_t workspace_size = get_workspace_qsi4_size(M, K);
    std::vector<uint8_t> workspace_data(workspace_size);

    kai_run_lhs_quant_pack_qai8dxp_f32(
        M, K,
        qsi4_ukernel.get_mr(), qsi4_ukernel.get_kr(), qsi4_ukernel.get_sr(),
        0,
        a_ptr, K * sizeof(float),
        workspace_data.data());


    const int m_step = qsi4_ukernel.get_m_step();
    const int n_step = qsi4_ukernel.get_n_step();
    const int block_len = 32;

    // #pragma omp parallel for
    #pragma omp parallel for collapse(2) num_threads(kai_thread_count)
    for (int m_start = 0; m_start < M; m_start += m_step) {
        for (int n_start = 0; n_start < N; n_start += n_step) {
            const int current_m = std::min(M - m_start, m_step);
            const int current_n = std::min(N - n_start, n_step);
            const void* a_packed_offset = (const char*)workspace_data.data() + qsi4_ukernel.get_lhs_packed_offset(m_start, K);
            const void* b_packed_offset = (const char*)packed_b_ptr + qsi4_ukernel.get_rhs_packed_offset(n_start, K, block_len);
            float* c_offset = c_ptr + m_start * N + n_start;
            qsi4_ukernel.run_matmul(
                current_m, current_n, K, block_len,
                a_packed_offset, b_packed_offset,
                c_offset, N * sizeof(float), sizeof(float),
                -FLT_MAX, FLT_MAX
            );
        }
    }
}
// #endif

// 【新增】
//#if defined(USE_QSI4_TO_FP16)
// ###################################################################### //
// ##         Implementation 1.5: QSI4 (INT4) -> FP16                  ##
// ###################################################################### //
void mllm_kleidai_gemm_qsi4_to_fp16(
    mllm_fp16_t* c_ptr, const float* a_ptr, const uint8_t* packed_b_ptr,
    int M, int N, int K) {
    
    // 1. 创建一个临时的 FP32 输出缓冲区
    std::vector<float> c_temp(M * N);

    // 2. 运行现有的 QSI4->FP32 GEMM 计算，将结果存入临时缓冲区
    size_t workspace_size = get_workspace_qsi4_size(M, K);
    std::vector<uint8_t> workspace_data(workspace_size);
    kai_run_lhs_quant_pack_qai8dxp_f32(M, K, qsi4_ukernel.get_mr(), qsi4_ukernel.get_kr(), qsi4_ukernel.get_sr(),
        0, a_ptr, K * sizeof(float), workspace_data.data());

    const int m_step = qsi4_ukernel.get_m_step();
    const int n_step = qsi4_ukernel.get_n_step();
    const int block_len = 32;

    // #pragma omp parallel for
    #pragma omp parallel for collapse(2) num_threads(kai_thread_count)
    for (int m_start = 0; m_start < M; m_start += m_step) {
        for (int n_start = 0; n_start < N; n_start += n_step) {
            const int current_m = std::min(M - m_start, m_step);
            const int current_n = std::min(N - n_start, n_step);
            const void* a_packed_offset = (const char*)workspace_data.data() + qsi4_ukernel.get_lhs_packed_offset(m_start, K);
            const void* b_packed_offset = (const char*)packed_b_ptr + qsi4_ukernel.get_rhs_packed_offset(n_start, K, block_len);
            float* c_offset = c_temp.data() + m_start * N + n_start;
            qsi4_ukernel.run_matmul(
                current_m, current_n, K, block_len,
                a_packed_offset, b_packed_offset,
                c_offset, N * sizeof(float), sizeof(float),
                -FLT_MAX, FLT_MAX
            );
        }
    }
    
    // 3. 并行地将 FP32 临时结果转换为 FP16 并写入最终输出
    // #pragma omp parallel for
    #pragma omp parallel for collapse(1) num_threads(kai_thread_count)
    for(int i = 0; i < M * N; ++i) {
        c_ptr[i] = static_cast<mllm_fp16_t>(c_temp[i]);
    }
}
// #endif


//#if defined(USE_FP16)
// ###################################################################### //
// ##                   Implementation 2: FP16                         ##
// ###################################################################### //
#include "kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p_interface.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon.h"

static const kai_matmul_clamp_f16_f16_f16p_ukernel fp16_ukernel = {
    .get_m_step = kai_get_m_step_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
    .get_n_step = kai_get_n_step_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
    .get_nr = kai_get_nr_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
    .get_kr = kai_get_kr_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
    .get_sr = kai_get_sr_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
    .run_matmul = kai_run_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
};

size_t mllm_kleidai_get_packed_b_fp16_size(int N, int K) {
    return kai_get_rhs_packed_size_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon(N, K);
}

void mllm_kleidai_pack_b_and_bias_fp16(mllm_fp16_t* packed_b_ptr, const mllm_fp16_t* b_ptr, const float* bias_ptr, int N, int K) {
    std::vector<mllm_fp16_t> bias_fp16_buffer(N);
    if (bias_ptr != nullptr) {
        for(int i = 0; i < N; ++i) {
            bias_fp16_buffer[i] = static_cast<mllm_fp16_t>(bias_ptr[i]);
        }
    } else {
        std::fill(bias_fp16_buffer.begin(), bias_fp16_buffer.end(), static_cast<mllm_fp16_t>(0.0f));
    }
    kai_run_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon(
        1, N, K, fp16_ukernel.get_nr(), fp16_ukernel.get_kr(), fp16_ukernel.get_sr(),
        N * sizeof(mllm_fp16_t), b_ptr, bias_fp16_buffer.data(), nullptr, packed_b_ptr, 0, nullptr);
}

void mllm_kleidai_gemm_fp16(float* c_ptr, const float* a_ptr, const mllm_fp16_t* packed_b_ptr, int M, int N, int K) {
    std::vector<mllm_fp16_t> a_fp16(M * K);
    for(int i = 0; i < M * K; ++i) {
        a_fp16[i] = static_cast<mllm_fp16_t>(a_ptr[i]);
    }

    std::vector<mllm_fp16_t> c_fp16(M * N);
    const int m_step = fp16_ukernel.get_m_step();
    const int n_step = fp16_ukernel.get_n_step();

    // #pragma omp parallel for
    #pragma omp parallel for collapse(2) num_threads(kai_thread_count)
    for (int m_start = 0; m_start < M; m_start += m_step) {
        for (int n_start = 0; n_start < N; n_start += n_step) {
            const int current_m = std::min(M - m_start, m_step);
            const int current_n = std::min(N - n_start, n_step);
            const mllm_fp16_t* a_offset = a_fp16.data() + m_start * K;
            const mllm_fp16_t* b_offset = packed_b_ptr + (n_start * (K + 1));
            mllm_fp16_t* c_offset = c_fp16.data() + m_start * N + n_start;
            fp16_ukernel.run_matmul(
                current_m, current_n, K, a_offset, K * sizeof(mllm_fp16_t),
                b_offset, c_offset, N * sizeof(mllm_fp16_t), sizeof(mllm_fp16_t),
                -FLT_MAX, FLT_MAX );
        }
    }
    for(int i = 0; i < M * N; ++i) {
        c_ptr[i] = static_cast<float>(c_fp16[i]);
    }
}
// #endif


//#if defined(USE_FP32)
// ###################################################################### //
// ##                   Implementation 3: FP32                         ##
// ###################################################################### //
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p_interface.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon.h"

static const kai_matmul_clamp_f32_f32_f32p_ukernel fp32_ukernel = {
    .get_m_step = kai_get_m_step_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
    .get_n_step = kai_get_n_step_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
    .get_nr = kai_get_nr_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
    .get_kr = kai_get_kr_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
    .get_sr = kai_get_sr_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
    .run_matmul = kai_run_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
};

size_t mllm_kleidai_get_packed_b_fp32_size(int N, int K) {
    return kai_get_rhs_packed_size_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(N, K);
}

void mllm_kleidai_pack_b_and_bias_fp32(float* packed_b_ptr, const float* b_ptr, const float* bias_ptr, int N, int K) {
    const float* bias_to_use = bias_ptr;
    std::vector<float> fake_bias;
    if (bias_to_use == nullptr) {
        fake_bias.assign(N, 0.0f);
        bias_to_use = fake_bias.data();
    }
    kai_run_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(
        1, N, K, fp32_ukernel.get_nr(), fp32_ukernel.get_kr(), fp32_ukernel.get_sr(),
        N * sizeof(float), b_ptr, bias_to_use, nullptr, packed_b_ptr, 0, nullptr);
}

void mllm_kleidai_gemm_fp32(float* c_ptr, const float* a_ptr, const float* packed_b_ptr, int M, int N, int K) {
    const int m_step = fp32_ukernel.get_m_step();
    const int n_step = fp32_ukernel.get_n_step();
    
    // #pragma omp parallel for
    #pragma omp parallel for collapse(2) num_threads(kai_thread_count)
    for (int m_start = 0; m_start < M; m_start += m_step) {
        for (int n_start = 0; n_start < N; n_start += n_step) {
            const int current_m = std::min(M - m_start, m_step);
            const int current_n = std::min(N - n_start, n_step);
            const float* a_offset = a_ptr + m_start * K;
            const float* b_offset = packed_b_ptr + (n_start * (K + 1));
            float* c_offset = c_ptr + m_start * N + n_start;
            fp32_ukernel.run_matmul(
                current_m, current_n, K,
                a_offset, K * sizeof(float), b_offset,
                c_offset, N * sizeof(float), sizeof(float),
                -FLT_MAX, FLT_MAX );
        }
    }
}
// #endif
#endif