
#if defined(__aarch64__) || defined(__arm__) || defined(__arm64__)

#include "GemmKleidiai.hpp"
#include "FeatureCheck.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cfloat>

#include <omp.h>

class WorkspaceManager {
public:
    static WorkspaceManager &get_instance() {
        static WorkspaceManager instance;
        return instance;
    }

private:
    WorkspaceManager() {
        int max_threads = kai_thread_count;
#ifdef _OPENMP
        max_threads = omp_get_max_threads();
#endif
        qsi4_workspaces_.resize(max_threads);
        qsi4_c_temp_buffers_.resize(max_threads);
        fp16_a_buffers_.resize(max_threads);
        fp16_c_buffers_.resize(max_threads);
    }

public:
    WorkspaceManager(const WorkspaceManager &) = delete;
    WorkspaceManager &operator=(const WorkspaceManager &) = delete;

    std::vector<uint8_t> &get_qsi4_workspace() {
        int thread_id = 0;
#ifdef _OPENMP
        thread_id = omp_get_thread_num();
#endif
        return qsi4_workspaces_[thread_id];
    }

    std::vector<float> &get_qsi4_c_temp_buffer() {
        int thread_id = 0;
#ifdef _OPENMP
        thread_id = omp_get_thread_num();
#endif
        return qsi4_c_temp_buffers_[thread_id];
    }

    std::vector<mllm_fp16_t> &get_fp16_a_buffer() {
        int thread_id = 0;
#ifdef _OPENMP
        thread_id = omp_get_thread_num();
#endif
        return fp16_a_buffers_[thread_id];
    }

    std::vector<mllm_fp16_t> &get_fp16_c_buffer() {
        int thread_id = 0;
#ifdef _OPENMP
        thread_id = omp_get_thread_num();
#endif
        return fp16_c_buffers_[thread_id];
    }

private:
    std::vector<std::vector<uint8_t>> qsi4_workspaces_;
    std::vector<std::vector<float>> qsi4_c_temp_buffers_;
    std::vector<std::vector<mllm_fp16_t>> fp16_a_buffers_;
    std::vector<std::vector<mllm_fp16_t>> fp16_c_buffers_;
};

#include <unordered_map>
#if defined(__linux__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif

#include "kai_matmul_clamp_f32_qai8dxp_qsi4c32p_interface.h"
#include "kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0.h"
#include "kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h"
#include "kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm.h"
#include "kai_matmul_clamp_f32_qai8dxp_qsi4c32p_interface.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0.h"

#include "kai/ukernels/matmul/matmul_clamp_f16_qai8dxp_qsi4cxp/kai_matmul_clamp_f16_qai8dxp_qsi4cxp_interface.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_qai8dxp_qsi4cxp/kai_matmul_clamp_f16_qai8dxp1x8_qsi4cxp4x8_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_qai8dxp_qsi4cxp/kai_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_16x4_neon_i8mm.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f16_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi4cxp_qs4cxs1s0.h"

enum class KleidiaiQsi4Tile {
    k1x8_4x8_1x4x32_dotprod,
    k4x8_4x8_8x4x32_i8mm,
};

namespace std {
template <>
struct hash<KleidiaiQsi4Tile> {
    std::size_t operator()(const KleidiaiQsi4Tile &k) const noexcept {
        return std::hash<std::underlying_type<KleidiaiQsi4Tile>::type>()(
            static_cast<std::underlying_type<KleidiaiQsi4Tile>::type>(k));
    }
};
} // namespace std

static const kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel dotprod_ukernel = {
    .get_m_step = kai_get_m_step_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
    .get_n_step = kai_get_n_step_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
    .get_mr = kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
    .get_nr = kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
    .get_kr = kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
    .get_sr = kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
    .get_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
    .get_rhs_packed_offset = kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
    .get_dst_offset = kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
    .run_matmul = kai_run_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod};

static const kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel i8mm_ukernel = {
    .get_m_step = kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
    .get_n_step = kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
    .get_mr = kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
    .get_nr = kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
    .get_kr = kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
    .get_sr = kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
    .get_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
    .get_rhs_packed_offset = kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
    .get_dst_offset = kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
    .run_matmul = kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm};

static const std::unordered_map<KleidiaiQsi4Tile, kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel> qsi4_ukernels = {
    {KleidiaiQsi4Tile::k1x8_4x8_1x4x32_dotprod, dotprod_ukernel},
    {KleidiaiQsi4Tile::k4x8_4x8_8x4x32_i8mm, i8mm_ukernel}};

static KleidiaiQsi4Tile kleidiai_get_best_qsi4_tile_config() {
    static const KleidiaiQsi4Tile best_tile = arm_is_i8mm_supported() ?
                                                  KleidiaiQsi4Tile::k4x8_4x8_8x4x32_i8mm :
                                                  KleidiaiQsi4Tile::k1x8_4x8_1x4x32_dotprod;
    return best_tile;
}

enum class KleidiaiQsi4TileF16 {
    k1x8_4x8_1x4_dotprod,
    k4x8_4x8_16x4_i8mm,
};

namespace std {
template <>
struct hash<KleidiaiQsi4TileF16> {
    std::size_t operator()(const KleidiaiQsi4TileF16 &k) const noexcept {
        return std::hash<std::underlying_type<KleidiaiQsi4TileF16>::type>()(
            static_cast<std::underlying_type<KleidiaiQsi4TileF16>::type>(k));
    }
};
} // namespace std

static const kai_matmul_clamp_f16_qai8dxp_qsi4cxp_ukernel f16_dotprod_ukernel = {
    .get_m_step = kai_get_m_step_matmul_clamp_f16_qai8dxp1x8_qsi4cxp4x8_1x4_neon_dotprod,
    .get_n_step = kai_get_n_step_matmul_clamp_f16_qai8dxp1x8_qsi4cxp4x8_1x4_neon_dotprod,
    .get_mr = kai_get_mr_matmul_clamp_f16_qai8dxp1x8_qsi4cxp4x8_1x4_neon_dotprod,
    .get_nr = kai_get_nr_matmul_clamp_f16_qai8dxp1x8_qsi4cxp4x8_1x4_neon_dotprod,
    .get_kr = kai_get_kr_matmul_clamp_f16_qai8dxp1x8_qsi4cxp4x8_1x4_neon_dotprod,
    .get_sr = kai_get_sr_matmul_clamp_f16_qai8dxp1x8_qsi4cxp4x8_1x4_neon_dotprod,
    .get_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_f16_qai8dxp1x8_qsi4cxp4x8_1x4_neon_dotprod,
    .get_rhs_packed_offset = kai_get_rhs_packed_offset_matmul_clamp_f16_qai8dxp1x8_qsi4cxp4x8_1x4_neon_dotprod,
    .get_dst_offset = kai_get_dst_offset_matmul_clamp_f16_qai8dxp1x8_qsi4cxp4x8_1x4_neon_dotprod,
    .run_matmul = kai_run_matmul_clamp_f16_qai8dxp1x8_qsi4cxp4x8_1x4_neon_dotprod};

static const kai_matmul_clamp_f16_qai8dxp_qsi4cxp_ukernel f16_i8mm_ukernel = {
    .get_m_step = kai_get_m_step_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_16x4_neon_i8mm,
    .get_n_step = kai_get_n_step_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_16x4_neon_i8mm,
    .get_mr = kai_get_mr_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_16x4_neon_i8mm,
    .get_nr = kai_get_nr_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_16x4_neon_i8mm,
    .get_kr = kai_get_kr_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_16x4_neon_i8mm,
    .get_sr = kai_get_sr_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_16x4_neon_i8mm,
    .get_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_16x4_neon_i8mm,
    .get_rhs_packed_offset = kai_get_rhs_packed_offset_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_16x4_neon_i8mm,
    .get_dst_offset = kai_get_dst_offset_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_16x4_neon_i8mm,
    .run_matmul = kai_run_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_16x4_neon_i8mm};

static const std::unordered_map<KleidiaiQsi4TileF16, kai_matmul_clamp_f16_qai8dxp_qsi4cxp_ukernel> qsi4_f16_ukernels = {
    {KleidiaiQsi4TileF16::k1x8_4x8_1x4_dotprod, f16_dotprod_ukernel},
    {KleidiaiQsi4TileF16::k4x8_4x8_16x4_i8mm, f16_i8mm_ukernel}};

static KleidiaiQsi4TileF16 kleidiai_get_best_qsi4_tile_config_f16() {
    static const KleidiaiQsi4TileF16 best_tile = arm_is_i8mm_supported() ?
                                                     KleidiaiQsi4TileF16::k4x8_4x8_16x4_i8mm :
                                                     KleidiaiQsi4TileF16::k1x8_4x8_1x4_dotprod;
    return best_tile;
}

size_t mllm_kleidai_get_packed_b_qsi4_size(int N, int K) {
    const auto tile_cfg = kleidiai_get_best_qsi4_tile_config();
    const auto &ukernel = qsi4_ukernels.at(tile_cfg);
    const int block_len = 32;
    return kai_get_rhs_packed_size_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0(
        N, K, ukernel.get_nr(), ukernel.get_kr(), ukernel.get_sr(),
        block_len, kai_dt_bf16);
}

size_t get_workspace_qsi4_size(int M, int K) {
    const auto tile_cfg = kleidiai_get_best_qsi4_tile_config();
    const auto &ukernel = qsi4_ukernels.at(tile_cfg);
    return kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(
        M, K, ukernel.get_mr(), ukernel.get_kr(), ukernel.get_sr());
}

void mllm_kleidai_pack_b_and_bias_qsi4(
    uint8_t *packed_b_ptr,
    const float *b_ptr,
    const float *bias_ptr,
    int N,
    int K) {
    const auto tile_cfg = kleidiai_get_best_qsi4_tile_config();
    const auto &ukernel = qsi4_ukernels.at(tile_cfg);

    const float *bias_to_use = bias_ptr;
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
    kai_run_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0(
        1, N, K, ukernel.get_nr(), ukernel.get_kr(), ukernel.get_sr(), block_len,
        temp_quantized_b.data(), N / 2, bias_to_use, (const uint8_t *)temp_scales.data(), num_blocks_k * sizeof(uint16_t),
        packed_b_ptr, 0, &params);
}

#ifndef KAI_FP16_CAL
void mllm_kleidai_gemm_qsi4(
    float *c_ptr, const float *a_ptr, const uint8_t *packed_b_ptr,
    int M, int N, int K) {
    kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel ukernel;
    if (M == 1) {
        ukernel = qsi4_ukernels.at(KleidiaiQsi4Tile::k1x8_4x8_1x4x32_dotprod);
    } else {
        const auto tile_cfg = arm_is_i8mm_supported() ?
                                  KleidiaiQsi4Tile::k4x8_4x8_8x4x32_i8mm :
                                  KleidiaiQsi4Tile::k1x8_4x8_1x4x32_dotprod;
        ukernel = qsi4_ukernels.at(tile_cfg);
    }

    auto &workspace_data = WorkspaceManager::get_instance().get_qsi4_workspace();
    size_t required_workspace_size = get_workspace_qsi4_size(M, K);
    if (workspace_data.size() < required_workspace_size) {
        workspace_data.resize(required_workspace_size);
    }

    kai_run_lhs_quant_pack_qai8dxp_f32(
        M, K,
        ukernel.get_mr(), ukernel.get_kr(), ukernel.get_sr(),
        0,
        a_ptr, K * sizeof(float),
        workspace_data.data());

    const int n_step = ukernel.get_n_step();
    const int block_len = 32;

#pragma omp parallel for num_threads(kai_thread_count)
    for (int n_start = 0; n_start < N; n_start += n_step) {
        const int current_n = std::min(N - n_start, n_step);

        const void *a_packed_ptr = workspace_data.data();
        const void *b_packed_offset = (const char *)packed_b_ptr + ukernel.get_rhs_packed_offset(n_start, K, block_len);
        float *c_offset = c_ptr + n_start;

        ukernel.run_matmul(
            M, current_n, K, block_len,
            a_packed_ptr, b_packed_offset,
            c_offset, N * sizeof(float),
            sizeof(float),
            -FLT_MAX, FLT_MAX);
    }
}

#else
void mllm_kleidai_gemm_qsi4_f16_compute(
    mllm_fp16_t *c_ptr, const mllm_fp16_t *a_ptr, const uint8_t *packed_b_ptr,
    int M, int N, int K) {
    kai_matmul_clamp_f16_qai8dxp_qsi4cxp_ukernel ukernel;
    if (M == 1) {
        ukernel = qsi4_f16_ukernels.at(KleidiaiQsi4TileF16::k1x8_4x8_1x4_dotprod);
    } else {
        const auto tile_cfg = arm_is_i8mm_supported() ?
                                  KleidiaiQsi4TileF16::k4x8_4x8_16x4_i8mm :
                                  KleidiaiQsi4TileF16::k1x8_4x8_1x4_dotprod;
        ukernel = qsi4_f16_ukernels.at(tile_cfg);
    }

    auto &workspace_data = WorkspaceManager::get_instance().get_qsi4_workspace();

    size_t required_workspace_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f16_neon(
        M, K, ukernel.get_mr(), ukernel.get_kr(), ukernel.get_sr());
    if (workspace_data.size() < required_workspace_size) {
        workspace_data.resize(required_workspace_size);
    }

    kai_run_lhs_quant_pack_qai8dxp_f16_neon(
        M, K,
        ukernel.get_mr(), ukernel.get_kr(), ukernel.get_sr(),
        0,
        a_ptr,
        K * sizeof(mllm_fp16_t),
        workspace_data.data());

    const int n_step = ukernel.get_n_step();

#pragma omp parallel for num_threads(kai_thread_count)
    for (int n_start = 0; n_start < N; n_start += n_step) {
        const int current_n = std::min(N - n_start, n_step);
        const void *a_packed_ptr = workspace_data.data();
        const void *b_packed_offset = (const char *)packed_b_ptr + ukernel.get_rhs_packed_offset(n_start, K);
        void *c_offset = c_ptr + n_start;

        ukernel.run_matmul(
            M, current_n, K,
            a_packed_ptr, b_packed_offset,
            c_offset, N * sizeof(mllm_fp16_t),
            sizeof(mllm_fp16_t),
            -FLT_MAX, FLT_MAX);
    }
}

void mllm_kleidai_gemm_qsi4(
    float *c_ptr, const float *a_ptr, const uint8_t *packed_b_ptr,
    int M, int N, int K) {
    auto &a_fp16 = WorkspaceManager::get_instance().get_fp16_a_buffer();
    if (a_fp16.size() < M * K) {
        a_fp16.resize(M * K);
    }

    auto &c_fp16 = WorkspaceManager::get_instance().get_fp16_c_buffer();
    if (c_fp16.size() < M * N) {
        c_fp16.resize(M * N);
    }

#pragma omp parallel for num_threads(kai_thread_count)
    for (int i = 0; i < M * K; ++i) {
        a_fp16[i] = static_cast<mllm_fp16_t>(a_ptr[i]);
    }

    mllm_kleidai_gemm_qsi4_f16_compute(c_fp16.data(), a_fp16.data(), packed_b_ptr, M, N, K);

#pragma omp parallel for num_threads(kai_thread_count)
    for (int i = 0; i <= (M * N) - 8; i += 8) {
        float16x8_t fp16_vec = vld1q_f16(reinterpret_cast<const __fp16 *>(c_fp16.data() + i));

        float32x4_t fp32_vec_low = vcvt_f32_f16(vget_low_f16(fp16_vec));
        float32x4_t fp32_vec_high = vcvt_f32_f16(vget_high_f16(fp16_vec));

        vst1q_f32(c_ptr + i, fp32_vec_low);
        vst1q_f32(c_ptr + i + 4, fp32_vec_high);
    }
    for (int i = (M * N) - ((M * N) % 8); i < M * N; ++i) {
        c_ptr[i] = static_cast<float>(c_fp16[i]);
    }
}
#endif

size_t mllm_kleidai_get_packed_b_qsi4_size_to_fp16(int N, int K) {
    const auto tile_cfg = kleidiai_get_best_qsi4_tile_config_f16();
    const auto &ukernel = qsi4_f16_ukernels.at(tile_cfg);

    return kai_get_rhs_packed_size_rhs_pack_kxn_qsi4cxp_qs4cxs1s0(
        N, K, ukernel.get_nr(), ukernel.get_kr(), ukernel.get_sr());
}

void mllm_kleidai_pack_b_and_bias_qsi4_to_fp16(
    uint8_t *packed_b_ptr,
    const float *b_ptr,
    const float *bias_ptr,
    int N,
    int K) {
    const auto tile_cfg = kleidiai_get_best_qsi4_tile_config_f16();
    const auto &ukernel = qsi4_f16_ukernels.at(tile_cfg);

    const float *bias_to_use = bias_ptr;
    std::vector<float> fake_bias;
    if (bias_to_use == nullptr) {
        fake_bias.assign(N, 0.0f);
        bias_to_use = fake_bias.data();
    }

    const size_t quantized_b_size = (size_t)K * N / 2;
    std::vector<uint8_t> temp_quantized_b(quantized_b_size, 0);
    std::vector<float> temp_scales_fp32(N);

    for (int n = 0; n < N; ++n) {
        float amax = 0.0f;
        for (int k = 0; k < K; ++k) {
            amax = std::max(amax, std::abs(b_ptr[k * N + n]));
        }

        const float scale = amax / 7.0f;
        temp_scales_fp32[n] = scale;
        const float inv_scale = (scale != 0.0f) ? 1.0f / scale : 0.0f;

        for (int k = 0; k < K; ++k) {
            const float val = b_ptr[k * N + n];
            int32_t q_val = static_cast<int32_t>(roundf(val * inv_scale));
            q_val = std::max(-8, std::min(7, q_val));

            uint8_t stored_val = static_cast<uint8_t>(q_val + 8);

            size_t byte_idx = (k * N + n) / 2;
            if ((k * N + n) % 2 == 0) {
                temp_quantized_b[byte_idx] = stored_val;
            } else {
                temp_quantized_b[byte_idx] |= (stored_val << 4);
            }
        }
    }

    struct kai_rhs_pack_kxn_qsi4cxp_qs4cxs1s0_params kxn_params = {};
    kxn_params.lhs_zero_point = 1;
    kxn_params.rhs_zero_point = 8;
    kai_run_rhs_pack_kxn_qsi4cxp_qs4cxs1s0(
        1, N, K, ukernel.get_nr(), ukernel.get_kr(), ukernel.get_sr(),
        temp_quantized_b.data(), // Pointer to quantized data
        bias_to_use,             // Pointer to bias data
        temp_scales_fp32.data(), // Pointer to fp32 scale data
        packed_b_ptr,            // Output packed data
        0,                       // Output stride (0 for contiguous)
        &kxn_params);
}

void mllm_kleidai_gemm_qsi4_f16_internal(
    mllm_fp16_t *c_ptr, const float *a_ptr, const uint8_t *packed_b_ptr,
    int M, int N, int K) {
    kai_matmul_clamp_f16_qai8dxp_qsi4cxp_ukernel ukernel;
    if (M == 1) {
        ukernel = qsi4_f16_ukernels.at(KleidiaiQsi4TileF16::k1x8_4x8_1x4_dotprod);
    } else {
        const auto tile_cfg = arm_is_i8mm_supported() ?
                                  KleidiaiQsi4TileF16::k4x8_4x8_16x4_i8mm :
                                  KleidiaiQsi4TileF16::k1x8_4x8_1x4_dotprod;
        ukernel = qsi4_f16_ukernels.at(tile_cfg);
    }

    auto &workspace_data = WorkspaceManager::get_instance().get_qsi4_workspace();

    size_t required_workspace_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f16_neon(
        M, K, ukernel.get_mr(), ukernel.get_kr(), ukernel.get_sr());
    if (workspace_data.size() < required_workspace_size) {
        workspace_data.resize(required_workspace_size);
    }
    kai_run_lhs_quant_pack_qai8dxp_f16_neon(
        M, K,
        ukernel.get_mr(), ukernel.get_kr(), ukernel.get_sr(),
        0,
        reinterpret_cast<const mllm_fp16_t *>(a_ptr),
        K * sizeof(float),
        workspace_data.data());

    const int n_step = ukernel.get_n_step();

#pragma omp parallel for num_threads(kai_thread_count)
    for (int n_start = 0; n_start < N; n_start += n_step) {
        const int current_n = std::min(N - n_start, n_step);

        const void *a_packed_ptr = workspace_data.data();
        const void *b_packed_offset = (const char *)packed_b_ptr + ukernel.get_rhs_packed_offset(n_start, K);

        uint16_t *c_offset = reinterpret_cast<uint16_t *>(c_ptr) + n_start * N + n_start;

        ukernel.run_matmul(
            M, current_n, K,
            a_packed_ptr, b_packed_offset,
            c_offset, N * sizeof(uint16_t),
            sizeof(uint16_t),
            -FLT_MAX, FLT_MAX);
    }
}

#ifndef KAI_FP16_CAL
void mllm_kleidai_gemm_qsi4_to_fp16(
    mllm_fp16_t *c_ptr, const float *a_ptr, const uint8_t *packed_b_ptr,
    int M, int N, int K) {
    auto &c_temp = WorkspaceManager::get_instance().get_qsi4_c_temp_buffer();
    if (c_temp.size() < M * N) {
        c_temp.resize(M * N);
    }

    mllm_kleidai_gemm_qsi4(c_temp.data(), a_ptr, packed_b_ptr, M, N, K);

#pragma omp parallel for num_threads(kai_thread_count)
    for (int i = 0; i <= (M * N) - 4; i += 4) {
        float32x4_t fp32_vec = vld1q_f32(c_temp.data() + i);
        float16x4_t fp16_vec = vcvt_f16_f32(fp32_vec);
        vst1_f16(reinterpret_cast<__fp16 *>(c_ptr + i), fp16_vec);
    }
    for (int i = (M * N) - ((M * N) % 4); i < M * N; ++i) {
        c_ptr[i] = static_cast<mllm_fp16_t>(c_temp[i]);
    }
}
#else
void mllm_kleidai_gemm_qsi4_to_fp16(
    mllm_fp16_t *c_ptr, const float *a_ptr, const uint8_t *packed_b_ptr,
    int M, int N, int K) {
    mllm_kleidai_gemm_qsi4_f16_internal(c_ptr, a_ptr, packed_b_ptr, M, N, K);
}
#endif

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

void mllm_kleidai_pack_b_and_bias_fp16(mllm_fp16_t *packed_b_ptr, const mllm_fp16_t *b_ptr, const float *bias_ptr, int N, int K) {
    std::vector<mllm_fp16_t> bias_fp16_buffer(N);
    if (bias_ptr != nullptr) {
        for (int i = 0; i < N; ++i) {
            bias_fp16_buffer[i] = static_cast<mllm_fp16_t>(bias_ptr[i]);
        }
    } else {
        std::fill(bias_fp16_buffer.begin(), bias_fp16_buffer.end(), static_cast<mllm_fp16_t>(0.0f));
    }
    kai_run_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon(
        1, N, K, fp16_ukernel.get_nr(), fp16_ukernel.get_kr(), fp16_ukernel.get_sr(),
        N * sizeof(mllm_fp16_t), b_ptr, bias_fp16_buffer.data(), nullptr, packed_b_ptr, 0, nullptr);
}

void mllm_kleidai_gemm_fp16(float *c_ptr, const float *a_ptr, const mllm_fp16_t *packed_b_ptr, int M, int N, int K) {
    auto &a_fp16 = WorkspaceManager::get_instance().get_fp16_a_buffer();
    if (a_fp16.size() < M * K) {
        a_fp16.resize(M * K);
    }

    auto &c_fp16 = WorkspaceManager::get_instance().get_fp16_c_buffer();
    if (c_fp16.size() < M * N) {
        c_fp16.resize(M * N);
    }

#pragma omp parallel for num_threads(kai_thread_count)
    for (int i = 0; i < M * K; ++i) {
        a_fp16[i] = static_cast<mllm_fp16_t>(a_ptr[i]);
    }

    const int m_step = fp16_ukernel.get_m_step();
    const int n_step = fp16_ukernel.get_n_step();

#pragma omp parallel for collapse(2) num_threads(kai_thread_count)
    for (int m_start = 0; m_start < M; m_start += m_step) {
        for (int n_start = 0; n_start < N; n_start += n_step) {
            const int current_m = std::min(M - m_start, m_step);
            const int current_n = std::min(N - n_start, n_step);

            const mllm_fp16_t *a_offset = a_fp16.data() + m_start * K;
            const mllm_fp16_t *b_offset = packed_b_ptr + (n_start * (K + 1));
            mllm_fp16_t *c_offset = c_fp16.data() + m_start * N + n_start;

            fp16_ukernel.run_matmul(
                current_m, current_n, K, a_offset, K * sizeof(mllm_fp16_t),
                b_offset, c_offset, N * sizeof(mllm_fp16_t), sizeof(mllm_fp16_t),
                -FLT_MAX, FLT_MAX);
        }
    }

#pragma omp parallel for num_threads(kai_thread_count)
    for (int i = 0; i < M * N; ++i) {
        c_ptr[i] = static_cast<float>(c_fp16[i]);
    }
}

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

void mllm_kleidai_pack_b_and_bias_fp32(float *packed_b_ptr, const float *b_ptr, const float *bias_ptr, int N, int K) {
    const float *bias_to_use = bias_ptr;
    std::vector<float> fake_bias;
    if (bias_to_use == nullptr) {
        fake_bias.assign(N, 0.0f);
        bias_to_use = fake_bias.data();
    }
    kai_run_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(
        1, N, K, fp32_ukernel.get_nr(), fp32_ukernel.get_kr(), fp32_ukernel.get_sr(),
        N * sizeof(float), b_ptr, bias_to_use, nullptr, packed_b_ptr, 0, nullptr);
}

void mllm_kleidai_gemm_fp32(float *c_ptr, const float *a_ptr, const float *packed_b_ptr, int M, int N, int K) {
    const int m_step = fp32_ukernel.get_m_step();
    const int n_step = fp32_ukernel.get_n_step();

#pragma omp parallel for collapse(2) num_threads(kai_thread_count)
    for (int m_start = 0; m_start < M; m_start += m_step) {
        for (int n_start = 0; n_start < N; n_start += n_step) {
            const int current_m = std::min(M - m_start, m_step);
            const int current_n = std::min(N - n_start, n_step);
            const float *a_offset = a_ptr + m_start * K;
            const float *b_offset = packed_b_ptr + (n_start * (K + 1));
            float *c_offset = c_ptr + m_start * N + n_start;
            fp32_ukernel.run_matmul(
                current_m, current_n, K,
                a_offset, K * sizeof(float), b_offset,
                c_offset, N * sizeof(float), sizeof(float),
                -FLT_MAX, FLT_MAX);
        }
    }
}

#include "Transpose2D.hpp"
void mllm_kleidai_pack_b_and_bias_fp32_transpose(float *packed_b_ptr, const float *b_ptr_nxk, const float *bias_ptr, int N, int K) {
    std::vector<float> b_temp_kxn(K * N);
    transpose_matrix_efficient(b_ptr_nxk, b_temp_kxn.data(), N, K);
    mllm_kleidai_pack_b_and_bias_fp32(packed_b_ptr, b_temp_kxn.data(), bias_ptr, N, K);
}

void mllm_kleidai_pack_b_and_bias_fp16_transpose(mllm_fp16_t *packed_b_ptr, const mllm_fp16_t *b_ptr_nxk, const float *bias_ptr, int N, int K) {
    std::vector<mllm_fp16_t> b_temp_kxn(K * N);

#if defined(__aarch64__)
    transpose_matrix_efficient_fp16(b_ptr_nxk, b_temp_kxn.data(), N, K);
#else
    const int BLOCK_DIM = 32;
    for (int i = 0; i < N; i += BLOCK_DIM) {
        for (int j = 0; j < K; j += BLOCK_DIM) {
            for (int bi = i; bi < i + BLOCK_DIM && bi < N; ++bi) {
                for (int bj = j; j < K && bj < j + BLOCK_DIM; ++bj) {
                    b_temp_kxn[bj * N + bi] = b_ptr_nxk[bi * K + bj];
                }
            }
        }
    }
#endif

    mllm_kleidai_pack_b_and_bias_fp16(packed_b_ptr, b_temp_kxn.data(), bias_ptr, N, K);
}
/*** no use ****/
void mllm_kleidai_gemm_fp32_transpose(float *c_ptr, const float *a_ptr, const float *b_ptr_nxk, const float *bias_ptr, int M, int N, int K) {
    size_t packed_b_size = mllm_kleidai_get_packed_b_fp32_size(N, K);
    std::vector<float> packed_b_data(packed_b_size);
    mllm_kleidai_pack_b_and_bias_fp32_transpose(packed_b_data.data(), b_ptr_nxk, bias_ptr, N, K);
    mllm_kleidai_gemm_fp32(c_ptr, a_ptr, packed_b_data.data(), M, N, K);
}

void mllm_kleidai_gemm_fp16_transpose(float *c_ptr, const float *a_ptr, const mllm_fp16_t *b_ptr_nxk, const float *bias_ptr, int M, int N, int K) {
    size_t packed_b_size = mllm_kleidai_get_packed_b_fp16_size(N, K);
    std::vector<mllm_fp16_t> packed_b_data(packed_b_size / sizeof(mllm_fp16_t));
    mllm_kleidai_pack_b_and_bias_fp16_transpose(packed_b_data.data(), b_ptr_nxk, bias_ptr, N, K);
    mllm_kleidai_gemm_fp16(c_ptr, a_ptr, packed_b_data.data(), M, N, K);
}

void mllm_kleidai_gemm_fp32_bshd(float *c_ptr, const float *a_ptr, const float *packed_b_ptr, int B, int H, int S_M, int S_N, int D_K) {
    const int M = S_M;
    const int K = D_K; // 在GEMM上下文中，K是BSHD布局中的D（dimension）

    // 为 BSHD (B,S,H,D/N) 布局计算跨距
    const long long stride_a_b = (long long)S_M * H * K;
    const long long stride_a_s = (long long)H * K;

    const long long stride_c_b = (long long)S_M * H * S_N;
    const long long stride_c_s = (long long)H * S_N;

    const int m_step = fp32_ukernel.get_m_step();
    const int n_step = fp32_ukernel.get_n_step();

    // 并行处理 batch 和 head 维度
#pragma omp parallel for collapse(2) num_threads(kai_thread_count)
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int m_start = 0; m_start < M; m_start += m_step) {
                for (int n_start = 0; n_start < S_N; n_start += n_step) {
                    const int current_m = std::min(M - m_start, m_step);
                    const int current_n = std::min(S_N - n_start, n_step);

                    // 计算当前块在BSHD布局中的A矩阵偏移
                    // 指向 A[b, m_start, h, 0]
                    const float *a_offset = a_ptr + b * stride_a_b + m_start * stride_a_s + h * K;

                    // B矩阵是预打包的，其偏移仅与N维度相关
                    const float *b_offset = packed_b_ptr + (n_start * (K + 1));

                    // 计算当前块在BSHD布局中的C矩阵偏移
                    // 指向 C[b, m_start, h, n_start]
                    float *c_offset = c_ptr + b * stride_c_b + m_start * stride_c_s + h * S_N + n_start;

                    // 调用微内核，传入正确的行跨距
                    fp32_ukernel.run_matmul(
                        current_m, current_n, K,
                        a_offset, stride_a_s * sizeof(float), // A矩阵的行跨距
                        b_offset,
                        c_offset, stride_c_s * sizeof(float), // C矩阵的行跨距
                        sizeof(float),
                        -FLT_MAX, FLT_MAX);
                }
            }
        }
    }
}

void mllm_kleidai_gemm_fp16_bshd(float *c_ptr, const float *a_ptr, const mllm_fp16_t *packed_b_ptr, int B, int H, int S_M, int S_N, int D_K) {
    const int M = S_M;
    const int K = D_K; // 在GEMM上下文中，K是BSHD布局中的D（dimension）

    // 为 BSHD (B,S,H,D/N) 布局计算跨距
    const long long stride_a_b = (long long)S_M * H * K;
    const long long stride_a_s = (long long)H * K;

    const long long stride_c_b = (long long)S_M * H * S_N;
    const long long stride_c_s = (long long)H * S_N;

    const int m_step = fp16_ukernel.get_m_step();
    const int n_step = fp16_ukernel.get_n_step();

    // 并行处理 batch 和 head 维度
#pragma omp parallel for collapse(2) num_threads(kai_thread_count)
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            // 从工作区管理器获取线程本地缓冲区
            auto &a_fp16 = WorkspaceManager::get_instance().get_fp16_a_buffer();
            if (a_fp16.size() < M * K) {
                a_fp16.resize(M * K);
            }

            auto &c_fp16 = WorkspaceManager::get_instance().get_fp16_c_buffer();
            if (c_fp16.size() < M * S_N) {
                c_fp16.resize(M * S_N);
            }

            // 1. 收集(Gather)和转换: 将非连续的 BSHD float A矩阵切片复制到连续的 fp16 缓冲区
            const float *a_bh_ptr = a_ptr + b * stride_a_b; // 指向批次 b 的起始位置
            for (int s = 0; s < S_M; ++s) {
                for (int d = 0; d < D_K; ++d) {
                    // 从 A[b,s,h,d] 读取
                    a_fp16[s * D_K + d] = static_cast<mllm_fp16_t>(a_bh_ptr[s * stride_a_s + h * D_K + d]);
                }
            }

            // 2. 计算: 在连续的缓冲区上执行GEMM
            for (int m_start = 0; m_start < M; m_start += m_step) {
                for (int n_start = 0; n_start < S_N; n_start += n_step) {
                    const int current_m = std::min(M - m_start, m_step);
                    const int current_n = std::min(S_N - n_start, n_step);

                    const mllm_fp16_t *a_offset = a_fp16.data() + m_start * K;
                    const mllm_fp16_t *b_offset = packed_b_ptr + (n_start * (K + 1));
                    mllm_fp16_t *c_offset = c_fp16.data() + m_start * S_N + n_start;

                    // 由于 a_fp16 和 c_fp16 是连续的，使用标准的行跨距
                    fp16_ukernel.run_matmul(
                        current_m, current_n, K,
                        a_offset, K * sizeof(mllm_fp16_t),
                        b_offset,
                        c_offset, S_N * sizeof(mllm_fp16_t), sizeof(mllm_fp16_t),
                        -FLT_MAX, FLT_MAX);
                }
            }

            // 3. 分散(Scatter)和转换: 将连续的 fp16 结果缓冲区复制回非连续的 BSHD float C矩阵
            float *c_bh_ptr = c_ptr + b * stride_c_b; // 指向批次 b 的起始位置
            for (int s = 0; s < S_M; ++s) {
                for (int n = 0; n < S_N; ++n) {
                    // 写入 C[b,s,h,n]
                    c_bh_ptr[s * stride_c_s + h * S_N + n] = static_cast<float>(c_fp16[s * S_N + n]);
                }
            }
        }
    }
}
#endif