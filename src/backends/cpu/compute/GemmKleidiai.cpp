
#if defined(__aarch64__) || defined(__arm__) || defined(__arm64__)

#include "GemmKleidiai.hpp"
#include "FeatureCheck.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cfloat>

// 引入 OpenMP 头文件
#include <omp.h>
//  ###################################################################### //
//  ##                   Implementation 1: QSI4 (INT4)                  ##
//  ###################################################################### //// 【新增】线程局部的全局工作区，用于内存复用
// 【新增】一个为OpenMP设计的、线程安全的工作区管理器
class WorkspaceManager {
public:
    // 构造函数：获取OpenMP最大线程数，并为每个线程创建一个工作区
    WorkspaceManager() {
        int max_threads = kai_thread_count; // 获取当前线程数;
#ifdef _OPENMP
        max_threads = omp_get_max_threads();
#endif
        qsi4_workspaces_.resize(max_threads);
        qsi4_c_temp_buffers_.resize(max_threads);
        fp16_a_buffers_.resize(max_threads);
        fp16_c_buffers_.resize(max_threads);
    }

    // 获取当前线程的 QSI4 工作区
    std::vector<uint8_t> &get_qsi4_workspace() {
        int thread_id = 0;
#ifdef _OPENMP
        thread_id = omp_get_thread_num();
#endif
        return qsi4_workspaces_[thread_id];
    }

    // 获取当前线程的 QSI4->FP16 临时C区
    std::vector<float> &get_qsi4_c_temp_buffer() {
        int thread_id = 0;
#ifdef _OPENMP
        thread_id = omp_get_thread_num();
#endif
        return qsi4_c_temp_buffers_[thread_id];
    }

    // 获取当前线程的 FP16 临时A区
    std::vector<mllm_fp16_t> &get_fp16_a_buffer() {
        int thread_id = 0;
#ifdef _OPENMP
        thread_id = omp_get_thread_num();
#endif
        return fp16_a_buffers_[thread_id];
    }

    // 获取当前线程的 FP16 临时C区
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

// 创建一个全局唯一的管理器实例
static WorkspaceManager g_workspace_manager;

//  ###################################################################### //
//  ##                   Implementation 1: QSI4 (INT4)                  ##
//  ###################################################################### //

// 【新增】包含用于map和CPU特性检测的头文件
#include <unordered_map>
#if defined(__linux__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif

// 【新增】包含所有需要的计算核心头文件，特别是 i8mm 版本
#include "kai_matmul_clamp_f32_qai8dxp_qsi4c32p_interface.h"
#include "kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0.h"
#include "kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h"
#include "kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm.h"
#include "kai_matmul_clamp_f32_qai8dxp_qsi4c32p_interface.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0.h"

// 【新增】在.cpp内部定义核心配置的枚举，外部不可见
enum class KleidiaiQsi4Tile {
    k1x8_4x8_1x4x32_dotprod, // 兼容性最好的 dotprod 核心
    k4x8_4x8_8x4x32_i8mm,    // 高性能的 i8mm 核心
};

// 【新增】为KleidiaiQsi4Tile定义哈希函数以用于unordered_map
namespace std {
template <>
struct hash<KleidiaiQsi4Tile> {
    std::size_t operator()(const KleidiaiQsi4Tile &k) const noexcept {
        return std::hash<std::underlying_type<KleidiaiQsi4Tile>::type>()(
            static_cast<std::underlying_type<KleidiaiQsi4Tile>::type>(k));
    }
};
} // namespace std

// 【新增】在.cpp内部定义一个静态map，存储所有可用的计算核心
// 【新增】第一步：将每个 ukernel 定义为独立的静态常量
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

// 第二步：使用定义好的常量来初始化 map，语法更简洁清晰
static const std::unordered_map<KleidiaiQsi4Tile, kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel> qsi4_ukernels = {
    {KleidiaiQsi4Tile::k1x8_4x8_1x4x32_dotprod, dotprod_ukernel},
    {KleidiaiQsi4Tile::k4x8_4x8_8x4x32_i8mm, i8mm_ukernel}};
    
// 【新增】根据CPU能力自动选择最佳核心，并缓存结果
static KleidiaiQsi4Tile kleidiai_get_best_qsi4_tile_config() {
    // 使用静态变量，这样CPU检测的逻辑只会在第一次调用时执行一次
    static const KleidiaiQsi4Tile best_tile = arm_is_i8mm_supported() ?
                                                  KleidiaiQsi4Tile::k4x8_4x8_8x4x32_i8mm :
                                                  KleidiaiQsi4Tile::k1x8_4x8_1x4x32_dotprod;
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
    uint8_t* packed_b_ptr,
    const float* b_ptr,
    const float* bias_ptr,
    int N,
    int K) {
    const auto tile_cfg = kleidiai_get_best_qsi4_tile_config();
    const auto &ukernel = qsi4_ukernels.at(tile_cfg);

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
        1, N, K, ukernel.get_nr(), ukernel.get_kr(), ukernel.get_sr(), block_len,
        temp_quantized_b.data(), N/2, bias_to_use, (const uint8_t*)temp_scales.data(), num_blocks_k * sizeof(uint16_t),
        packed_b_ptr, 0, &params);
}

// --- 【替换】旧的 mllm_kleidai_gemm_qsi4 函数 ---// --- 【替换】旧的 mllm_kleidai_gemm_qsi4 函数 ---
void mllm_kleidai_gemm_qsi4(
    float *c_ptr, const float *a_ptr, const uint8_t *packed_b_ptr,
    int M, int N, int K) {
    // 1. 内部自动选择最佳计算核心 (i8mm 或 dotprod)
    const auto tile_cfg = kleidiai_get_best_qsi4_tile_config();
    const auto &ukernel = qsi4_ukernels.at(tile_cfg);

    // 2. 从全局管理器获取当前线程专属的工作区，如果空间不足则自动扩容
    auto &workspace_data = g_workspace_manager.get_qsi4_workspace();
    size_t required_workspace_size = get_workspace_qsi4_size(M, K);
    if (workspace_data.size() < required_workspace_size) {
        workspace_data.resize(required_workspace_size);
    }

    // 3. 在获取到的工作区中，对左手矩阵 A 进行量化和打包
    kai_run_lhs_quant_pack_qai8dxp_f32(
        M, K,
        ukernel.get_mr(), ukernel.get_kr(), ukernel.get_sr(),
        0,
        a_ptr, K * sizeof(float),
        workspace_data.data());

    const int m_step = ukernel.get_m_step();
    const int n_step = ukernel.get_n_step();
    const int block_len = 32;

    // 4. 使用选择的最佳核心，在 OpenMP 并行循环中执行矩阵乘法
#pragma omp parallel for collapse(2) num_threads(kai_thread_count)
    for (int m_start = 0; m_start < M; m_start += m_step) {
        for (int n_start = 0; n_start < N; n_start += n_step) {
            const int current_m = std::min(M - m_start, m_step);
            const int current_n = std::min(N - n_start, n_step);

            // 使用来自工作区的指针
            const void *a_packed_offset = (const char *)workspace_data.data() + ukernel.get_lhs_packed_offset(m_start, K);
            const void *b_packed_offset = (const char *)packed_b_ptr + ukernel.get_rhs_packed_offset(n_start, K, block_len);
            float *c_offset = c_ptr + m_start * N + n_start;

            ukernel.run_matmul(
                current_m, current_n, K, block_len,
                a_packed_offset, b_packed_offset,
                c_offset, N * sizeof(float), sizeof(float),
                -FLT_MAX, FLT_MAX);
        }
    }
}

void mllm_kleidai_gemm_qsi4_to_fp16(
    mllm_fp16_t *c_ptr, const float *a_ptr, const uint8_t *packed_b_ptr,
    int M, int N, int K) {
    // 【优化】为临时的FP32输出缓冲区使用线程局部存储
    auto &c_temp = g_workspace_manager.get_qsi4_c_temp_buffer();
    if (c_temp.size() < M * N) {
        c_temp.resize(M * N);
    }

    mllm_kleidai_gemm_qsi4(c_temp.data(), a_ptr, packed_b_ptr, M, N, K);

    // ... 后续的 FP32 -> FP16 转换代码保持不变 ...
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

//  ###################################################################### //
//  ##                   Implementation 2: FP16                         ##
//  ###################################################################### //
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

// --- 【替换】旧的 mllm_kleidai_gemm_fp16 函数 ---
// --- 【替换】旧的 mllm_kleidai_gemm_fp16 函数 ---
void mllm_kleidai_gemm_fp16(float *c_ptr, const float *a_ptr, const mllm_fp16_t *packed_b_ptr, int M, int N, int K) {
    // 【优化】从全局管理器中，获取当前线程专属的缓冲区
    // a_fp16 用于存放 a_ptr 从 FP32 转换到 FP16 的结果
    auto &a_fp16 = g_workspace_manager.get_fp16_a_buffer();
    if (a_fp16.size() < M * K) {
        a_fp16.resize(M * K);
    }

    // c_fp16 用于存放 FP16 矩阵乘法的中间结果
    auto &c_fp16 = g_workspace_manager.get_fp16_c_buffer();
    if (c_fp16.size() < M * N) {
        c_fp16.resize(M * N);
    }

    // 1. 将输入的 FP32 矩阵 A 并行转换为 FP16，存入线程专属的 a_fp16 缓冲区
#pragma omp parallel for num_threads(kai_thread_count)
    for (int i = 0; i < M * K; ++i) {
        a_fp16[i] = static_cast<mllm_fp16_t>(a_ptr[i]);
    }

    const int m_step = fp16_ukernel.get_m_step();
    const int n_step = fp16_ukernel.get_n_step();

    // 2. 执行 FP16 矩阵乘法，结果存入线程专属的 c_fp16 缓冲区
#pragma omp parallel for collapse(2) num_threads(kai_thread_count)
    for (int m_start = 0; m_start < M; m_start += m_step) {
        for (int n_start = 0; n_start < N; n_start += n_step) {
            const int current_m = std::min(M - m_start, m_step);
            const int current_n = std::min(N - n_start, n_step);

            // 计算时使用缓冲区的指针
            const mllm_fp16_t *a_offset = a_fp16.data() + m_start * K;
            const mllm_fp16_t *b_offset = packed_b_ptr + (n_start * (K + 1));
            mllm_fp16_t *c_offset = c_fp16.data() + m_start * N + n_start;

            fp16_ukernel.run_matmul(
                current_m, current_n, K, a_offset, K * sizeof(mllm_fp16_t),
                b_offset, c_offset, N * sizeof(mllm_fp16_t), sizeof(mllm_fp16_t),
                -FLT_MAX, FLT_MAX);
        }
    }

    // 3. 将 FP16 的中间结果并行转换为最终的 FP32 输出
#pragma omp parallel for num_threads(kai_thread_count)
    for (int i = 0; i < M * N; ++i) {
        c_ptr[i] = static_cast<float>(c_fp16[i]);
    }
}
//  ###################################################################### //
//  ##                   Implementation 3: FP32                         ##
//  ###################################################################### //
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

// #pragma omp parallel for
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

// --- BEGIN: Transpose-and-Pack Functions ---
#include "Transpose2D.hpp"
void mllm_kleidai_pack_b_and_bias_fp32_transpose(float *packed_b_ptr, const float *b_ptr_nxk, const float *bias_ptr, int N, int K) {
    // 1. Create a temporary buffer for the transposed KxN matrix
    std::vector<float> b_temp_kxn(K * N);

    // 2. Call the efficient transpose function to fill the buffer
    transpose_matrix_efficient(b_ptr_nxk, b_temp_kxn.data(), N, K);

    // 3. Call the original packing function with the now correctly-ordered data
    mllm_kleidai_pack_b_and_bias_fp32(packed_b_ptr, b_temp_kxn.data(), bias_ptr, N, K);
}

void mllm_kleidai_pack_b_and_bias_fp16_transpose(mllm_fp16_t *packed_b_ptr, const mllm_fp16_t *b_ptr_nxk, const float *bias_ptr, int N, int K) {
    // 1. Create a temporary buffer
    std::vector<mllm_fp16_t> b_temp_kxn(K * N);

    // 2. Perform a cache-friendly transpose for fp16
    // (transpose_matrix_efficient is for float32, so we use its blocking logic here)
#if defined(__aarch64__)
    // 2. 【高效路径】在ARM平台上，调用为__fp16优化的NEON SIMD转置函数
    transpose_matrix_efficient_fp16(b_ptr_nxk, b_temp_kxn.data(), N, K);
#else
    // 2. 【通用路径】在非ARM平台，使用缓存优化的C++转置
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

    // 3. Call the original fp16 packing function
    mllm_kleidai_pack_b_and_bias_fp16(packed_b_ptr, b_temp_kxn.data(), bias_ptr, N, K);
}

// --- END: Transpose-and-Pack Functions ---
// --- BEGIN: High-Level Transposed GEMM APIs ---

void mllm_kleidai_gemm_fp32_transpose(float *c_ptr, const float *a_ptr, const float *b_ptr_nxk, const float *bias_ptr, int M, int N, int K) {
    // Allocate space for the packed B matrix
    size_t packed_b_size = mllm_kleidai_get_packed_b_fp32_size(N, K);
    std::vector<float> packed_b_data(packed_b_size);

    // Call the new transpose-and-pack function
    mllm_kleidai_pack_b_and_bias_fp32_transpose(packed_b_data.data(), b_ptr_nxk, bias_ptr, N, K);

    // Call the original GEMM compute function with the packed data
    mllm_kleidai_gemm_fp32(c_ptr, a_ptr, packed_b_data.data(), M, N, K);
}

void mllm_kleidai_gemm_fp16_transpose(float *c_ptr, const float *a_ptr, const mllm_fp16_t *b_ptr_nxk, const float *bias_ptr, int M, int N, int K) {
    // Allocate space for the packed B matrix
    size_t packed_b_size = mllm_kleidai_get_packed_b_fp16_size(N, K);
    std::vector<mllm_fp16_t> packed_b_data(packed_b_size / sizeof(mllm_fp16_t));

    // Call the new transpose-and-pack function
    mllm_kleidai_pack_b_and_bias_fp16_transpose(packed_b_data.data(), b_ptr_nxk, bias_ptr, N, K);

    // Call the original GEMM compute function with the packed data
    mllm_kleidai_gemm_fp16(c_ptr, a_ptr, packed_b_data.data(), M, N, K);
}

// --- END: High-Level Transposed GEMM APIs ---

#endif