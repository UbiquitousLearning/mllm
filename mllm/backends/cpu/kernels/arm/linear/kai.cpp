// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <limits>

#include "mllm/core/Parallel.hpp"
#include "mllm/backends/cpu/kernels/arm/linear/kai.hpp"

// for fp32
#include "kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla.h"
#include "kai_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon.h"

// for pack_kxn_fp16_w_bias
#include "kai_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla.h"
#include "kai_matmul_clamp_f16_f16_f16p_interface.h"
#include "kai_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon.h"

// for f32_qai8dxp_qsi4c32
#include "kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod.h"
#include "kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h"
#include "kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod.h"
#include "kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm.h"
#include "kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm.h"
#include "kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm.h"
#include "kai_matmul_clamp_f32_qai8dxp_qsi4c32p_interface.h"
#include "kai_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0.h"
#include "kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0.h"

// for qsi8d32p_qai4c32p. The Scale of LHS is fp32 not fp16!!!
#include "kai_matmul_clamp_f16_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod.h"
#include "kai_matmul_clamp_f16_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm.h"
#include "kai_lhs_quant_pack_qsi8d32pscalef32_f16_neon.h"
#include "kai_lhs_quant_pack_qsi8d32pscalef32_f32_neon.h"
#include "kai_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon.h"
#include "mllm/utils/Common.hpp"

namespace mllm::cpu::arm {

kai_matmul_clamp_f32_f32_f32p_ukernel KaiLinear_fp32_fp32_fp32p_mxk_kxn::ukernel_ = {
    .get_m_step = kai_get_m_step_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
    .get_n_step = kai_get_n_step_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
    .get_nr = kai_get_nr_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
    .get_kr = kai_get_kr_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
    .get_sr = kai_get_sr_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
    .get_lhs_offset = kai_get_lhs_offset_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
    .get_rhs_packed_offset = kai_get_rhs_packed_offset_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
    .get_dst_offset = kai_get_dst_offset_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
    .get_dst_size = kai_get_dst_size_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
    .run_matmul = kai_run_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla};

size_t KaiLinear_fp32_fp32_fp32p_mxk_kxn::workspace_size(int M, int K) { return 0; }

size_t KaiLinear_fp32_fp32_fp32p_mxk_kxn::quant_pack_rhs_size(int K, int N) {
  const size_t nr = ukernel_.get_nr();
  const size_t kr = ukernel_.get_kr();
  const size_t sr = ukernel_.get_sr();
  const size_t rhs_packed_size = kai_get_rhs_packed_size_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(N, K);
  return rhs_packed_size;
}

void KaiLinear_fp32_fp32_fp32p_mxk_kxn::quant_pack_rhs_offline(uint8_t* __restrict__ packed_weight,
                                                               const float* __restrict__ rhs, const float* __restrict__ bias,
                                                               int K, int N) {
  const size_t nr = ukernel_.get_nr();
  const size_t kr = ukernel_.get_kr();
  const size_t sr = ukernel_.get_sr();
  const size_t rhs_stride = N * sizeof(float);
  float* new_bias = nullptr;
  if (bias == nullptr) {
    new_bias = new float[N];
    memset(new_bias, 0, N * sizeof(float));
  }
  kai_run_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(1, N, K, nr, kr, sr,     // Packing arguments
                                                   rhs_stride,              // RHS stride
                                                   rhs,                     // RHS
                                                   bias ? bias : new_bias,  // Bias
                                                   nullptr,                 // Scale
                                                   packed_weight,           // RHS packed
                                                   0, nullptr);
  delete[] new_bias;
}

void KaiLinear_fp32_fp32_fp32p_mxk_kxn::matmul(float* __restrict__ dst, const float* __restrict__ lhs_fp32,
                                               const uint8_t* packed_weight_bias, void* workspace, int M, int K, int N,
                                               int thread_count) {
  const size_t lhs_stride = K * sizeof(float);
  const size_t rhs_stride = N * sizeof(float);
  const size_t dst_stride_row = N * sizeof(float);
  const size_t dst_stride_col = sizeof(float);

  const size_t m_step = ukernel_.get_m_step();  // Scheduling along M
  const size_t n_step = ukernel_.get_n_step();  // Scheduling along N

  MLLM_CONDITIONAL_PARALLEL_FOR(thread_count > 1, thread_count, i_m_step, 0, M, m_step, {
    for (size_t i_n_step = 0; i_n_step < N; i_n_step += n_step) {
      // Support functions return offset in bytes
      const uint8_t* lhs_ptr = (const uint8_t*)lhs_fp32 + (ukernel_.get_lhs_offset(i_m_step, K * sizeof(float)));
      const uint8_t* rhs_ptr = (const uint8_t*)packed_weight_bias + (ukernel_.get_rhs_packed_offset(i_n_step, K));
      uint8_t* dst_ptr = (uint8_t*)dst + (ukernel_.get_dst_offset(i_m_step, i_n_step, N * sizeof(float)));
      const size_t actual_m = std::min((size_t)(M - i_m_step), m_step);
      const size_t actual_n = std::min((size_t)(N - i_n_step), n_step);
      ukernel_.run_matmul(actual_m, actual_n, K,  // Dimensions
                          lhs_ptr,                // LHS
                          lhs_stride,             // LHS stride
                          rhs_ptr,                // RHS packed
                          dst_ptr,                // DST
                          dst_stride_row,         // DST stride (row)
                          dst_stride_col,         // DST stride (col)
                          -std::numeric_limits<float>::max(),
                          std::numeric_limits<float>::max()  // Min and max for the clamp operation
      );
    }
  });
}

kai_matmul_clamp_f16_f16_f16p_ukernel KaiLinear_fp16_fp16_fp16p_mxk_kxn::ukernel_ = {
    .get_m_step = kai_get_m_step_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
    .get_n_step = kai_get_n_step_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
    .get_nr = kai_get_nr_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
    .get_kr = kai_get_kr_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
    .get_sr = kai_get_sr_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
    .get_lhs_packed_offset = kai_get_lhs_offset_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
    .get_rhs_packed_offset = kai_get_rhs_packed_offset_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
    .get_dst_offset = kai_get_dst_offset_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
    .get_dst_size = kai_get_dst_size_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
    .run_matmul = kai_run_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla};

size_t KaiLinear_fp16_fp16_fp16p_mxk_kxn::pack_rhs_size(int K, int N) {
  return kai_get_rhs_packed_size_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon(N, K);
}

void KaiLinear_fp16_fp16_fp16p_mxk_kxn::pack_rhs_offline(float16_t* __restrict__ rhs_packed, const float16_t* __restrict__ rhs,
                                                         const float16_t* bias, int K, int N) {
  bool has_bias = bias != nullptr;
  float16_t* fake_bias = nullptr;

  if (!has_bias) {
    fake_bias = new float16_t[N];
    for (int i = 0; i < N; ++i) fake_bias[i] = 0;
  }

  const size_t nr = ukernel_.get_nr();
  const size_t kr = ukernel_.get_kr();
  const size_t sr = ukernel_.get_sr();

  const size_t rhs_stride = N * sizeof(float16_t);

  kai_run_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon(1, N, K, nr, kr, sr,          // Packing arguments
                                                    rhs_stride,                   // RHS stride
                                                    rhs,                          // RHS
                                                    has_bias ? bias : fake_bias,  // Bias
                                                    nullptr,                      // Scale
                                                    rhs_packed,                   // RHS packed
                                                    0, nullptr);
  if (!has_bias) { delete[] fake_bias; }
}

void KaiLinear_fp16_fp16_fp16p_mxk_kxn::matmul(float16_t* __restrict__ dst, const float16_t* __restrict__ lhs,
                                               const float16_t* __restrict__ rhs, int M, int K, int N, int thread_count) {
  const int lhs_stride = K * sizeof(float16_t);
  const int dst_stride_row = N * sizeof(float16_t);
  const int dst_stride_col = sizeof(float16_t);

  const int m_step = ukernel_.get_m_step();  // Scheduling along M
  const int n_step = ukernel_.get_n_step();  // Scheduling along N

  MLLM_CONDITIONAL_PARALLEL_FOR(thread_count > 1, thread_count, i_m_step, 0, M, m_step, {
    for (int i_n_step = 0; i_n_step < N; i_n_step += n_step) {
      // Support functions return offset in bytes
      const uint8_t* lhs_ptr = (const uint8_t*)lhs + (ukernel_.get_lhs_packed_offset(i_m_step, K * sizeof(uint16_t)));
      const uint8_t* rhs_ptr = (const uint8_t*)rhs + (ukernel_.get_rhs_packed_offset(i_n_step, K));
      uint8_t* dst_ptr = (uint8_t*)dst + (ukernel_.get_dst_offset(i_m_step, i_n_step, N * sizeof(uint16_t)));

      const int actual_m = std::min(M - (int)i_m_step, m_step);
      const int actual_n = std::min(N - i_n_step, n_step);

      ukernel_.run_matmul(actual_m, actual_n, K,  // Dimensions
                          lhs_ptr,                // LHS
                          lhs_stride,             // LHS stride
                          rhs_ptr,                // RHS packed
                          dst_ptr,                // DST
                          dst_stride_row,         // DST stride (row)
                          dst_stride_col,         // DST stride (col)
                          -std::numeric_limits<float>::max(),
                          std::numeric_limits<float>::max()  // Min and max for the clamp operation
      );
    }
  });
}

std::unordered_map<KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles, kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel>
    KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::ukernels_ = {
        {KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp1x8_qsi4c32p4x8_1x4x32,
         {.get_m_step = kai_get_m_step_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_n_step = kai_get_n_step_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_mr = kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_nr = kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_kr = kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_sr = kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_rhs_packed_offset = kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_dst_offset = kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_dst_size = kai_get_dst_size_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .run_matmul = kai_run_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod}},
        {KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp1x8_qsi4c32p8x8_1x8x32,
         {.get_m_step = kai_get_m_step_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
          .get_n_step = kai_get_n_step_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
          .get_mr = kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
          .get_nr = kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
          .get_kr = kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
          .get_sr = kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
          .get_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
          .get_rhs_packed_offset = kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
          .get_dst_offset = kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
          .get_dst_size = kai_get_dst_size_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
          .run_matmul = kai_run_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod}},
        {KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp4x8_qsi4c32p4x8_8x4x32,
         {.get_m_step = kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
          .get_n_step = kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
          .get_mr = kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
          .get_nr = kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
          .get_kr = kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
          .get_sr = kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
          .get_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
          .get_rhs_packed_offset = kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
          .get_dst_offset = kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
          .get_dst_size = kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
          .run_matmul = kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm}},
        {KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp4x8_qsi4c32p4x8_16x4x32,
         {.get_m_step = kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
          .get_n_step = kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
          .get_mr = kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
          .get_nr = kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
          .get_kr = kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
          .get_sr = kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
          .get_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
          .get_rhs_packed_offset = kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
          .get_dst_offset = kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
          .get_dst_size = kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
          .run_matmul = kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm}},
        {KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp4x8_qsi4c32p8x8_4x8x32,
         {.get_m_step = kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
          .get_n_step = kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
          .get_mr = kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
          .get_nr = kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
          .get_kr = kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
          .get_sr = kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
          .get_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
          .get_rhs_packed_offset = kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
          .get_dst_offset = kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
          .get_dst_size = kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
          .run_matmul = kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm}},
        {KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp1x4_qsi4c32p4x4_1x4,
         {.get_m_step = kai_get_m_step_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
          .get_n_step = kai_get_n_step_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
          .get_mr = kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
          .get_nr = kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
          .get_kr = kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
          .get_sr = kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
          .get_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
          .get_rhs_packed_offset = kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
          .get_dst_offset = kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
          .get_dst_size = kai_get_dst_size_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
          .run_matmul = kai_run_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod}}};

void KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::quant_nxk_qs4c32_f32(size_t n, size_t k, size_t bl, const float* rhs_f32,
                                                                  uint8_t* rhs_qs4c32, uint16_t* rhs_scales_bf16) {
  constexpr int INT4_MIN = -8;
  constexpr int INT4_MAX = 7;

  const size_t num_blocks_row = get_num_blocks_per_row(k, bl);
  const size_t rhs_qs4c32_stride = get_rhs_native_stride(k);

  // Make sure the output is filled with zeros
  std::memset(rhs_qs4c32, 0, n * rhs_qs4c32_stride);

  for (size_t row_idx = 0; row_idx < n; ++row_idx) {
    const float* src_ptr = rhs_f32 + row_idx * k;

    for (size_t block_idx = 0; block_idx < num_blocks_row; ++block_idx) {
      float amax = 0.0f;
      float max = 0.0f;

      for (size_t b = 0; b < bl; ++b) {
        const size_t k_idx = block_idx * bl + b;

        if (k_idx >= k) { break; }

        const float src0_0 = src_ptr[k_idx];
        const float asrc0_0 = fabsf(src0_0);

        if (amax < asrc0_0) {
          amax = asrc0_0;
          max = src0_0;
        }
      }

      const float scale = max / -8.0;
      const float recip_scale = scale ? 1.0f / scale : 0.0f;

      // Store the scale in the dedicated buffer
      *rhs_scales_bf16 = kai_cast_bf16_f32(scale);

      rhs_scales_bf16 += 1;

      for (size_t i = 0; i < bl; ++i) {
        const size_t k_idx = block_idx * bl + i;

        if (k_idx >= k) { break; }

        const float src0_0 = src_ptr[k_idx];

        // Scale the values
        int32_t v0_s32 = (int32_t)(round(src0_0 * recip_scale));

        // Maximum/minimum int4 values
        v0_s32 = std::max(v0_s32, INT4_MIN);
        v0_s32 = std::min(v0_s32, INT4_MAX);

        const uint8_t v0_u8 = (uint8_t)(v0_s32 + 8);

        const size_t dst_addr = (k_idx / 2) + row_idx * rhs_qs4c32_stride;
        uint8_t rhs_v0 = rhs_qs4c32[dst_addr];

        if ((k_idx % 2) == 0) {
          rhs_v0 = v0_u8;
        } else {
          rhs_v0 |= (v0_u8 << 4);
        }

        rhs_qs4c32[dst_addr] = rhs_v0;
      }
    }
  }
}

size_t KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::workspace_size(int M, int K,
                                                              KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles tile_cfg) {
  const size_t mr = ukernels_[tile_cfg].get_mr();
  const size_t kr = ukernels_[tile_cfg].get_kr();
  const size_t sr = ukernels_[tile_cfg].get_sr();

  return kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(M, K, mr, kr, sr);
}

size_t KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::quant_pack_rhs_size(int N, int K,
                                                                   KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles tile_cfg) {
  const size_t nr = ukernels_[tile_cfg].get_nr();
  const size_t kr = ukernels_[tile_cfg].get_kr();
  const size_t sr = ukernels_[tile_cfg].get_sr();
  return kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(N, K, nr, kr, sr, 32, kai_dt_bf16);
}

void KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::quant_pack_rhs_offline(uint8_t* __restrict__ packed_weight,
                                                                    const float* __restrict__ rhs,
                                                                    const float* __restrict__ bias, int N, int K,
                                                                    KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles tile_cfg) {
  // meta info
  const size_t rhs_native_size_f32 = N * K * sizeof(float);
  const size_t rhs_native_size_qs4c32 = N * get_rhs_native_stride(K);
  const size_t rhs_scales_size_bf16 = N * get_rhs_scale_stride(K, 32);

  const size_t nr = ukernels_[tile_cfg].get_nr();
  const size_t kr = ukernels_[tile_cfg].get_kr();
  const size_t sr = ukernels_[tile_cfg].get_sr();

  uint8_t* rhs_qs4c32 = new uint8_t[rhs_native_size_qs4c32];
  uint8_t* rhs_scales_bf16 = new uint8_t[rhs_scales_size_bf16];

  // quant
  quant_nxk_qs4c32_f32(N, K, 32, rhs, rhs_qs4c32, (uint16_t*)rhs_scales_bf16);

  // pack
  kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0_params params;
  params.lhs_zero_point = 1;
  params.rhs_zero_point = 8;
  params.scale_dt = kai_dt_bf16;
  kai_run_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(1, N, K,                       // Dimensions
                                            nr, kr, sr,                    // Packing arguments
                                            32,                            // Block length
                                            (const uint8_t*)(rhs_qs4c32),  // RHS
                                            get_rhs_native_stride(K),      // RHS stride
                                            bias,                          // Bias
                                            rhs_scales_bf16,               // Scale
                                            get_rhs_scale_stride(K, 32),   // Scale stride
                                            packed_weight,                 // RHS packed
                                            0, &params);

  delete[] rhs_qs4c32;
  delete[] rhs_scales_bf16;
}

void KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::matmul(float* __restrict__ dst, const float* __restrict__ lhs_fp32,
                                                    const uint8_t* packed_weight_bias, void* workspace, int M, int K, int N,
                                                    KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles tile_cfg, int thread_count) {
  const size_t nr = ukernels_[tile_cfg].get_nr();
  const size_t kr = ukernels_[tile_cfg].get_kr();
  const size_t sr = ukernels_[tile_cfg].get_sr();
  const size_t mr = ukernels_[tile_cfg].get_mr();

  kai_run_lhs_quant_pack_qai8dxp_f32(M, K,                    // Dimensions
                                     mr, kr, sr, 0,           // Packing arguments
                                     (const float*)lhs_fp32,  // LHS
                                     K * sizeof(float),       // LHS stride
                                     workspace);              // LHS packed

  // matmul
  {
    const size_t dst_stride = N * sizeof(float);
    const int m_step = ukernels_[tile_cfg].get_m_step();  // Scheduling along M
    const int n_step = ukernels_[tile_cfg].get_n_step();  // Scheduling along N

    std::vector<std::pair<int, int>> tile_splits;
    for (int i_m_step = 0; i_m_step < M; i_m_step += m_step) {
      for (int i_n_step = 0; i_n_step < N; i_n_step += n_step) { tile_splits.emplace_back(i_m_step, i_n_step); }
    }
    auto tile_sizes = tile_splits.size();

    MLLM_CONDITIONAL_PARALLEL_FOR(thread_count > 1, thread_count, tile_idx, 0, tile_sizes, 1, {
      auto [i_m_step, i_n_step] = tile_splits[tile_idx];

      // Support functions return offset in bytes
      const void* lhs_ptr = (const void*)((const char*)workspace + (ukernels_[tile_cfg].get_lhs_packed_offset(i_m_step, K)));
      const void* rhs_ptr =
          (const void*)((const char*)packed_weight_bias + (ukernels_[tile_cfg].get_rhs_packed_offset(i_n_step, K, 32)));
      float* dst_ptr = (float*)((uint8_t*)dst + (ukernels_[tile_cfg].get_dst_offset(i_m_step, i_n_step, dst_stride)));

      const int actual_m = std::min(M - i_m_step, m_step);
      const int actual_n = std::min(N - i_n_step, n_step);

      ukernels_[tile_cfg].run_matmul(actual_m, actual_n, K,  // Dimensions
                                     32,                     // Block length
                                     lhs_ptr,                // LHS packed
                                     rhs_ptr,                // RHS packed
                                     dst_ptr,                // DST
                                     dst_stride,             // DST stride (row)
                                     sizeof(float),          // DST stride (col)
                                     -std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    });

    // MLLM_CONDITIONAL_PARALLEL_FOR(thread_count > 1, thread_count, i_m_step, 0, M, m_step, {
    //   for (int i_n_step = 0; i_n_step < N; i_n_step += n_step) {
    //     // Support functions return offset in bytes
    //     const void* lhs_ptr = (const void*)((const char*)workspace + (ukernels_[tile_cfg].get_lhs_packed_offset(i_m_step,
    //     K))); const void* rhs_ptr =
    //         (const void*)((const char*)packed_weight_bias + (ukernels_[tile_cfg].get_rhs_packed_offset(i_n_step, K, 32)));
    //     float* dst_ptr = (float*)((uint8_t*)dst + (ukernels_[tile_cfg].get_dst_offset(i_m_step, i_n_step, dst_stride)));

    //     const int actual_m = std::min(M - (int)i_m_step, m_step);
    //     const int actual_n = std::min(N - i_n_step, n_step);

    //     ukernels_[tile_cfg].run_matmul(actual_m, actual_n, K,  // Dimensions
    //                                    32,                     // Block length
    //                                    lhs_ptr,                // LHS packed
    //                                    rhs_ptr,                // RHS packed
    //                                    dst_ptr,                // DST
    //                                    dst_stride,             // DST stride (row)
    //                                    sizeof(float),          // DST stride (col)
    //                                    -std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    //   }
    // });
  }
}

std::unordered_map<KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::Tiles, kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel>
    KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::ukernels_ = {
        {KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::Tiles::qai8dxp1x8_qsi4c32p4x8_1x4x32,
         {.get_m_step = kai_get_m_step_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_n_step = kai_get_n_step_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_mr = kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_nr = kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_kr = kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_sr = kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_rhs_packed_offset = kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_dst_offset = kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .get_dst_size = kai_get_dst_size_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
          .run_matmul = kai_run_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod

         }},
        {KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::Tiles::qai8dxp1x8_qsi4c32p8x8_1x8x32,
         {.get_m_step = kai_get_m_step_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
          .get_n_step = kai_get_n_step_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
          .get_mr = kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
          .get_nr = kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
          .get_kr = kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
          .get_sr = kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
          .get_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
          .get_rhs_packed_offset = kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
          .get_dst_offset = kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
          .get_dst_size = kai_get_dst_size_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
          .run_matmul = kai_run_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod}},
        {KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::Tiles::qai8dxp4x8_qsi4c32p4x8_8x4x32,
         {.get_m_step = kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
          .get_n_step = kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
          .get_mr = kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
          .get_nr = kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
          .get_kr = kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
          .get_sr = kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
          .get_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
          .get_rhs_packed_offset = kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
          .get_dst_offset = kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
          .get_dst_size = kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
          .run_matmul = kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm}},
        {KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::Tiles::qai8dxp4x8_qsi4c32p4x8_16x4x32,
         {.get_m_step = kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
          .get_n_step = kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
          .get_mr = kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
          .get_nr = kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
          .get_kr = kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
          .get_sr = kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
          .get_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
          .get_rhs_packed_offset = kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
          .get_dst_offset = kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
          .get_dst_size = kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
          .run_matmul = kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm}},
        {KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::Tiles::qai8dxp4x8_qsi4c32p8x8_4x8x32,
         {.get_m_step = kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
          .get_n_step = kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
          .get_mr = kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
          .get_nr = kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
          .get_kr = kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
          .get_sr = kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
          .get_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
          .get_rhs_packed_offset = kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
          .get_dst_offset = kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
          .get_dst_size = kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
          .run_matmul = kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm}},
        {KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::Tiles::qai8dxp1x4_qsi4c32p4x4_1x4,
         {.get_m_step = kai_get_m_step_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
          .get_n_step = kai_get_n_step_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
          .get_mr = kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
          .get_nr = kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
          .get_kr = kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
          .get_sr = kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
          .get_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
          .get_rhs_packed_offset = kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
          .get_dst_offset = kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
          .get_dst_size = kai_get_dst_size_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
          .run_matmul = kai_run_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod}}};

size_t KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::workspace_size(int M, int K,
                                                              KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::Tiles tile_cfg) {
  const size_t mr = ukernels_[tile_cfg].get_mr();
  const size_t kr = ukernels_[tile_cfg].get_kr();
  const size_t sr = ukernels_[tile_cfg].get_sr();

  return kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(M, K, mr, kr, sr);
}

size_t KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::quant_pack_rhs_size(int K, int N,
                                                                   KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::Tiles tile_cfg) {
  const size_t nr = ukernels_[tile_cfg].get_nr();
  const size_t kr = ukernels_[tile_cfg].get_kr();
  const size_t sr = ukernels_[tile_cfg].get_sr();

  return kai_get_rhs_packed_size_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0(N, K, nr, kr, sr, 32, kai_dt_bf16);
}

void KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::quant_pack_rhs_offline(uint8_t* __restrict__ packed_weight,
                                                                    const float* __restrict__ rhs,
                                                                    const float* __restrict__ bias, int K, int N,
                                                                    KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::Tiles tile_cfg) {
  // meta info
  const size_t rhs_native_size_f32 = N * K * sizeof(float);
  const size_t rhs_native_size_qs4c32 = N * get_rhs_native_stride(K);
  const size_t rhs_scales_size_bf16 = N * get_rhs_scale_stride(K, 32);

  const size_t nr = ukernels_[tile_cfg].get_nr();
  const size_t kr = ukernels_[tile_cfg].get_kr();
  const size_t sr = ukernels_[tile_cfg].get_sr();

  uint8_t* rhs_qs4c32 = new uint8_t[rhs_native_size_qs4c32];
  uint8_t* rhs_scales_bf16 = new uint8_t[rhs_scales_size_bf16];

  // quant
  quant_kxn_qs4c32_f32(N, K, 32, rhs, rhs_qs4c32, (uint16_t*)rhs_scales_bf16);

  // pack
  kai_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0_params params;
  params.lhs_zero_point = 1;
  params.rhs_zero_point = 8;
  params.scale_dt = kai_dt_bf16;
  kai_run_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0(1, N, K,                       // Dimensions
                                            nr, kr, sr,                    // Packing arguments
                                            32,                            // Block length
                                            (const uint8_t*)(rhs_qs4c32),  // RHS
                                            get_rhs_native_stride(K),      // RHS stride
                                            bias,                          // Bias
                                            rhs_scales_bf16,               // Scale
                                            get_rhs_scale_stride(K, 32),   // Scale stride
                                            packed_weight,                 // RHS packed
                                            0, &params);

  delete[] rhs_qs4c32;
  delete[] rhs_scales_bf16;
}

void KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::matmul(float* __restrict__ dst, const float* __restrict__ lhs_fp32,
                                                    const uint8_t* packed_weight_bias, void* workspace, int M, int K, int N,
                                                    KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::Tiles tile_cfg, int thread_count) {
  const size_t nr = ukernels_[tile_cfg].get_nr();
  const size_t kr = ukernels_[tile_cfg].get_kr();
  const size_t sr = ukernels_[tile_cfg].get_sr();
  const size_t mr = ukernels_[tile_cfg].get_mr();

  kai_run_lhs_quant_pack_qai8dxp_f32(M, K,                    // Dimensions
                                     mr, kr, sr, 0,           // Packing arguments
                                     (const float*)lhs_fp32,  // LHS
                                     K * sizeof(float),       // LHS stride
                                     workspace);              // LHS packed

  // matmul
  {
    const size_t dst_stride = N * sizeof(float);
    const int m_step = ukernels_[tile_cfg].get_m_step();  // Scheduling along M
    const int n_step = ukernels_[tile_cfg].get_n_step();  // Scheduling along N

    MLLM_CONDITIONAL_PARALLEL_FOR(thread_count > 1, thread_count, i_m_step, 0, M, m_step, {
      for (int i_n_step = 0; i_n_step < N; i_n_step += n_step) {
        // Support functions return offset in bytes
        const void* lhs_ptr = (const void*)((const char*)workspace + (ukernels_[tile_cfg].get_lhs_packed_offset(i_m_step, K)));
        const void* rhs_ptr =
            (const void*)((const char*)packed_weight_bias + (ukernels_[tile_cfg].get_rhs_packed_offset(i_n_step, K, 32)));
        float* dst_ptr = (float*)((uint8_t*)dst + (ukernels_[tile_cfg].get_dst_offset(i_m_step, i_n_step, dst_stride)));

        const int actual_m = std::min(M - (int)i_m_step, m_step);
        const int actual_n = std::min(N - i_n_step, n_step);

        ukernels_[tile_cfg].run_matmul(actual_m, actual_n, K,  // Dimensions
                                       32,                     // Block length
                                       lhs_ptr,                // LHS packed
                                       rhs_ptr,                // RHS packed
                                       dst_ptr,                // DST
                                       dst_stride,             // DST stride (row)
                                       sizeof(float),          // DST stride (col)
                                       -std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
      }
    });
  }
}

void KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::quant_kxn_qs4c32_f32(size_t n, size_t k, size_t bl, const float* rhs_f32,
                                                                  uint8_t* rhs_qs4c32, uint16_t* rhs_scales_bf16) {
  constexpr int INT4_MIN = -8;
  constexpr int INT4_MAX = 7;

  const size_t num_blocks_row = get_num_blocks_per_row(k, bl);
  const size_t rhs_qs4c32_stride = get_rhs_native_stride(n);

  // Make sure the output is filled with zeros
  std::memset(rhs_qs4c32, 0, k * rhs_qs4c32_stride);

  for (size_t row_idx = 0; row_idx < n; ++row_idx) {
    const float* src_ptr = rhs_f32 + row_idx * k;

    for (size_t block_idx = 0; block_idx < num_blocks_row; ++block_idx) {
      float amax = 0.0f;
      float max = 0.0f;

      for (size_t b = 0; b < bl; ++b) {
        const size_t k_idx = block_idx * bl + b;

        if (k_idx >= k) { break; }

        const float src0_0 = src_ptr[k_idx];
        const float asrc0_0 = fabsf(src0_0);

        if (amax < asrc0_0) {
          amax = asrc0_0;
          max = src0_0;
        }
      }

      const float scale = max / -8.0;
      const float recip_scale = scale ? 1.0f / scale : 0.0f;

      // Store the scale in the dedicated buffer
      *rhs_scales_bf16 = kai_cast_bf16_f32(scale);

      rhs_scales_bf16 += 1;

      for (size_t i = 0; i < bl; ++i) {
        const size_t k_idx = block_idx * bl + i;

        if (k_idx >= k) { break; }

        const float src0_0 = src_ptr[k_idx];

        // Scale the values
        int32_t v0_s32 = (int32_t)(round(src0_0 * recip_scale));

        // Maximum/minimum int4 values
        v0_s32 = std::max(v0_s32, INT4_MIN);
        v0_s32 = std::min(v0_s32, INT4_MAX);

        const uint8_t v0_u8 = (uint8_t)(v0_s32 + 8);

        const size_t dst_addr = (row_idx / 2) + k_idx * rhs_qs4c32_stride;
        uint8_t rhs_v0 = rhs_qs4c32[dst_addr];

        if ((row_idx % 2) == 0) {
          rhs_v0 = v0_u8;
        } else {
          rhs_v0 |= (v0_u8 << 4);
        }

        rhs_qs4c32[dst_addr] = rhs_v0;
      }
    }
  }
}

size_t KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk::workspace_size(int M, int K, DataTypes lhs_dtype,
                                                               KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk::Tiles tile_cfg) {
  const size_t mr = ukernels_[tile_cfg].get_mr();
  const size_t kr = ukernels_[tile_cfg].get_kr();
  const size_t sr = ukernels_[tile_cfg].get_sr();

  // Pack bl size per block along K axis. Will generate k / bl blocks per line.
  switch (lhs_dtype) {
    case kFloat16:
      // FIXME:
      // is bl=32 ok?
      return kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32pscalef32_f16_neon(M, K, 32, mr, kr, sr);
    case kFloat32:
      // FIXME:
      // is bl=32 ok?
      return kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32pscalef32_f32_neon(M, K, 32, mr, kr, sr);
    default: MLLM_ERROR_EXIT(ExitCode::kCoreError, "Unsupported lhs_dtype");
  }
}

size_t KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk::quant_pack_rhs_size(int N, int K,
                                                                    KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk::Tiles tile_cfg) {
  const size_t nr = ukernels_[tile_cfg].get_nr();
  const size_t kr = ukernels_[tile_cfg].get_kr();
  const size_t sr = ukernels_[tile_cfg].get_sr();

  return kai_get_rhs_packed_size_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon(N, K, nr, kr, 32);
}

void KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk::quant_nxk_qai4c32_f32(size_t n, size_t k, size_t bl, const float* rhs_f32,
                                                                    uint8_t* rhs_qai4c32, float* rhs_zeros_fp32,
                                                                    float* rhs_scales_fp32) {
  constexpr int INT4_MIN = -8;
  constexpr int INT4_MAX = 7;
  constexpr float INT4_MIN_IN_FP32 = -8;
  constexpr float INT4_MAX_IN_FP32 = 7;

  const size_t num_blocks_row = get_num_blocks_per_row(k, bl);
  const size_t rhs_qs4c32_stride = get_rhs_native_stride(k);

  auto shadow_rhs_zeros_fp32 = rhs_zeros_fp32;
  auto shadow_rhs_scales_fp32 = rhs_scales_fp32;

  // Make sure the output is filled with zeros
  std::memset(rhs_qai4c32, 0, n * rhs_qs4c32_stride);

  // Loop on rows.
  for (size_t row_idx = 0; row_idx < n; ++row_idx) {
    const float* src_ptr = rhs_f32 + row_idx * k;

    // Loop each blocks in this row.
    for (size_t block_idx = 0; block_idx < num_blocks_row; ++block_idx) {
      auto this_block_min_value = std::numeric_limits<float>::max();
      auto this_block_max_value = std::numeric_limits<float>::lowest();

      // Find min and max
      for (size_t b = 0; b < bl; ++b) {
        const size_t x = block_idx * bl + b;
        if (x >= k) { break; }

        const auto value = src_ptr[x];
        this_block_min_value = std::min(this_block_min_value, value);
        this_block_max_value = std::max(this_block_max_value, value);
      }

      // Find zero point and scale point
      if (this_block_min_value > 0) { this_block_min_value = 0; }
      if (this_block_max_value < 0) { this_block_max_value = 0; }

      const float inv_scale = this_block_max_value != this_block_min_value
                                  ? (INT4_MAX_IN_FP32 - INT4_MIN_IN_FP32) / (this_block_max_value - this_block_min_value)
                                  : 1.0F;
      const float scale = 1.0F / inv_scale;
      const float scaled_min = this_block_min_value / scale;
      const float scaled_max = this_block_max_value / scale;
      const float zero_point_f = -(scaled_min + INT4_MIN_IN_FP32) < scaled_max + INT4_MAX_IN_FP32
                                     ? scaled_min - INT4_MIN_IN_FP32
                                     : scaled_max - INT4_MAX_IN_FP32;
      const int32_t zero_point = -round_nearest_from_fp32_2_int32(zero_point_f);

      // Store the scale and zero point in the dedicated buffer
      *shadow_rhs_zeros_fp32 = zero_point;
      *shadow_rhs_scales_fp32 = scale;
      shadow_rhs_zeros_fp32 += 1;
      shadow_rhs_scales_fp32 += 1;

      // Do scale and quant
      for (size_t i = 0; i < bl; ++i) {
        const size_t x = block_idx * bl + i;
        if (x >= k) { break; }

        const auto value_f = src_ptr[x];
        const auto this_inv_scale = scale != 0 ? 1.0F / scale : 0.0F;
        int32_t value_q_i32 = round_nearest_from_fp32_2_int32(value_f * this_inv_scale) + zero_point;

        value_q_i32 = std::max(value_q_i32, INT4_MIN);
        value_q_i32 = std::min(value_q_i32, INT4_MAX);

        // Convert to unsigned, so that we can move bits
        uint8_t value_q_u8 = (uint8_t)(value_q_i32 + 8);

        // Combine s0(int4) and s1(int4) into one int8 value
        const size_t dst_addr = (x / 2) + row_idx * rhs_qs4c32_stride;
        uint8_t rhs_v0 = rhs_qai4c32[dst_addr];
        if ((x % 2) == 0) {
          rhs_v0 = value_q_u8;
        } else {
          rhs_v0 |= (value_q_u8 << 4);
        }
        rhs_qai4c32[dst_addr] = rhs_v0;
      }
    }
  }

  // rescale zero point
  int zeros_scales_block_cnt = 0;
  for (size_t row_idx = 0; row_idx < n; ++row_idx) {
    for (size_t block_idx = 0; block_idx < num_blocks_row; ++block_idx) {
      rhs_zeros_fp32[zeros_scales_block_cnt] =
          (-rhs_zeros_fp32[zeros_scales_block_cnt]) * rhs_scales_fp32[zeros_scales_block_cnt];
      zeros_scales_block_cnt++;
    }
  }

  // The value in rhs_qai4c32 is s1 s0 right now. We need to change it to s0 s1
  for (int idx = 0; idx < n * rhs_qs4c32_stride; ++idx) {
    uint8_t v = rhs_qai4c32[idx];
    uint8_t v_low = v & 0x0F;  // 0000 1111
    uint8_t v_high = v >> 4;   // 1111 0000
    uint8_t v_s0s1 = (v_low << 4) | v_high;
    rhs_qai4c32[idx] = v_s0s1;
  }
}

void KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk::quant_pack_rhs_offline(uint8_t* __restrict__ packed_weight,
                                                                     const float* __restrict__ rhs,
                                                                     const float* __restrict__ bias, int K, int N,
                                                                     KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk::Tiles tile_cfg) {
  // meta info
  const size_t rhs_native_size_f32 = N * K * sizeof(float);
  const size_t rhs_native_size_qs4c32 = N * get_rhs_native_stride(K);
  const size_t rhs_scales_size_fp32 = N * get_rhs_scale_stride(K, 32);

  const size_t nr = ukernels_[tile_cfg].get_nr();
  const size_t kr = ukernels_[tile_cfg].get_kr();
  const size_t sr = ukernels_[tile_cfg].get_sr();

  uint8_t* rhs_qai4c32 = new uint8_t[rhs_native_size_qs4c32];
  uint8_t* rhs_zeros_fp32 = new uint8_t[rhs_scales_size_fp32];
  uint8_t* rhs_scales_fp32 = new uint8_t[rhs_scales_size_fp32];

  quant_nxk_qai4c32_f32(N, K, 32, rhs, rhs_qai4c32, (float*)rhs_zeros_fp32, (float*)rhs_scales_fp32);

  kai_rhs_pack_nxk_qai4c32p_params params;
  params.lhs_zero_point = 1;
  params.rhs_zero_point = 8;

  // Use kai to pack
  kai_run_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon(1, N, K, nr, kr, sr, 32, rhs_qai4c32, rhs_zeros_fp32, bias,
                                                             rhs_scales_fp32, packed_weight, 0, &params);

  delete[] rhs_qai4c32;
  delete[] rhs_zeros_fp32;
  delete[] rhs_scales_fp32;
}

void KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk::matmul(float16_t* __restrict__ dst, const float* __restrict__ lhs_fp32,
                                                     const uint8_t* packed_weight_bias, void* workspace, int M, int K, int N,
                                                     KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk::Tiles tile_cfg,
                                                     int thread_count) {
  const size_t nr = ukernels_[tile_cfg].get_nr();
  const size_t kr = ukernels_[tile_cfg].get_kr();
  const size_t sr = ukernels_[tile_cfg].get_sr();
  const size_t mr = ukernels_[tile_cfg].get_mr();

  // FIXME:
  // Using m_idx_start ? Fuse this packing to run_matmul loops with m_idx_start.
  kai_run_lhs_quant_pack_qsi8d32pscalef32_f32_neon(M, K, 32, mr, kr, sr, 0, lhs_fp32, K * sizeof(float), workspace);

  // matmul
  {
    const size_t dst_stride = N * sizeof(float16_t);
    const int m_step = ukernels_[tile_cfg].get_m_step();  // Scheduling along M
    const int n_step = ukernels_[tile_cfg].get_n_step();  // Scheduling along N

    MLLM_CONDITIONAL_PARALLEL_FOR(thread_count > 1, thread_count, i_m_step, 0, M, m_step, {
      for (int i_n_step = 0; i_n_step < N; i_n_step += n_step) {
        // Support functions return offset in bytes
        const void* lhs_ptr =
            (const void*)((const char*)workspace + (ukernels_[tile_cfg].get_lhs_packed_offset(i_m_step, K, 32)));
        const void* rhs_ptr =
            (const void*)((const char*)packed_weight_bias + (ukernels_[tile_cfg].get_rhs_packed_offset(i_n_step, K, 32)));
        float16_t* dst_ptr = (float16_t*)((uint8_t*)dst + (ukernels_[tile_cfg].get_dst_offset(i_m_step, i_n_step, dst_stride)));

        const int actual_m = std::min(M - (int)i_m_step, m_step);
        const int actual_n = std::min(N - i_n_step, n_step);

        ukernels_[tile_cfg].run_matmul(actual_m, actual_n, K,  // Dimensions
                                       32,                     // Block length
                                       lhs_ptr,                // LHS packed
                                       rhs_ptr,                // RHS packed
                                       dst_ptr,                // DST
                                       dst_stride,             // DST stride (row)
                                       sizeof(float16_t),      // DST stride (col)
                                       -65504, 65504);
      }
    });
  }
}

void KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk::matmul(float16_t* __restrict__ dst, const float16_t* __restrict__ lhs_fp16,
                                                     const uint8_t* packed_weight_bias, void* workspace, int M, int K, int N,
                                                     KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk::Tiles tile_cfg,
                                                     int thread_count) {
  const size_t nr = ukernels_[tile_cfg].get_nr();
  const size_t kr = ukernels_[tile_cfg].get_kr();
  const size_t sr = ukernels_[tile_cfg].get_sr();
  const size_t mr = ukernels_[tile_cfg].get_mr();

  // FIXME:
  // Using m_idx_start ? Fuse this packing to run_matmul loops with m_idx_start.
  kai_run_lhs_quant_pack_qsi8d32pscalef32_f16_neon(M, K, 32, mr, kr, sr, 0, lhs_fp16, K * sizeof(float16_t), workspace);

  // matmul
  {
    const size_t dst_stride = N * sizeof(float16_t);
    const int m_step = ukernels_[tile_cfg].get_m_step();  // Scheduling along M
    const int n_step = ukernels_[tile_cfg].get_n_step();  // Scheduling along N

    MLLM_CONDITIONAL_PARALLEL_FOR(thread_count > 1, thread_count, i_m_step, 0, M, m_step, {
      for (int i_n_step = 0; i_n_step < N; i_n_step += n_step) {
        // Support functions return offset in bytes
        const void* lhs_ptr =
            (const void*)((const char*)workspace + (ukernels_[tile_cfg].get_lhs_packed_offset(i_m_step, K, 32)));
        const void* rhs_ptr =
            (const void*)((const char*)packed_weight_bias + (ukernels_[tile_cfg].get_rhs_packed_offset(i_n_step, K, 32)));
        float16_t* dst_ptr = (float16_t*)((uint8_t*)dst + (ukernels_[tile_cfg].get_dst_offset(i_m_step, i_n_step, dst_stride)));

        const int actual_m = std::min(M - (int)i_m_step, m_step);
        const int actual_n = std::min(N - i_n_step, n_step);

        ukernels_[tile_cfg].run_matmul(actual_m, actual_n, K,  // Dimensions
                                       32,                     // Block length
                                       lhs_ptr,                // LHS packed
                                       rhs_ptr,                // RHS packed
                                       dst_ptr,                // DST
                                       dst_stride,             // DST stride (row)
                                       sizeof(float16_t),      // DST stride (col)
                                       -std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
      }
    });
  }
}

std::unordered_map<KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk::Tiles, kai_matmul_clamp_f16_qsi8d32p_qai4c32p_ukernel>
    KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk::ukernels_ = {
        {KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk::Tiles::qsi8d32p1x8_qai4c32p4x8_1x4,
         {.get_m_step = kai_get_m_step_matmul_clamp_f16_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod,
          .get_n_step = kai_get_n_step_matmul_clamp_f16_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod,
          .get_mr = kai_get_mr_matmul_clamp_f16_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod,
          .get_nr = kai_get_nr_matmul_clamp_f16_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod,
          .get_kr = kai_get_kr_matmul_clamp_f16_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod,
          .get_sr = kai_get_sr_matmul_clamp_f16_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod,
          .get_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_f16_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod,
          .get_rhs_packed_offset = kai_get_rhs_packed_offset_matmul_clamp_f16_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod,
          .get_dst_offset = kai_get_dst_offset_matmul_clamp_f16_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod,
          .get_dst_size = kai_get_dst_size_matmul_clamp_f16_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod,
          .run_matmul = kai_run_matmul_clamp_f16_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod}},
        {KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk::Tiles::qsi8d32p4x4_qai4c32p4x4_8x4,
         {.get_m_step = nullptr,
          .get_n_step = nullptr,
          .get_mr = nullptr,
          .get_nr = nullptr,
          .get_kr = nullptr,
          .get_sr = nullptr,
          .get_lhs_packed_offset = nullptr,
          .get_rhs_packed_offset = nullptr,
          .get_dst_offset = nullptr,
          .get_dst_size = nullptr,
          .run_matmul = nullptr}},
        {KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk::Tiles::qsi8d32p4x8_qai4c32p4x8_8x4_i8mm,
         {.get_m_step = kai_get_m_step_matmul_clamp_f16_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm,
          .get_n_step = kai_get_n_step_matmul_clamp_f16_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm,
          .get_mr = kai_get_mr_matmul_clamp_f16_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm,
          .get_nr = kai_get_nr_matmul_clamp_f16_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm,
          .get_kr = kai_get_kr_matmul_clamp_f16_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm,
          .get_sr = kai_get_sr_matmul_clamp_f16_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm,
          .get_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_f16_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm,
          .get_rhs_packed_offset = kai_get_rhs_packed_offset_matmul_clamp_f16_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm,
          .get_dst_offset = kai_get_dst_offset_matmul_clamp_f16_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm,
          .get_dst_size = kai_get_dst_size_matmul_clamp_f16_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm,
          .run_matmul = kai_run_matmul_clamp_f16_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm}}};

}  // namespace mllm::cpu::arm
