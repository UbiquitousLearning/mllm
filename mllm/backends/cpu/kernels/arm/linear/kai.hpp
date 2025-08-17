// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

// TERM is same with kleidiai:
// see https://gitlab.arm.com/kleidi/kleidiai/-/blob/main/kai/ukernels/matmul/README.md
// f32 -  Floating-point 32-bit
// q : Quantized
// s :  Symmetric
// a : Asymmetric
// i : Signed integer
// u : Unsigned integer
// 4 : 4-bit Quantized
// 8 : 8-bit Quantized
// dx : Per dimension quantization
// cx : Per channel quantization
// c32 : Per block quantization, with block length multiple of 32 scale
// f16 : Scale factors are stores as floating-point 16-bit
// p : Matrix is packed
//
// e.g.:
// qsi4cxp :
//      qs - Quantized symmetric
//      i4 - Signed Integer 4-bit
//      cx - Per channel quantized
//      p - packed
// Some other examples :
//      s16s0 - Packing order of data is interleaved
//      s1s0 - Packing order of data is sequential
//      fp16 - Floating-point 16-bit data type

#include <arm_neon.h>
#include <cstdint>
#include <unordered_map>

// for pack_kxn_fp16_w_bias

#include "kai_matmul_clamp_f16_f16_f16p_interface.h"

// for f32_qai8dxp_qsi4c32
#include "kai_matmul_clamp_f32_qai8dxp_qsi4c32p_interface.h"

// for f16_qsi8d32p_qai4c32p
#include "kai_matmul_clamp_f16_qsi8d32p_qai4c32p_interface.h"

// mllm
#include "mllm/core/DataTypes.hpp"

namespace mllm::cpu::arm {

/// WARN: Do not use This kai kernel has precision error
struct KaiLinear_fp16_fp16_fp16p_mxk_kxn {
  [[deprecated("This kernel has precision error, use other alternatives")]] inline bool need_pack_lhs() { return false; }

  [[deprecated("This kernel has precision error, use other alternatives")]] inline bool need_pack_rhs() { return true; }

  [[deprecated("This kernel has precision error, use other alternatives")]] size_t pack_rhs_size(int K, int N);

  [[deprecated("This kernel has precision error, use other alternatives")]] void pack_rhs_offline(
      float16_t* __restrict__ rhs_packed, const float16_t* __restrict__ rhs, const float16_t* bias, int K, int N);

  [[deprecated("This kernel has precision error, use other alternatives")]] void matmul(float16_t* __restrict__ dst,
                                                                                        const float16_t* __restrict__ lhs,
                                                                                        const float16_t* __restrict__ rhs,
                                                                                        int M, int K, int N, int thread_count);

 private:
  static kai_matmul_clamp_f16_f16_f16p_ukernel ukernel_;
};

struct KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk {
  enum class Tiles : uint8_t {
    qai8dxp1x8_qsi4c32p4x8_1x4x32,
    qai8dxp1x8_qsi4c32p8x8_1x8x32,
    qai8dxp4x8_qsi4c32p4x8_8x4x32,
    qai8dxp4x8_qsi4c32p4x8_16x4x32,
    qai8dxp4x8_qsi4c32p8x8_4x8x32,
    qai8dxp1x4_qsi4c32p4x4_1x4,
  };

  inline bool need_pack_lhs() { return true; }

  inline bool need_pack_rhs() { return true; }

  size_t workspace_size(int M, int K, KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles tile_cfg);

  size_t quant_pack_rhs_size(int N, int K, KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles tile_cfg);

  void quant_pack_rhs_offline(uint8_t* __restrict__ packed_weight, const float* __restrict__ rhs,
                              const float* __restrict__ bias, int N, int K,
                              KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles tile_cfg);

  void matmul(float* __restrict__ dst, const float* __restrict__ lhs_fp32, const uint8_t* packed_weight_bias, void* workspace,
              int M, int K, int N, KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles tile_cfg, int thread_count);

 private:
  void quant_nxk_qs4c32_f32(size_t n, size_t k, size_t bl, const float* rhs_f32, uint8_t* rhs_qs4c32,
                            uint16_t* rhs_scales_bf16);

  inline size_t roundup(size_t a, size_t b) { return ((a + b - 1) / b) * b; }

  inline size_t get_num_blocks_per_row(size_t k, size_t bl) { return roundup(k, bl) / bl; }

  inline size_t get_rhs_native_stride(size_t x) { return roundup(x, 2) / 2; }

  inline size_t get_rhs_scale_stride(size_t k, size_t bl) {
    const size_t num_blocks_per_row = get_num_blocks_per_row(k, bl);
    return num_blocks_per_row * sizeof(uint16_t);
  }

  static std::unordered_map<Tiles, kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel> ukernels_;
};

struct KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn {
  enum class Tiles : uint8_t {
    qai8dxp1x8_qsi4c32p4x8_1x4x32,
    qai8dxp1x8_qsi4c32p8x8_1x8x32,
    qai8dxp4x8_qsi4c32p4x8_8x4x32,
    qai8dxp4x8_qsi4c32p4x8_16x4x32,
    qai8dxp4x8_qsi4c32p8x8_4x8x32,
    qai8dxp1x4_qsi4c32p4x4_1x4,
  };

  inline bool need_pack_lhs() { return true; }

  inline bool need_pack_rhs() { return true; }

  size_t workspace_size(int M, int K, KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::Tiles tile_cfg);

  size_t quant_pack_rhs_size(int K, int N, KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::Tiles tile_cfg);

  void quant_pack_rhs_offline(uint8_t* __restrict__ packed_weight, const float* __restrict__ rhs,
                              const float* __restrict__ bias, int K, int N,
                              KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::Tiles tile_cfg);

  void matmul(float* __restrict__ dst, const float* __restrict__ lhs_fp32, const uint8_t* packed_weight_bias, void* workspace,
              int M, int K, int N, KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::Tiles tile_cfg, int thread_count);

 private:
  void quant_kxn_qs4c32_f32(size_t n, size_t k, size_t bl, const float* rhs_f32, uint8_t* rhs_qs4c32,
                            uint16_t* rhs_scales_bf16);

  inline size_t roundup(size_t a, size_t b) { return ((a + b - 1) / b) * b; }

  inline size_t get_num_blocks_per_row(size_t k, size_t bl) { return roundup(k, bl) / bl; }

  inline size_t get_rhs_native_stride(size_t x) { return roundup(x, 2) / 2; }

  inline size_t get_rhs_scale_stride(size_t k, size_t bl) {
    const size_t num_blocks_per_row = get_num_blocks_per_row(k, bl);
    return num_blocks_per_row * sizeof(uint16_t);
  }

  static std::unordered_map<Tiles, kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel> ukernels_;
};

/// WARN: Do not use. This kai kernel has precision error
struct KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk {
 public:
  enum class Tiles : uint8_t {
    qsi8d32p1x8_qai4c32p4x8_1x4,
    qsi8d32p4x4_qai4c32p4x4_8x4,
    qsi8d32p4x8_qai4c32p4x8_8x4_i8mm,
  };

  [[deprecated("This kernel has precision error, use other alternatives")]] inline bool need_pack_lhs() { return true; }

  [[deprecated("This kernel has precision error, use other alternatives")]] inline bool need_pack_rhs() { return true; }

  [[deprecated("This kernel has precision error, use other alternatives")]] size_t workspace_size(
      int M, int K, DataTypes lhs_dtype, KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk::Tiles tile_cfg);

  [[deprecated("This kernel has precision error, use other alternatives")]] size_t quant_pack_rhs_size(
      int N, int K, KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk::Tiles tile_cfg);

  [[deprecated("This kernel has precision error, use other alternatives")]] void quant_pack_rhs_offline(
      uint8_t* __restrict__ packed_weight, const float* __restrict__ rhs, const float* __restrict__ bias, int K, int N,
      KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk::Tiles tile_cfg);

  [[deprecated("This kernel has precision error, use other alternatives")]] void matmul(
      float16_t* __restrict__ dst, const float* __restrict__ lhs_fp32, const uint8_t* packed_weight_bias, void* workspace,
      int M, int K, int N, KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk::Tiles tile_cfg, int thread_count);

  [[deprecated("This kernel has precision error, use other alternatives")]] void matmul(
      float16_t* __restrict__ dst, const float16_t* __restrict__ lhs_fp16, const uint8_t* packed_weight_bias, void* workspace,
      int M, int K, int N, KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk::Tiles tile_cfg, int thread_count);

 private:
  inline int32_t round_nearest_from_fp32_2_int32(const float v) {
    int32_t result;
    asm volatile("fcvtns %w[res], %s[val]\n" : [res] "=r"(result) : [val] "w"(v));
    return result;
  }

  inline size_t roundup(size_t a, size_t b) { return ((a + b - 1) / b) * b; }

  inline size_t get_num_blocks_per_row(size_t k, size_t bl) { return roundup(k, bl) / bl; }

  inline size_t get_rhs_native_stride(size_t x) { return roundup(x, 2) / 2; }

  inline size_t get_rhs_scale_stride(size_t k, size_t bl) {
    // NOTE:
    // Scale should be float type.
    const size_t num_blocks_per_row = get_num_blocks_per_row(k, bl);
    return num_blocks_per_row * sizeof(float);
  }

  void quant_nxk_qai4c32_f32(size_t n, size_t k, size_t bl, const float* rhs_f32, uint8_t* rhs_qai4c32, float* rhs_zeros_fp32,
                             float* rhs_scales_fp32);

  static std::unordered_map<Tiles, kai_matmul_clamp_f16_qsi8d32p_qai4c32p_ukernel> ukernels_;
};

}  // namespace mllm::cpu::arm
